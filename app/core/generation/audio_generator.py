"""SA3 inference engine.

Thin wrapper around stable_audio_3.StableAudioModel.from_pretrained() that
caches the loaded model between requests (eviction on model_id change),
auto-detects the device, and writes 44.1 kHz stereo int16 WAV.

Cancellation is wired via `request_stop()` for API parity, but SA3's
generate() doesn't expose a per-step callback yet — the flag is checked
between calls, not inside them. A finer-grained cancel hook is a Phase
3.1 follow-up.
"""
import math
import os
import platform
import re
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from utils.logger import get_logger

logger = get_logger("AudioGenerator")


# Live progress from the SA3 sampler. SA3's `model.generate(**sampler_kwargs)`
# forwards `callback=fn` into the sampler, which fires it per ODE step with
# `{'i': step_index, ...}`. We mirror that into this dict so the frontend can
# poll real progress instead of a fake ticker. Reset on each new generation.
_generation_state: Dict[str, Any] = {
    "is_generating": False,
    # idle | loading | sampling | loop_stitching | decoding | complete | failed
    "phase": "idle",
    "step": 0,
    "total_steps": 0,
    "progress": 0,          # 0-100, derived
    "batch_index": 0,
    "batch_total": 0,
    "started_at": None,
    "ended_at": None,
    "error": None,
}
_generation_state_lock = threading.Lock()


def get_generation_progress() -> Dict[str, Any]:
    """Snapshot of the current generation's live progress. Cheap to call."""
    with _generation_state_lock:
        return dict(_generation_state)


def _set_progress(**kwargs: Any) -> None:
    """Merge fields into _generation_state under the lock. Recomputes
    `progress` automatically when step/total_steps land in the same update."""
    with _generation_state_lock:
        _generation_state.update(kwargs)
        total = int(_generation_state.get("total_steps") or 0)
        step = int(_generation_state.get("step") or 0)
        _generation_state["progress"] = (
            int(round(100 * step / total)) if total > 0 else 0
        )


def _reset_progress() -> None:
    with _generation_state_lock:
        _generation_state.update({
            "is_generating": False, "phase": "idle",
            "step": 0, "total_steps": 0, "progress": 0,
            "batch_index": 0, "batch_total": 0,
            "started_at": None, "ended_at": None, "error": None,
        })

# Vendored SA3 lives at <repo>/vendor/stable-audio-3 — put it on sys.path so
# `import stable_audio_3` resolves without a global pip install.
_SA3_VENDOR = Path(__file__).resolve().parents[3] / "vendor" / "stable-audio-3"
if str(_SA3_VENDOR) not in sys.path:
    sys.path.insert(0, str(_SA3_VENDOR))


# model_id -> (sa3_name passed to StableAudioModel.from_pretrained,
#              "user-visible or base" tag, max duration seconds).
# Kept in sync manually with _SA3_CATALOG in app/core/model_manager.py.
_MODEL_INFO: Dict[str, Tuple[str, str, int]] = {
    "sa3-small-music":      ("small-music",      "post", 120),
    "sa3-small-sfx":        ("small-sfx",        "post", 120),
    "sa3-medium":           ("medium",           "post", 380),
    "sa3-small-music-base": ("small-music-base", "base", 120),
    "sa3-small-sfx-base":   ("small-sfx-base",   "base", 120),
    "sa3-medium-base":      ("medium-base",      "base", 380),
}


class GenerationStopped(Exception):
    """Raised when an in-flight generation is interrupted by a stop request."""


def _slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", text or "")
    return s[:max_len].strip("_").lower() or "audio"


def _autodetect_device() -> str:
    """cuda → mps → cpu, with FRAGMENTA_FORCE_DEVICE override."""
    override = os.environ.get("FRAGMENTA_FORCE_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AudioGenerator:
    """One-model warm cache. Reload only when model_id changes."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.model: Any = None
        self._model_id: Optional[str] = None
        self._device: Optional[str] = None
        self._stop_requested: bool = False
        # Tracks LoRAs currently injected into self.model. List of
        # {"path": str, "strength": float}. Empty when no LoRAs are active.
        self._loaded_loras: list = []

    # --- cooperative cancel ---------------------------------------------------
    def request_stop(self) -> bool:
        if self._stop_requested:
            return False
        self._stop_requested = True
        return True

    # --- model load -----------------------------------------------------------
    def _ensure_model(
        self,
        model_id: str,
        device: Optional[str] = None,
        half: bool = True,
    ) -> None:
        if model_id not in _MODEL_INFO:
            raise ValueError(f"Unknown SA3 model_id: {model_id}")
        sa3_name, _kind, _max_dur = _MODEL_INFO[model_id]

        if model_id in ("sa3-medium", "sa3-medium-base"):
            if platform.system() == "Windows":
                raise RuntimeError(
                    "sa3-medium requires Flash Attention 2, which doesn't have "
                    "Windows wheels. Use sa3-small-music / sa3-small-sfx, or run "
                    "Fragmenta via Docker on WSL2."
                )
            try:
                import flash_attn  # noqa: F401
            except ImportError as err:
                raise RuntimeError(
                    "sa3-medium needs Flash Attention 2 (flash_attn) but the "
                    f"current install is unusable: {err}.\n"
                    "Pick the wheel matching your torch+ABI+Python+CUDA from\n"
                    "  https://github.com/Dao-AILab/flash-attention/releases\n"
                    "and install with `pip install --no-deps <wheel-url>`. "
                    "See the note next to flash-attn in requirements.txt for an example."
                ) from err

        device = device or _autodetect_device()
        if (
            self.model is not None
            and self._model_id == model_id
            and self._device == device
        ):
            return  # warm cache hit

        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Two layouts to support during the unification transition:
        #   1. Canonical (post-Phase 5c): HF cache layout rooted at
        #      <app>/models/pretrained/sa3/hub/. model_manager sets
        #      HF_HUB_CACHE to that path, so StableAudioModel.from_pretrained
        #      finds files there without going to ~/.cache/huggingface.
        #   2. Legacy: <app>/models/pretrained/sa3/<model_id>/ flat layout
        #      from earlier downloads. We fall back to direct load so
        #      pre-existing users don't have to re-download.
        #
        # Defense-in-depth: re-force the HF cache vars here too. model_manager
        # sets them at construction, but if generation is reached via an
        # alternate code path or the env was clobbered later, we still
        # guarantee resolution into <pretrained>/sa3/hub/.
        hub_dir = self.config.get_path("models_pretrained") / "sa3" / "hub"
        hf_env_keys = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
                       "TRANSFORMERS_CACHE", "HF_HUB_OFFLINE")
        prev_env = {k: os.environ.get(k) for k in hf_env_keys}
        os.environ["HF_HUB_CACHE"] = str(hub_dir)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(hub_dir)
        os.environ["HF_HUB_OFFLINE"] = "1"
        # huggingface_hub captures HF_HUB_CACHE and HF_HUB_OFFLINE as
        # module-level constants AT IMPORT TIME. The Flask backend imports
        # huggingface_hub (transitively, via model_manager.py) before we ever
        # set these env vars, so the constants point at ~/.cache/huggingface/
        # and offline=False. Setting os.environ now has no effect on already-
        # captured constants. We have to monkey-patch them directly.
        # Same trick we used for the CLAP loader.
        prev_hub_constants = {}
        try:
            import huggingface_hub.constants as _hf_const
            prev_hub_constants = {
                "HF_HUB_CACHE": _hf_const.HF_HUB_CACHE,
                "HF_HUB_OFFLINE": _hf_const.HF_HUB_OFFLINE,
            }
            _hf_const.HF_HUB_CACHE = str(hub_dir)
            _hf_const.HF_HUB_OFFLINE = True
        except Exception:
            _hf_const = None
        try:
            try:
                from stable_audio_3 import StableAudioModel
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model = StableAudioModel.from_pretrained(
                        sa3_name, device=device, model_half=half,
                    )
            except (FileNotFoundError, OSError) as primary_err:
                # HF cache miss — fall back to flat layout.
                legacy_dir = self.config.get_path("models_pretrained") / "sa3" / model_id
                config_path = legacy_dir / "model_config.json"
                ckpt_path = legacy_dir / "model.safetensors"
                if not (config_path.exists() and ckpt_path.exists()):
                    raise FileNotFoundError(
                        f"Checkpoint '{model_id}' not found in HF cache "
                        f"({os.environ.get('HF_HUB_CACHE')}) or legacy flat "
                        f"layout ({legacy_dir}). Download it from the "
                        f"Checkpoint Manager."
                    ) from primary_err
                import json
                with open(config_path) as fh:
                    model_config = json.load(fh)
                from stable_audio_3.loading_utils import load_diffusion_cond
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    inner = load_diffusion_cond(
                        model_config, str(ckpt_path),
                        device=device, model_half=half,
                    )
                    inner.use_lora = False
                    inner.lora_names = []
                    self.model = StableAudioModel(inner, model_config, device, half)
        finally:
            for k, v in prev_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            # Restore the patched constants so we don't permanently alter
            # global huggingface_hub state for anything else in-process.
            if _hf_const is not None and prev_hub_constants:
                _hf_const.HF_HUB_CACHE = prev_hub_constants["HF_HUB_CACHE"]
                _hf_const.HF_HUB_OFFLINE = prev_hub_constants["HF_HUB_OFFLINE"]
        self._model_id = model_id
        self._device = device

    # --- LoRA stack -----------------------------------------------------------
    def _apply_loras(self, loras: list) -> None:
        """Inject the given LoRA stack into self.model (idempotent).

        loras: [{"path": str, "strength": float}, ...]

        Strategy:
          * Same paths in same order → just update strengths in place.
          * Different paths → remove all, load fresh.
        """
        if self.model is None:
            return

        new_paths = [l["path"] for l in loras]
        cur_paths = [l["path"] for l in self._loaded_loras]

        if new_paths == cur_paths:
            # Path-set unchanged; only strengths may have moved.
            for i, l in enumerate(loras):
                self.model.set_lora_strength(l["strength"], lora_index=i)
            self._loaded_loras = list(loras)
            return

        # Path-set changed. Remove any currently loaded, then load the new set.
        if cur_paths:
            try:
                from stable_audio_3.models.lora import remove_lora
                # SA3 applies LoRA to the DiffusionCond's DiT (.model) and
                # conditioner (.conditioner) — mirror StableAudioModel's own
                # set_lora_strength which iterates both submodules.
                # `self.model` is StableAudioModel; `self.model.model` is the
                # inner DiffusionCond.
                #
                # remove_lora() strips *every* LoRA parametrization in one
                # pass. We use it instead of remove_lora_by_index(..., 0) in a
                # loop: removal does NOT renumber the remaining adapters, so
                # repeatedly popping index 0 only ever clears the first LoRA
                # and leaves indices 1..n-1 stranded — stale adapters then
                # contaminate every later generation with a different stack.
                inner = self.model.model
                remove_lora(inner.model)
                remove_lora(inner.conditioner)
            except Exception as exc:
                # If removal fails (e.g. an upstream API change), force a
                # base-model reload so we don't carry stale adapters. KEEP
                # _model_id intact — _ensure_model needs it to know what to
                # reload. (Previous code zeroed it; the reload then raised
                # "Unknown SA3 model_id: None".)
                logger.warning(
                    "LoRA removal failed (%s); reloading base model %s",
                    exc, self._model_id,
                )
                self.model = None

        if self.model is None and self._model_id is not None:
            # Forced full reload (only if remove failed above).
            self._ensure_model(self._model_id, device=self._device, half=True)

        if loras:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.load_lora(new_paths)
            for i, l in enumerate(loras):
                self.model.set_lora_strength(l["strength"], lora_index=i)

        self._loaded_loras = list(loras)

    def set_lora_strength(self, index: int, strength: float) -> bool:
        """Live-update one slot's strength. Returns False if index invalid."""
        if not self.model or index < 0 or index >= len(self._loaded_loras):
            return False
        self.model.set_lora_strength(float(strength), lora_index=index)
        self._loaded_loras[index]["strength"] = float(strength)
        return True

    # --- public entry ---------------------------------------------------------
    def generate_audio(
        self,
        prompt: str,
        *,
        model_id: str,
        duration: float = 10.0,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: int = -1,
        negative_prompt: Optional[str] = None,
        batch_size: int = 1,
        device: Optional[str] = None,
        half: bool = True,
        chunked_decode: Optional[bool] = None,
        loop_mode: bool = False,                 # bars-mode passthrough
        loras: Optional[list] = None,            # [{path, strength}, ...]
        # Phase 7: audio-to-audio + inpainting -----------------------------
        init_audio_path: Optional[str] = None,
        init_noise_level: float = 1.0,
        inpaint_audio_path: Optional[str] = None,
        inpaint_starts: Optional[list] = None,   # list[float], seconds
        inpaint_ends: Optional[list] = None,
        # Phase 7: seamless looping ----------------------------------------
        loop_stitch: Optional[str] = None,       # "inpaint" | "crossfade" | None
        loop_bars: Optional[int] = None,
        loop_bpm: Optional[float] = None,
        **_ignored_legacy_kwargs: Any,
    ) -> Path:
        self._stop_requested = False
        if self._stop_requested:                  # honour pre-call stop
            raise GenerationStopped()

        # Validate loop args before doing any heavy work.
        if loop_stitch is not None:
            if loop_stitch not in ("inpaint", "crossfade"):
                raise ValueError(
                    f"loop_stitch must be 'inpaint', 'crossfade', or None; got {loop_stitch!r}"
                )
            if not loop_bars or not loop_bpm:
                raise ValueError(
                    "loop_stitch requires loop_bars and loop_bpm "
                    "(seamless loops are tempo-aware)"
                )
            if loop_stitch == "inpaint" and inpaint_audio_path:
                raise ValueError(
                    "loop_stitch='inpaint' is incompatible with inpaint_audio_path "
                    "(the loop algorithm itself uses inpainting as a second pass)"
                )

        _set_progress(
            is_generating=True, phase="loading",
            step=0, total_steps=0, error=None,
            started_at=time.time(), ended_at=None,
        )

        self._ensure_model(model_id, device=device, half=half)
        self._apply_loras(loras or [])

        init_audio = self._load_audio(init_audio_path) if init_audio_path else None
        inpaint_audio = self._load_audio(inpaint_audio_path) if inpaint_audio_path else None

        _, kind, max_dur = _MODEL_INFO[model_id]
        is_base = (kind == "base")

        # Defaults differ by model kind. Post-trained models distilled CFG
        # away; we force cfg=1.0 there even if the caller overrides.
        effective_steps = int(steps) if steps else (50 if is_base else 8)
        effective_cfg = float(cfg_scale) if (cfg_scale is not None and is_base) else (
            7.0 if is_base else 1.0
        )

        duration = float(min(max(1.0, float(duration)), float(max_dur)))

        # For loop_stitch="inpaint" we need slack for TWO operations:
        # tempo conform (up to 15% stretch in safe range) AND head-trim
        # to the first detected beat (up to 1.5 s). 50% headroom covers
        # both; after align+crop we have exactly `duration` of tempo-
        # correct, phase-aligned audio.
        target_samples = int(round(duration * 44100))
        gen_duration = duration
        if loop_stitch == "inpaint":
            gen_duration = min(float(max_dur), duration * 1.5 + 1.5)

        # Loop-inpaint runs the sampler twice; surface that in the progress
        # bar so 100% only lands after both passes finish.
        total_steps_logical = (
            2 * effective_steps if loop_stitch == "inpaint" else effective_steps
        )

        if self._stop_requested:                  # one more check before the heavy call
            raise GenerationStopped()

        # Sampler callback — fires per ODE step. Also gives us a cheap
        # cancellation hook: raising mid-callback aborts the sampler.
        # The offset lets us continue progress across the loop-inpaint pass.
        def _make_callback(offset: int) -> Callable[[Dict[str, Any]], None]:
            def _cb(info: Dict[str, Any]) -> None:
                if self._stop_requested:
                    raise GenerationStopped()
                i = info.get("i")
                if isinstance(i, int):
                    _set_progress(step=min(offset + i + 1, total_steps_logical))
            return _cb

        _set_progress(phase="sampling", total_steps=int(total_steps_logical), step=0)

        gen_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            duration=gen_duration,
            steps=effective_steps,
            cfg_scale=effective_cfg,
            seed=int(seed),
            batch_size=int(batch_size),
            chunked_decode=chunked_decode,
            callback=_make_callback(0),
        )
        if init_audio is not None:
            gen_kwargs["init_audio"] = init_audio
            gen_kwargs["init_noise_level"] = float(init_noise_level)
        if inpaint_audio is not None:
            gen_kwargs["inpaint_audio"] = inpaint_audio
            if inpaint_starts is not None and len(inpaint_starts) > 0:
                # SA3 accepts a single float or a list for multi-region.
                gen_kwargs["inpaint_mask_start_seconds"] = (
                    list(inpaint_starts) if len(inpaint_starts) > 1 else float(inpaint_starts[0])
                )
            if inpaint_ends is not None and len(inpaint_ends) > 0:
                gen_kwargs["inpaint_mask_end_seconds"] = (
                    list(inpaint_ends) if len(inpaint_ends) > 1 else float(inpaint_ends[0])
                )

        try:
            audio = self.model.generate(**gen_kwargs)
            # audio: torch.Tensor[B, channels=2, samples] in [-1, 1] @ 44.1 kHz
        except GenerationStopped:
            _set_progress(phase="idle", is_generating=False, ended_at=time.time())
            raise
        except Exception as exc:
            _set_progress(phase="failed", is_generating=False,
                          error=str(exc), ended_at=time.time())
            raise

        # --- Seamless loop processing (Phase 7) ------------------------------
        if loop_stitch == "inpaint":
            _set_progress(phase="loop_stitching", step=effective_steps)
            # Phase-align AND tempo-conform the baseline BEFORE the inpaint
            # pass. align_for_loop: tempo-stretches (uniform, so seam
            # relationship survives), head-trims to first detected beat
            # (puts the downbeat at sample 0 so multiple channels lock at
            # the bar boundary), and crops to exact target_samples. The
            # inpaint pass then smooths the resulting boundary.
            audio = self._align_baseline_for_loop(
                audio,
                target_samples=target_samples,
                target_bpm=float(loop_bpm),
            )
            try:
                audio = self._make_loop_inpaint(
                    audio,
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    duration=duration,
                    steps=effective_steps,
                    cfg_scale=effective_cfg,
                    seed=int(seed),
                    batch_size=int(batch_size),
                    chunked_decode=chunked_decode,
                    bars=int(loop_bars),
                    bpm=float(loop_bpm),
                    callback=_make_callback(effective_steps),
                )
            except GenerationStopped:
                _set_progress(phase="idle", is_generating=False, ended_at=time.time())
                raise
            except Exception as exc:
                _set_progress(phase="failed", is_generating=False,
                              error=str(exc), ended_at=time.time())
                raise
        elif loop_stitch == "crossfade":
            _set_progress(phase="loop_stitching", step=effective_steps)
            audio = self._crossfade_seam(audio, fade_sec=0.1, sr=44100)

        _set_progress(phase="decoding", step=total_steps_logical)
        try:
            return self._finalize(audio, prompt=prompt, model_id=model_id)
        finally:
            _set_progress(phase="complete", is_generating=False,
                          step=total_steps_logical, ended_at=time.time())

    # --- Phase 7: seamless loop helpers --------------------------------------
    def _align_baseline_for_loop(
        self,
        audio: torch.Tensor,
        *,
        target_samples: int,
        target_bpm: float,
    ) -> torch.Tensor:
        """Phase-align + tempo-conform a baseline tensor to exact target_samples.

        Delegates the librosa work to `align_for_loop` in audio_post_process
        (numpy-domain). Returns a torch tensor of shape `[B=1, C, target_samples]`
        on the same device and dtype as the input.

        Falls back to a plain crop/pad of the input if alignment fails — the
        wrap-and-inpaint that runs next still smooths the seam, only the phase
        / tempo accuracy is lost.
        """
        if audio.shape[-1] < target_samples:
            pad = target_samples - audio.shape[-1]
            return torch.nn.functional.pad(audio, (0, pad))

        # SCOPE: this method is called ONLY from the loop_stitch=="inpaint"
        # path in generate(), which is set ONLY by Performance tab → Bars
        # mode + looping=true. Generation tab and Performance Sec mode
        # never set loop_stitch, so the loop_quantizer flag below only
        # affects Performance Bars output.
        try:
            # SA3 returns [B, C, T] — pull the first batch into [T, C]
            # for the numpy-domain aligners.
            full = audio[0].detach().cpu().numpy().astype(np.float32).T  # [T, C]
            from app.core.loop_quantizer import loop_quantizer_enabled
            if loop_quantizer_enabled():
                import os as _os
                from app.core.loop_quantizer import AubioDetector, quantize_to_loop
                # Recover bars from target_samples; the canonical-grid
                # formula is invertible for the small integer bar counts
                # the UI exposes (1–16).
                target_bars = int(round(
                    target_samples * float(target_bpm) / (4.0 * 60.0 * 44100)
                ))
                grid_div = int(_os.environ.get("FRAGMENTA_LOOP_QUANTIZER_GRID", "16"))
                _truthy = ("1", "true", "yes", "on")
                hierarchical = _os.environ.get(
                    "FRAGMENTA_LOOP_QUANTIZER_HIER", "0"
                ).strip().lower() in _truthy
                tempo_only = _os.environ.get(
                    "FRAGMENTA_LOOP_QUANTIZER_TEMPO_ONLY", "0"
                ).strip().lower() in _truthy
                onset_thr = _os.environ.get("FRAGMENTA_LOOP_QUANTIZER_ONSET_THRESHOLD")
                onset_method = _os.environ.get(
                    "FRAGMENTA_LOOP_QUANTIZER_ONSET_METHOD"
                )
                if onset_thr is not None or onset_method is not None:
                    det_kwargs = {}
                    if onset_thr is not None:
                        det_kwargs["threshold"] = float(onset_thr)
                    if onset_method is not None:
                        det_kwargs["method"] = onset_method
                    detector = AubioDetector(**det_kwargs)
                else:
                    detector = None
                hier_tol = _os.environ.get(
                    "FRAGMENTA_LOOP_QUANTIZER_HIER_TOLERANCE_MS"
                )
                hier_kwargs = (
                    {"hierarchical_tolerance_ms": float(hier_tol)}
                    if hier_tol is not None else {}
                )
                no_stretcher = _os.environ.get(
                    "FRAGMENTA_LOOP_QUANTIZER_NO_STRETCHER", "0"
                ).strip().lower() in _truthy
                if no_stretcher:
                    from app.core.loop_quantizer import NO_STRETCHER as _NO_STRETCHER
                    hier_kwargs["stretcher"] = _NO_STRETCHER
                no_tempo_conform = _os.environ.get(
                    "FRAGMENTA_LOOP_QUANTIZER_NO_TEMPO_CONFORM", "0"
                ).strip().lower() in _truthy
                # Beat-track is the production default — it locks anchors
                # to the periodic pulse and falls back to hierarchical
                # onset snap when aubio.tempo fails its BPM/confidence
                # gates. Opt out with FRAGMENTA_LOOP_QUANTIZER_NO_BEAT_TRACK=1
                # to measure pure onset behaviour.
                no_beat_track = _os.environ.get(
                    "FRAGMENTA_LOOP_QUANTIZER_NO_BEAT_TRACK", "0"
                ).strip().lower() in _truthy
                beat_track = not no_beat_track
                beat_tol = _os.environ.get(
                    "FRAGMENTA_LOOP_QUANTIZER_BEAT_TOLERANCE_MS"
                )
                if beat_tol is not None:
                    hier_kwargs["beat_track_tolerance_ms"] = float(beat_tol)
                aligned = quantize_to_loop(
                    full,
                    bpm=float(target_bpm),
                    bars=target_bars,
                    grid=grid_div,
                    sample_rate=44100,
                    detector=detector,
                    hierarchical=hierarchical,
                    tempo_only=tempo_only,
                    tempo_conform=not no_tempo_conform,
                    beat_track=beat_track,
                    **hier_kwargs,
                )
            else:
                # DEPRECATED legacy path — safety net per AUDIT.md §9.
                from app.core.generation.audio_post_process import align_for_loop
                aligned = align_for_loop(
                    full,
                    sr=44100,
                    target_samples=target_samples,
                    target_bpm=target_bpm,
                )
            # Back to [B=1, C, T]
            result = (
                torch.from_numpy(np.ascontiguousarray(aligned.T))
                .unsqueeze(0)
                .to(audio.device)
                .to(audio.dtype)
            )
        except Exception as exc:
            logger.warning(
                "Loop-alignment failed (%s); falling back to plain crop", exc,
            )
            return audio[..., :target_samples]

        if result.shape[-1] != target_samples:
            # Defensive guard against align_for_loop returning the wrong size.
            if result.shape[-1] > target_samples:
                result = result[..., :target_samples]
            else:
                pad = target_samples - result.shape[-1]
                result = torch.nn.functional.pad(result, (0, pad))
        return result

    def _make_loop_inpaint(
        self,
        baseline: torch.Tensor,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        duration: float,
        steps: int,
        cfg_scale: float,
        seed: int,
        batch_size: int,
        chunked_decode: Optional[bool],
        bars: int,
        bpm: float,
        callback: Callable[[Dict[str, Any]], None],
    ) -> torch.Tensor:
        """Wrap-and-inpaint seamless looping (Phase 7).

        SA3 has no circular sampling, so we exploit inpainting:
          1. Roll the baseline by N/2 so the boundary between sample[N-1]
             and sample[0] sits at the centre of the wrapped audio.
          2. Inpaint a small region around the centre — SA3 regenerates it
             with full context from the rest of the (now-shifted) clip.
          3. Roll back. The new boundary is spectrally consistent because it
             was generated as a continuous waveform inside the wrap.

        Mask half-width = 0.5 bar, clamped to [0.25, 2.0] seconds.
        """
        if baseline.ndim != 3:
            raise RuntimeError(f"Unexpected baseline shape {tuple(baseline.shape)}")

        sr = 44100
        try:
            sr = int(self.model.model_config.get("sample_rate", 44100))
        except Exception:
            pass

        N = baseline.shape[-1]
        half_samples = N // 2

        wrapped = torch.roll(baseline, shifts=half_samples, dims=-1)
        # SA3 inpaint takes a single (sr, [C, T]) tuple; drop the batch dim
        # (one conditioning produces one consistent seam — multi-batch here
        # would just diverge the seam regen per batch member without help).
        wrapped_first = wrapped[0]

        bar_sec = 60.0 / float(bpm) * 4.0
        # Mask half-width: 0.5 bar by default, floored at 0.25 s (SA3's
        # minimum useful inpaint span), capped absolutely at 2.0 s so long
        # bars don't lose too much musical content, AND capped at 1/4 of
        # the clip duration so the inpaint never covers more than half the
        # clip. Tighter caps starve the model of context near the seam and
        # collapse the inpainted region to near-silence — empirically the
        # 25% cap is the sweet spot.
        r_sec = max(
            0.25,
            min(2.0, 0.5 * bar_sec, 0.25 * duration),
        )
        seam_t = half_samples / float(sr)
        mask_start = max(0.0, seam_t - r_sec)
        mask_end = min(duration, seam_t + r_sec)

        inpainted = self.model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration=duration,
            steps=steps,
            cfg_scale=cfg_scale,
            # Reuse the seed; SA3 advances global RNG during the first call,
            # so the inpaint pass naturally samples different noise.
            seed=seed,
            batch_size=batch_size,
            chunked_decode=chunked_decode,
            inpaint_audio=(sr, wrapped_first),
            inpaint_mask_start_seconds=float(mask_start),
            inpaint_mask_end_seconds=float(mask_end),
            callback=callback,
        )

        # Match the baseline length exactly before unrolling. Lengths usually
        # match (same duration request), but SA3 rounds duration up to chunk
        # alignment internally — guard against off-by-one.
        if inpainted.shape[-1] > N:
            inpainted = inpainted[..., :N]
        elif inpainted.shape[-1] < N:
            pad = N - inpainted.shape[-1]
            inpainted = torch.nn.functional.pad(inpainted, (0, pad))

        return torch.roll(inpainted, shifts=-half_samples, dims=-1)

    @staticmethod
    def _crossfade_seam(audio: torch.Tensor, fade_sec: float, sr: int) -> torch.Tensor:
        """Equal-power loop crossfade — fallback to wrap-and-inpaint.

        Blends the first `fade_sec` of audio into the last `fade_sec` so a
        looping player spectrally morphs from tail content into head content
        across the fade. Not bit-exact at the boundary, but fast and never
        fails. For true seamlessness use loop_stitch='inpaint'.

        Length unchanged; the input tensor is not modified in place.
        """
        n = audio.shape[-1]
        fade_n = max(1, int(fade_sec * sr))
        if fade_n >= n // 2:
            return audio  # clip too short for a useful fade

        head = audio[..., :fade_n]
        tail = audio[..., -fade_n:]

        t = torch.linspace(
            0.0, math.pi / 2.0, fade_n,
            device=audio.device, dtype=audio.dtype,
        )
        fade_in = torch.sin(t) ** 2
        fade_out = torch.cos(t) ** 2

        out = audio.clone()
        out[..., -fade_n:] = tail * fade_out + head * fade_in
        return out

    # --- audio loader (a2a + inpaint inputs) ----------------------------------
    @staticmethod
    def _load_audio(path: str):
        """Load a wav/mp3/flac into the (sample_rate, tensor) tuple SA3 expects.

        Returns a stereo float32 tensor of shape (channels, samples). Mono
        inputs are duplicated to stereo (SA3 expects 2 channels); ≥3-channel
        inputs are truncated to the first 2.
        """
        import torchaudio
        wav, sr = torchaudio.load(str(path))   # (channels, samples), float32
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]
        return int(sr), wav

    # --- output --------------------------------------------------------------
    def _finalize(self, audio: torch.Tensor, *, prompt: str, model_id: str) -> Path:
        audio = audio.detach().clamp_(-1.0, 1.0).cpu()
        if audio.ndim != 3:
            raise RuntimeError(f"Unexpected SA3 output shape {tuple(audio.shape)}")
        first = audio[0]                           # [C, samples]
        pcm = (first.numpy() * 32767.0).astype(np.int16).T  # → [samples, C]

        out_dir = self.config.get_path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{ts}_{model_id}_{_slugify(prompt)}.wav"
        sf.write(str(out_path), pcm, 44100, subtype="PCM_16")
        return out_path
