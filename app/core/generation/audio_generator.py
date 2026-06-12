"""SA3 inference engine.

Thin wrapper around stable_audio_3.StableAudioModel.from_pretrained() that
caches the loaded model between requests (eviction on model_id change),
auto-detects the device, and writes 44.1 kHz stereo int16 WAV.

Generations are serialized: `generate_audio` holds an internal lock for the
whole run, so concurrent requests (e.g. a Performance-channel generation
overlapping a main-tab one — Flask runs threaded) queue instead of evicting
each other's model mid-sampling or interleaving the shared progress state.

Cancellation is per-run: each generation gets its own stop event, set by
`request_stop()`. The SA3 sampler fires our per-ODE-step callback, which
checks the event and aborts mid-run. Starting a new generation can never
clear a stop aimed at the previous one.
"""
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
    # idle | loading | sampling | decoding | complete | failed
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
        # Tracks LoRAs currently injected into self.model. List of
        # {"path": str, "strength": float}. Empty when no LoRAs are active.
        self._loaded_loras: list = []
        # Serializes whole generations (and model unloads) — see module
        # docstring. _state_lock only guards the _current_stop swap so
        # request_stop() never blocks behind a running generation.
        self._gen_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._current_stop: Optional[threading.Event] = None

    # --- cooperative cancel ---------------------------------------------------
    def request_stop(self) -> bool:
        """Signal the in-flight generation, if any, to stop.

        Returns True only when a running generation was newly signalled.
        A stop with nothing running is a no-op — it must NOT arm a flag
        that a later, unrelated run would inherit.
        """
        with self._state_lock:
            ev = self._current_stop
        if ev is None or ev.is_set():
            return False
        ev.set()
        return True

    # --- unload ----------------------------------------------------------------
    def unload_model(self) -> bool:
        """Drop the cached model and every piece of bookkeeping tied to it.

        Returns False (and does nothing) when a generation is in flight —
        yanking the model out from under the sampler would crash it.

        The LoRA bookkeeping must die with the model: the adapters live as
        parametrizations inside the model object, so after a drop the
        `_apply_loras` fast path ("same paths → just update strengths") would
        otherwise no-op against a fresh, adapter-less model while the UI still
        claims the LoRA is active.
        """
        if not self._gen_lock.acquire(blocking=False):
            return False
        try:
            self.model = None
            self._model_id = None
            self._loaded_loras = []
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        finally:
            self._gen_lock.release()

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
            # Medium normally requires Flash Attention 2 for its long-form (up
            # to 380s) sliding-window attention. FRAGMENTA_MEDIUM_NO_FLASH=1 is
            # the Path-B validation switch: it lets medium load WITHOUT
            # flash_attn and fall back to PyTorch-native attention
            # (flex_attention -> chunked-halo SDPA -> masked SDPA; see
            # transformer.apply_attn). Output is math-equivalent, but VRAM is
            # higher and sampling slower at long durations. Off by default, so
            # the shipped behaviour is unchanged until the fallback is validated.
            allow_no_flash = os.environ.get("FRAGMENTA_MEDIUM_NO_FLASH") == "1"
            try:
                import flash_attn  # noqa: F401
                have_flash = True
            except ImportError as err:
                have_flash = False
                _flash_err = err

            if not have_flash and not allow_no_flash:
                if platform.system() == "Windows":
                    raise RuntimeError(
                        "sa3-medium requires Flash Attention 2, which doesn't "
                        "have Windows wheels. Use sa3-small-music / sa3-small-sfx, "
                        "run Fragmenta via Docker on WSL2, or set "
                        "FRAGMENTA_MEDIUM_NO_FLASH=1 to run on the (slower, "
                        "higher-memory) PyTorch attention fallback."
                    ) from _flash_err
                raise RuntimeError(
                    "sa3-medium needs Flash Attention 2 (flash_attn) but the "
                    f"current install is unusable: {_flash_err}.\n"
                    "Pick the wheel matching your torch+ABI+Python+CUDA from\n"
                    "  https://github.com/Dao-AILab/flash-attention/releases\n"
                    "and install with `pip install --no-deps <wheel-url>`. "
                    "See the note next to flash-attn in requirements.txt for an example.\n"
                    "Or set FRAGMENTA_MEDIUM_NO_FLASH=1 to use the PyTorch "
                    "attention fallback."
                ) from _flash_err

            if not have_flash:
                logger.warning(
                    "sa3-medium loading WITHOUT Flash Attention 2 "
                    "(FRAGMENTA_MEDIUM_NO_FLASH=1). Using the PyTorch-native "
                    "attention fallback — expect higher VRAM and slower sampling "
                    "at long durations. Validate memory headroom before "
                    "generating long-form (up to 380s) clips."
                )

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
            # The LoRA stack was injected into the evicted model's
            # parametrizations and is gone with it. Without this reset the
            # next _apply_loras call with the same paths takes the
            # strengths-only fast path and silently applies nothing.
            self._loaded_loras = []
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
    def generate_audio(self, prompt: str, **kwargs: Any) -> Path:
        """Serialized public entry. See _generate_audio_impl for parameters.

        Holds the generation lock for the whole run (concurrent callers
        queue) and owns the per-run stop event's lifecycle.
        """
        with self._gen_lock:
            stop_event = threading.Event()
            with self._state_lock:
                self._current_stop = stop_event
            try:
                return self._generate_audio_impl(
                    prompt, stop_event=stop_event, **kwargs)
            except BaseException as exc:
                # Sampling failures and stops reset the progress state at
                # their raise site, but a failure before sampling (a model
                # load error, a bad init-audio file) would leave the
                # frontend's progress poller stuck on is_generating=True.
                if get_generation_progress().get("is_generating"):
                    _set_progress(phase="failed", is_generating=False,
                                  error=str(exc), ended_at=time.time())
                raise
            finally:
                with self._state_lock:
                    self._current_stop = None

    def _generate_audio_impl(
        self,
        prompt: str,
        *,
        stop_event: threading.Event,
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
        # `loop_stitch` / `loop_bars` / `loop_bpm` are accepted for API
        # compatibility but ignored — the seamless-loop pipeline was
        # removed because user A/B testing showed it degraded audio
        # quality on every prompt class. We deliver raw model output.

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

        target_samples = int(round(duration * 44100))
        gen_duration = duration
        total_steps_logical = effective_steps

        # The vendored generate() defaults sample_size=5292032 (=120s) and
        # _adapt_sample_size CLAMPS the seconds_total-derived length to it. Since
        # we never passed sample_size, every model — including medium — was
        # capped at 120s, so medium's advertised 380s never materialised. Pass
        # the model's native sample_size as the ceiling so _adapt sizes output to
        # the requested duration (plus its 6s padding) within the real maximum.
        try:
            native_sample_size = int(self.model.model_config.get("sample_size"))
        except Exception:
            native_sample_size = 0
        sample_size_ceiling = max(native_sample_size, target_samples)

        if stop_event.is_set():                   # stopped during model load
            _set_progress(phase="idle", is_generating=False, ended_at=time.time())
            raise GenerationStopped()

        # Sampler callback — fires per ODE step. Also gives us a cheap
        # cancellation hook: raising mid-callback aborts the sampler.
        def _cb(info: Dict[str, Any]) -> None:
            if stop_event.is_set():
                raise GenerationStopped()
            i = info.get("i")
            if isinstance(i, int):
                _set_progress(step=min(i + 1, total_steps_logical))

        _set_progress(phase="sampling", total_steps=int(total_steps_logical), step=0)

        gen_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            duration=gen_duration,
            sample_size=sample_size_ceiling,
            steps=effective_steps,
            cfg_scale=effective_cfg,
            seed=int(seed),
            batch_size=int(batch_size),
            chunked_decode=chunked_decode,
            callback=_cb,
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

        # Seamless-loop processing (quantize, inpaint, crossfade) was
        # removed: the user A/B-compared raw SA3 output against the full
        # pipeline and confirmed the post-processing made every prompt
        # worse — silence-at-start on percussion, smeared transients,
        # off-grid anchoring. We now deliver the raw model output. The
        # `loop_stitch` / `loop_bars` / `loop_bpm` parameters are still
        # accepted from the frontend for API compatibility but are
        # ignored. Performance-Bars looping will have an audible click
        # at the wrap point and multi-channel stacks will not be
        # sample-aligned — both acceptable trade-offs vs. the artifacts
        # the quantizer was introducing.
        # A stop that lands after the final ODE step (the sampler callback
        # has already fired for the last time) would otherwise be ignored and
        # a ghost fragment written. Honour it before touching disk.
        if stop_event.is_set():
            _set_progress(phase="idle", is_generating=False, ended_at=time.time())
            raise GenerationStopped()

        _set_progress(phase="decoding", step=total_steps_logical)
        try:
            return self._finalize(audio, prompt=prompt, model_id=model_id)
        finally:
            _set_progress(phase="complete", is_generating=False,
                          step=total_steps_logical, ended_at=time.time())

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
        first = audio[0].numpy()                   # [C, samples]
        # clamp_ bounds ±inf but lets NaN through (NaN compares false), and a
        # NaN hits int16 conversion as undefined behaviour — scrub it here so
        # a numerically unstable run yields silence, not a corrupt WAV.
        first = np.nan_to_num(first, nan=0.0, posinf=1.0, neginf=-1.0)
        pcm = (first * 32767.0).astype(np.int16).T  # → [samples, C]

        out_dir = self.config.get_path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"{ts}_{model_id}_{_slugify(prompt)}"
        # The timestamp has 1 s resolution; back-to-back fast generations of
        # the same prompt would silently overwrite each other without a
        # uniquing suffix.
        out_path = out_dir / f"{base}.wav"
        counter = 2
        while out_path.exists():
            out_path = out_dir / f"{base}_{counter}.wav"
            counter += 1
        sf.write(str(out_path), pcm, 44100, subtype="PCM_16")
        return out_path
