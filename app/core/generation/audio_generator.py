"""Generation: everything that isn't model sampling.

Builds the request, reads input audio, reports progress, handles stop, and writes
the 44.1 kHz stereo int16 WAV with its tags. The sampling itself belongs to an
engine (engine.py); this picks the one that runs the requested model.

Generations are serialized: `generate_audio` holds an internal lock for the
whole run, so concurrent requests (e.g. a Performance-channel generation
overlapping a main-tab one — Flask runs threaded) queue instead of evicting
each other's model mid-sampling or interleaving the shared progress state.

Cancellation is per-run: each generation gets its own stop event, set by
`request_stop()`. The SA3 sampler fires our per-ODE-step callback, which
checks the event and aborts mid-run. Starting a new generation can never
clear a stop aimed at the previous one.
"""
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.generation.engine import Capabilities, Engine, GenerationRequest
from app.core.generation.models import MODEL_INFO as _MODEL_INFO, engine_for
from app.core.generation.provenance import generation_tags, write_tagged_wav
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

class GenerationStopped(Exception):
    """Raised when an in-flight generation is interrupted by a stop request."""


def _slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", text or "")
    return s[:max_len].strip("_").lower() or "audio"


# (engine, its distilled model, its base model) for the real-model contract test.
ENGINE_TEST_MODELS = (("torch", "sa3-small-music", "sa3-small-music-base"),)


def _engine_class(name: str) -> Any:
    # Imported here: an engine's framework only loads on the machines that use it.
    from app.core.generation.torch_engine import TorchEngine
    return TorchEngine


def _make_engine(name: str, config: Any) -> Engine:
    return _engine_class(name)(config)


def engine_available(name: str) -> bool:
    return _engine_class(name).available()


def engine_capabilities(name: str) -> Capabilities:
    return _engine_class(name).capabilities


class AudioGenerator:
    """Runs generations on one engine at a time."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self._engines: Dict[str, Engine] = {}
        self._active: Optional[Engine] = None
        # Serializes whole generations (and model unloads) — see module
        # docstring. _state_lock only guards the _current_stop swap so
        # request_stop() never blocks behind a running generation.
        self._gen_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._current_stop: Optional[threading.Event] = None

    # --- engines --------------------------------------------------------------
    def _engine(self, model_id: str) -> Engine:
        if model_id not in _MODEL_INFO:
            raise ValueError(f"Unknown SA3 model_id: {model_id}")
        name = engine_for(model_id)
        engine = self._engines.get(name)
        if engine is None:
            engine = self._engines[name] = _make_engine(name, self.config)
        # Engines can share memory (unified memory on a Mac), so only one holds a model.
        if self._active is not None and self._active is not engine:
            self._active.unload()
        self._active = engine
        return engine

    @property
    def model(self) -> Any:
        return self._active.model if self._active else None

    @property
    def _model_id(self) -> Optional[str]:
        return self._active.model_id if self._active else None

    @property
    def _loaded_loras(self) -> List[Dict[str, Any]]:
        return self._active.loaded_loras if self._active else []

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

        The engine's LoRA bookkeeping dies with the model: the adapters live as
        parametrizations inside the model object, so after a drop the
        `apply_loras` fast path ("same paths → just update strengths") would
        otherwise no-op against a fresh, adapter-less model while the UI still
        claims the LoRA is active.
        """
        if not self._gen_lock.acquire(blocking=False):
            return False
        try:
            if self._active is not None:
                self._active.unload()
            return True
        finally:
            self._gen_lock.release()

    def set_lora_strength(self, index: int, strength: float) -> bool:
        """Live-update one slot's strength. Returns False if index invalid."""
        return bool(self._active) and self._active.set_lora_strength(index, strength)

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

        engine = self._engine(model_id)
        engine.load(model_id, device=device, half=half)
        engine.apply_loras(loras or [])

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

        total_steps_logical = effective_steps

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

        request = GenerationRequest(
            prompt=prompt,
            model_id=model_id,
            duration=duration,
            steps=effective_steps,
            cfg_scale=effective_cfg,
            seed=int(seed),
            negative_prompt=negative_prompt or None,
            batch_size=int(batch_size),
            chunked_decode=chunked_decode,
            init_audio=init_audio,
            init_noise_level=float(init_noise_level),
            inpaint_audio=inpaint_audio,
            inpaint_starts=list(inpaint_starts) if inpaint_starts is not None else None,
            inpaint_ends=list(inpaint_ends) if inpaint_ends is not None else None,
        )

        try:
            audio = engine.generate(request, _cb)
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
        """Load any accepted audio file as (sample_rate, samples).

        Returns stereo float32 samples, shape (channels, samples). Mono inputs
        are duplicated to stereo (SA3 expects 2 channels); ≥3-channel inputs are
        truncated to the first 2.
        """
        from app.core.audio.decode import read_audio
        wav, sr = read_audio(path)   # (channels, samples), float32
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]
        return int(sr), wav.numpy()

    # --- output --------------------------------------------------------------
    def _finalize(self, audio: np.ndarray, *, prompt: str, model_id: str) -> Path:
        if audio.ndim != 2:
            raise RuntimeError(f"Unexpected engine output shape {tuple(audio.shape)}")
        first = np.clip(audio, -1.0, 1.0)          # [C, samples]
        # clip bounds ±inf but lets NaN through (NaN compares false), and a
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
        # Machine-readable AI-output marking (EU AI Act Art. 50) — see provenance.py.
        write_tagged_wav(
            out_path, pcm, 44100,
            generation_tags(model_id, [l["path"] for l in self._loaded_loras]),
        )
        return out_path
