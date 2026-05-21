"""SA3 inference engine.

Thin wrapper around stable_audio_3.StableAudioModel.from_pretrained() that
caches the loaded model between requests (eviction on model_id change),
auto-detects the device, and writes 44.1 kHz stereo int16 WAV.

Cancellation is wired via `request_stop()` for API parity, but SA3's
generate() doesn't expose a per-step callback yet — the flag is checked
between calls, not inside them. A finer-grained cancel hook is a Phase
3.1 follow-up.
"""
import os
import platform
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

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

        # If the snapshot is already in the HF cache, force offline mode for
        # the load so SA3's hf_hub_download stops doing etag HEAD requests on
        # every file. Restore the previous setting afterwards so a later
        # snapshot_download (different model) still has network access.
        prev_offline = os.environ.get("HF_HUB_OFFLINE")
        if self._is_cached(sa3_name):
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            from stable_audio_3 import StableAudioModel
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = StableAudioModel.from_pretrained(
                    sa3_name, device=device, model_half=half,
                )
        finally:
            if prev_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_offline
        self._model_id = model_id
        self._device = device

    @staticmethod
    def _is_cached(sa3_name: str) -> bool:
        """Heuristic: does the HF cache already have the safetensors for this model?"""
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return False
        repo_id = f"stabilityai/stable-audio-3-{sa3_name}"
        # try_to_load_from_cache returns a path string when present, None when missing,
        # _CACHED_NO_EXIST sentinel when explicitly known-not-on-the-hub.
        result = try_to_load_from_cache(repo_id=repo_id, filename="model.safetensors")
        return isinstance(result, str)

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
        **_ignored_legacy_kwargs: Any,
    ) -> Path:
        self._stop_requested = False
        if self._stop_requested:                  # honour pre-call stop
            raise GenerationStopped()

        self._ensure_model(model_id, device=device, half=half)

        _, kind, max_dur = _MODEL_INFO[model_id]
        is_base = (kind == "base")

        # Defaults differ by model kind. Post-trained models distilled CFG
        # away; we force cfg=1.0 there even if the caller overrides.
        effective_steps = int(steps) if steps else (50 if is_base else 8)
        effective_cfg = float(cfg_scale) if (cfg_scale is not None and is_base) else (
            7.0 if is_base else 1.0
        )

        duration = float(min(max(1.0, float(duration)), float(max_dur)))

        if self._stop_requested:                  # one more check before the heavy call
            raise GenerationStopped()

        audio = self.model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            duration=duration,
            steps=effective_steps,
            cfg_scale=effective_cfg,
            seed=int(seed),
            batch_size=int(batch_size),
            chunked_decode=chunked_decode,
        )
        # audio: torch.Tensor[B, channels=2, samples] in [-1, 1] @ 44.1 kHz

        return self._finalize(audio, prompt=prompt, model_id=model_id)

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
