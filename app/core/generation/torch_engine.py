"""The PyTorch engine: Stable Audio 3 through stable_audio_3.StableAudioModel.

Caches the loaded model between requests (eviction on model_id change) and
auto-detects the device. CUDA, MPS or CPU.
"""
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from app.core.generation.engine import Capabilities, Engine, GenerationRequest, StepCallback
from app.core.generation.models import MODEL_INFO
from utils.logger import get_logger

logger = get_logger("AudioGenerator")

# Vendored SA3 lives at <repo>/vendor/stable-audio-3 — put it on sys.path so
# `import stable_audio_3` resolves without a global pip install.
_SA3_VENDOR = Path(__file__).resolve().parents[3] / "vendor" / "stable-audio-3"
if str(_SA3_VENDOR) not in sys.path:
    sys.path.insert(0, str(_SA3_VENDOR))


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


def _as_tensor(audio):
    sample_rate, samples = audio
    return sample_rate, torch.from_numpy(samples)


class TorchEngine(Engine):
    """One-model warm cache. Reload only when model_id changes."""

    name = "torch"
    capabilities = Capabilities(training=True, bending=True)
    handled_fields = frozenset({
        "prompt", "model_id", "duration", "steps", "cfg_scale", "seed", "negative_prompt",
        "batch_size", "chunked_decode", "init_audio", "init_noise_level", "inpaint_audio",
        "inpaint_starts", "inpaint_ends",
    })

    def unload(self) -> None:
        self.model = None
        self.model_id = None
        self.loaded_loras = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- model load -----------------------------------------------------------
    def load(self, model_id: str, device: Optional[str] = None, half: bool = True) -> None:
        if model_id not in MODEL_INFO:
            raise ValueError(f"Unknown SA3 model_id: {model_id}")
        sa3_name, _kind, _max_dur = MODEL_INFO[model_id]

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
                raise RuntimeError(
                    "sa3-medium needs Flash Attention 2 (flash_attn) but the "
                    f"current install is unusable: {_flash_err}.\n"
                    "Fragmenta installs it on Linux and Windows with an NVIDIA "
                    "GPU; run the installer again to repair it.\n"
                    "Or use sa3-small-music / sa3-small-sfx, or set "
                    "FRAGMENTA_MEDIUM_NO_FLASH=1 to run on the (slower, "
                    "higher-memory) PyTorch attention fallback."
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
            and self.model_id == model_id
            and self.device == device
        ):
            return  # warm cache hit

        if self.model is not None:
            del self.model
            self.model = None
            # The LoRA stack was injected into the evicted model's
            # parametrizations and is gone with it. Without this reset the
            # next apply_loras call with the same paths takes the
            # strengths-only fast path and silently applies nothing.
            self.loaded_loras = []
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
                with open(config_path, encoding="utf-8") as fh:
                    model_config = json.load(fh)
                from stable_audio_3.model_configs import localize_conditioners
                localize_conditioners(model_config, str(config_path))
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
        self.model_id = model_id
        self.device = device

    # --- LoRA stack -----------------------------------------------------------
    def apply_loras(self, loras: List[Dict[str, Any]]) -> None:
        """Inject the given LoRA stack into self.model (idempotent).

        loras: [{"path": str, "strength": float}, ...]

        Strategy:
          * Same paths in same order → just update strengths in place.
          * Different paths → remove all, load fresh.
        """
        if self.model is None:
            return

        new_paths = [l["path"] for l in loras]
        cur_paths = [l["path"] for l in self.loaded_loras]

        if new_paths == cur_paths:
            # Path-set unchanged; only strengths may have moved.
            for i, l in enumerate(loras):
                self.model.set_lora_strength(l["strength"], lora_index=i)
            self.loaded_loras = list(loras)
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
                # model_id intact — load needs it to know what to
                # reload. (Previous code zeroed it; the reload then raised
                # "Unknown SA3 model_id: None".)
                logger.warning(
                    "LoRA removal failed (%s); reloading base model %s",
                    exc, self.model_id,
                )
                self.model = None

        if self.model is None and self.model_id is not None:
            # Forced full reload (only if remove failed above).
            self.load(self.model_id, device=self.device, half=True)

        if loras:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.load_lora(new_paths)
            for i, l in enumerate(loras):
                self.model.set_lora_strength(l["strength"], lora_index=i)

        self.loaded_loras = list(loras)

    def set_lora_strength(self, index: int, strength: float) -> bool:
        """Live-update one slot's strength. Returns False if index invalid."""
        if not self.model or index < 0 or index >= len(self.loaded_loras):
            return False
        self.model.set_lora_strength(float(strength), lora_index=index)
        self.loaded_loras[index]["strength"] = float(strength)
        return True

    # --- sampling -------------------------------------------------------------
    def generate(self, request: GenerationRequest, on_step: StepCallback) -> np.ndarray:
        # The vendored generate() defaults sample_size=5292032 (=120s) and
        # _adapt_sample_size CLAMPS the seconds_total-derived length to it. Since
        # we never passed sample_size, every model — including medium — was
        # capped at 120s, so medium's advertised 380s never materialised. Pass
        # the model's native sample_size as the ceiling so _adapt sizes output to
        # the requested duration (plus its 6s padding) within the real maximum.
        target_samples = int(round(request.duration * 44100))
        try:
            native_sample_size = int(self.model.model_config.get("sample_size"))
        except Exception:
            native_sample_size = 0
        sample_size_ceiling = max(native_sample_size, target_samples)

        gen_kwargs = dict(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            duration=request.duration,
            sample_size=sample_size_ceiling,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=int(request.seed),
            batch_size=int(request.batch_size),
            chunked_decode=request.chunked_decode,
            callback=on_step,
        )
        if request.init_audio is not None:
            gen_kwargs["init_audio"] = _as_tensor(request.init_audio)
            gen_kwargs["init_noise_level"] = float(request.init_noise_level)
        if request.inpaint_audio is not None:
            gen_kwargs["inpaint_audio"] = _as_tensor(request.inpaint_audio)
            starts, ends = request.inpaint_starts, request.inpaint_ends
            if starts is not None and len(starts) > 0:
                # SA3 accepts a single float or a list for multi-region.
                gen_kwargs["inpaint_mask_start_seconds"] = (
                    list(starts) if len(starts) > 1 else float(starts[0])
                )
            if ends is not None and len(ends) > 0:
                gen_kwargs["inpaint_mask_end_seconds"] = (
                    list(ends) if len(ends) > 1 else float(ends[0])
                )

        audio = self.model.generate(**gen_kwargs)
        # audio: torch.Tensor[B, channels=2, samples] in [-1, 1] @ 44.1 kHz
        if audio.ndim != 3:
            raise RuntimeError(f"Unexpected SA3 output shape {tuple(audio.shape)}")
        return audio[0].detach().cpu().numpy()
