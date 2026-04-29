import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import re
import sys
import threading
import time
import warnings
from datetime import datetime


class GenerationStopped(Exception):
    """Raised by the per-step callback when a stop has been requested."""
    pass


def _slugify_prompt(text: str, max_len: int = 40) -> str:
    s = re.sub(r'[^a-zA-Z0-9]+', '_', text.strip().lower())
    s = re.sub(r'_+', '_', s).strip('_')
    return s[:max_len] or 'untitled'

sys.path.append(
    str(Path(__file__).parent.parent.parent.parent / "stable-audio-tools"))


warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models import create_model_from_config

logger = logging.getLogger(__name__)


class AudioGenerator:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_model_name = None
        self.current_model_path = None
        # Caller-facing key for the currently loaded weights. Used to short-
        # circuit reloads when generate_audio is invoked repeatedly with the
        # same model — without this, every Generate click reloads from disk.
        self.current_model_key = None
        self.is_distilled_small = False
        self._stop_event = threading.Event()
        logger.info(f"Using device: {self.device}")

    def request_stop(self) -> bool:
        """Signal the in-flight diffusion loop (if any) to abort at the next step."""
        already_set = self._stop_event.is_set()
        self._stop_event.set()
        return not already_set

    def load_local_base_model(self, model_name: str = "stable-audio-open-small") -> bool:
        try:
            logger.info(f"Loading local base model: {model_name}")
            
            self.current_model_name = model_name
            
            from stable_audio_tools.models.factory import create_model_from_config
            from stable_audio_tools.models.utils import load_ckpt_state_dict
            if "small" in model_name:
                config_file = "model_config_small.json"
            else:
                config_file = "model_config.json"
            self.is_distilled_small = "small" in model_name.lower()

            config_path = Path(__file__).parent.parent.parent.parent / "models" / "config" / config_file
            logger.info(f"Using config file: {config_path}")
            
            with open(config_path, 'r') as f:
                import json
                model_config = json.load(f)
            
            self.model = create_model_from_config(model_config)
            if model_name == 'stable-audio-open-small':
                model_file_name = 'stable-audio-open-small-model.safetensors'
            elif model_name == 'stable-audio-open-1.0':
                model_file_name = 'stable-audio-open-model.safetensors'
            else:
                model_file_name = f"{model_name}-model.safetensors"
            
            model_file = Path(__file__).parent.parent.parent.parent / "models" / "pretrained" / model_file_name
            self.current_model_path = str(model_file)
            logger.info(f"Loading weights from: {model_file}")
            
            if not model_file.exists():
                raise FileNotFoundError(f"Local model file not found: {model_file}")
            
            state_dict = load_ckpt_state_dict(str(model_file))
            self.model.load_state_dict(state_dict, strict=False)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model.requires_grad_(False)
            if self.device.startswith("cuda"):
                self.model = torch.compile(self.model, mode="reduce-overhead")

            logger.info("Local base model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load local base model: {e}")
            return False

    def load_model(self, model_path: Optional[Path] = None) -> bool:
        try:
            print(f"Loading model from {model_path}")

            if model_path is None:
                return self.load_local_base_model("stable-audio-open-small")
            else:
                safetensors_files = list(model_path.glob("*.safetensors"))
                if safetensors_files:
                    unwrapped_path = str(safetensors_files[0])
                    print(f"Found safetensors file: {unwrapped_path}")
                    return self.load_unwrapped_model(unwrapped_path)
                else:
                    print(f"No safetensors files found in {model_path}, using local base model")
                    return self.load_local_base_model("stable-audio-open-small")

        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def load_unwrapped_model(self, unwrapped_model_path: str, config_file: str = None) -> bool:
        try:
            print(f"Loading unwrapped model from {unwrapped_model_path}")

            self.current_model_path = unwrapped_model_path

            from stable_audio_tools.models.factory import create_model_from_config
            from stable_audio_tools.models.utils import load_ckpt_state_dict
            if config_file is None:
                config_file = "model_config_small.json"
            self.is_distilled_small = "small" in config_file.lower()

            config_path = Path(__file__).parent.parent.parent.parent / \
                "models" / "config" / config_file
            print(f"Using config file: {config_path}")

            with open(config_path, 'r') as f:
                import json
                model_config = json.load(f)

            self.model = create_model_from_config(model_config)

            state_dict = load_ckpt_state_dict(unwrapped_model_path)
            self.model.load_state_dict(state_dict, strict=False)

            self.model = self.model.to(self.device)
            self.model.eval()
            self.model.requires_grad_(False)

            if self.device.startswith("cuda"):
                self.model = torch.compile(self.model, mode="reduce-overhead")

            print(f"AUDIO GENERATOR: Unwrapped model loaded successfully")
            return True

        except Exception as e:
            print(f"Failed to load unwrapped model: {e}")
            return False

    def generate_audio(
        self,
        prompt: str,
        model_path: Optional[Path] = None,
        unwrapped_model_path: Optional[str] = None,
        config_file: Optional[str] = None,
        duration: float = 10.0,
        cfg_scale: float = 7.0,
        steps: int = 250,
        seed: int = -1,
        output_path: Optional[Path] = None,
        batch_index: int = 1,
        batch_total: int = 1,
        loop_mode: bool = False,
    ) -> Path:
        print(f"\nAUDIO GENERATOR: generate_audio called")
        print(f"   - Prompt: '{prompt}'")
        print(f"   - Duration: {duration}s")

        # Build a cache key for the requested model so we can reuse weights
        # across consecutive Generate clicks on the same model.
        if unwrapped_model_path:
            target_key = ('unwrapped', str(unwrapped_model_path))
        elif model_path:
            target_key = ('path', str(model_path))
        else:
            target_key = ('default', 'stable-audio-open-small')

        if self.model is not None and self.current_model_key == target_key:
            print(f"AUDIO GENERATOR: Reusing already-loaded model")
        else:
            print(f"AUDIO GENERATOR: Loading new model")

            if unwrapped_model_path:
                print(f"AUDIO GENERATOR: Loading unwrapped model from {unwrapped_model_path}")
                if not self.load_unwrapped_model(unwrapped_model_path, config_file):
                    raise ValueError(f"Failed to load unwrapped model from {unwrapped_model_path}")
            elif model_path:
                model_path_str = str(model_path)
                print(f"AUDIO GENERATOR: Checking model path: {model_path_str}")

                if "stable-audio-open-small" in model_path_str:
                    print(f"AUDIO GENERATOR: Loading local small base model")
                    if not self.load_local_base_model("stable-audio-open-small"):
                        raise ValueError("Failed to load local small base model")
                elif "stable-audio-open-model" in model_path_str:
                    print(f"AUDIO GENERATOR: Loading local large base model")
                    if not self.load_local_base_model("stable-audio-open-1.0"):
                        raise ValueError("Failed to load local large base model")
                else:
                    print(f"AUDIO GENERATOR: Loading fine-tuned model from {model_path}")
                    if not self.load_model(model_path):
                        raise ValueError(f"Failed to load model from {model_path}")
            else:
                print(f"AUDIO GENERATOR: Loading default local small base model")
                if not self.load_local_base_model("stable-audio-open-small"):
                    raise ValueError("Failed to load default local base model")

            self.current_model_key = target_key

        print(f"AUDIO GENERATOR: Model loaded successfully")

        # A new generation invalidates any prior stop request.
        self._stop_event.clear()

        def _stop_callback(state):
            if self._stop_event.is_set():
                raise GenerationStopped("Stop requested mid-diffusion")

        try:
            # Stable Audio Open Small is an adversarially-distilled checkpoint
            # that requires the pingpong sampler at 8 steps with CFG 1.0.
            # Running the dpmpp-3m-sde recipe on it produces noise.
            if self.is_distilled_small:
                effective_sampler = "pingpong"
                effective_steps = 8
                effective_cfg = 1.0
                sigma_kwargs = {}
            else:
                effective_sampler = "dpmpp-3m-sde"
                effective_steps = steps
                effective_cfg = cfg_scale
                sigma_kwargs = {"sigma_min": 0.03, "sigma_max": 1000}

            print(f"Generating audio for prompt: '{prompt}'")
            print(
                f"Duration: {duration}s, CFG scale: {effective_cfg}, "
                f"Steps: {effective_steps}, Sampler: {effective_sampler}"
                + (" (distilled small overrides applied)" if self.is_distilled_small else "")
            )
            requested_sample_size = int(duration * self.model.sample_rate)
            max_sample_size = None
            try:
                max_sample_size = self.model.sample_size
            except AttributeError:
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'sample_size'):
                    max_sample_size = self.model.model.sample_size
                else:
                    config_path = Path(__file__).parent.parent.parent.parent / "models" / "config"
                    if hasattr(self, 'current_model_name') and self.current_model_name:
                        if 'small' in self.current_model_name:
                            config_file = config_path / "model_config_small.json"
                        else:
                            config_file = config_path / "model_config.json"
                    else:
                        if hasattr(self, 'current_model_path') and self.current_model_path:
                            model_file = Path(self.current_model_path)
                            if model_file.exists():
                                file_size_gb = model_file.stat().st_size / (1024**3)
                                if file_size_gb < 2.0:
                                    config_file = config_path / "model_config_small.json"
                                else:
                                    config_file = config_path / "model_config.json"
                            else:
                                config_file = config_path / "model_config_small.json"
                        else:
                            config_file = config_path / "model_config_small.json"

                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            import json
                            config_data = json.load(f)
                            max_sample_size = config_data.get('sample_size', 44100 * 10)
                    else:
                        max_sample_size = 44100 * 10
            if max_sample_size and requested_sample_size > max_sample_size:
                print(f"Requested duration {duration}s exceeds model maximum. Truncating.")
                requested_sample_size = max_sample_size
                duration = requested_sample_size / self.model.sample_rate

            if seed == -1:
                import numpy as np
                seed = np.random.randint(0, 2**32 - 1, dtype=np.int64)

            print(f"Using seed: {seed}")

            # In loop_mode (bars mode in the performance panel), tell the model
            # the song is much longer than what we render. SAO 1.0 was trained
            # to fade out as it approaches `seconds_total`, so matching it to
            # the requested duration bakes a song-ending fade into the clip.
            # We still get back exactly `requested_sample_size` samples — the
            # model just thinks they're the opening of a longer piece.
            if loop_mode and max_sample_size:
                song_seconds = max(int(duration),
                                   int(max_sample_size / self.model.sample_rate))
            else:
                song_seconds = int(duration)

            conditioning = [{
                "prompt": prompt,
                "seconds_start": 0,
                "seconds_total": song_seconds,
            }]

            device = next(self.model.parameters()).device
            print(f"Using device: {device}")

            with warnings.catch_warnings():
                # Known torchsde float-boundary chatter from dpmpp-3m-sde.
                warnings.filterwarnings(
                    "ignore",
                    message=r"Should have tb<=t1 but got tb=.*",
                    category=UserWarning,
                    module=r"torchsde\._brownian\.brownian_interval",
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r"Should have ta>=t0 but got ta=.*",
                    category=UserWarning,
                    module=r"torchsde\._brownian\.brownian_interval",
                )

                audio = generate_diffusion_cond(
                    model=self.model,
                    steps=effective_steps,
                    cfg_scale=effective_cfg,
                    conditioning=conditioning,
                    batch_size=1,
                    sample_size=requested_sample_size,
                    seed=seed,
                    device=str(device),
                    sampler_type=effective_sampler,
                    callback=_stop_callback,
                    **sigma_kwargs,
                )

            print(f"Generation complete, audio shape: {audio.shape}")

            from einops import rearrange
            audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32)
            audio = audio / audio.abs().max()
            audio_int16 = (audio.clamp(-1, 1) * 32767).to(torch.int16).cpu()

            if output_path is None:
                output_dir = Path(__file__).parent.parent.parent.parent / "output"
                output_dir.mkdir(exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                slug = _slugify_prompt(prompt)
                suffix = f"_{batch_index}" if batch_total > 1 else ""
                output_path = output_dir / f"fragmenta_{ts}_{slug}{suffix}.wav"

            self.save_audio(audio_int16, output_path, self.model.sample_rate)

            print(f"AUDIO GENERATOR: Generation complete")
            print(f"   - Output file: {output_path}")
            print(f"   - Output file size: {output_path.stat().st_size} bytes")

            return output_path

        except GenerationStopped:
            print("AUDIO GENERATOR: Generation stopped by user request")
            raise
        except Exception as e:
            print(f"AUDIO GENERATOR: Error during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self._stop_event.clear()

    def generate_batch(
        self,
        prompts: List[str],
        duration: float = 10.0,
        cfg_scale: float = 6.0,
        steps: int = 250,
        seed: int = -1,
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        results = []

        for i, prompt in enumerate(prompts):
            print(f"Generating audio {i+1}/{len(prompts)}")

            current_seed = seed if seed != -1 else seed + i
            output_path = None
            if output_dir:
                output_dir.mkdir(exist_ok=True, parents=True)
                output_path = output_dir / f"generated_{i+1:03d}.wav"

            try:
                output_path = self.generate_audio(
                    prompt=prompt,
                    duration=duration,
                    cfg_scale=cfg_scale,
                    steps=steps,
                    seed=current_seed,
                    output_path=output_path
                )
                results.append(output_path)

            except Exception as e:
                print(f"Failed to generate audio for prompt {i+1}: {e}")
                results.append(None)

        return results

    def save_audio(self, audio: torch.Tensor, output_path: Path, sample_rate: int):
        output_path.parent.mkdir(exist_ok=True, parents=True)
        audio_np = audio.detach().cpu().transpose(0, 1).numpy()
        sf.write(str(output_path), audio_np, sample_rate, subtype="PCM_16")

    def get_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {"status": "no_model_loaded"}

        return {
            "status": "loaded",
            "sample_rate": self.model.sample_rate,
            "device": str(self.device),
            "model_type": getattr(self.model, 'model_type', 'unknown'),
            "io_channels": getattr(self.model, 'io_channels', 'unknown')
        }
