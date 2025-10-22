import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import sys
import time

sys.path.append(
    str(Path(__file__).parent.parent.parent.parent / "stable-audio-tools"))

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
        logger.info(f"Using device: {self.device}")

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
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate audio from a text prompt

        Args:
            prompt: Text description of the audio to generate
            model_path: Path to fine-tuned model directory
            unwrapped_model_path: Path to unwrapped .safetensors file
            config_file: Model config file to use (small or large)
            duration: Duration in seconds
            cfg_scale: Classifier-free guidance scale
            steps: Number of diffusion steps
            seed: Random seed (-1 for random)
            output_path: Optional path to save the generated audio

        Returns:
            Path to the generated audio file
        """
        print(f"\nAUDIO GENERATOR: generate_audio called")
        print(f"   - Prompt: '{prompt}'")
        print(f"   - Duration: {duration}s")

        if self.model is None or model_path is not None or unwrapped_model_path is not None:
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
        else:
            print(f"AUDIO GENERATOR: Using existing model")

        print(f"AUDIO GENERATOR: Model loaded successfully")

        try:
            print(f"Generating audio for prompt: '{prompt}'")
            print(f"Duration: {duration}s, CFG scale: {cfg_scale}, Steps: {steps}")
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
            conditioning = [{
                "prompt": prompt,
                "seconds_start": 0,
                "seconds_total": int(duration)
            }]

            device = next(self.model.parameters()).device
            print(f"Using device: {device}")

            audio = generate_diffusion_cond(
                model=self.model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=conditioning,
                batch_size=1,
                sample_size=requested_sample_size,
                seed=seed,
                device=str(device),
                sigma_min=0.03,
                sigma_max=1000,
                sampler_type="dpmpp-3m-sde"
            )

            print(f"Generation complete, audio shape: {audio.shape}")

            from einops import rearrange
            audio = rearrange(audio, "b d n -> d (b n)").to(torch.float32)
            audio = audio / audio.abs().max()
            audio_int16 = (audio.clamp(-1, 1) * 32767).to(torch.int16).cpu()

            if output_path is None:
                output_dir = Path(__file__).parent.parent.parent.parent / "output"
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"generated_{int(time.time())}.wav"

            self.save_audio(audio_int16, output_path, self.model.sample_rate)

            print(f"AUDIO GENERATOR: Generation complete")
            print(f"   - Output file: {output_path}")
            print(f"   - Output file size: {output_path.stat().st_size} bytes")

            return output_path

        except Exception as e:
            print(f"AUDIO GENERATOR: Error during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

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
        torchaudio.save(str(output_path), audio, sample_rate)

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
