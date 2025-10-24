import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def fast_scandir(dir_path, ext_list):
    import os
    subfolders, files = [], []
    # add starting period to extensions if needed
    ext_list = ['.'+x if x[0] != '.' else x for x in ext_list]

    try:
        for f in os.scandir(dir_path):
            try:
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")

                    if file_ext in ext_list and not is_hidden:
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext_list)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


class SimpleAudioProcessor:

    def __init__(self, model_config_path: Optional[Path] = None):
        self.audio_extensions = (".wav", ".mp3", ".flac", ".m4a")
        
        # Load model config for info only
        if model_config_path and model_config_path.exists():
            with open(model_config_path, 'r') as f:
                model_config = json.load(f)
            self.sample_size = model_config.get("sample_size", 2097152)
            self.sample_rate = model_config.get("sample_rate", 44100)
            self.audio_channels = model_config.get("audio_channels", 2)
        else:
            # Defaults
            self.sample_size = 2097152
            self.sample_rate = 44100
            self.audio_channels = 2

    def load_prompts(self, prompts_file: Path) -> Dict[str, str]:
        prompts = {}
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '|' in line:
                        filename, prompt = line.split('|', 1)
                        prompts[filename.strip()] = prompt.strip()
        except Exception as e:
            logger.error(f"Error loading prompts file: {e}")
        return prompts

    def create_dataset_config(
        self,
        input_dir: Path,
        output_dir: Path,
        prompts_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        # Find audio files
        audio_files = []
        for ext in self.audio_extensions:
            _, files = fast_scandir(str(input_dir), [ext[1:]])
            audio_files.extend(files)

        if not audio_files:
            raise ValueError(f"No audio files found in {input_dir}")

        logger.info(f"Found {len(audio_files)} audio files")

        # Create output directory
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Copy files to output directory (only if different directories)
        if input_dir != output_dir:
            import shutil
            for audio_file in audio_files:
                src_path = Path(audio_file)
                dst_path = output_dir / src_path.name
                
                if not dst_path.exists() or dst_path.stat().st_size != src_path.stat().st_size:
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Copied {src_path.name}")
        else:
            logger.info("Input and output directories are the same - no copying needed")

        # Create simple dataset config
        dataset_config = {
            "dataset_type": "audio_dir",
            "datasets": [
                {
                    "id": "custom_dataset", 
                    "path": str(output_dir),
                    "custom_metadata_module": "custom_metadata"
                }
            ],
            "random_crop": True,  # CRITICAL - enables random cropping during training
            "drop_last": True
        }

        # Save prompts if provided
        if prompts_file and prompts_file.exists():
            prompts = self.load_prompts(prompts_file)
            if prompts:
                metadata_file = output_dir / "prompts_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump([{"file_name": k, "prompt": v} for k, v in prompts.items()], f, indent=2)
                logger.info(f"Saved prompts metadata")

        return {
            "dataset_config": dataset_config,
            "file_count": len(audio_files),
            "sample_size": self.sample_size,
            "sample_rate": self.sample_rate,
            "audio_channels": self.audio_channels
        }
