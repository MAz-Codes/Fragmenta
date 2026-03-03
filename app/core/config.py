import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json

class ProjectConfig:

    def __init__(self, project_root: Optional[Path] = None) -> None:
        if getattr(sys, 'frozen', False):
            # Running in PyInstaller bundle
            self.frozen = True
            # sys._MEIPASS is where PyInstaller unpacks the bundle
            self.project_root = Path(sys._MEIPASS)
            
            # For writable data, use a user directory
            if sys.platform == "win32":
                self.user_data_dir = Path(os.environ["APPDATA"]) / "FragmentaDesktop"
            elif sys.platform == "darwin":
                self.user_data_dir = Path.home() / "Library" / "Application Support" / "FragmentaDesktop"
            else:
                # Linux/Unix
                self.user_data_dir = Path.home() / ".local" / "share" / "FragmentaDesktop"
                
            self.user_data_dir.mkdir(parents=True, exist_ok=True)
            print(f"Running in frozen mode. Project root: {self.project_root}")
            print(f"User data directory: {self.user_data_dir}")
            
        else:
            self.frozen = False
            if project_root is None:
                config_file_dir = Path(__file__).parent.parent.parent
                if (config_file_dir / "requirements.txt").exists() and (config_file_dir / "app" / "frontend").exists():
                    project_root = config_file_dir
                else:
                    current_dir = Path.cwd()
                    for parent in [current_dir] + list(current_dir.parents):
                        if (parent / "requirements.txt").exists() and (parent / "app" / "frontend").exists():
                            project_root = parent
                            break
                    else:
                        project_root = config_file_dir
            
            self.project_root: Path = Path(project_root).resolve()
            self.user_data_dir = self.project_root

        self.paths: Dict[str, Path] = {
            # Writable paths - go to user_data_dir in frozen mode
            "models": self.user_data_dir / "models",
            "models_config": self.user_data_dir / "models" / "config",
            "models_pretrained": self.user_data_dir / "models" / "pretrained",
            "models_fine_tuned": self.user_data_dir / "models" / "fine_tuned",
            "data": self.user_data_dir / "data",
            "logs": self.user_data_dir / "logs",
            "output": self.user_data_dir / "output",
            
            # Read-only attributes/codebase - stay in project_root
            "application": self.project_root,
            "backend": self.project_root / "app" / "backend",
            "frontend": self.project_root / "app" / "frontend",
            "stable_audio_tools": self.project_root / "stable-audio-tools",
            "venv": self.project_root / "venv",
        }

        self._ensure_directories()
        self.model_configs: Dict[str, Dict[str, str]
                                 ] = self._load_model_configs()

    def _ensure_directories(self) -> None:

        for path_name, path in self.paths.items():
            if path_name.endswith(('_fine_tuned', 'data')):
                path.mkdir(parents=True, exist_ok=True)

    def _load_model_configs(self) -> Dict[str, Dict[str, str]]:

        return {
            "stable-audio-open-1.0": {
                "config": str(self.paths["models_config"] / "model_config.json"),
                "ckpt": str(self.paths["models_pretrained"] / "stable-audio-open-model.safetensors")
            },
            "stable-audio-open-small": {
                "config": str(self.paths["models_config"] / "model_config_small.json"),
                "ckpt": str(self.paths["models_pretrained"] / "stable-audio-open-small-model.safetensors")
            },
            "custom": {
                "config": str(self.paths["models_config"] / "model_config_small.json"),
                "ckpt": str(self.paths["models_pretrained"] / "stable-audio-open-small-model.safetensors")
            }
        }

    def get_path(self, path_name: str) -> Path:
        if path_name not in self.paths:
            raise ValueError(f"Unknown path name: {path_name}")
        return self.paths[path_name]

    def get_model_config(self, model_name: str) -> Dict[str, str]:
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        return self.model_configs[model_name]

    def get_dataset_config_path(self) -> str:
        return str(self.paths["models_config"] / "dataset-config.json")

    def get_custom_metadata_path(self) -> str:
        return str(self.project_root / "stable-audio-tools" / "custom_metadata.py")

    def get_metadata_json_path(self) -> str:
        return str(self.paths["data"] / "metadata.json")

    def update_dataset_config(self) -> None:
        from app.backend.data.simple_audio_processor import SimpleAudioProcessor
        
        try:
            processor = SimpleAudioProcessor(
                model_config_path=self.paths["models_config"] / "model_config.json"
            )
            
            result = processor.create_dataset_config(
                input_dir=self.paths["data"],
                output_dir=self.paths["data"]
            )
            
            target_config = self.paths["models_config"] / "dataset-config.json"
            with open(target_config, 'w') as f:
                json.dump(result["dataset_config"], f, indent=4)
            
            print(f"Updated dataset config: {target_config}")
            print(f"Points to {result['file_count']} original audio files")
            print(f"Sample size: {result['sample_size']} samples ({result['sample_size']/result['sample_rate']:.1f}s)")
            print(f"Random cropping during training (correct!)")
            
        except Exception as e:
            print(f"Failed to update dataset config: {e}")
            print("Falling back to basic dataset config...")
            
            dataset_config: Dict[str, Any] = {
                "dataset_type": "audio_dir",
                "datasets": [
                    {
                        "id": "fine_tune_data",
                        "path": str(self.paths["data"].relative_to(self.project_root)),
                        "custom_metadata_module": "custom_metadata"
                    }
                ],
                "random_crop": True
            }

            config_path = self.paths["models_config"] / "dataset-config.json"
            with open(config_path, 'w') as f:
                json.dump(dataset_config, f, indent=4)

            print(f"Updated fallback dataset config: {config_path}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "paths": {name: str(path) for name, path in self.paths.items()},
            "model_configs": self.model_configs
        }

    def print_paths(self) -> None:
        print("Project Configuration:")
        print(f"Project Root: {self.project_root}")
        print("\nPaths:")
        for name, path in self.paths.items():
            print(f"{name}: {path}")
        print("\nModel Configs:")
        for name, config in self.model_configs.items():
            print(f"{name}: {config}")


_config_instance: Optional[ProjectConfig] = None


def get_config() -> ProjectConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = ProjectConfig()
    return _config_instance


def set_config(config: ProjectConfig) -> None:
    global _config_instance
    _config_instance = config


def get_default_config() -> ProjectConfig:
    return get_config()
