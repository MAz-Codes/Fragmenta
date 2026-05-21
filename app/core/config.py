import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json


class ProjectConfig:

    def __init__(self, project_root: Optional[Path] = None) -> None:
        if getattr(sys, 'frozen', False):
            self.frozen = True
            # PyInstaller unpacks the bundle to sys._MEIPASS; writable data lives elsewhere.
            self.project_root = Path(sys._MEIPASS)

            if sys.platform == "win32":
                self.user_data_dir = Path(os.environ["APPDATA"]) / "FragmentaDesktop"
            elif sys.platform == "darwin":
                self.user_data_dir = Path.home() / "Library" / "Application Support" / "FragmentaDesktop"
            else:
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

        fine_tuned_override = os.environ.get("FRAGMENTA_FINE_TUNED_DIR")
        fine_tuned_dir = Path(fine_tuned_override) if fine_tuned_override else self.user_data_dir / "models" / "fine_tuned"

        data_override = os.environ.get("FRAGMENTA_DATA_DIR")
        data_dir = Path(data_override) if data_override else self.user_data_dir / "data"

        self.paths: Dict[str, Path] = {
            "models": self.user_data_dir / "models",
            "models_config": self.user_data_dir / "models" / "config",
            "models_pretrained": self.user_data_dir / "models" / "pretrained",
            "models_fine_tuned": fine_tuned_dir,
            "data": data_dir,
            "logs": self.user_data_dir / "logs",
            "output": self.user_data_dir / "output",

            "application": self.project_root,
            "backend": self.project_root / "app" / "backend",
            "frontend": self.project_root / "app" / "frontend",
            "stable_audio_3": self.project_root / "vendor" / "stable-audio-3",
            "venv": self.project_root / "venv",
        }

        self._ensure_directories()
        # SA3-TODO(Phase 2): populated by the Checkpoint Manager catalog. Empty until then.
        self.model_configs: Dict[str, Dict[str, str]] = {}

    def _ensure_directories(self) -> None:

        for path_name, path in self.paths.items():
            if path_name.endswith(('_fine_tuned', 'data')):
                path.mkdir(parents=True, exist_ok=True)

    def get_path(self, path_name: str) -> Path:
        if path_name not in self.paths:
            raise ValueError(f"Unknown path name: {path_name}")
        return self.paths[path_name]

    def get_model_config(self, model_name: str) -> Dict[str, str]:
        # SA3-TODO(Phase 2): Checkpoint Manager owns the catalog now.
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name} (SA3 catalog not yet wired)")
        return self.model_configs[model_name]

    def get_dataset_config_path(self) -> str:
        # SA3-TODO(Phase 5): SA3 reads <basename>.txt sidecars; legacy dataset-config.json
        # is obsolete. Path is preserved for callers but the file is no longer generated.
        return str(self.paths["models_config"] / "dataset-config.json")

    def get_custom_metadata_path(self) -> str:
        # SA3-TODO(Phase 5): replaced by app/core/training/sa3_lora_runner.py caption materializer.
        return ""

    def get_metadata_json_path(self) -> str:
        return str(self.paths["data"] / "metadata.json")

    def update_dataset_config(self) -> None:
        # SA3-TODO(Phase 5): SA3 datasets use <basename>.txt sidecars (see Appendix B
        # in SA3_INTEGRATION_PLAN.md). Fragmenta no longer maintains a dataset-config.json.
        # This is a no-op pending the new caption materializer in sa3_lora_runner.py.
        return

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
