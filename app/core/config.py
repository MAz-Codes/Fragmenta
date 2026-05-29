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
        # The SA3 catalog lives in app/core/model_manager.py. This dict stays
        # empty; it's retained only because to_dict()/print_paths() and the
        # config validator still reference it.
        self.model_configs: Dict[str, Dict[str, str]] = {}

    def _ensure_directories(self) -> None:

        for path_name, path in self.paths.items():
            if path_name.endswith(('_fine_tuned', 'data')):
                path.mkdir(parents=True, exist_ok=True)

    def get_path(self, path_name: str) -> Path:
        if path_name not in self.paths:
            raise ValueError(f"Unknown path name: {path_name}")
        return self.paths[path_name]

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
