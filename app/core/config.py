import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json


def _default_user_data_dir() -> Path:
    """Writable per-user data dir for packaged builds.

    MUST stay in sync with ``install.py``'s ``_user_data_dir`` so the venv,
    models, output and logs all resolve to the same place.
    """
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "FragmentaDesktop"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "FragmentaDesktop"
    return Path.home() / ".local" / "share" / "FragmentaDesktop"


class ProjectConfig:

    def __init__(self, project_root: Optional[Path] = None) -> None:
        frozen = bool(getattr(sys, 'frozen', False))
        # "Packaged" = shipped desktop build. Two shapes resolve here:
        #   * PyInstaller-frozen process (sys.frozen) — code at sys._MEIPASS.
        #   * The bootstrapper launcher: start.py runs under the venv's normal
        #     Python (NOT frozen), so the native launcher sets FRAGMENTA_PACKAGED=1
        #     to flag that the code sits in a read-only bundle and data must go
        #     to the writable user-data dir.
        packaged = frozen or os.environ.get("FRAGMENTA_PACKAGED") == "1"

        if packaged:
            self.frozen = frozen
            if frozen:
                # PyInstaller unpacks the bundle to sys._MEIPASS.
                self.project_root = Path(sys._MEIPASS)
            else:
                # Read-only bundle: this file is app/core/config.py, so the code
                # root (holding app/, vendor/, requirements.txt) is three up —
                # …/Resources on macOS, the install dir on Windows.
                self.project_root = Path(__file__).resolve().parent.parent.parent

            self.user_data_dir = _default_user_data_dir()
            self.user_data_dir.mkdir(parents=True, exist_ok=True)
            print(f"Running packaged (frozen={frozen}). Project root: {self.project_root}")
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

        # Scratch area for browser folder uploads (/api/upload-folder). The
        # SA2-era "data" dataset directory is gone in 0.2.0 — datasets are now
        # Dataset Workbench projects under projects/.
        uploads_override = os.environ.get("FRAGMENTA_UPLOADS_DIR")
        uploads_dir = Path(uploads_override) if uploads_override else self.user_data_dir / "uploads"

        self.paths: Dict[str, Path] = {
            "models": self.user_data_dir / "models",
            "models_config": self.user_data_dir / "models" / "config",
            "models_pretrained": self.user_data_dir / "models" / "pretrained",
            "models_fine_tuned": fine_tuned_dir,
            "uploads": uploads_dir,
            "logs": self.user_data_dir / "logs",
            "output": self.user_data_dir / "output",

            "application": self.project_root,
            "backend": self.project_root / "app" / "backend",
            "frontend": self.project_root / "app" / "frontend",
            "stable_audio_3": self.project_root / "vendor" / "stable-audio-3",
            # venv lives with the writable data (== project_root in source mode,
            # the user-data dir in a packaged build).
            "venv": self.user_data_dir / "venv",
        }

        self._ensure_directories()
        # The SA3 catalog lives in app/core/model_manager.py. This dict stays
        # empty; it's retained only because to_dict()/print_paths() and the
        # config validator still reference it.
        self.model_configs: Dict[str, Dict[str, str]] = {}

    def _ensure_directories(self) -> None:

        for path_name, path in self.paths.items():
            if path_name.endswith(('_fine_tuned', 'uploads')):
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
