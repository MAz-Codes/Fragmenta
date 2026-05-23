"""On-disk project model for SA3 sidecar-native datasets.

A *project* is a folder under `<user_data_dir>/projects/<name>/` (or the
location pointed at by `FRAGMENTA_PROJECTS_DIR`) containing:

  - audio files (.wav/.mp3/.flac/.m4a/.ogg/.aac)
  - one `<basename>.txt` sidecar per audio file (SA3's caption format)
  - a hidden `.project.json` with Fragmenta metadata (template, ingest
    history, per-clip lock flags)

The on-disk state is the source of truth. There's no in-memory session —
reads scan the folder, writes update sidecars + metadata immediately. This
removes the "export" step entirely: training can point at the project
folder unchanged.

See DATASET_PREP_REDESIGN.md for the full design.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.backend.data.auto_annotator import AUDIO_EXTENSIONS, _iter_audio_files

logger = logging.getLogger(__name__)

PROJECT_METADATA_FILENAME = ".project.json"
DEFAULT_INGEST_MODE = "copy"  # copy | symlink
INGEST_MODES = ("copy", "symlink")

# Names must look like reasonable filesystem folders.
_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _\-.]{0,99}$")


# ---------- Locations -------------------------------------------------------


def get_projects_dir() -> Path:
    """Resolve the projects root.

    Honors `FRAGMENTA_PROJECTS_DIR` for power users; otherwise sits next to
    `data/` and `models/` under the configured user_data_dir.
    """
    override = os.environ.get("FRAGMENTA_PROJECTS_DIR")
    if override:
        root = Path(override).expanduser()
    else:
        # Imported lazily to keep this module callable from tests that
        # don't need the full config bootstrap.
        from app.core.config import get_config
        root = get_config().user_data_dir / "projects"
    root.mkdir(parents=True, exist_ok=True)
    return root


def project_path(name: str) -> Path:
    return get_projects_dir() / name


def project_metadata_path(name: str) -> Path:
    return project_path(name) / PROJECT_METADATA_FILENAME


# ---------- Validation ------------------------------------------------------


def sanitize_project_name(raw: Any) -> str:
    """Validate a user-typed project name.

    The name becomes a folder name, so reject anything that would let the
    user escape the projects dir or create a hidden / unreadable folder.
    """
    if not isinstance(raw, str):
        raise ValueError("Project name must be a string.")
    name = raw.strip()
    if not name:
        raise ValueError("Project name cannot be empty.")
    if name in (".", ".."):
        raise ValueError("Invalid project name.")
    if not _VALID_NAME_RE.match(name):
        raise ValueError(
            "Project name must start with a letter or digit and may only "
            "contain letters, digits, spaces, dashes, underscores, and dots."
        )
    return name


# ---------- Metadata --------------------------------------------------------


def _default_metadata(name: str) -> Dict[str, Any]:
    now = time.time()
    return {
        "name": name,
        "created_at": now,
        "modified_at": now,
        "ingest_mode": DEFAULT_INGEST_MODE,
        "prompt_template": "",
        "source_folders": [],
        "clips": {},  # file_name -> {"locked": bool, ...}
    }


def _read_metadata(name: str) -> Dict[str, Any]:
    path = project_metadata_path(name)
    if not path.exists():
        return _default_metadata(name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read project metadata %s: %s; using defaults.", path, exc)
        return _default_metadata(name)
    # Backfill missing keys so older projects don't error.
    defaults = _default_metadata(name)
    for k, v in defaults.items():
        data.setdefault(k, v)
    return data


def _write_metadata(name: str, metadata: Dict[str, Any]) -> None:
    metadata["modified_at"] = time.time()
    path = project_metadata_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    os.replace(tmp, path)


# ---------- Sidecars --------------------------------------------------------


def _sidecar_for(audio_path: Path) -> Path:
    return audio_path.with_suffix(".txt")


def read_sidecar(audio_path: Path) -> str:
    txt = _sidecar_for(audio_path)
    if not txt.exists():
        return ""
    try:
        return txt.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def write_sidecar(audio_path: Path, prompt: str) -> None:
    _sidecar_for(audio_path).write_text(prompt or "", encoding="utf-8")


# ---------- CRUD ------------------------------------------------------------


@dataclass
class ClipView:
    """Lightweight view of a clip — recomputed on each read from disk."""

    file_name: str
    path: str
    prompt: str
    locked: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "path": self.path,
            "prompt": self.prompt,
            "locked": self.locked,
        }


def list_projects() -> List[Dict[str, Any]]:
    root = get_projects_dir()
    out: List[Dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        try:
            meta = _read_metadata(entry.name)
        except Exception as exc:
            logger.warning("Skipping project %s: %s", entry.name, exc)
            continue
        clip_count = sum(
            1 for f in entry.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        )
        out.append({
            "name": entry.name,
            "created_at": meta.get("created_at"),
            "modified_at": meta.get("modified_at"),
            "clip_count": clip_count,
        })
    return out


def create_project(name: str) -> Dict[str, Any]:
    name = sanitize_project_name(name)
    path = project_path(name)
    if path.exists():
        raise FileExistsError(f"Project '{name}' already exists.")
    path.mkdir(parents=True)
    metadata = _default_metadata(name)
    _write_metadata(name, metadata)
    return get_project(name)


def get_project(name: str) -> Dict[str, Any]:
    path = project_path(name)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Project not found: {name}")
    meta = _read_metadata(name)

    clips: List[ClipView] = []
    for audio_path in sorted(path.iterdir()):
        if not audio_path.is_file():
            continue
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        prompt = read_sidecar(audio_path)
        locked = bool(meta.get("clips", {}).get(audio_path.name, {}).get("locked"))
        clips.append(ClipView(
            file_name=audio_path.name,
            path=str(audio_path),
            prompt=prompt,
            locked=locked,
        ))

    return {
        "name": name,
        "created_at": meta.get("created_at"),
        "modified_at": meta.get("modified_at"),
        "ingest_mode": meta.get("ingest_mode", DEFAULT_INGEST_MODE),
        "prompt_template": meta.get("prompt_template", ""),
        "source_folders": list(meta.get("source_folders", [])),
        "clips": [c.to_dict() for c in clips],
        "clip_count": len(clips),
    }


def _stage_file(src: Path, dst: Path, mode: str) -> str:
    """Place `src` at `dst` using the requested ingest mode.

    Returns a short tag describing what happened: 'copied' | 'symlinked' |
    'skipped'. Existing files at `dst` are left alone (idempotent ingest).
    """
    if dst.exists() or dst.is_symlink():
        return "skipped"
    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return "symlinked"
        except OSError as exc:
            # Filesystem may not support symlinks (e.g. some Windows mounts);
            # fall back to copy so the user still gets a working ingest.
            logger.warning("Symlink failed for %s -> %s: %s; falling back to copy.", src, dst, exc)
            shutil.copy2(src, dst)
            return "copied"
    else:
        shutil.copy2(src, dst)
        return "copied"


def ingest_folder(name: str, source_folder: Path, mode: str) -> Dict[str, Any]:
    """Add every audio file under `source_folder` to project `name`.

    `mode` is 'copy' or 'symlink'. Audio that already exists at the
    destination is skipped (re-ingest is idempotent).
    """
    if mode not in INGEST_MODES:
        raise ValueError(f"Invalid ingest mode: {mode}")
    if not source_folder.exists() or not source_folder.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")

    proj_path = project_path(name)
    if not proj_path.exists():
        raise FileNotFoundError(f"Project not found: {name}")

    files = _iter_audio_files(source_folder)
    if not files:
        raise ValueError(f"No audio files found in {source_folder}")

    copied = 0
    symlinked = 0
    skipped = 0
    for src in files:
        dst = proj_path / src.name
        tag = _stage_file(src, dst, mode)
        if tag == "copied":
            copied += 1
        elif tag == "symlinked":
            symlinked += 1
        else:
            skipped += 1

    meta = _read_metadata(name)
    meta["ingest_mode"] = mode
    src_abs = str(source_folder.resolve())
    if src_abs not in meta["source_folders"]:
        meta["source_folders"].append(src_abs)
    _write_metadata(name, meta)

    return {
        "copied": copied,
        "symlinked": symlinked,
        "skipped": skipped,
        "added": copied + symlinked,
    }


def update_clip(name: str, file_name: str, *, prompt: Optional[str] = None, locked: Optional[bool] = None) -> Dict[str, Any]:
    """Edit a single clip's prompt and/or lock state.

    Writes the .txt sidecar to disk immediately. Lock state lives in
    `.project.json` since the .txt only holds the prompt itself.
    """
    proj_path = project_path(name)
    audio_path = proj_path / file_name
    if not audio_path.exists():
        raise FileNotFoundError(f"Clip not found in project '{name}': {file_name}")

    if prompt is not None:
        write_sidecar(audio_path, prompt)

    if locked is not None:
        meta = _read_metadata(name)
        clips = meta.setdefault("clips", {})
        clip_meta = clips.setdefault(file_name, {})
        clip_meta["locked"] = bool(locked)
        _write_metadata(name, meta)

    # Return the canonical view of this clip after the write.
    meta = _read_metadata(name)
    return ClipView(
        file_name=file_name,
        path=str(audio_path),
        prompt=read_sidecar(audio_path),
        locked=bool(meta.get("clips", {}).get(file_name, {}).get("locked")),
    ).to_dict()


def delete_clip(name: str, file_name: str) -> None:
    """Remove a clip's audio + sidecar from a project."""
    proj_path = project_path(name)
    audio_path = proj_path / file_name
    txt_path = _sidecar_for(audio_path)
    if audio_path.exists():
        audio_path.unlink()
    if txt_path.exists():
        txt_path.unlink()
    meta = _read_metadata(name)
    meta.get("clips", {}).pop(file_name, None)
    _write_metadata(name, meta)
