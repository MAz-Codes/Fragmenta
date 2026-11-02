"""The Performance page's state as files in the data folder.

Session, presets, MIDI settings and the audio of every channel fragment used to
live only in the window's own storage, which a new install, Import and
Uninstall can't see. Here they are plain files under ``<data>/performance``:

    session.json             the live session
    midi.json                MIDI devices, channels, takeover, mappings
    presets/<name>.json      one file per preset
    fragments/<scope>/<id>.<ext>
                             channel audio; scope is session-ch<N> or
                             preset-<name>-ch<N>, as the page names them

Names become file names with every unsafe character %-escaped. Off on the
Space and in Docker, where visitors share one backend: the page keeps its
state in the window there.
"""
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote

from app.core.config import get_config

STATE_NAMES = ("session", "midi")
_EXT = {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav", "audio/mpeg": ".mp3",
        "audio/flac": ".flac", "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/aac": ".aac"}
_TYPE = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg",
         ".m4a": "audio/mp4", ".aac": "audio/aac"}


def available() -> bool:
    return not (os.environ.get("FRAGMENTA_DOCKER", "0") == "1"
                or os.environ.get("FRAGMENTA_HF_SPACE", "0") == "1"
                or bool(os.environ.get("SPACE_ID")))


def root() -> Path:
    return get_config().get_path("performance")


def file_name(name: str) -> str:
    if not isinstance(name, str) or not name or len(name) > 200:
        raise ValueError("bad name")
    # quote leaves '.' alone, so '.' and '..' would still walk out of the folder.
    return quote(name, safe="").replace(".", "%2E")


def preset_scope(name: str, channel: int) -> str:
    """The page's presetChannelScope(): encodeURIComponent of the name."""
    encoded = quote(name, safe="-_.!~*'()")
    return f"preset-{encoded}-ch{channel}"


def _write(path: Path, data: bytes) -> None:
    """Whole file or nothing: a crash mid-write leaves the old file. Not
    mkstemp, which makes the file private (0600) where the rest of the data
    folder follows the umask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp-{os.getpid()}-{threading.get_ident()}-{path.name}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load() -> Dict[str, Any]:
    """Everything but the audio. A missing file is None, so the page knows to
    move its own copy over; presets is None until the folder exists."""
    base = root()
    state: Dict[str, Any] = {name: _read_json(base / f"{name}.json") for name in STATE_NAMES}
    presets_dir = base / "presets"
    if presets_dir.is_dir():
        state["presets"] = {}
        for path in sorted(presets_dir.glob("*.json")):
            value = _read_json(path)
            if value is not None:
                state["presets"][unquote(path.stem)] = value
    else:
        state["presets"] = None
    return state


def save_state(name: str, value: Any) -> None:
    if name not in STATE_NAMES:
        raise ValueError("bad name")
    _write(root() / f"{name}.json", json.dumps(value).encode("utf-8"))


def delete_state(name: str) -> None:
    if name not in STATE_NAMES:
        raise ValueError("bad name")
    (root() / f"{name}.json").unlink(missing_ok=True)


def save_preset(name: str, value: Any) -> None:
    _write(root() / "presets" / f"{file_name(name)}.json", json.dumps(value).encode("utf-8"))


def delete_preset(name: str) -> None:
    (root() / "presets" / f"{file_name(name)}.json").unlink(missing_ok=True)
    # The folder stays, even empty: it says the presets have been moved over.


def _scope_dir(scope: str) -> Path:
    return root() / "fragments" / file_name(scope)


def _fragment_path(scope: str, fragment_id: str) -> Optional[Path]:
    folder = _scope_dir(scope)
    stem = file_name(fragment_id)
    if not folder.is_dir():
        return None
    for path in folder.iterdir():
        if path.name.rsplit(".", 1)[0] == stem:
            return path
    return None


def put_fragment(scope: str, fragment_id: str, data: bytes, content_type: str) -> None:
    old = _fragment_path(scope, fragment_id)
    ext = _EXT.get((content_type or "").split(";")[0].strip().lower(), ".bin")
    path = _scope_dir(scope) / f"{file_name(fragment_id)}{ext}"
    _write(path, data)
    if old is not None and old != path:
        old.unlink(missing_ok=True)


def get_fragment(scope: str, fragment_id: str) -> Optional[Path]:
    return _fragment_path(scope, fragment_id)


def content_type(path: Path) -> str:
    return _TYPE.get(path.suffix.lower(), "application/octet-stream")


def delete_fragment(scope: str, fragment_id: str) -> None:
    path = _fragment_path(scope, fragment_id)
    if path is not None:
        path.unlink(missing_ok=True)


def clear_scope(scope: str) -> None:
    shutil.rmtree(_scope_dir(scope), ignore_errors=True)


def copy_scope(src: str, dst: str) -> int:
    """Copy every fragment of one scope into another; existing ones there are
    replaced, others kept, as the page's IndexedDB copy did."""
    src_dir, dst_dir = _scope_dir(src), _scope_dir(dst)
    if not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in src_dir.iterdir():
        if path.name.startswith(".tmp-") or not path.is_file():
            continue
        stem = path.name.rsplit(".", 1)[0]
        for old in dst_dir.glob(f"{stem}.*"):
            if old.name.rsplit(".", 1)[0] == stem:
                old.unlink(missing_ok=True)
        shutil.copy2(path, dst_dir / path.name)
        count += 1
    return count


def list_fragments() -> List[Dict[str, str]]:
    base = root() / "fragments"
    found: List[Dict[str, str]] = []
    if not base.is_dir():
        return found
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and not path.name.startswith(".tmp-"):
                found.append({"scope": unquote(folder.name), "id": unquote(path.name.rsplit(".", 1)[0])})
    return found
