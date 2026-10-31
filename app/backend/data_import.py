"""Bring models, projects and outputs over from another Fragmenta folder.

So nobody re-downloads 2-9 GB after moving to an installed 2.0: point this at an
old source folder or data folder, pick what to bring, and it moves it (same
drive) or copies it. Nothing already here is ever overwritten. Copying never
changes the old folder.
"""
import errno
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from app.backend.data.projects import get_projects_dir
from app.core.config import get_config

DISK_MARGIN_BYTES = 1_000_000_000
_CHUNK = 8 * 1024 * 1024
# Half-finished downloads and their locks are no use anywhere else.
_LEFT_BEHIND_SUFFIXES = (".incomplete", ".lock")
_LEFT_BEHIND_NAMES = {"xet", ".locks", ".gitkeep", "__pycache__"}


class _Cancelled(Exception):
    pass


_lock = threading.Lock()
_job: Dict[str, Any] = {"state": "idle"}
_cancel = threading.Event()


def _items() -> List[Dict[str, Any]]:
    """What can come over. ``merge`` fills gaps inside folders (a model cache is
    named by content, so that's safe); otherwise each entry comes whole or not at
    all (two projects with one name aren't the same project)."""
    config = get_config()
    pretrained = config.get_path("models_pretrained")
    return [
        {"id": "models", "label": "Generation models", "src": "models/pretrained/sa3",
         "dst": pretrained / "sa3", "merge": True},
        {"id": "clap", "label": "CLAP annotation model", "src": "models/pretrained/clap",
         "dst": pretrained / "clap", "merge": True},
        {"id": "adapters", "label": "Trained LoRAs", "src": "models/fine_tuned",
         "dst": config.get_path("models_fine_tuned")},
        {"id": "projects", "label": "Dataset projects", "src": "projects", "dst": get_projects_dir()},
        {"id": "output", "label": "Generated audio", "src": "output", "dst": config.get_path("output")},
        {"id": "vocabulary", "label": "CLAP vocabulary", "src": "config/annotator_labels.user.json",
         "dst": Path(config.user_data_dir) / "config" / "annotator_labels.user.json"},
    ]


def _left_behind(path: Path) -> bool:
    return path.name in _LEFT_BEHIND_NAMES or path.name.endswith(_LEFT_BEHIND_SUFFIXES)


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _size(path: Path) -> int:
    """Bytes in this file or folder, not following links."""
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            full = Path(root) / name
            if not full.is_symlink() and not _left_behind(full):
                try:
                    total += full.stat().st_size
                except OSError:
                    pass
    return total


def _merge_steps(src: Path, dst: Path) -> Iterator[Tuple[Path, Path]]:
    if _left_behind(src):
        return
    if not _exists(dst):
        yield src, dst
    elif src.is_dir() and not src.is_symlink() and dst.is_dir():
        for child in sorted(src.iterdir()):
            yield from _merge_steps(child, dst / child.name)


def _steps(item: Dict[str, Any], source: Path) -> Dict[str, Any]:
    """The moves this item needs, plus how many of its entries are already here."""
    src, dst = source / item["src"], item["dst"]
    steps: List[Tuple[Path, Path]] = []
    entries = 0
    already = 0
    if src.is_file():
        entries = 1
        if _exists(dst):
            already = 1
        else:
            steps.append((src, dst))
    elif src.is_dir():
        for child in sorted(src.iterdir()):
            if _left_behind(child) or child.name.startswith(".") or child.name == "CACHEDIR.TAG":
                continue
            entries += 1
            target = dst / child.name
            found = list(_merge_steps(child, target)) if item.get("merge") else (
                [] if _exists(target) else [(child, target)])
            if found:
                steps += found
            else:
                already += 1
    return {"steps": steps, "entries": entries, "already": already}


def _deployed() -> bool:
    return (os.environ.get("FRAGMENTA_DOCKER", "0") == "1"
            or os.environ.get("FRAGMENTA_HF_SPACE", "0") == "1"
            or bool(os.environ.get("SPACE_ID")))


def _source_folder(path: Any) -> Tuple[Optional[Path], Optional[str]]:
    if not path or not isinstance(path, str):
        return None, "Choose a folder first."
    try:
        source = Path(path).expanduser().resolve()
    except (OSError, ValueError, RuntimeError):
        return None, "That folder can't be read."
    if not source.is_dir():
        return None, "That folder doesn't exist."
    here = Path(get_config().user_data_dir).resolve()
    if source == here or source in here.parents or here in source.parents:
        return None, "That's this Fragmenta's own folder. Choose the older one."
    return source, None


def scan(path: Any) -> Tuple[Dict[str, Any], int]:
    """What the folder holds that isn't here yet. Changes nothing."""
    if _deployed():
        return {"error": "Importing isn't available here."}, 403
    source, problem = _source_folder(path)
    if problem:
        return {"error": problem}, 400
    items = []
    for item in _items():
        plan = _steps(item, source)
        if plan["entries"]:
            items.append({
                "id": item["id"],
                "label": item["label"],
                "count": plan["entries"] - plan["already"],
                "already": plan["already"],
                "bytes": sum(_size(s) for s, _d in plan["steps"]),
            })
    if not items:
        return {"error": "No Fragmenta models, projects or audio in that folder."}, 400
    here = Path(get_config().user_data_dir)
    here.mkdir(parents=True, exist_ok=True)
    return {
        "source": str(source),
        "items": items,
        "same_drive": source.stat().st_dev == here.stat().st_dev,
        "free_bytes": shutil.disk_usage(here).free,
        "margin_bytes": DISK_MARGIN_BYTES,
    }, 200


def _update(**changes: Any) -> None:
    with _lock:
        _job.update(changes)


def _copy_file(src: Path, dst: Path) -> None:
    with open(src, "rb") as reader, open(dst, "wb") as writer:
        while True:
            if _cancel.is_set():
                raise _Cancelled()
            chunk = reader.read(_CHUNK)
            if not chunk:
                break
            writer.write(chunk)
            with _lock:
                _job["done_bytes"] += len(chunk)
    shutil.copystat(src, dst)


def _copy_link(src: Path, dst: Path) -> None:
    try:
        os.symlink(os.readlink(src), dst)
    except OSError:
        # Windows without the right to make links: bring the file it points at.
        if src.is_file():
            _copy_file(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True)
    for child in sorted(src.iterdir()):
        if _left_behind(child):
            continue
        target = dst / child.name
        if child.is_symlink():
            _copy_link(child, target)
        elif child.is_dir():
            _copy_tree(child, target)
        else:
            _copy_file(child, target)


def _copy_step(src: Path, dst: Path) -> None:
    # Under a temporary name, so a stopped copy never looks finished.
    temp = dst.with_name(f".importing-{dst.name}")
    try:
        if src.is_symlink():
            _copy_link(src, temp)
        elif src.is_dir():
            _copy_tree(src, temp)
        else:
            _copy_file(src, temp)
        os.replace(temp, dst)
    except BaseException:
        if temp.is_dir() and not temp.is_symlink():
            shutil.rmtree(temp, ignore_errors=True)
        elif _exists(temp):
            temp.unlink(missing_ok=True)
        raise


def _move_step(src: Path, dst: Path, size: int) -> None:
    try:
        os.rename(src, dst)
    except OSError:
        before = _job["done_bytes"]
        try:
            _copy_step(src, dst)
        except BaseException:
            _update(done_bytes=before)
            raise
        if src.is_dir() and not src.is_symlink():
            shutil.rmtree(src, ignore_errors=True)
        else:
            src.unlink(missing_ok=True)
        return
    with _lock:
        _job["done_bytes"] += size


def _run(steps: List[Tuple[Path, Path, int]], move: bool) -> None:
    imported = 0
    failed: List[Dict[str, str]] = []
    state = "done"
    error = None
    try:
        for src, dst, size in steps:
            if _cancel.is_set():
                raise _Cancelled()
            _update(current=src.name)
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if _exists(dst):  # appeared since the scan: still never overwrite
                    continue
                if move:
                    _move_step(src, dst, size)
                else:
                    _copy_step(src, dst)
                imported += 1
            except OSError as exc:
                if exc.errno == errno.ENOSPC:
                    raise
                failed.append({"name": src.name, "error": str(exc)})
    except _Cancelled:
        state = "cancelled"
    except OSError as exc:
        state = "failed"
        error = "The disk is full. Free up some space and try again." if exc.errno == errno.ENOSPC else str(exc)
    _update(state=state, error=error, imported=imported, failed=failed, current=None)


def start(path: Any, item_ids: Any, keep_old: Any, manager: Any) -> Tuple[Dict[str, Any], int]:
    if _deployed():
        return {"error": "Importing isn't available here."}, 403
    source, problem = _source_folder(path)
    if problem:
        return {"error": problem}, 400
    if any(j["status"] in ("queued", "running") for j in manager.list_jobs()):
        return {"error": "A model is downloading. Wait for it to finish, or cancel it first."}, 409

    wanted = set(item_ids) if isinstance(item_ids, list) else set()
    steps = []
    for item in _items():
        if item["id"] in wanted:
            steps += [(s, d, _size(s)) for s, d in _steps(item, source)["steps"]]
    if not steps:
        return {"error": "Nothing to import."}, 400

    here = Path(get_config().user_data_dir)
    here.mkdir(parents=True, exist_ok=True)
    move = not keep_old and source.stat().st_dev == here.stat().st_dev
    total = sum(size for _s, _d, size in steps)
    if not move:
        free = shutil.disk_usage(here).free
        if free < total + DISK_MARGIN_BYTES:
            needed = (total + DISK_MARGIN_BYTES) / 1e9
            return {"error": f"Not enough space (need {needed:.1f} GB, have {free / 1e9:.1f} GB)."}, 409

    with _lock:
        if _job.get("state") == "running":
            return {"error": "An import is already running."}, 409
        _cancel.clear()
        _job.clear()
        _job.update(state="running", mode="move" if move else "copy", total_bytes=total, done_bytes=0,
                    current=None, imported=0, failed=[], error=None)
    threading.Thread(target=_run, args=(steps, move), daemon=True, name="data-import").start()
    return status(), 200


def status() -> Dict[str, Any]:
    with _lock:
        return dict(_job)


def cancel() -> Dict[str, Any]:
    _cancel.set()
    return status()
