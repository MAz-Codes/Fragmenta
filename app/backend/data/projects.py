"""On-disk projects + buffered in-memory editing for SA3 sidecar datasets.

A *project* is a folder under `<user_data_dir>/projects/<name>/` (or wherever
`FRAGMENTA_PROJECTS_DIR` points) holding audio + `.txt` sidecar pairs plus a
hidden `.project.json` with Fragmenta metadata. The on-disk folder is the
**committed** dataset — what training reads, what survives across app
restarts.

The UI works against an **in-memory session** per loaded project. Prompt
edits, auto-annotate output, and just-ingested audio all live in memory
until the user explicitly persists them via:

  Save    → write `.draft.json` (transient, hidden). Survives app restart
            but is not the SA3 deliverable.
  Commit  → flush prompts to `.txt` sidecars, mark current audio as
            committed in `.project.json`, delete `.draft.json`.
  Discard → drop the in-memory session, delete `.draft.json`, remove any
            audio files added since the last commit.

See DATASET_PREP_REDESIGN.md for the full design and rationale.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.backend.data.auto_annotator import AUDIO_EXTENSIONS, _iter_audio_files

logger = logging.getLogger(__name__)

PROJECT_METADATA_FILENAME = ".project.json"
PROJECT_DRAFT_FILENAME = ".draft.json"
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
        from app.core.config import get_config
        root = get_config().user_data_dir / "projects"
    root.mkdir(parents=True, exist_ok=True)
    return root


def project_path(name: str) -> Path:
    return get_projects_dir() / name


def project_metadata_path(name: str) -> Path:
    return project_path(name) / PROJECT_METADATA_FILENAME


def project_draft_path(name: str) -> Path:
    return project_path(name) / PROJECT_DRAFT_FILENAME


# ---------- Validation ------------------------------------------------------


def sanitize_project_name(raw: Any) -> str:
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


# ---------- Disk persistence: committed state -------------------------------


def _default_metadata(name: str) -> Dict[str, Any]:
    now = time.time()
    return {
        "name": name,
        "created_at": now,
        "modified_at": now,
        "committed_at": None,
        "ingest_mode": DEFAULT_INGEST_MODE,
        "prompt_template": "",
        "source_folders": [],
        "committed_files": [],  # files written to disk + already committed
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


def _sidecar_for(audio_path: Path) -> Path:
    return audio_path.with_suffix(".txt")


def _read_sidecar(audio_path: Path) -> str:
    txt = _sidecar_for(audio_path)
    if not txt.exists():
        return ""
    try:
        return txt.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _write_sidecar(audio_path: Path, prompt: str) -> None:
    _sidecar_for(audio_path).write_text(prompt or "", encoding="utf-8")


# ---------- Disk persistence: draft state -----------------------------------


def _read_draft(name: str) -> Optional[Dict[str, Any]]:
    path = project_draft_path(name)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read draft %s: %s; treating as no draft.", path, exc)
        return None


def _write_draft(name: str, draft: Dict[str, Any]) -> None:
    draft["saved_at"] = time.time()
    path = project_draft_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(draft, f, indent=2)
    os.replace(tmp, path)


def _delete_draft(name: str) -> None:
    path = project_draft_path(name)
    if path.exists():
        path.unlink()


# ---------- In-memory session ----------------------------------------------


@dataclass
class ClipState:
    """One clip in an active project session.

    `prompt` is the live in-memory value (what the UI shows). `committed_prompt`
    is what's on disk in the sidecar — used to compute dirtiness.

    `parent` is the original clip's file_name if this clip was produced by a
    slice operation in the current session. In-memory only; not persisted
    across restart (yet). Future merge-back will need disk-level lineage.
    """
    file_name: str
    path: str
    prompt: str = ""
    committed_prompt: str = ""
    committed: bool = True   # False if audio was added since last commit
    parent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "path": self.path,
            "prompt": self.prompt,
            "committed_prompt": self.committed_prompt,
            "committed": self.committed,
            "dirty": self.prompt != self.committed_prompt,
            "parent": self.parent,
        }


@dataclass
class ProjectSession:
    """In-memory view of a project. One per loaded project name.

    Loading happens lazily on first GET. The session stays alive until
    the user discards, commits, or the process exits.
    """
    name: str
    clips: Dict[str, ClipState] = field(default_factory=dict)  # by file_name
    saved_at: Optional[float] = None        # last time .draft.json was written
    last_save_snapshot: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    # file_name -> (peaks, duration). Lazily filled by get_or_compute_peaks.
    # Cleared on Discard. Survives an annotate; safe to recompute on miss.
    peaks_cache: Dict[str, Tuple[List[float], float]] = field(default_factory=dict)
    # file_name -> duration_sec. Same lifecycle, but populated cheaply via
    # soundfile.info() instead of waiting for a peaks fetch.
    duration_cache: Dict[str, float] = field(default_factory=dict)
    # file_name -> sample_rate (Hz) and approximate loudness in dB.
    # Both populated lazily during a health check; cleared with the session.
    samplerate_cache: Dict[str, int] = field(default_factory=dict)
    loudness_cache: Dict[str, float] = field(default_factory=dict)

    def _draft_snapshot(self) -> Dict[str, str]:
        """Map file_name -> prompt, only for clips whose prompt differs from
        the committed sidecar. Used both to decide if a Save is needed and
        to compute the on-disk draft contents."""
        return {c.file_name: c.prompt for c in self.clips.values() if c.prompt != c.committed_prompt}

    def has_dirty_prompts(self) -> bool:
        return any(c.prompt != c.committed_prompt for c in self.clips.values())

    def has_uncommitted_files(self) -> bool:
        return any(not c.committed for c in self.clips.values())

    def has_unsaved_changes(self) -> bool:
        """True if the in-memory state differs from the saved draft."""
        return self._draft_snapshot() != self.last_save_snapshot

    def to_dict(self) -> Dict[str, Any]:
        ordered = sorted(self.clips.values(), key=lambda c: c.file_name)
        return {
            "name": self.name,
            "created_at": self.metadata.get("created_at"),
            "modified_at": self.metadata.get("modified_at"),
            "committed_at": self.metadata.get("committed_at"),
            "ingest_mode": self.metadata.get("ingest_mode", DEFAULT_INGEST_MODE),
            "prompt_template": self.metadata.get("prompt_template", ""),
            "source_folders": list(self.metadata.get("source_folders", [])),
            "saved_at": self.saved_at,
            "dirty": self.has_dirty_prompts() or self.has_uncommitted_files(),
            "has_unsaved_changes": self.has_unsaved_changes(),
            "uncommitted_files": [c.file_name for c in ordered if not c.committed],
            "clips": [c.to_dict() for c in ordered],
            "clip_count": len(self.clips),
        }


# Registry of active sessions keyed by project name.
_sessions: Dict[str, ProjectSession] = {}
_sessions_lock = threading.Lock()


def _get_or_load_session(name: str) -> ProjectSession:
    """Return the active session for `name`, loading from disk if needed."""
    with _sessions_lock:
        existing = _sessions.get(name)
        if existing is not None:
            return existing

    # Validate folder exists.
    path = project_path(name)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Project not found: {name}")

    metadata = _read_metadata(name)
    committed_files = set(metadata.get("committed_files") or [])

    # Build clip states from the disk layout. `committed_prompt` is whatever's
    # in the .txt sidecar today.
    clips: Dict[str, ClipState] = {}
    for audio_path in sorted(path.iterdir()):
        if not audio_path.is_file():
            continue
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        committed_prompt = _read_sidecar(audio_path)
        is_committed = audio_path.name in committed_files
        clips[audio_path.name] = ClipState(
            file_name=audio_path.name,
            path=str(audio_path),
            prompt=committed_prompt,
            committed_prompt=committed_prompt,
            committed=is_committed,
        )

    session = ProjectSession(name=name, clips=clips, metadata=metadata)

    # Overlay any draft prompts on top of committed values.
    draft = _read_draft(name)
    if draft:
        for file_name, prompt in (draft.get("prompts") or {}).items():
            clip = session.clips.get(file_name)
            if clip is not None:
                clip.prompt = prompt
        session.saved_at = draft.get("saved_at")
        session.last_save_snapshot = dict(draft.get("prompts") or {})

    with _sessions_lock:
        # Race: another thread may have loaded concurrently. Use whichever
        # got in first.
        existing = _sessions.get(name)
        if existing is not None:
            return existing
        _sessions[name] = session
        return session


def _drop_session(name: str) -> None:
    with _sessions_lock:
        _sessions.pop(name, None)


# ---------- CRUD ------------------------------------------------------------


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
        has_draft = project_draft_path(entry.name).exists()
        out.append({
            "name": entry.name,
            "created_at": meta.get("created_at"),
            "modified_at": meta.get("modified_at"),
            "committed_at": meta.get("committed_at"),
            "clip_count": clip_count,
            "has_draft": has_draft,
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
    session = _get_or_load_session(name)
    with session.lock:
        return session.to_dict()


def _stage_file(src: Path, dst: Path, mode: str) -> str:
    """Place `src` at `dst` using the requested ingest mode."""
    if dst.exists() or dst.is_symlink():
        return "skipped"
    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return "symlinked"
        except OSError as exc:
            logger.warning("Symlink failed for %s -> %s: %s; falling back to copy.", src, dst, exc)
            shutil.copy2(src, dst)
            return "copied"
    else:
        shutil.copy2(src, dst)
        return "copied"


def ingest_folder(name: str, source_folder: Path, mode: str) -> Dict[str, Any]:
    """Add every audio file under `source_folder` to project `name`.

    Audio is written to disk immediately (we don't buffer gigabytes). The
    new files are flagged as uncommitted in the session so a later Discard
    can remove them.
    """
    if mode not in INGEST_MODES:
        raise ValueError(f"Invalid ingest mode: {mode}")
    if not source_folder.exists() or not source_folder.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")

    session = _get_or_load_session(name)
    proj_path = project_path(name)

    files = _iter_audio_files(source_folder)
    if not files:
        raise ValueError(f"No audio files found in {source_folder}")

    copied = 0
    symlinked = 0
    skipped = 0
    with session.lock:
        for src in files:
            dst = proj_path / src.name
            tag = _stage_file(src, dst, mode)
            if tag == "copied":
                copied += 1
            elif tag == "symlinked":
                symlinked += 1
            else:
                skipped += 1
            if tag != "skipped" and src.name not in session.clips:
                # Newly added file — uncommitted.
                session.clips[src.name] = ClipState(
                    file_name=src.name,
                    path=str(dst),
                    prompt="",
                    committed_prompt="",
                    committed=False,
                )

        session.metadata["ingest_mode"] = mode
        src_abs = str(source_folder.resolve())
        if src_abs not in session.metadata.setdefault("source_folders", []):
            session.metadata["source_folders"].append(src_abs)

    return {
        "copied": copied,
        "symlinked": symlinked,
        "skipped": skipped,
        "added": copied + symlinked,
    }


def update_clip_prompt(name: str, file_name: str, prompt: str) -> Dict[str, Any]:
    """In-memory only. Disk is not touched until Save or Commit."""
    session = _get_or_load_session(name)
    with session.lock:
        clip = session.clips.get(file_name)
        if clip is None:
            raise FileNotFoundError(f"Clip not found in project '{name}': {file_name}")
        clip.prompt = prompt or ""
        return clip.to_dict()


def delete_clip(name: str, file_name: str) -> None:
    """Remove a clip immediately (audio + sidecar + session entry).

    Treated like ingest: the disk change happens now, since carrying a
    pending-deletion in memory complicates everything for no real win.
    Discard cannot recover deleted files.
    """
    session = _get_or_load_session(name)
    proj_path = project_path(name)
    with session.lock:
        audio_path = proj_path / file_name
        txt_path = _sidecar_for(audio_path)
        if audio_path.exists():
            audio_path.unlink()
        if txt_path.exists():
            txt_path.unlink()
        session.clips.pop(file_name, None)
        # Evict any cached peaks for this file (regardless of N).
        for key in list(session.peaks_cache):
            if key.startswith(f"{file_name}:"):
                del session.peaks_cache[key]
        session.duration_cache.pop(file_name, None)
        session.samplerate_cache.pop(file_name, None)
        session.loudness_cache.pop(file_name, None)
        committed = session.metadata.get("committed_files") or []
        if file_name in committed:
            session.metadata["committed_files"] = [f for f in committed if f != file_name]


# ---------- Save / Commit / Discard -----------------------------------------


def save_project(name: str) -> Dict[str, Any]:
    """Persist the current in-memory prompt diffs as a hidden draft."""
    session = _get_or_load_session(name)
    with session.lock:
        snapshot = session._draft_snapshot()
        draft = {
            "prompts": snapshot,
            "uncommitted_files": [c.file_name for c in session.clips.values() if not c.committed],
        }
        _write_draft(name, draft)
        session.saved_at = time.time()
        session.last_save_snapshot = dict(snapshot)
        return session.to_dict()


def commit_project(name: str) -> Dict[str, Any]:
    """Flush in-memory state to disk as the canonical SA3 dataset.

    Overwrites existing sidecars. Marks all current audio as committed.
    Deletes any draft.
    """
    session = _get_or_load_session(name)
    proj_path = project_path(name)
    with session.lock:
        # Write a sidecar for every clip, even if the prompt didn't change.
        # This guarantees the on-disk state is exactly the in-memory state
        # after Commit, no surprises.
        for clip in session.clips.values():
            audio_path = proj_path / clip.file_name
            _write_sidecar(audio_path, clip.prompt)
            clip.committed_prompt = clip.prompt
            clip.committed = True

        session.metadata["committed_files"] = sorted(session.clips.keys())
        session.metadata["committed_at"] = time.time()
        _write_metadata(name, session.metadata)
        _delete_draft(name)
        session.saved_at = None
        session.last_save_snapshot = {}
        return session.to_dict()


def delete_project(name: str) -> None:
    """Permanently remove a project — folder, sidecars, drafts, session.

    Destructive: there is no recovery path. Caller should confirm with
    the user before invoking.
    """
    proj_path = project_path(name)
    if not proj_path.exists():
        raise FileNotFoundError(f"Project not found: {name}")

    # Cancel any in-flight annotate first, drop the session, then nuke
    # the folder. Order matters: if we rm the folder while another
    # thread is writing to it (e.g. annotate writing prompts to memory
    # is fine, but the audio-stream endpoint could be holding a file
    # handle), at least the session is gone so no fresh writes happen.
    with _sessions_lock:
        existing = _sessions.pop(name, None)
    if existing is not None:
        existing.cancel_event.set()
    shutil.rmtree(proj_path, ignore_errors=True)


def discard_project(name: str) -> Dict[str, Any]:
    """Throw away all uncommitted work.

    - Delete the draft.
    - Delete audio files added since the last commit (and their sidecars).
    - Drop the in-memory session so the next GET rebuilds from disk.
    """
    session = _get_or_load_session(name)
    proj_path = project_path(name)
    with session.lock:
        # Cancel any in-flight annotate before we tear state apart.
        session.cancel_event.set()

        uncommitted = [c.file_name for c in session.clips.values() if not c.committed]
        for file_name in uncommitted:
            audio_path = proj_path / file_name
            txt_path = _sidecar_for(audio_path)
            if audio_path.exists():
                audio_path.unlink()
            if txt_path.exists():
                txt_path.unlink()

        _delete_draft(name)

    _drop_session(name)
    return get_project(name)


# ---------- Annotate cancellation handle ------------------------------------


def get_session_handle(name: str) -> ProjectSession:
    """Used by the annotate endpoint to share a cancel handle + clip dict."""
    return _get_or_load_session(name)


def reset_cancel(session: ProjectSession) -> None:
    session.cancel_event.clear()


# ---------- Waveform peaks --------------------------------------------------


def _compute_peaks(audio_path: Path, n: int) -> Tuple[List[float], float]:
    """Return N normalized peak amplitudes + duration in seconds.

    Reads N short blocks at evenly spaced offsets via soundfile.seek instead
    of decoding the whole file. ~40x faster than librosa.load on a typical
    30s clip; bounded I/O regardless of file length (a 5-minute clip costs
    the same as a 30s one).

    Falls back to a librosa-based decode for formats soundfile can't open
    on this build (typically m4a/aac without ffmpeg-libsndfile).
    """
    import numpy as np
    try:
        import soundfile as sf
        with sf.SoundFile(str(audio_path)) as src:
            total = src.frames
            sr = src.samplerate
            if total == 0:
                return ([0.0] * n, 0.0)
            duration = float(total / sr)
            # ~6 buckets-worth of samples per probe gives stable peaks without
            # devolving into "read the whole file."
            block = max(256, total // (n * 6))
            peaks = np.zeros(n, dtype="float32")
            for i in range(n):
                center = int((i + 0.5) * total / n)
                start = max(0, center - block // 2)
                src.seek(start)
                data = src.read(block, dtype="float32", always_2d=False)
                if data.ndim > 1:
                    data = data.max(axis=1)
                if len(data):
                    peaks[i] = float(np.abs(data).max())
            max_peak = float(peaks.max())
            if max_peak > 0:
                peaks = peaks / max_peak
            return (peaks.tolist(), duration)
    except Exception as exc:
        logger.debug("soundfile peak path failed for %s (%s); falling back to librosa", audio_path.name, exc)

    # Fallback: librosa.load handles every codec we register, at the cost of
    # a full-file decode + resample. Slower but bulletproof.
    import librosa
    y, sr = librosa.load(str(audio_path), sr=8000, mono=True)
    if len(y) == 0:
        return ([0.0] * n, 0.0)
    duration = float(len(y) / sr)
    chunks = np.array_split(y, n)
    peaks = np.array([float(np.abs(c).max()) if len(c) else 0.0 for c in chunks])
    max_peak = peaks.max()
    if max_peak > 0:
        peaks = peaks / max_peak
    return (peaks.tolist(), duration)


def get_or_compute_peaks(
    session: ProjectSession,
    file_name: str,
    audio_path: Path,
    n: int = 200,
) -> Tuple[List[float], float]:
    """Memoized per-session peak computation. Cache key is `file_name:N`."""
    cache_key = f"{file_name}:{n}"
    cached = session.peaks_cache.get(cache_key)
    if cached is not None:
        return cached
    result = _compute_peaks(audio_path, n)
    session.peaks_cache[cache_key] = result
    return result


# ---------- Health checks ---------------------------------------------------


def _clip_duration_sec(audio_path: Path) -> Optional[float]:
    """Cheap duration probe via soundfile.info() — header read, no decode."""
    try:
        import soundfile as sf
        info = sf.info(str(audio_path))
        if info.samplerate <= 0:
            return None
        return float(info.frames / info.samplerate)
    except Exception:
        return None


def _clip_samplerate(audio_path: Path) -> Optional[int]:
    """Header-only sample-rate probe."""
    try:
        import soundfile as sf
        return int(sf.info(str(audio_path)).samplerate)
    except Exception:
        return None


def _clip_loudness_db(audio_path: Path) -> Optional[float]:
    """Approximate RMS dB via the same N-probe stride as the peak computer.

    Not LUFS — we don't have pyloudnorm in the venv and it's not worth the
    extra dep for a health-check rough indicator. Reads ~200 short blocks
    spread across the file (~1-2 ms per clip regardless of duration).
    """
    try:
        import soundfile as sf
        import numpy as np
        with sf.SoundFile(str(audio_path)) as src:
            total = src.frames
            if total == 0:
                return None
            n_probes = 200
            probe_size = max(64, total // (n_probes * 6))
            sq_sum = 0.0
            samples = 0
            for i in range(n_probes):
                center = int((i + 0.5) * total / n_probes)
                start = max(0, center - probe_size // 2)
                src.seek(start)
                data = src.read(probe_size, dtype="float32", always_2d=False)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if len(data):
                    sq_sum += float(np.sum(data ** 2))
                    samples += len(data)
            if samples == 0:
                return None
            mean_sq = sq_sum / samples
            if mean_sq <= 1e-12:
                return -120.0
            return float(20.0 * np.log10(np.sqrt(mean_sq)))
    except Exception:
        return None


def compute_health(
    name: str,
    long_threshold_sec: float = 30.0,
    short_threshold_sec: float = 1.0,
    loudness_outlier_db: float = 6.0,
) -> Dict[str, Any]:
    """Per-clip checks that surface dataset problems before training.

    Duration defaults match the SA3 LoRA --duration default (30s) and the
    practical "this is so short it's mostly silence-padding" floor (1s).
    Loudness outliers: |dB - median_dB| > loudness_outlier_db (default 6).
    """
    import statistics
    from collections import defaultdict

    session = _get_or_load_session(name)
    with session.lock:
        clips = list(session.clips.values())

    empty_prompts: List[str] = []
    too_long: List[str] = []
    too_short: List[str] = []
    sr_by_file: Dict[str, int] = {}
    loudness_by_file: Dict[str, float] = {}
    prompt_groups: Dict[str, List[str]] = defaultdict(list)

    for c in clips:
        if not (c.prompt or "").strip():
            empty_prompts.append(c.file_name)
        else:
            prompt_groups[c.prompt.strip().lower()].append(c.file_name)

        # Duration (header-only, ~free)
        dur = session.duration_cache.get(c.file_name)
        if dur is None:
            dur = _clip_duration_sec(Path(c.path))
            if dur is not None:
                session.duration_cache[c.file_name] = dur
        if dur is not None:
            if dur > long_threshold_sec:
                too_long.append(c.file_name)
            elif dur < short_threshold_sec:
                too_short.append(c.file_name)

        # Sample rate (header-only)
        sr = session.samplerate_cache.get(c.file_name)
        if sr is None:
            sr = _clip_samplerate(Path(c.path))
            if sr is not None:
                session.samplerate_cache[c.file_name] = sr
        if sr is not None:
            sr_by_file[c.file_name] = sr

        # Loudness (cheap stride read, ~1-2 ms per clip)
        loud = session.loudness_cache.get(c.file_name)
        if loud is None:
            loud = _clip_loudness_db(Path(c.path))
            if loud is not None:
                session.loudness_cache[c.file_name] = loud
        if loud is not None:
            loudness_by_file[c.file_name] = loud

    # --- Mixed sample rates: dominant SR is the majority; minority gets flagged.
    sr_counts: Dict[int, int] = {}
    for sr in sr_by_file.values():
        sr_counts[sr] = sr_counts.get(sr, 0) + 1
    dominant_sr = max(sr_counts, key=sr_counts.get) if sr_counts else None
    sr_minority = (
        sorted([f for f, sr in sr_by_file.items() if sr != dominant_sr])
        if dominant_sr is not None and len(sr_counts) > 1
        else []
    )

    # --- Loudness outliers: |x - median| > threshold dB
    loudness_outliers: List[str] = []
    median_db = None
    if loudness_by_file:
        median_db = float(statistics.median(loudness_by_file.values()))
        loudness_outliers = sorted([
            f for f, v in loudness_by_file.items()
            if abs(v - median_db) > loudness_outlier_db
        ])

    # --- Duplicate annotations: any non-empty prompt shared by 2+ clips.
    dup_groups = [files for files in prompt_groups.values() if len(files) > 1]
    dup_files = sorted({f for group in dup_groups for f in group})

    empty_prompts.sort()
    too_long.sort()
    too_short.sort()

    return {
        "total_clips": len(clips),
        "empty_prompts": {"count": len(empty_prompts), "files": empty_prompts},
        "too_long": {
            "count": len(too_long),
            "threshold_sec": long_threshold_sec,
            "files": too_long,
        },
        "too_short": {
            "count": len(too_short),
            "threshold_sec": short_threshold_sec,
            "files": too_short,
        },
        "mixed_sample_rates": {
            "count": len(sr_minority),
            "dominant_sr": dominant_sr,
            "rates": sorted(sr_counts.keys()),
            "files": sr_minority,
        },
        "loudness_outliers": {
            "count": len(loudness_outliers),
            "median_db": median_db,
            "threshold_db": loudness_outlier_db,
            "files": loudness_outliers,
        },
        "duplicate_annotations": {
            "count": len(dup_files),
            "group_count": len(dup_groups),
            "files": dup_files,
        },
    }


# ---------- Slicing ---------------------------------------------------------


def slice_clip(
    name: str,
    file_name: str,
    target_sec: float,
    overlap_sec: float,
    strategy: str,
) -> Dict[str, Any]:
    """Split one clip into N children. Disk-level — happens immediately.

    The parent file (and its sidecar) is deleted. Each child:
      - lives in the project folder as `<stem>__NNN.wav`
      - inherits the parent's in-memory prompt verbatim
      - is uncommitted (so Discard rolls it back)
      - keeps `parent=<parent_file_name>` in its session state

    Discard cannot recover the parent file from children — same rule as
    delete_clip. Commit makes the slice permanent.
    """
    from app.backend.data.slicing import plan_slices, write_slices

    session = _get_or_load_session(name)
    proj_path = project_path(name)
    audio_path = proj_path / file_name

    if not audio_path.exists():
        raise FileNotFoundError(f"Clip not on disk: {file_name}")

    plans = plan_slices(audio_path, target_sec, overlap_sec, strategy)
    if len(plans) <= 1:
        raise ValueError(
            f"{file_name} is shorter than the target duration "
            f"({target_sec:.1f}s); nothing to slice."
        )

    stem = audio_path.stem
    children = write_slices(audio_path, plans, proj_path, stem)
    if not children:
        raise RuntimeError("Slice produced no children — check the audio file.")

    with session.lock:
        parent_clip = session.clips.get(file_name)
        inherited_prompt = parent_clip.prompt if parent_clip else ""

        # Remove the parent from session + disk.
        session.clips.pop(file_name, None)
        for key in list(session.peaks_cache):
            if key.startswith(f"{file_name}:"):
                del session.peaks_cache[key]
        session.duration_cache.pop(file_name, None)
        session.samplerate_cache.pop(file_name, None)
        session.loudness_cache.pop(file_name, None)
        sidecar = _sidecar_for(audio_path)
        if audio_path.exists():
            audio_path.unlink()
        if sidecar.exists():
            sidecar.unlink()
        committed = session.metadata.get("committed_files") or []
        if file_name in committed:
            session.metadata["committed_files"] = [f for f in committed if f != file_name]

        # Register children as uncommitted clips with parent linkage.
        for child_path in children:
            session.clips[child_path.name] = ClipState(
                file_name=child_path.name,
                path=str(child_path),
                prompt=inherited_prompt,
                committed_prompt="",
                committed=False,
                parent=file_name,
            )

    return {
        "parent": file_name,
        "children": [
            {"file_name": p.name, "start_sec": pl.start_sec, "end_sec": pl.end_sec}
            for p, pl in zip(children, plans)
        ],
        "project": get_project(name),
    }
