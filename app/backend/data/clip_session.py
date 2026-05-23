"""In-memory dataset session for the SA3 dataset-prep redesign.

A `ClipSession` is a working set of audio clips being prepared for SA3 training.
Sessions live only in process memory for Phase 1 — there is no persistence to
disk yet. The session is the *source of truth* for the dataset-prep UI; the
shipped artifact (SA3 sidecar folder) is produced by an explicit export step,
not by any in-place editing of `data/metadata.json`.

See `DATASET_PREP_REDESIGN.md` for the full design.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from app.backend.data.auto_annotator import _iter_audio_files


@dataclass
class Clip:
    """A single audio clip in a dataset session.

    `path` is the absolute filesystem path the clip currently lives at (the
    user's original location — we never modify source files in place). `prompt`
    is the caption that will be written to `<basename>.txt` on export. `locked`
    means the prompt was hand-edited or hand-imported and should not be
    silently overwritten by a bulk auto-annotate.
    """

    path: str
    file_name: str
    prompt: str = ""
    locked: bool = False


@dataclass
class ClipSession:
    id: str
    created_at: float = field(default_factory=time.time)
    clips: List[Clip] = field(default_factory=list)

    def add_audio_folder(self, folder: Path) -> int:
        """Append all audio files under `folder` to the session.

        Returns the number of new clips added. Existing clips (same `path`)
        are not duplicated — re-adding a folder is a no-op for already-present
        files, which keeps the ingest button idempotent.
        """
        existing = {c.path for c in self.clips}
        added = 0
        for audio_path in _iter_audio_files(folder):
            path_str = str(audio_path)
            if path_str in existing:
                continue
            self.clips.append(Clip(path=path_str, file_name=audio_path.name))
            added += 1
        return added

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "clip_count": len(self.clips),
            "clips": [asdict(c) for c in self.clips],
        }


_sessions: Dict[str, ClipSession] = {}
_sessions_lock = threading.Lock()


def create_session() -> ClipSession:
    """Mint a new empty session with a short random id."""
    session_id = uuid.uuid4().hex[:12]
    session = ClipSession(id=session_id)
    with _sessions_lock:
        _sessions[session_id] = session
    return session


def get_session(session_id: str) -> Optional[ClipSession]:
    with _sessions_lock:
        return _sessions.get(session_id)
