"""The Bending Log — Kotowski & Font's compositional technique as a store.

Every bent generation is appended automatically (layer/operation/result
row); the user adds the one-line sonic-result note afterwards. Entries are
per-model filterable and any row can be recalled into the rack (the full
patch is stored with each entry).

Storage: bends/log.json — plain JSON so the log doubles as research
documentation the user can read and version outside the app. Writes are
serialized with a process-local lock (Flask threads); the file is small
(capped) so read-modify-write is fine.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

MAX_ENTRIES = 500
_NOTE_MAX = 500


class BendLog:
    def __init__(self, bends_dir: Path):
        self.path = Path(bends_dir) / "log.json"
        self._lock = threading.Lock()

    def _read(self) -> List[Dict[str, Any]]:
        try:
            with open(self.path) as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except FileNotFoundError:
            return []
        except Exception:
            # A corrupt log must never block generation; start fresh but
            # keep the broken file for forensics.
            try:
                self.path.rename(self.path.with_suffix(".json.corrupt"))
            except Exception:
                pass
            return []

    def _write(self, entries: List[Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump(entries, fh, indent=1)
        tmp.replace(self.path)

    def append(self, *, model_id: str, patch: Dict[str, Any], prompt: str,
               seed: int, duration: float, steps: Optional[int],
               fragment: Optional[str],
               warnings: Optional[List[str]] = None) -> str:
        entry = {
            "id": uuid.uuid4().hex[:12],
            "ts": time.time(),
            "model_id": model_id,
            "patch": patch,
            "prompt": prompt,
            "seed": int(seed),
            "duration": float(duration),
            "steps": int(steps) if steps else None,
            "fragment": fragment,
            "warnings": list(warnings or []),
            "note": "",
        }
        with self._lock:
            entries = self._read()
            entries.append(entry)
            if len(entries) > MAX_ENTRIES:
                entries = entries[-MAX_ENTRIES:]
            self._write(entries)
        return entry["id"]

    def set_note(self, entry_id: str, note: str) -> bool:
        with self._lock:
            entries = self._read()
            for e in entries:
                if e.get("id") == entry_id:
                    e["note"] = str(note)[:_NOTE_MAX]
                    self._write(entries)
                    return True
        return False

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            entries = self._read()
            kept = [e for e in entries if e.get("id") != entry_id]
            if len(kept) == len(entries):
                return False
            self._write(kept)
            return True

    def list(self, model_id: Optional[str] = None,
             limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            entries = self._read()
        if model_id:
            entries = [e for e in entries if e.get("model_id") == model_id]
        return entries[-max(1, min(limit, MAX_ENTRIES)):][::-1]  # newest first
