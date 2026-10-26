"""Fingerprint of the backend code on disk.

Shared by the backend (which bakes it in at import) and the launcher (which
computes it fresh), so a running backend can be told apart from the code next
to it: a `git pull` keeps VERSION unchanged but leaves newer files behind.
Deliberately dependency-free — start.py imports it before anything else.
"""

import os
from pathlib import Path

# The .py files a backend process runs.
_CODE_DIRS = ("app/backend", "app/core", "utils", "vendor/stable-audio-3/stable_audio_3",
              "vendor/stable-audio-3/scripts")


def code_stamp(root: Path) -> float:
    """Newest modification time of any backend .py file under ``root``."""
    newest = 0.0
    for rel in _CODE_DIRS:
        for dirpath, dirnames, filenames in os.walk(Path(root) / rel):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in filenames:
                if name.endswith(".py"):
                    try:
                        newest = max(newest, os.stat(os.path.join(dirpath, name)).st_mtime)
                    except OSError:
                        pass
    return newest
