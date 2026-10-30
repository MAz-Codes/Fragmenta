"""Checksum-verified downloads and safe extraction for the bundled runtime.

Stdlib-only. Used by python_standalone.py and uv_standalone.py.
"""
from __future__ import annotations

import hashlib
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


def download(url: str, dest: Path, sha256: str, retries: int = 3) -> Path:
    """Fetch ``url`` to ``dest``; refuse it unless its SHA-256 matches."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as fh:
                while chunk := r.read(1 << 20):
                    h.update(chunk)
                    fh.write(chunk)
            if h.hexdigest() != sha256:
                raise ValueError(f"checksum mismatch for {url}: got {h.hexdigest()}, pinned {sha256}")
            part.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as exc:
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise SystemExit(f"[fetch] {exc}")
            print(f"[fetch] attempt {attempt} failed ({exc}); retrying…")
            time.sleep(2 * attempt)
    return dest


def extract(archive: Path, into: Path) -> Path:
    """Extract a .tar.gz / .zip into ``into`` (emptied first)."""
    if into.exists():
        shutil.rmtree(into)
    into.mkdir(parents=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(into)          # zipfile drops absolute and ../ paths
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(into, filter="data")   # no links or paths outside `into`
    return into
