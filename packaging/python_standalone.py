#!/usr/bin/env python3
"""Resolve & fetch a relocatable standalone CPython 3.11 (python-build-standalone).

We ship our own Python so the user needs nothing preinstalled. These are
Astral's `python-build-standalone` "install_only" builds — self-contained and
relocatable. install.py runs under this interpreter and builds the venv from it.

Pins live at the top of this file. To bump: pick a newer release from
https://github.com/astral-sh/python-build-standalone/releases and update
``PBS_RELEASE`` + ``PY_VERSION`` (and verify the asset names still match).

Stdlib-only so it can run on a bare system Python.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# --- Pins (verify against the releases page when bumping) --------------------
PBS_RELEASE = "20250612"      # python-build-standalone release tag (a date)
PY_VERSION = "3.11.13"        # CPython version within that release
_BASE = "https://github.com/astral-sh/python-build-standalone/releases/download"

# target key -> (asset triple, archive extension)
TARGETS = {
    "macos-arm64": ("aarch64-apple-darwin", "tar.gz"),
    "windows-x64": ("x86_64-pc-windows-msvc", "tar.gz"),
    # add "macos-x64": ("x86_64-apple-darwin", "tar.gz") if Intel is ever needed
}


def default_target() -> str:
    if sys.platform == "darwin":
        return "macos-arm64"
    if sys.platform.startswith("win"):
        return "windows-x64"
    raise SystemExit(f"no standalone-Python target mapped for platform {sys.platform!r}")


def asset_url(target: str) -> str:
    if target not in TARGETS:
        raise SystemExit(f"unknown target {target!r}; known: {', '.join(TARGETS)}")
    triple, ext = TARGETS[target]
    name = f"cpython-{PY_VERSION}+{PBS_RELEASE}-{triple}-install_only.{ext}"
    return f"{_BASE}/{PBS_RELEASE}/{name}"


def fetch(target: str, dest: Path) -> Path:
    """Download + extract standalone Python into ``dest`` (normalised layout).

    The archive extracts to a top-level ``python/`` dir; we move it to ``dest``
    so the result is ``dest/bin/python3.11`` (unix) or ``dest/python.exe`` (win).
    """
    url = asset_url(target)
    triple, ext = TARGETS[target]
    tmp = dest.parent / f".pbs-download.{ext}"
    extract_root = dest.parent / ".pbs-extract"

    print(f"[python] fetching {url}")
    urllib.request.urlretrieve(url, tmp)

    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True)

    if ext == "zip":
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(extract_root)
    else:
        with tarfile.open(tmp) as tf:
            tf.extractall(extract_root)

    inner = extract_root / "python"
    if not inner.exists():
        raise SystemExit(f"unexpected archive layout: no 'python/' under {extract_root}")

    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(inner), str(dest))

    tmp.unlink(missing_ok=True)
    shutil.rmtree(extract_root, ignore_errors=True)
    print(f"[python] ready at {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch standalone CPython 3.11")
    ap.add_argument("--target", default=None, help="macos-arm64 | windows-x64 (default: auto)")
    ap.add_argument("--dest", required=True, type=Path, help="destination dir (becomes python-3.11/)")
    ap.add_argument("--url-only", action="store_true", help="print the asset URL and exit")
    args = ap.parse_args()

    target = args.target or default_target()
    if args.url_only:
        print(asset_url(target))
        return
    fetch(target, args.dest)


if __name__ == "__main__":
    main()
