#!/usr/bin/env python3
"""Fetch the relocatable standalone CPython the bundle ships (python-build-standalone).

We ship our own Python so the user needs nothing preinstalled. These are
Astral's `python-build-standalone` "install_only" builds — self-contained and
relocatable. install.py runs under this interpreter and builds the venv from it.

Pins live at the top of this file. To bump: pick a release from
https://github.com/astral-sh/python-build-standalone/releases, update
``PBS_RELEASE`` + ``PY_VERSION``, and copy each target's line from that
release's ``SHA256SUMS`` into ``SHA256``. A download that doesn't match is
refused.

Stdlib-only so it can run on a bare system Python.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import download, extract  # noqa: E402

# --- Pins (from the release's SHA256SUMS) ------------------------------------
PBS_RELEASE = "20261026"      # python-build-standalone release tag (a date)
PY_VERSION = "3.14.10"        # CPython version within that release
_BASE = "https://github.com/astral-sh/python-build-standalone/releases/download"

# target key -> (asset triple, sha256 of the install_only .tar.gz)
TARGETS = {
    "macos-arm64": ("aarch64-apple-darwin", "30a18bcb9ab75e485b4bf4e5d783dde165571e7e42c56dff47a2d2c5345183a4"),
    "windows-x64": ("x86_64-pc-windows-msvc", "98bbe6a0d798bcbad142e96e5930a330e4dc407638e898b515c51eb73a27580d"),
    "linux-x86_64": ("x86_64-unknown-linux-gnu", "1b38024924eb7f4cf0cab45d7b871d1f1991354190fccd2625ac7529a43786e2"),
}


def default_target() -> str:
    if sys.platform == "darwin":
        return "macos-arm64"
    if sys.platform.startswith("win"):
        return "windows-x64"
    if sys.platform.startswith("linux"):
        return "linux-x86_64"
    raise SystemExit(f"no standalone-Python target mapped for platform {sys.platform!r}")


def asset_url(target: str) -> str:
    if target not in TARGETS:
        raise SystemExit(f"unknown target {target!r}; known: {', '.join(TARGETS)}")
    triple, _ = TARGETS[target]
    name = f"cpython-{PY_VERSION}+{PBS_RELEASE}-{triple}-install_only.tar.gz"
    return f"{_BASE}/{PBS_RELEASE}/{name}"


def python_exe(dest: Path, target: str) -> Path:
    return dest / "python.exe" if target.startswith("windows") else dest / "bin" / "python3"


def fetch(target: str, dest: Path) -> Path:
    """Download, verify and extract standalone Python into ``dest``.

    The archive extracts to a top-level ``python/`` dir; we move it to ``dest``
    so the result is ``dest/bin/python3`` (unix) or ``dest/python.exe`` (win).
    """
    url = asset_url(target)
    _, sha256 = TARGETS[target]
    archive = dest.parent / ".pbs-download.tar.gz"
    extract_root = dest.parent / ".pbs-extract"

    print(f"[python] fetching {url}")
    download(url, archive, sha256)
    extract(archive, extract_root)

    inner = extract_root / "python"
    if not inner.exists():
        raise SystemExit(f"unexpected archive layout: no 'python/' under {extract_root}")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(inner), str(dest))
    archive.unlink(missing_ok=True)
    shutil.rmtree(extract_root, ignore_errors=True)

    exe = python_exe(dest, target)
    if not exe.exists():
        raise SystemExit(f"[python] {exe} missing after extraction")
    print(f"[python] {PY_VERSION} ready at {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch standalone CPython 3.14")
    ap.add_argument("--target", default=None, help=f"{' | '.join(TARGETS)} (default: auto)")
    ap.add_argument("--dest", type=Path, help="destination dir (e.g. <bundle>/runtime/python); required unless --url-only")
    ap.add_argument("--url-only", action="store_true", help="print the asset URL and exit")
    args = ap.parse_args()

    target = args.target or default_target()
    if args.url_only:
        print(asset_url(target))
        return
    if args.dest is None:
        ap.error("--dest is required (unless --url-only)")
    fetch(target, args.dest)


if __name__ == "__main__":
    main()
