#!/usr/bin/env python3
"""Fetch the pinned uv binary the bundle ships (install.py's installer engine).

To bump: pick a release from https://github.com/astral-sh/uv/releases, update
``UV_VERSION`` and copy each target's ``uv-<triple>.<ext>.sha256`` into
``TARGETS``. Re-test a lock sync with it: install.py relies on
``uv pip sync --require-hashes --index-strategy unsafe-best-match --python …``
and the UV_SYSTEM_CERTS / UV_HTTP_TIMEOUT / UV_CACHE_DIR variables.

Stdlib-only.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import download, extract  # noqa: E402

UV_VERSION = "0.13.5"
_BASE = "https://github.com/astral-sh/uv/releases/download"

# target key -> (asset name, sha256)
TARGETS = {
    "macos-arm64": ("uv-aarch64-apple-darwin.tar.gz", "743f85035a1996ecb350659e01cd9faea9709829be09b1b4b03c73eaf3c9cd73"),
    "windows-x64": ("uv-x86_64-pc-windows-msvc.zip", "a38b8d1902b7cfbc76b423ba1bb456f623f42064cdc72628b6075b89afd1019a"),
    "linux-x86_64": ("uv-x86_64-unknown-linux-gnu.tar.gz", "f68d291c75783122dad58ab77def0e13f3233f358731db5d6d7e8715603b928f"),
}


def fetch(target: str, runtime_dir: Path) -> Path:
    """Put the uv binary at ``runtime_dir/uv`` (``uv.exe`` on Windows)."""
    if target not in TARGETS:
        raise SystemExit(f"unknown target {target!r}; known: {', '.join(TARGETS)}")
    asset, sha256 = TARGETS[target]
    exe = "uv.exe" if target.startswith("windows") else "uv"
    archive = runtime_dir / f".uv-download-{asset}"
    unpacked = runtime_dir / ".uv-extract"

    url = f"{_BASE}/{UV_VERSION}/{asset}"
    print(f"[uv] fetching {url}")
    download(url, archive, sha256)
    extract(archive, unpacked)
    found = next((p for p in unpacked.rglob(exe) if p.is_file()), None)
    if found is None:
        raise SystemExit(f"[uv] no {exe} inside {asset}")
    dest = runtime_dir / exe
    shutil.copy2(found, dest)
    os.chmod(dest, 0o755)
    archive.unlink(missing_ok=True)
    shutil.rmtree(unpacked, ignore_errors=True)
    print(f"[uv] {UV_VERSION} ready at {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch the pinned uv binary")
    ap.add_argument("--target", required=True, help=" | ".join(TARGETS))
    ap.add_argument("--dest", type=Path, required=True, help="runtime dir (uv lands at <dest>/uv)")
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    fetch(args.target, args.dest)


if __name__ == "__main__":
    main()
