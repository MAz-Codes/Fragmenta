#!/usr/bin/env python3
"""Assemble the Fragmenta bundle payload (OS-agnostic).

Produces the staging tree that the per-OS packagers (macos/build_dmg.sh,
windows/fragmenta.iss) wrap into a `.dmg` / `.exe`:

    <staging>/                     ← the "payload"
    ├── python-3.11/               standalone CPython (unless --skip-python)
    ├── install.py  start.py  requirements.txt
    ├── app/  vendor/  utils/  config/
    ├── LICENSE  NOTICE.md  VERSION
    └── manifest.txt               build provenance

App code comes from ``git archive`` (tracked files only) so node_modules,
venv, models/pretrained, __pycache__, output/, logs/, projects/ etc. are
excluded automatically — then a small prune list drops tracked-but-unneeded
paths. The frozen native launcher (packaging/launcher.py) is NOT placed here;
PyInstaller bakes it into the shipped executable separately.

Stdlib-only. Run from the repo root.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Tracked paths we do NOT ship in the bundle (runtime data, dev-only, or heavy
# non-runtime vendor extras). Pruned after `git archive` extraction.
PRUNE = [
    "models",                       # created under the user-data dir at runtime
    "fragmenta.sh", "fragmenta.bat", "fragmenta.command",  # source launchers
    "packaging",                    # build scripts; not part of the app
    "distribution.md",
    ".gitignore", ".gitattributes", ".python-version",
    "vendor/stable-audio-3/tests",
    "vendor/stable-audio-3/docs",
    "vendor/stable-audio-3/.github",
    "vendor/stable-audio-3/uv.lock",
    "vendor/stable-audio-3/stable-audio-3.png",
    "app/frontend/src",             # only the built bundle is served at runtime
]

# Sanity check: these MUST exist in the payload after assembly.
REQUIRED = [
    "install.py", "start.py", "requirements.txt",
    "app/backend", "app/core", "app/frontend/build", "app/frontend/public",
    "vendor/stable-audio-3/stable_audio_3", "vendor/stable-audio-3/LICENSE",
    "config", "LICENSE", "NOTICE.md",
]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("[assemble] $ " + " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


def export_tracked(ref: str, staging: Path) -> None:
    """Extract the tracked tree at ``ref`` into ``staging`` via git archive."""
    tar_path = staging.parent / ".fragmenta-archive.tar"
    with open(tar_path, "wb") as fh:
        run(["git", "-C", str(REPO), "archive", "--format=tar", ref], stdout=fh)
    with tarfile.open(tar_path) as tf:
        tf.extractall(staging)
    tar_path.unlink(missing_ok=True)


def prune(staging: Path) -> None:
    for rel in PRUNE:
        p = staging / rel
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()


def verify(staging: Path) -> None:
    missing = [r for r in REQUIRED if not (staging / r).exists()]
    if missing:
        raise SystemExit("[assemble] ERROR: payload missing required paths:\n  " + "\n  ".join(missing))


def write_manifest(staging: Path, ref: str, target: str, version: str) -> None:
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", ref],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        sha = ref
    (staging / "manifest.txt").write_text(
        f"fragmenta {version}\ntarget {target}\nref {ref}\ncommit {sha}\n"
    )


def main() -> None:
    version = (REPO / "VERSION").read_text().strip()
    ap = argparse.ArgumentParser(description="Assemble the Fragmenta bundle payload")
    ap.add_argument("--ref", default="HEAD", help="git ref to ship (default HEAD). Must be committed.")
    ap.add_argument("--target", default=None, help="macos-arm64 | windows-x64 (default: auto)")
    ap.add_argument("--out", type=Path, default=REPO / "packaging" / "build" / "payload")
    ap.add_argument("--skip-python", action="store_true", help="don't fetch standalone Python")
    args = ap.parse_args()

    staging = args.out.resolve()
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print(f"[assemble] fragmenta {version} → {staging}")
    export_tracked(args.ref, staging)
    prune(staging)
    verify(staging)

    if not args.skip_python:
        import python_standalone as pbs
        target = args.target or pbs.default_target()
        pbs.fetch(target, staging / "python-3.11")
    else:
        target = args.target or "skipped"
        print("[assemble] --skip-python: not fetching standalone Python")

    write_manifest(staging, args.ref, target, version)
    print(f"[assemble] done. payload at {staging}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
