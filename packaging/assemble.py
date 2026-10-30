#!/usr/bin/env python3
"""Assemble the Fragmenta bundle (OS-agnostic).

Produces the tree that the per-OS packagers (macos/build_dmg.sh,
windows/fragmenta.iss) wrap into a `.dmg` / `.exe`. Its directory names don't
carry a Python or app version, so an update replaces files in place:

    <bundle>/                      ← macOS: Contents/Resources · Windows: install dir
    ├── runtime/
    │   ├── python/                standalone CPython (python_standalone.py)
    │   └── uv[.exe]               pinned uv, install.py's installer (uv_standalone.py)
    └── payload/
        ├── install.py  start.py  requirements.txt  .python-version
        ├── locks/                 hashed dependency locks (packaging/lock.py)
        ├── app/  vendor/  utils/  config/  licenses/
        ├── LICENSE  NOTICE.md  VERSION
        └── manifest.json          build provenance

App code comes from ``git archive`` (tracked files only) so node_modules,
venv, models/pretrained, __pycache__, output/, logs/, projects/ etc. are
excluded automatically — then a small prune list drops tracked-but-unneeded
paths. The frozen native launcher (packaging/launcher.py) is NOT placed here;
PyInstaller bakes it into the shipped executable separately.

Stdlib-only. Run from the repo root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Tracked paths we do NOT ship in the payload (runtime data, dev-only, or heavy
# non-runtime extras). Pruned after `git archive` extraction.
PRUNE = [
    "models",                       # created under the user-data dir at runtime
    "fragmenta.sh", "fragmenta.bat", "fragmenta.command",  # source launchers
    "packaging", "tests", ".github",  # build and test tooling (dev/2.0 only)
    "docs/fragmenta.gif", "docs/Dataset.png", "docs/Generation.png",
    "docs/Performance.png", "docs/Training.png",  # README media
    ".gitignore", ".gitattributes",
    "vendor/stable-audio-3/tests",
    "vendor/stable-audio-3/docs",
    "vendor/stable-audio-3/.github",
    "vendor/stable-audio-3/uv.lock",
    "vendor/stable-audio-3/stable-audio-3.png",
    "app/frontend/src",             # only the built bundle is served at runtime
]

# Sanity check: these MUST exist in the payload after assembly.
REQUIRED = [
    "install.py", "start.py", "requirements.txt", ".python-version", "VERSION",
    "locks/linux-x86_64-cu128.txt", "locks/windows-x86_64-cu128.txt", "locks/macos-arm64.txt",
    "app/backend", "app/core", "app/frontend/build", "app/frontend/public",
    "vendor/stable-audio-3/stable_audio_3", "vendor/stable-audio-3/LICENSE",
    "config", "licenses", "LICENSE", "NOTICE.md",
]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("[assemble] $ " + " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


def export_tracked(ref: str, payload: Path) -> None:
    """Extract the tracked tree at ``ref`` into ``payload`` via git archive."""
    tar_path = payload.parent / ".fragmenta-archive.tar"
    with open(tar_path, "wb") as fh:
        run(["git", "-C", str(REPO), "archive", "--format=tar", ref], stdout=fh)
    with tarfile.open(tar_path) as tf:
        tf.extractall(payload, filter="data")
    tar_path.unlink(missing_ok=True)


def prune(payload: Path) -> None:
    for rel in PRUNE:
        p = payload / rel
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()


def verify(payload: Path) -> None:
    missing = [r for r in REQUIRED if not (payload / r).exists()]
    if missing:
        raise SystemExit("[assemble] ERROR: payload missing required paths:\n  " + "\n  ".join(missing))


def write_manifest(payload: Path, ref: str, target: str, version: str, runtime: dict) -> None:
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", ref],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        sha = ref
    locks = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((payload / "locks").glob("*.txt"))}
    manifest = {
        "version": version, "ref": ref, "commit": sha, "target": target,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **runtime, "locks": locks,
    }
    (payload / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import python_standalone as pbs
    import uv_standalone as uvs

    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    ap = argparse.ArgumentParser(description="Assemble the Fragmenta bundle")
    ap.add_argument("--ref", default="HEAD", help="git ref to ship (default HEAD). Must be committed.")
    ap.add_argument("--target", default=None, help=f"{' | '.join(pbs.TARGETS)} (default: auto)")
    ap.add_argument("--out", type=Path, default=REPO / "packaging" / "build" / "bundle")
    ap.add_argument("--skip-runtime", action="store_true", help="don't fetch standalone Python and uv")
    args = ap.parse_args()

    bundle = args.out.resolve()
    if bundle.exists():
        shutil.rmtree(bundle)
    payload = bundle / "payload"
    payload.mkdir(parents=True)
    target = args.target or pbs.default_target()

    print(f"[assemble] fragmenta {version} ({target}) → {bundle}")
    export_tracked(args.ref, payload)
    prune(payload)
    verify(payload)

    runtime = {}
    if not args.skip_runtime:
        pbs.fetch(target, bundle / "runtime" / "python")
        uvs.fetch(target, bundle / "runtime")
        runtime = {"python": {"version": pbs.PY_VERSION, "pbs_release": pbs.PBS_RELEASE},
                   "uv": uvs.UV_VERSION}
    else:
        print("[assemble] --skip-runtime: not fetching standalone Python or uv")

    write_manifest(payload, args.ref, target, version, runtime)
    print(f"[assemble] done. bundle at {bundle}")


if __name__ == "__main__":
    main()
