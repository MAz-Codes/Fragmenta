#!/usr/bin/env python3
"""Fragmenta bootstrapping installer (cross-platform, stdlib-only).

This is the single source of truth for setting up Fragmenta's Python
environment. The platform launchers (`fragmenta.sh`, `fragmenta.command`,
`fragmenta.bat`) only acquire Python 3.11 (and any OS-level runtime libs),
then hand off to this script — so the venv + dependency logic lives in one
place instead of being triplicated and drifting across three shells.

Design goals
------------
* **Idempotent / fast relaunch.** Dependencies are (re)installed only when
  `requirements.txt` actually changes. A stamp file under the venv records the
  hash of what was last installed; a matching stamp turns a relaunch into a
  near-instant no-op. (The old launchers ran `pip install -r requirements.txt`
  on *every* launch — minutes each time.) Force a fresh install with
  `--reinstall`.
* **No third-party imports.** This runs under a *bare* system Python 3.11
  before any deps exist, so it uses only the standard library.
* **SA3-aware.** `requirements.txt` already pins torch==2.7.1, declares the
  CUDA wheel index, and gates flash-attn behind `sys_platform == 'linux'`, so
  one `pip install -r requirements.txt` resolves the whole graph. The Stable
  Audio 3 vendor at `vendor/stable-audio-3` is loaded via `sys.path` at
  runtime — nothing to pip-install for it.

Usage
-----
    python3.11 install.py            # ensure the environment is ready
    python3.11 install.py --launch   # ensure, then start the app
    python3.11 install.py --reinstall  # force a dependency reinstall
    python3.11 install.py --check    # report status, change nothing
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PATH = PROJECT_ROOT / "venv"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
WHEELS_DIR = PROJECT_ROOT / "utils" / "vendor" / "wheels"
STAMP_PATH = VENV_PATH / ".fragmenta-install-stamp"
# laion-clap is installed with --no-deps (its numpy<2 pin conflicts with SA3's
# numpy>=2.2.6, but it works fine at runtime on numpy 2.x). Bumping this string
# forces a reinstall via the stamp, same as editing requirements.txt.
LAION_CLAP_SPEC = "laion-clap>=1.1.6"


def log(msg: str) -> None:
    print(f"[install] {msg}", flush=True)


def fail(msg: str, code: int = 1) -> "None":
    print(f"[install] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def venv_python(path: Path = VENV_PATH) -> Path:
    """Path to the interpreter inside a venv, per-platform."""
    if os.name == "nt":
        return path / "Scripts" / "python.exe"
    return path / "bin" / "python"


def is_py311(python: Path) -> bool:
    try:
        out = subprocess.run(
            [str(python), "-c",
             "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=30,
        )
        return out.returncode == 0 and out.stdout.strip() == "3.11"
    except Exception:
        return False


def require_py311_host() -> None:
    """We're running under the system Python the launcher chose; insist 3.11.

    Fragmenta pins torch==2.7.1 + flash-attn cp311 wheels, which ship only for
    Python 3.11. The launchers try to hand us a 3.11; if something else slipped
    through, stop with a clear message rather than producing a broken venv.
    """
    if sys.version_info[:2] != (3, 11):
        fail(
            f"this installer must run under Python 3.11 "
            f"(got {sys.version_info.major}.{sys.version_info.minor}). "
            "Install 3.11 from https://www.python.org/downloads/release/python-3119/ "
            "and run the launcher again."
        )


def ensure_venv() -> Path:
    """Create the venv if missing; recreate it if it's the wrong Python."""
    py = venv_python()
    if VENV_PATH.exists():
        if py.exists() and is_py311(py):
            return py
        log("existing venv is not Python 3.11 — recreating…")
        shutil.rmtree(VENV_PATH, ignore_errors=True)
    log(f"creating virtual environment at {VENV_PATH} …")
    venv.EnvBuilder(with_pip=True, clear=False).create(str(VENV_PATH))
    return venv_python()


def requirements_hash() -> str:
    """Stamp value: hash of requirements.txt + the laion-clap spec + py tag.

    Any change to the dependency set (or the Python minor) invalidates the
    stamp and triggers a reinstall on the next run.
    """
    h = hashlib.sha256()
    h.update(REQUIREMENTS.read_bytes())
    h.update(LAION_CLAP_SPEC.encode())
    h.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    return h.hexdigest()


def deps_up_to_date() -> bool:
    if not STAMP_PATH.exists():
        return False
    try:
        return STAMP_PATH.read_text().strip() == requirements_hash()
    except Exception:
        return False


def pip(py: Path, *args: str) -> None:
    cmd = [str(py), "-m", "pip", *args]
    log("pip " + " ".join(args[:4]) + (" …" if len(args) > 4 else ""))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        fail(f"`pip {' '.join(args)}` failed (exit {result.returncode}).")


def install_dependencies(py: Path, force: bool = False) -> None:
    if not force and deps_up_to_date():
        log("dependencies already up to date (requirements.txt unchanged) — skipping install.")
        return

    log("installing dependencies (first run / requirements changed — this can take a few minutes)…")
    pip(py, "install", "--upgrade", "pip", "setuptools<70", "wheel", "build")

    install_args = ["install", "-r", str(REQUIREMENTS), "--prefer-binary"]
    if WHEELS_DIR.is_dir():
        install_args += ["--find-links", str(WHEELS_DIR)]
    pip(py, *install_args)

    # laion-clap with --no-deps (see LAION_CLAP_SPEC note). Non-fatal: the
    # auto-annotator's Rich tier degrades without it, everything else works.
    log("installing laion-clap (auto-annotator) with --no-deps…")
    rc = subprocess.run([str(py), "-m", "pip", "install", LAION_CLAP_SPEC, "--no-deps"]).returncode
    if rc != 0:
        log("WARNING: laion-clap install failed — Rich auto-annotation may be unavailable.")

    STAMP_PATH.write_text(requirements_hash())
    log("dependencies installed; stamp written.")


def verify(py: Path) -> None:
    checks = (
        ("torch", "import torch; print('PyTorch', torch.__version__, '· CUDA', torch.cuda.is_available())"),
        ("pywebview", "import webview; print('pywebview ready')"),
        ("stable_audio_3", "import sys; sys.path.insert(0, r'%s'); import stable_audio_3; print('stable_audio_3 importable')"
         % str(PROJECT_ROOT / "vendor" / "stable-audio-3")),
        ("flash_attn", "import flash_attn; print('Flash Attention', flash_attn.__version__)"),
    )
    log("verifying key packages…")
    for name, snippet in checks:
        r = subprocess.run([str(py), "-c", snippet], capture_output=True, text=True)
        if r.returncode == 0:
            print("  ✓ " + r.stdout.strip())
        else:
            note = " (Linux + CUDA only)" if name == "flash_attn" else ""
            print(f"  · {name} unavailable{note}")


def launch(py: Path) -> "None":
    log("starting Fragmenta…")
    os.chdir(PROJECT_ROOT)
    sys.exit(subprocess.run([str(py), str(PROJECT_ROOT / "start.py")]).returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fragmenta bootstrapping installer")
    ap.add_argument("--launch", action="store_true", help="start the app after ensuring the environment")
    ap.add_argument("--reinstall", action="store_true", help="force a dependency reinstall")
    ap.add_argument("--check", action="store_true", help="report status only; change nothing")
    args = ap.parse_args()

    if not REQUIREMENTS.exists():
        fail(f"requirements.txt not found at {REQUIREMENTS} — run from the Fragmenta folder.")

    if args.check:
        py = venv_python()
        ok = VENV_PATH.exists() and py.exists() and is_py311(py)
        print(f"venv:           {'present (py3.11)' if ok else 'missing / wrong python'}")
        print(f"dependencies:   {'up to date' if ok and deps_up_to_date() else 'install needed'}")
        sys.exit(0 if ok and deps_up_to_date() else 1)

    require_py311_host()
    py = ensure_venv()
    install_dependencies(py, force=args.reinstall)
    verify(py)
    if args.launch:
        launch(py)
    else:
        log("environment ready. Launch with: python install.py --launch")


if __name__ == "__main__":
    main()
