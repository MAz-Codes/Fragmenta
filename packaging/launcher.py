#!/usr/bin/env python3
"""Fragmenta native launcher (stdlib-only).

This is the tiny entry point that PyInstaller freezes into the shipped
executable (`Fragmenta.app` on macOS, `fragmenta.exe` on Windows). It does
NOT contain torch or any app dependency — it only:

  1. locates the bundled standalone Python 3.11 and the app code (the
     "payload") next to itself,
  2. sets ``FRAGMENTA_PACKAGED=1`` so install.py / config.py resolve the
     writable user-data dir,
  3. runs ``<bundled-python> install.py --launch`` — which builds the venv on
     first run (idempotent stamp afterwards) and starts the app.

Everything heavy (torch, SA3, flash-attn) is pip-installed by install.py at
first run, so this launcher and its frozen binary stay small.

Layout it expects (see packaging/assemble.py):

    <payload>/
    ├── python-3.11/        standalone CPython (bin/python3.11 | python.exe)
    ├── install.py  start.py  requirements.txt
    ├── app/  vendor/  utils/  config/
    └── LICENSE  NOTICE.md

On macOS the launcher exe lives in ``Fragmenta.app/Contents/MacOS/`` and the
payload in ``Fragmenta.app/Contents/Resources/``; on Windows they sit in the
same install directory.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _payload_dir() -> Path:
    """Directory holding the bundled Python + app code."""
    override = os.environ.get("FRAGMENTA_PAYLOAD")
    if override:
        return Path(override).resolve()

    if getattr(sys, "frozen", False):
        exe = Path(sys.executable).resolve()
        # macOS .app: …/Contents/MacOS/fragmenta  ->  …/Contents/Resources
        if sys.platform == "darwin" and exe.parent.name == "MacOS":
            return exe.parent.parent / "Resources"
        # Windows / Linux one-dir: payload sits beside the launcher exe.
        return exe.parent

    # Not frozen (running this script directly, e.g. local testing): assume the
    # payload is the directory this file lives in. Override with FRAGMENTA_PAYLOAD.
    return Path(__file__).resolve().parent


def _bundled_python(payload: Path) -> Path:
    if os.name == "nt":
        return payload / "python-3.11" / "python.exe"
    return payload / "python-3.11" / "bin" / "python3.11"


def main() -> int:
    payload = _payload_dir()
    python = _bundled_python(payload)
    install_py = payload / "install.py"

    missing = [str(p) for p in (python, install_py) if not p.exists()]
    if missing:
        sys.stderr.write(
            "Fragmenta launcher: corrupt installation — missing:\n  "
            + "\n  ".join(missing)
            + "\nPlease reinstall Fragmenta.\n"
        )
        return 2

    env = dict(os.environ)
    env["FRAGMENTA_PACKAGED"] = "1"

    # Run install.py with the bundled interpreter. install.py creates/updates the
    # venv in the user-data dir, then start.py launches the app. stdio is
    # inherited so first-run pip/setup output is visible. (A richer first-run
    # progress UI is tracked separately — distribution.md §10.)
    proc = subprocess.run([str(python), str(install_py), "--launch"], cwd=str(payload), env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
