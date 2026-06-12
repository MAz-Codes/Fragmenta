#!/usr/bin/env python3
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


def _user_data_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "FragmentaDesktop"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "FragmentaDesktop"
    return Path.home() / ".local" / "share" / "FragmentaDesktop"


PACKAGED = os.environ.get("FRAGMENTA_PACKAGED") == "1"
VENV_PATH = (_user_data_dir() / "venv") if PACKAGED else (PROJECT_ROOT / "venv")
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
WHEELS_DIR = PROJECT_ROOT / "utils" / "vendor" / "wheels"
STAMP_PATH = VENV_PATH / ".fragmenta-install-stamp"
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
    if sys.version_info[:2] != (3, 11):
        hint = (
            "the bundled Python looks wrong — reinstall Fragmenta."
            if PACKAGED else
            "Install 3.11 from https://www.python.org/downloads/release/python-3119/ "
            "and run the launcher again."
        )
        fail(
            f"this installer must run under Python 3.11 "
            f"(got {sys.version_info.major}.{sys.version_info.minor}). " + hint
        )


def _macos_link_libpython(venv_path: Path) -> None:
    if sys.platform != "darwin":
        return
    src = Path(sys.base_prefix) / "lib" / "libpython3.11.dylib"
    dst = venv_path / "lib" / "libpython3.11.dylib"
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)


def ensure_venv() -> Path:
    """Create the venv if missing; recreate it if it's the wrong Python."""
    py = venv_python()
    if VENV_PATH.exists():
        if py.exists() and is_py311(py):
            return py
        log("existing venv is not Python 3.11 — recreating…")
        shutil.rmtree(VENV_PATH, ignore_errors=True)
    log(f"creating virtual environment at {VENV_PATH} …")
    VENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=False, clear=False).create(str(VENV_PATH))
    _macos_link_libpython(VENV_PATH)
    subprocess.check_call(
        [str(venv_python()), "-m", "ensurepip", "--upgrade", "--default-pip"]
    )
    return venv_python()


def requirements_hash() -> str:
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
