#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _user_data_dir() -> Path:
    # MUST stay identical to app/core/config.py's _default_user_data_dir.
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "FragmentaDesktop"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "FragmentaDesktop"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "FragmentaDesktop"


PACKAGED = os.environ.get("FRAGMENTA_PACKAGED") == "1"
DATA_DIR = _user_data_dir() if PACKAGED else PROJECT_ROOT
VENV_PATH = DATA_DIR / "venv"
LOG_DIR = DATA_DIR / "logs"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
LOCKS = PROJECT_ROOT / "locks"
WHEELS_DIR = PROJECT_ROOT / "utils" / "vendor" / "wheels"
STAMP_PATH = VENV_PATH / ".fragmenta-install-stamp"
LAION_CLAP_SPEC = "laion-clap>=1.1.6"
# Left out of the Linux lock (the packaged app opens in Chromium); a source
# install still wants them for the native-window fallback.
LINUX_GTK = {"pycairo", "pygobject"}
KEEP_LOGS = 10

# A first install is the ~8 GB CUDA venv and uv cache plus room to unpack them
# (the Mac has no CUDA stack). An update reuses most of the cache.
DISK_FIRST_INSTALL_GB = 4 if sys.platform == "darwin" else 12
DISK_UPDATE_GB = 2 if sys.platform == "darwin" else 5
DISK_MODEL_GB = 4  # the smallest base model is ~3.5 GB
DOWNLOAD_HOSTS = ("pypi.org", "files.pythonhosted.org", "download.pytorch.org", "github.com")
HOST_TIMEOUT = 5
MIN_NVIDIA_DRIVER = 570  # the cu128 torch builds need R570
MIN_MACOS = (14, 0)
MIN_WINDOWS_BUILD = 17763  # Windows 10 1809
MIN_GLIBC = (2, 28)


def _required_python() -> tuple[int, int]:
    # .python-version is the one place the version lives; the launchers
    # look for the same one.
    try:
        major, minor = (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8").strip().split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return (3, 14)


REQUIRED_PY = _required_python()
REQUIRED_PY_STR = "%d.%d" % REQUIRED_PY

_log_file = None


def open_log() -> Path:
    """Persistent install log: logs/install-YYYYMMDD-HHMMSS.log, last few kept."""
    global _log_file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / time.strftime("install-%Y%m%d-%H%M%S.log")
    _log_file = open(path, "w", encoding="utf-8")
    for old in sorted(LOG_DIR.glob("install-*.log"))[:-KEEP_LOGS]:
        try:
            old.unlink()
        except OSError:
            pass
    return path


def _write_log(text: str) -> None:
    if _log_file:
        _log_file.write(text)
        _log_file.flush()


def log(msg: str) -> None:
    print(f"[install] {msg}", flush=True)
    _write_log(f"[install] {msg}\n")


def fail(msg: str, code: int = 1) -> "None":
    print(f"[install] ERROR: {msg}", file=sys.stderr, flush=True)
    _write_log(f"[install] ERROR: {msg}\n")
    if _log_file:
        print(f"[install] full log: {_log_file.name}", file=sys.stderr, flush=True)
    sys.exit(code)


def run(cmd: list[str], env: dict | None = None) -> int:
    """Run a command, streaming its output to the console and the log."""
    _write_log("$ " + " ".join(cmd) + "\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", cwd=PROJECT_ROOT,
                            env={**os.environ, **(env or {})})
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        _write_log(line)
    return proc.wait()


def venv_python(path: Path = VENV_PATH) -> Path:
    if os.name == "nt":
        return path / "Scripts" / "python.exe"
    return path / "bin" / "python"


def python_version_of(python: Path) -> tuple[int, ...] | None:
    try:
        out = subprocess.run(
            [str(python), "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"],
            capture_output=True, text=True, timeout=30,
        )
        return tuple(int(x) for x in out.stdout.strip().split(".")) if out.returncode == 0 else None
    except Exception:
        return None


def is_required_py(python: Path) -> bool:
    v = python_version_of(python)
    return v is not None and v[:2] == REQUIRED_PY


def require_host_py() -> None:
    if sys.version_info[:2] != REQUIRED_PY:
        hint = (
            "the bundled Python looks wrong — reinstall Fragmenta."
            if PACKAGED else
            f"Install Python {REQUIRED_PY_STR} from https://www.python.org/downloads/ "
            "and run the launcher again."
        )
        fail(
            f"this installer must run under Python {REQUIRED_PY_STR} "
            f"(got {sys.version_info.major}.{sys.version_info.minor}). " + hint
        )


def _macos_link_libpython(venv_path: Path) -> None:
    if sys.platform != "darwin":
        return
    name = "libpython%d.%d.dylib" % sys.version_info[:2]
    src = Path(sys.base_prefix) / "lib" / name
    dst = venv_path / "lib" / name
    if dst.is_symlink() and not dst.exists():
        dst.unlink()  # points into a Python that moved or was replaced
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)


def _venv_home_missing() -> bool:
    cfg = VENV_PATH / "pyvenv.cfg"
    try:
        for line in cfg.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "home":
                return not Path(value.strip()).exists()
    except OSError:
        return True
    return False


def ensure_venv() -> Path:
    py = venv_python()
    if VENV_PATH.exists():
        v = python_version_of(py) if py.exists() else None
        if v is None or v[:2] != REQUIRED_PY:
            log(f"existing venv is not Python {REQUIRED_PY_STR} — recreating…")
            shutil.rmtree(VENV_PATH, ignore_errors=True)
        else:
            if v != tuple(sys.version_info[:3]) or _venv_home_missing():
                # Same minor version, but the base Python changed or moved
                # (an app update ships a newer 3.14.x): refresh the venv's
                # copied interpreter instead of rebuilding every package.
                log("refreshing the venv for this Python…")
                rc = run([sys.executable, "-m", "venv", "--upgrade", "--copies", str(VENV_PATH)])
                if rc != 0:
                    fail(f"could not refresh the venv at {VENV_PATH} (exit {rc}).")
            _macos_link_libpython(VENV_PATH)
            return py
    log(f"creating virtual environment at {VENV_PATH} …")
    VENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Copies, not symlinks: the base Python may live inside an app bundle
    # that moves or is replaced on update.
    venv.EnvBuilder(with_pip=False, clear=False, symlinks=False).create(str(VENV_PATH))
    _macos_link_libpython(VENV_PATH)
    subprocess.check_call(
        [str(venv_python()), "-m", "ensurepip", "--upgrade", "--default-pip"]
    )
    return venv_python()


def platform_lock() -> Path | None:
    """The committed lock for this OS and CPU (packaging/lock.py), if any."""
    machine = platform.machine().lower()
    name = {
        ("linux", "x86_64"): "linux-x86_64-cu128.txt",
        ("win32", "amd64"): "windows-x86_64-cu128.txt",
        ("win32", "x86_64"): "windows-x86_64-cu128.txt",
        ("darwin", "arm64"): "macos-arm64.txt",
    }.get((sys.platform, machine))
    path = LOCKS / name if name else None
    return path if path and path.exists() else None


def find_uv() -> str | None:
    """Bundled uv (packaged: runtime/uv beside payload/), else uv on PATH."""
    exe = "uv.exe" if os.name == "nt" else "uv"
    candidates = [os.environ.get("FRAGMENTA_UV"),
                  str(PROJECT_ROOT.parent / "runtime" / exe),
                  shutil.which("uv")]
    for c in candidates:
        if c and Path(c).is_file():
            try:
                if subprocess.run([c, "--version"], capture_output=True, timeout=30).returncode == 0:
                    return c
            except Exception:
                pass
    return None


def requirements_hash(lock: Path | None) -> str:
    h = hashlib.sha256()
    h.update((lock or REQUIREMENTS).read_bytes())
    if lock is None:
        h.update(LAION_CLAP_SPEC.encode())
    if sys.platform.startswith("linux") and not PACKAGED:
        h.update(b"gtk")
    h.update(("py%d.%d.%d" % sys.version_info[:3]).encode())
    version = PROJECT_ROOT / "VERSION"
    if version.exists():
        h.update(version.read_bytes())
    return h.hexdigest()


def deps_up_to_date(lock: Path | None) -> bool:
    if not STAMP_PATH.exists():
        return False
    try:
        return STAMP_PATH.read_text(encoding="utf-8").strip() == requirements_hash(lock)
    except Exception:
        return False


def pip(py: Path, *args: str) -> None:
    log("pip " + " ".join(args[:4]) + (" …" if len(args) > 4 else ""))
    rc = run([str(py), "-m", "pip", *args])
    if rc != 0:
        fail(f"`pip {' '.join(args)}` failed (exit {rc}).")


def _requirement_lines(names: set[str]) -> list[str]:
    out = []
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if m and not line.lstrip().startswith(("#", "-")) and m.group(1).lower() in names:
            out.append(line.strip())
    return out


def install_from_lock(py: Path, lock: Path) -> None:
    uv = find_uv()
    if uv:
        log(f"syncing to {lock.relative_to(PROJECT_ROOT)} with uv…")
        env = {"UV_SYSTEM_CERTS": "1", "UV_HTTP_TIMEOUT": "120"}
        if PACKAGED:
            env["UV_CACHE_DIR"] = str(DATA_DIR / "cache" / "uv")
        # The PyTorch index mirrors some PyPI packages at older versions;
        # every file is hash-pinned, so the best match across indexes is safe.
        rc = run([uv, "pip", "sync", "--python", str(py), "--require-hashes",
                  "--index-strategy", "unsafe-best-match",
                  "--find-links", str(WHEELS_DIR), str(lock)], env=env)
        if rc != 0:
            fail(f"uv pip sync failed (exit {rc}).")
        installer = [uv, "pip", "install", "--python", str(py)]
    else:
        log(f"uv not found — installing {lock.relative_to(PROJECT_ROOT)} with pip…")
        pip(py, "install", "--require-hashes", "--no-deps", "--find-links", str(WHEELS_DIR), "-r", str(lock))
        installer = [str(py), "-m", "pip", "install"]
    if sys.platform.startswith("linux") and not PACKAGED:
        extras = _requirement_lines(LINUX_GTK)
        if extras:
            log("adding the GTK window libraries for the native-window fallback…")
            if run([*installer, *extras]) != 0:
                log("WARNING: PyGObject / pycairo didn't install — the app opens in Chromium or a browser instead.")


def install_from_requirements(py: Path) -> None:
    # No lock for this platform: resolve requirements.txt as before.
    pip(py, "install", "--upgrade", "pip", "setuptools", "wheel", "build")
    install_args = ["install", "-r", str(REQUIREMENTS), "--prefer-binary"]
    if WHEELS_DIR.is_dir():
        install_args += ["--find-links", str(WHEELS_DIR)]
    pip(py, *install_args)
    log("installing laion-clap (auto-annotator) with --no-deps…")
    if run([str(py), "-m", "pip", "install", LAION_CLAP_SPEC, "--no-deps"]) != 0:
        log("WARNING: laion-clap install failed — Rich auto-annotation may be unavailable.")


def install_dependencies(py: Path, force: bool = False) -> None:
    lock = platform_lock()
    if not force and deps_up_to_date(lock):
        log("dependencies already up to date — skipping install.")
        return
    log("installing dependencies (first run / update — this can take a few minutes)…")
    if lock:
        install_from_lock(py, lock)
    else:
        log(f"no lock for {sys.platform}/{platform.machine()} — resolving requirements.txt.")
        install_from_requirements(py)
    STAMP_PATH.write_text(requirements_hash(lock), encoding="utf-8")
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
        r = subprocess.run([str(py), "-c", snippet], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if r.returncode == 0:
            lines = r.stdout.strip().splitlines()
            line = "  ✓ " + (lines[-1] if lines else name)
        else:
            note = " (Linux + CUDA only)" if name == "flash_attn" else ""
            line = f"  · {name} unavailable{note}"
        print(line)
        _write_log(line + "\n")


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", text))


def _os_problem() -> str | None:
    if sys.platform == "darwin":
        found = platform.mac_ver()[0]
        if found and _version_tuple(found) < MIN_MACOS:
            return f"Fragmenta needs macOS {MIN_MACOS[0]} or newer (this is {found})."
    elif sys.platform == "win32":
        build = sys.getwindowsversion().build
        if build < MIN_WINDOWS_BUILD:
            return f"Fragmenta needs Windows 10 version 1809 or newer (this is build {build})."
    else:
        libc, found = platform.libc_ver()
        if libc == "glibc" and found and _version_tuple(found) < MIN_GLIBC:
            return f"Fragmenta needs glibc {'.'.join(map(str, MIN_GLIBC))} or newer (this is {found})."
    return None


def _data_dir_problem() -> str | None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=DATA_DIR):
            pass
    except OSError as exc:
        return f"Fragmenta can't write to {DATA_DIR}: {exc}"
    return None


def _free_gb() -> float:
    path = DATA_DIR
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def _reaches(host: str) -> bool:
    """Does anything answer at this host? Any HTTP status counts, and the
    certificate isn't checked: uv does that on the real download."""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    request = urllib.request.Request(f"https://{host}/", method="HEAD")
    try:
        urllib.request.urlopen(request, timeout=HOST_TIMEOUT, context=context).close()
    except urllib.error.HTTPError:
        pass
    except Exception:
        return False
    return True


def _reachable_hosts() -> dict[str, bool]:
    results = dict.fromkeys(DOWNLOAD_HOSTS, False)

    def probe(host: str) -> None:
        results[host] = _reaches(host)

    # Daemon threads: a stuck DNS lookup ignores the socket timeout.
    threads = [threading.Thread(target=probe, args=(h,), daemon=True) for h in DOWNLOAD_HOSTS]
    for t in threads:
        t.start()
    deadline = time.monotonic() + HOST_TIMEOUT + 1
    for t in threads:
        t.join(max(0, deadline - time.monotonic()))
    return dict(results)


def _nvidia_gpu() -> tuple[str, str] | None:
    """(name, driver version) of the first NVIDIA GPU, if there is one."""
    if sys.platform == "darwin":
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
        )
        name, driver = out.stdout.splitlines()[0].rsplit(",", 1)
        return name.strip(), driver.strip()
    except Exception:
        return None


def preflight(full: bool = False) -> dict:
    """What would stop the app from installing or starting, before anything
    changes. The network and GPU checks only run when an install is due (or
    with ``full``), so an ordinary launch doesn't wait for them."""
    failures: list[dict] = []
    warnings: list[str] = []
    notes: list[str] = []

    def failed(code: str, message: str) -> None:
        failures.append({"code": code, "message": message})

    lock = platform_lock()
    py = venv_python()
    install_needed = not (py.exists() and is_required_py(py) and deps_up_to_date(lock))

    version = PROJECT_ROOT / "VERSION"
    system: dict = {
        "version": version.read_text(encoding="utf-8").strip() if version.exists() else None,
        "os": platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "lock": lock.name if lock else None,
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock else None,
    }

    problem = _os_problem()
    if problem:
        failed("os", problem)
    if not lock:
        message = f"There is no Fragmenta build for {sys.platform} on {platform.machine()}."
        if PACKAGED:
            failed("os", message)
        else:
            warnings.append(message + " Resolving requirements.txt instead.")

    problem = _data_dir_problem()
    if problem:
        failed("permission", problem)

    free = _free_gb()
    system["disk_free_gb"] = round(free, 1)
    if install_needed:
        need = DISK_UPDATE_GB if VENV_PATH.exists() else DISK_FIRST_INSTALL_GB
        if free < need:
            failed("disk", f"Fragmenta needs about {need} GB of free space to install, and the drive "
                           f"holding {DATA_DIR} has {free:.1f} GB free. Free up some space and try again.")
        elif free < need + DISK_MODEL_GB:
            warnings.append(f"Only {free:.1f} GB is free. Fragmenta will install, but models need "
                            f"at least {DISK_MODEL_GB} GB more.")

    proxies = sorted(k for k in os.environ if k.upper() in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"))
    if proxies:
        notes.append("Proxy settings found: " + ", ".join(proxies))

    if install_needed or full:
        hosts = _reachable_hosts()
        system["hosts"] = hosts
        down = [h for h, up in hosts.items() if not up]
        if install_needed and len(down) == len(hosts):
            failed("network", "Fragmenta needs the internet to finish installing, and couldn't reach "
                              "any download server. Check your connection, firewall or proxy, and try again.")
        elif install_needed and down:
            warnings.append(f"Couldn't reach {', '.join(down)}. The install may not finish.")

        gpu = _nvidia_gpu()
        if gpu:
            system["gpu"], system["driver"] = gpu
            major = _version_tuple(gpu[1])[:1]
            if major and major[0] < MIN_NVIDIA_DRIVER:
                warnings.append(f"Your NVIDIA driver ({gpu[1]}) is older than version {MIN_NVIDIA_DRIVER}. "
                                "Fragmenta won't use the GPU until it's updated.")

    report = {
        "ok": not failures,
        "install_needed": install_needed,
        "full": install_needed or full,
        "failures": failures,
        "warnings": warnings,
        "notes": notes,
        "system": system,
        "log_dir": str(LOG_DIR),
    }
    if report["full"]:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            (LOG_DIR / "preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
    return report


def launch(py: Path) -> "None":
    log("starting Fragmenta…")
    if _log_file:
        _log_file.close()
    os.chdir(PROJECT_ROOT)
    sys.exit(subprocess.run([str(py), str(PROJECT_ROOT / "start.py")]).returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fragmenta bootstrapping installer")
    ap.add_argument("--launch", action="store_true", help="start the app after ensuring the environment")
    ap.add_argument("--reinstall", action="store_true", help="force a dependency reinstall")
    ap.add_argument("--check", action="store_true", help="report status only; change nothing")
    ap.add_argument("--preflight", action="store_true", help="print what would stop an install or launch, as JSON")
    ap.add_argument("--full", action="store_true", help="with --preflight: run the network and GPU checks anyway")
    args = ap.parse_args()

    if args.preflight:
        report = preflight(full=args.full)
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["ok"] else 1)

    if not REQUIREMENTS.exists():
        fail(f"requirements.txt not found at {REQUIREMENTS} — run from the Fragmenta folder.")

    if args.check:
        py = venv_python()
        lock = platform_lock()
        ok = VENV_PATH.exists() and py.exists() and is_required_py(py)
        print(f"venv:           {f'present (py{REQUIRED_PY_STR})' if ok else 'missing / wrong python'}")
        print(f"lock:           {lock.relative_to(PROJECT_ROOT) if lock else 'none for this platform (requirements.txt)'}")
        print(f"installer:      {'uv' if find_uv() else 'pip'}")
        print(f"dependencies:   {'up to date' if ok and deps_up_to_date(lock) else 'install needed'}")
        sys.exit(0 if ok and deps_up_to_date(lock) else 1)

    log(f"install log: {open_log()}")
    require_host_py()
    py = ensure_venv()
    install_dependencies(py, force=args.reinstall)
    verify(py)
    if args.launch:
        launch(py)
    else:
        log("environment ready. Launch with: python install.py --launch")


if __name__ == "__main__":
    main()
