"""Uninstall Fragmenta from inside the app (Settings → About).

Only the installed app can do this. It removes the app and its data folder, or,
when the user keeps their models, projects and outputs, only what Fragmenta
downloaded for itself (the Python packages and their download cache).

Windows can't delete files a running program uses, so the work is handed to a
small script outside the app: it waits for Fragmenta to quit, deletes the data,
then removes the app. macOS moves it to the Trash, Windows runs the uninstaller
Setup installed (silently), Linux deletes the AppImage or the unpacked folder.
The backend then exits with ``EXIT_CODE``, and start.py closes the window.
"""
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.core.config import get_config

# Must stay in sync with start.py's UNINSTALL_EXIT_CODE.
EXIT_CODE = 86
# Rebuilt by the app at its next launch, so they go even when data is kept.
PACKAGES = ("venv", "cache")
# start.py writes these on Linux; they would point at an app that's gone. The
# last two are the window's names on Wayland (start.wayland_app_id).
LINUX_ENTRIES = (
    Path.home() / ".local" / "share" / "applications" / "fragmenta.desktop",
    Path.home() / ".local" / "share" / "icons" / "fragmenta.png",
    Path.home() / ".local" / "share" / "applications" / "brave-127.0.0.1__-Default.desktop",
    Path.home() / ".local" / "share" / "applications" / "chrome-127.0.0.1__-Default.desktop",
)
WAIT_SECONDS = 60


def _deployed() -> bool:
    return (os.environ.get("FRAGMENTA_DOCKER", "0") == "1"
            or os.environ.get("FRAGMENTA_HF_SPACE", "0") == "1"
            or bool(os.environ.get("SPACE_ID")))


def _data_dir() -> Path:
    return Path(get_config().user_data_dir)


def _app() -> Optional[Path]:
    """What to remove besides the data: the .app, the install folder, or the
    AppImage. None unless it looks exactly like an installed Fragmenta."""
    bundle = Path(get_config().project_root).parent  # holds runtime/ and payload/
    if not ((bundle / "runtime" / "python").is_dir() and (bundle / "payload" / "install.py").is_file()):
        return None
    if sys.platform == "darwin":
        app = bundle.parents[2]  # Fragmenta.app/Contents/Resources/fragmenta
        return app if app.suffix == ".app" and (app / "Contents" / "Info.plist").is_file() else None
    if sys.platform == "win32":
        return bundle if (bundle / "fragmenta.exe").is_file() and any(bundle.glob("unins*.exe")) else None
    appimage = os.environ.get("APPIMAGE")
    if appimage:
        return Path(appimage) if Path(appimage).is_file() else None
    return bundle if (bundle / "AppRun").is_file() else None  # the unpacked tarball


def _sizes(data: Path) -> Tuple[int, int]:
    """(packages, everything else) in bytes, each hard-linked file counted once:
    on Linux the packages are hard links into the download cache."""
    seen, sizes = set(), [0, 0]
    for top in data.iterdir():
        which = 0 if top.name in PACKAGES else 1
        walk = os.walk(top) if top.is_dir() and not top.is_symlink() else [(str(data), [], [top.name])]
        for root, _dirs, files in walk:
            for name in files:
                try:
                    st = os.lstat(os.path.join(root, name))
                except OSError:
                    continue
                if (st.st_dev, st.st_ino) not in seen:
                    seen.add((st.st_dev, st.st_ino))
                    sizes[which] += st.st_size
    return sizes[0], sizes[1]


def _unavailable() -> Optional[str]:
    if _deployed() or os.environ.get("FRAGMENTA_PACKAGED") != "1":
        return "Only the installed app can uninstall itself."
    if _data_dir().name != "FragmentaDesktop":
        return "Fragmenta's data folder isn't where the installed app keeps it."
    if _app() is None:
        return "Couldn't find the installed app to remove."
    return None


def plan(sizes: bool = False) -> Dict[str, Any]:
    """Whether this copy can uninstall itself; ``sizes`` adds what each choice
    removes (a walk through the packages, about a second)."""
    reason = _unavailable()
    if reason:
        return {"available": False, "reason": reason}
    data = _data_dir()
    result = {"available": True, "app": str(_app()), "data_dir": str(data)}
    if sizes:
        result["packages_bytes"], result["yours_bytes"] = _sizes(data)
    return result


def _posix_script(targets, app: Path, pids) -> str:
    q = shlex.quote
    if sys.platform == "darwin":
        # Through Foundation, so macOS doesn't ask to let Fragmenta control Finder.
        trash = ('ObjC.import("Foundation"); $.NSFileManager.defaultManager'
                 f'.trashItemAtURLResultingItemURLError($.NSURL.fileURLWithPath({json.dumps(str(app))}), null, null)')
        remove_app = f"/usr/bin/osascript -l JavaScript -e {q(trash)} >/dev/null"
    else:
        remove_app = f"rm -rf -- {q(str(app))}"
    lines = ["#!/bin/sh", "# Fragmenta's uninstall: written by the app, deletes itself."]
    for pid in pids:
        lines.append(f"n=0; while kill -0 {pid} 2>/dev/null && [ $n -lt {WAIT_SECONDS * 2} ]; do sleep 0.5; n=$((n+1)); done")
    lines += ["sleep 1",
              "rm -rf -- " + " ".join(q(str(t)) for t in targets),
              remove_app,
              'rm -f -- "$0"']
    return "\n".join(lines) + "\n"


def _windows_script(targets, app: Path) -> str:
    def q(text) -> str:
        return "'" + str(text).replace("'", "''") + "'"
    # Everything Fragmenta runs comes from the install folder or the venv.
    roots = [q(os.path.join(str(folder), "")) for folder in (app, _data_dir() / "venv")]
    return "\n".join([
        "# Fragmenta's uninstall: written by the app, deletes itself.",
        "$ErrorActionPreference = 'SilentlyContinue'",
        f"$roots = @({', '.join(roots)})",
        f"$deadline = (Get-Date).AddSeconds({WAIT_SECONDS})",
        "do {",
        "    $busy = Get-Process | Where-Object { $p = $_.Path; $p -and ($roots | Where-Object { $p.StartsWith($_, 'OrdinalIgnoreCase') }) }",
        "    if ($busy) { Start-Sleep -Milliseconds 500 }",
        "} while ($busy -and (Get-Date) -lt $deadline)",
        f"foreach ($target in @({', '.join(q(t) for t in targets)})) {{",
        "    for ($i = 0; $i -lt 10 -and (Test-Path -LiteralPath $target); $i++) {",
        "        Remove-Item -LiteralPath $target -Recurse -Force",
        "        if (Test-Path -LiteralPath $target) { Start-Sleep -Seconds 1 }",
        "    }",
        "}",
        f"$uninstaller = Get-ChildItem -LiteralPath {q(app)} -Filter 'unins*.exe' | Select-Object -First 1",
        "if ($uninstaller) { Start-Process -FilePath $uninstaller.FullName -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART' }",
        "Remove-Item -LiteralPath $PSCommandPath -Force",
    ]) + "\n"


def start(keep_data: bool, busy: Optional[str]) -> Tuple[Dict[str, Any], int]:
    """Hand the uninstall to its script. The caller then exits with EXIT_CODE."""
    reason = _unavailable()
    if reason:
        return {"error": reason}, 403
    if busy:
        return {"error": busy}, 409
    data, app = _data_dir(), _app()
    targets = [data / name for name in PACKAGES] if keep_data else [data]
    if sys.platform.startswith("linux"):
        targets += [p for p in LINUX_ENTRIES if p.exists()]

    if sys.platform == "win32":
        fd, script = tempfile.mkstemp(prefix="fragmenta-uninstall-", suffix=".ps1")
        with os.fdopen(fd, "w", encoding="utf-8-sig") as f:  # the BOM tells PowerShell 5 it's UTF-8
            f.write(_windows_script(targets, app))
        cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
               "-WindowStyle", "Hidden", "-File", script]
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        fd, script = tempfile.mkstemp(prefix="fragmenta-uninstall-", suffix=".sh")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_posix_script(targets, app, (os.getpid(), os.getppid())))
        cmd, flags = ["/bin/sh", script], {"start_new_session": True}
    subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     close_fds=True, **flags)
    return {"started": True, "keep_data": keep_data}, 202
