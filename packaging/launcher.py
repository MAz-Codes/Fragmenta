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
     first run (idempotent stamp afterwards) and starts the app,
  4. shows a lightweight first-run progress splash while that work happens, so
     the user sees live status instead of a silently-bouncing dock icon.

First-run setup (build venv → pip-install torch/SA3 → download models) takes
minutes, during which no app window exists yet. To keep the app responsive and
informative we pipe the child's output, forward it to our own stdout (so a
terminal/Console launch still shows everything), and — only if startup is slower
than ``SPLASH_AFTER_SECONDS`` — pop a Tkinter splash with an indeterminate
progress bar and a live status line. The splash auto-dismisses as soon as
start.py prints ``UI_READY_SENTINEL`` (backend healthy, window about to appear),
so warm launches never flash it and the handoff is seamless.

Everything heavy (torch, SA3, flash-attn) is pip-installed by install.py at
first run, so this launcher and its frozen binary stay small (stdlib + Tk only).

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
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

# Printed by start.py once the backend is healthy and the window is opening.
# Must stay in sync with start.py:announce_ui_ready().
UI_READY_SENTINEL = "__FRAGMENTA_UI_READY__"

# Only show the splash if the app isn't ready within this many seconds, so fast
# (warm) launches never flash a splash that immediately disappears.
SPLASH_AFTER_SECONDS = 2.5

# Brand palette (matches start.py's window background).
_BG = "#0D1117"
_FG = "#E6EDF3"
_SUBFG = "#8B949E"
_ACCENT = "#2F81F7"


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


def _status_from_line(line: str) -> "str | None":
    """Map a raw child-output line to a friendly splash status, or None to keep
    the current one. Recognises install.py's curated ``[install]`` messages plus
    a couple of pip milestones; anything else leaves the status unchanged."""
    low = line.lower()
    if "creating virtual environment" in low:
        return "Creating Python environment…"
    if "ensurepip" in low or "upgrade pip" in low:
        return "Preparing package installer…"
    if "installing dependencies" in low:
        return "Downloading components — this can take a few minutes…"
    if "collecting torch" in low or "torch-2.7.1" in low:
        return "Installing PyTorch…"
    if "laion-clap" in low:
        return "Installing audio model tools…"
    if "verifying key packages" in low:
        return "Verifying installation…"
    if "starting fragmenta" in low or "dependencies already up to date" in low:
        return "Starting Fragmenta…"
    return None


def _echo(line: str) -> None:
    """Forward a child line to our stdout when there is one. In a Finder-launched
    --windowed build sys.stdout can be None; never let that crash the reader."""
    out = sys.stdout
    if out is None:
        return
    try:
        out.write(line)
        out.flush()
    except Exception:
        pass


def _dbg(msg: str) -> None:
    """Diagnostic line to stderr (visible when launched from a terminal, e.g.
    .../Contents/MacOS/fragmenta). No-ops in a Finder-launched --windowed build
    where stderr is None, so it never interferes with normal use."""
    err = sys.stderr
    if err is None:
        return
    try:
        err.write(f"[launcher] {msg}\n")
        err.flush()
    except Exception:
        pass


def _reader(proc: subprocess.Popen, q: "queue.Queue", ready: threading.Event) -> None:
    """Drain the child's merged stdout on a worker thread: forward every line to
    our own stdout (so terminal/Console launches still see all output) and feed
    status updates to the splash until the app signals readiness."""
    assert proc.stdout is not None
    for line in proc.stdout:
        _echo(line)
        if UI_READY_SENTINEL in line:
            if not ready.is_set():
                _dbg("readiness sentinel received from child")
                ready.set()
                q.put(("ready", None))
        elif not ready.is_set():
            q.put(("line", line))
    q.put(("exit", proc.poll()))


def _drain_until_ready(q: "queue.Queue", timeout: float) -> "tuple[str, str]":
    """Phase 1 — wait up to ``timeout`` for readiness (warm start) without a
    splash, tracking the latest friendly status so the splash (if needed) starts
    from the right text. Returns (kind, status) where kind is
    'ready' | 'exit' | 'timeout'."""
    deadline = time.monotonic() + timeout
    status = "Starting Fragmenta…"
    while time.monotonic() < deadline:
        try:
            kind, payload = q.get(timeout=0.1)
        except queue.Empty:
            continue
        if kind in ("ready", "exit"):
            return kind, status
        if kind == "line":
            mapped = _status_from_line(payload)
            if mapped:
                status = mapped
    return "timeout", status


def _run_splash(q: "queue.Queue", initial_status: str) -> "tuple[str, int] | None":
    """Phase 2 — show the progress splash until the app is ready, or the child
    exits (setup failed). Returns ('ready', 0) | ('error', rc), or None if Tk is
    unavailable (caller then just waits, output still streaming to stdout)."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        return None

    root = tk.Tk()
    root.title("Fragmenta")
    root.configure(bg=_BG)
    root.resizable(False, False)
    w, h = 460, 200
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 3}")
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    tk.Label(root, text="Fragmenta", bg=_BG, fg=_FG,
             font=("Helvetica", 26, "bold")).pack(pady=(40, 8))
    status_var = tk.StringVar(value=initial_status)
    tk.Label(root, textvariable=status_var, bg=_BG, fg=_SUBFG,
             font=("Helvetica", 13)).pack(pady=(0, 22))

    bar_style = {}
    try:
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Frag.Horizontal.TProgressbar", troughcolor="#161B22",
                        background=_ACCENT, bordercolor=_BG,
                        lightcolor=_ACCENT, darkcolor=_ACCENT)
        bar_style = {"style": "Frag.Horizontal.TProgressbar"}
    except Exception:
        pass
    bar = ttk.Progressbar(root, mode="indeterminate", length=340, **bar_style)
    bar.pack()
    bar.start(12)
    _dbg("splash shown")

    # macOS Aqua Tk needs a real mainloop() to stay responsive (a bare
    # root.update() loop leaves the window frozen). The dismiss races the app's
    # webview window stealing focus — once this launcher is in the background
    # macOS App Nap throttles its Tk timers and a late dismiss stalls. So poll
    # tightly and tear the window down the instant we see readiness, while this
    # process is still frontmost. start.py also yields briefly after printing the
    # readiness sentinel (packaged mode) to guarantee that head start.
    state = {"result": ("ready", 0), "done": False}

    def finish(result: "tuple[str, int]") -> None:
        if state["done"]:
            return
        state["done"] = True
        state["result"] = result
        _dbg(f"splash dismissing ({result[0]})")
        # Take the window off-screen and flush that to the window server *while the
        # event loop is still spinning* (we're inside a mainloop() callback, and
        # start.py keeps us frontmost for ~0.6s here). A bare root.destroy() leaves
        # a ghost on macOS: mainloop() returns and the process immediately blocks
        # in proc.wait() with no run loop left to clear the pixels, so the dead
        # splash stays frozen on screen. withdraw()+update() removes it now.
        try:
            root.withdraw()
            root.update()
        except tk.TclError:
            pass
        try:
            root.destroy()
        except tk.TclError:
            pass

    def poll() -> None:
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "ready":
                    finish(("ready", 0))
                    return
                if kind == "exit":
                    rc = payload if isinstance(payload, int) else 1
                    bar.stop()
                    status_var.set("Setup failed — please reopen Fragmenta.")
                    root.after(2400, lambda: finish(("error", rc)))
                    return
                if kind == "line":
                    mapped = _status_from_line(payload)
                    if mapped:
                        status_var.set(mapped)
        except queue.Empty:
            pass
        if not state["done"]:
            root.after(30, poll)

    root.after(0, poll)
    root.mainloop()
    _dbg("splash closed")
    return state["result"]


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

    # Pipe the child so we can drive the splash from its progress and still echo
    # everything to our stdout. start.py opens the real window; this process
    # stays alive (blocking on the child) for the whole app session.
    proc = subprocess.Popen(
        [str(python), str(install_py), "--launch"],
        cwd=str(payload),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    q: "queue.Queue" = queue.Queue()
    ready = threading.Event()
    threading.Thread(target=_reader, args=(proc, q, ready), daemon=True).start()

    # Phase 1: warm launches reach readiness fast — don't flash a splash.
    kind, status = _drain_until_ready(q, SPLASH_AFTER_SECONDS)
    if kind in ("ready", "exit"):
        proc.wait()
        return proc.returncode

    # Phase 2: slow / first run — show the splash until ready (or setup fails).
    outcome = _run_splash(q, status)
    if outcome is not None and outcome[0] == "error":
        proc.wait()
        return outcome[1] or proc.returncode or 1

    proc.wait()
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
