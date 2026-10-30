#!/usr/bin/env python3
"""Fragmenta native launcher (stdlib-only).

This is the tiny entry point that PyInstaller freezes into the shipped
executable (`Fragmenta.app` on macOS, `fragmenta.exe` on Windows). It does
NOT contain torch or any app dependency — it only:

  1. locates the bundled standalone Python 3.14 and the app code (the
     "payload") next to itself,
  2. sets ``FRAGMENTA_PACKAGED=1`` so install.py / config.py resolve the
     writable user-data dir,
  3. runs ``install.py --preflight`` (disk, internet, OS, GPU driver), so a
     problem shows up before a long install rather than halfway through it,
  4. runs ``<bundled-python> install.py --launch`` — which builds the venv on
     first run (idempotent stamp afterwards) and starts the app,
  5. shows a lightweight first-run progress splash while that work happens, so
     the user sees live status instead of a silently-bouncing dock icon. If
     setup fails it turns into an error window: what went wrong, plus
     [Retry] [Open log] [Copy diagnostics] [Quit].

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

Layout it expects (see packaging/assemble.py), with no version in any name:

    <bundle>/
    ├── runtime/python/     standalone CPython (bin/python3 | python.exe)
    ├── runtime/uv[.exe]    install.py's installer
    └── payload/            install.py  start.py  locks/  app/  vendor/ …

On macOS the launcher exe lives in ``Fragmenta.app/Contents/MacOS/`` and the
bundle is ``Fragmenta.app/Contents/Resources/``; on Windows the bundle is the
install directory, beside the exe.
"""
from __future__ import annotations

import collections
import json
import math
import os
import platform
import queue
import re
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

# Child output kept for the failure log, and how much of it is searched for the
# reason and copied with the diagnostics.
TAIL_LINES = 2000
CLASSIFY_LINES = 60
DIAGNOSTIC_LINES = 40
PREFLIGHT_TIMEOUT = 90

# The first group matching the end of the child's output decides. Certificate
# errors come before "network" because uv reports them as failed fetches too.
_FAILURE_PATTERNS = (
    ("disk", r"no space left|not enough space on the disk|os error 112|enospc",
     "The disk is full. Free up some space and try again."),
    ("certificate", r"certificate_verify_failed|invalid peer certificate|unknownissuer|self[- ]signed certificate",
     "A security certificate was rejected, usually by a company proxy or antivirus. "
     "Try again, or ask whoever runs your network."),
    ("download", r"hash mismatch|do not match the hashes",
     "A download arrived damaged. Try again."),
    ("network", r"failed to fetch|error sending request|dns error|failed to lookup|name or service not known"
                r"|name resolution|getaddrinfo failed|connection (refused|reset|aborted)|connection attempt failed"
                r"|timed out|network is unreachable|no route to host|max retries exceeded"
                r"|could not find a version that satisfies",
     "Couldn't reach the download servers. Check your internet connection, firewall or proxy, and try again."),
)
_GENERIC_FAILURE = "Something went wrong while starting Fragmenta. The log has the details."

# Brand palette (matches start.py's window background).
_BG = "#F2EDE1"
_FG = "#35392F"
_SUBFG = "#66625B"
_ACCENT = "#A0B2D5"
_ACCENT_HOVER = "#8EA2CA"
_CARD = "#F8F7F1"
_CONTROL = "#EEE9E0"
_CONTROL_HOVER = "#E4DED2"
_DISABLED = "#908C84"
# The system's own font: the bundled Tk can't load the app's fonts on Linux. Its
# fonts there also lack the ellipsis and the long dash (it draws symbols instead),
# so text shown in the window sticks to plain punctuation.
_FONT = {"win32": "Segoe UI", "darwin": "Helvetica Neue"}.get(sys.platform, "Helvetica")


def _bundle_root() -> Path:
    """Directory holding runtime/ and payload/."""
    override = os.environ.get("FRAGMENTA_BUNDLE")
    if override:
        return Path(override).resolve()

    if getattr(sys, "frozen", False):
        exe = Path(sys.executable).resolve()
        # macOS .app: …/Contents/MacOS/fragmenta  ->  …/Contents/Resources
        if sys.platform == "darwin" and exe.parent.name == "MacOS":
            return exe.parent.parent / "Resources"
        # Windows / Linux one-dir: the bundle sits beside the launcher exe.
        return exe.parent

    # Not frozen (running this script directly, e.g. local testing): assume the
    # bundle is the directory this file lives in. Override with FRAGMENTA_BUNDLE.
    return Path(__file__).resolve().parent


def _bundled_python(bundle: Path) -> Path:
    if os.name == "nt":
        return bundle / "runtime" / "python" / "python.exe"
    return bundle / "runtime" / "python" / "bin" / "python3"


_LONGEST_STATUS = "Downloading components - this can take a few minutes..."


def _status_from_line(line: str) -> "str | None":
    """Map a raw child-output line to a friendly splash status, or None to keep
    the current one. Recognises install.py's curated ``[install]`` messages plus
    a couple of pip milestones; anything else leaves the status unchanged."""
    low = line.lower()
    if "creating virtual environment" in low or "refreshing the venv" in low:
        return "Creating Python environment..."
    if "ensurepip" in low or "upgrade pip" in low:
        return "Preparing package installer..."
    if "installing dependencies" in low or "syncing to locks" in low:
        return _LONGEST_STATUS
    # uv prints "Downloading torch (...)"; pip "Collecting torch==..."
    if re.search(r"(downloading|collecting) torch\b", low):
        return "Downloading PyTorch..."
    if re.match(r"\s*(prepared|installed) \d+ packages", low):
        return "Installing components..."
    if line.startswith("[preflight]"):
        return "Checking your system..."
    if "laion-clap" in low:
        return "Installing audio model tools..."
    if "verifying key packages" in low:
        return "Verifying installation..."
    if "starting fragmenta" in low or "dependencies already up to date" in low:
        return "Starting Fragmenta..."
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


def _macos_hide_dock_icon() -> None:
    """Drop this launcher's Dock icon. It's marked LSUIElement (UI agent) in the
    .app Info.plist, but Tk's macOS startup forces the process to a 'regular'
    (Dock-showing) activation policy, re-adding the icon. Override it back to
    'accessory' at runtime so only the real app window's (child) process shows in
    the Dock. stdlib-only via ctypes + the Obj-C runtime; best-effort (any failure
    just leaves the icon — never crashes)."""
    if sys.platform != "darwin":
        return
    try:
        import ctypes

        objc = ctypes.CDLL("/usr/lib/libobjc.dylib")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        msg = objc.objc_msgSend

        # app = [NSApplication sharedApplication]   (id)
        msg.restype = ctypes.c_void_p
        msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        app = msg(objc.objc_getClass(b"NSApplication"),
                  objc.sel_registerName(b"sharedApplication"))
        # [app setActivationPolicy:NSApplicationActivationPolicyAccessory(=1)] (BOOL)
        msg.restype = ctypes.c_bool
        msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        msg(app, objc.sel_registerName(b"setActivationPolicy:"), 1)
    except Exception as exc:
        _dbg(f"could not set accessory policy: {exc}")


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


def _classify_failure(lines: "list[str]") -> "tuple[str, str]":
    """(code, friendly message) for a child that failed, from its last output."""
    text = "\n".join(lines[-CLASSIFY_LINES:]).lower()
    for code, pattern, message in _FAILURE_PATTERNS:
        if re.search(pattern, text):
            return code, message
    return "unknown", _GENERIC_FAILURE


def _run_preflight(python: Path, install_py: Path, env: dict, full: bool = False) -> "dict | None":
    """install.py's preflight report, or None if it couldn't run: a broken check
    must not stop the app, so the caller just carries on."""
    # A windowed launcher would otherwise flash a console for this short run.
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    cmd = [str(python), str(install_py), "--preflight"] + (["--full"] if full else [])
    try:
        out = subprocess.run(cmd, cwd=str(install_py.parent), env=env, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=PREFLIGHT_TIMEOUT, **flags)
        report = json.loads(out.stdout)
    except Exception as exc:
        _dbg(f"preflight didn't run: {exc}")
        return None
    return report if isinstance(report, dict) else None


class _Run:
    """One try at starting the app, on a worker thread: preflight, then
    ``install.py --launch``. Pipes the child so the window can follow its progress
    and everything still reaches our stdout. Posts to ``q``: ('line', text),
    ('warning', None), ('ready', None), ('failed', None) with ``failure`` set, or
    ('exit', code)."""

    def __init__(self, bundle: Path) -> None:
        self.bundle = bundle
        self.q: "queue.Queue" = queue.Queue()
        self.ready = threading.Event()
        self.tail: "collections.deque[str]" = collections.deque(maxlen=TAIL_LINES)
        self.report: "dict | None" = None
        self.warnings: "list[str]" = []
        self.failure: "dict | None" = None
        self.returncode = 0
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def _note(self, line: str) -> None:
        _echo(line + "\n")
        self.tail.append(line + "\n")

    def _fail(self, code: str, message: str, returncode: int = 1) -> None:
        self.returncode = returncode
        self.failure = {"code": code, "message": message, "log": self._write_failure_log(code, message)}
        self.q.put(("failed", None))

    def _write_failure_log(self, code: str, message: str) -> "str | None":
        # install.py's own log stops where start.py takes over, so keep everything
        # this try printed in one file for [Open log].
        log_dir = (self.report or {}).get("log_dir")
        if not log_dir:
            return None
        path = Path(log_dir) / "startup-failure.log"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{code}: {message}\n\n" + "".join(self.tail), encoding="utf-8")
        except OSError as exc:
            _dbg(f"could not write {path}: {exc}")
            return None
        return str(path)

    def _work(self) -> None:
        try:
            self._start_app()
        except Exception as exc:
            self._note(f"[launcher] {exc!r}")
            self._fail("unknown", _GENERIC_FAILURE)

    def _start_app(self) -> None:
        payload = self.bundle / "payload"
        python = _bundled_python(self.bundle)
        install_py = payload / "install.py"

        missing = [str(p) for p in (python, install_py) if not p.exists()]
        if missing:
            self._note("Missing: " + ", ".join(missing))
            self._fail("corrupt", "Fragmenta's installation is damaged. Please reinstall it.", returncode=2)
            return

        env = dict(os.environ)
        env["FRAGMENTA_PACKAGED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        self.q.put(("line", "[preflight] checking\n"))
        self.report = _run_preflight(python, install_py, env)
        if self.report:
            for text in self.report.get("notes", []) + self.report.get("warnings", []):
                self._note(f"[preflight] {text}")
            self.warnings = list(self.report.get("warnings", []))
            if self.warnings:
                self.q.put(("warning", None))
            failures = self.report.get("failures", [])
            if failures:
                message = "\n\n".join(f["message"] for f in failures)
                self._note(f"[preflight] {message}")
                self._fail(failures[0]["code"], message)
                return

        # start.py opens the real window; this thread stays alive (draining the
        # child's output) for the whole app session.
        proc = subprocess.Popen(
            [str(python), str(install_py), "--launch"],
            cwd=str(payload),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            _echo(line)
            if UI_READY_SENTINEL in line:
                if not self.ready.is_set():
                    _dbg("readiness sentinel received from child")
                    self.ready.set()
                    self.q.put(("ready", None))
            elif not self.ready.is_set():
                self.tail.append(line)
                self.q.put(("line", line))
        self.returncode = proc.wait()

        if self.ready.is_set() or self.returncode == 0:
            self.q.put(("exit", self.returncode))
            return
        # An ordinary launch skipped the slow checks; they may name the cause, and
        # the diagnostics want the GPU and driver.
        if not (self.report or {}).get("full"):
            self.report = _run_preflight(python, install_py, env, full=True) or self.report
        code, message = _classify_failure(list(self.tail))
        self._fail(code, message, self.returncode)


def _diagnostics(run: _Run) -> str:
    """Plain text for the clipboard, to paste into a bug report."""
    try:
        manifest = json.loads((run.bundle / "payload" / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    report = run.report or {}
    system = report.get("system", {})
    failure = run.failure or {}
    bundled = (manifest.get("python") or {}).get("version")
    lines = [
        f"Fragmenta {manifest.get('version') or system.get('version') or 'unknown'}",
        f"Commit: {manifest.get('commit', 'unknown')}",
        f"Target: {manifest.get('target', 'unknown')}",
        f"OS: {system.get('os') or platform.platform()}",
        f"Arch: {system.get('arch') or platform.machine()}",
        f"Python: {system.get('python') or bundled or 'unknown'}",
        f"Lock: {system.get('lock')} sha256 {system.get('lock_sha256')}",
        f"GPU: {system.get('gpu', 'none found')}, driver {system.get('driver', 'n/a')}",
        f"Free disk: {system.get('disk_free_gb', 'unknown')} GB",
        f"Download servers: {system.get('hosts', 'not checked')}",
        f"Install needed: {report.get('install_needed', 'unknown')}",
        f"Problem: {failure.get('code', 'none')}: {failure.get('message', '')}",
        f"Exit code: {run.returncode}",
        f"Log: {failure.get('log')}",
        "",
        "Last output:",
        *(line.rstrip("\n") for line in list(run.tail)[-DIAGNOSTIC_LINES:]),
    ]
    return "\n".join(lines).replace(str(Path.home()), "~")


def _open_path(path: str) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])
    except Exception as exc:
        _dbg(f"could not open {path}: {exc}")


def _drain_until_ready(q: "queue.Queue", timeout: float) -> "tuple[str, str]":
    """Phase 1 — wait up to ``timeout`` for readiness (warm start) without a
    splash, tracking the latest friendly status so the splash (if needed) starts
    from the right text. Returns (kind, status) where kind is
    'ready' | 'failed' | 'exit' | 'timeout'."""
    deadline = time.monotonic() + timeout
    status = "Starting Fragmenta..."
    while time.monotonic() < deadline:
        try:
            kind, payload = q.get(timeout=0.1)
        except queue.Empty:
            continue
        if kind in ("ready", "failed", "exit"):
            return kind, status
        if kind == "line":
            mapped = _status_from_line(payload)
            if mapped:
                status = mapped
    return "timeout", status


def _rgb(colour: str) -> "tuple[int, int, int]":
    return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)


def _shape_rows(w: int, h: int, radius: float, fill: str, bg: str, shadow: int = 0) -> "list[str]":
    """A rounded rectangle on a solid background, as rows of colours for a Tk
    image. Smoothed here because Tk draws jagged curves on Linux and Windows.
    ``shadow`` keeps that many px around the shape for a soft drop shadow."""
    fill_rgb, bg_rgb, ink_rgb = _rgb(fill), _rgb(bg), _rgb(_FG)
    half_w, half_h = (w - 2 * shadow) / 2, (h - 2 * shadow) / 2
    drop = shadow / 4

    def edge_distance(qx: float, qy: float) -> float:
        return ((max(qx, 0) ** 2 + max(qy, 0) ** 2) ** 0.5 + min(max(qx, qy), 0)) - radius

    rows: "dict[tuple[float, float], str]" = {}
    out = []
    for y in range(h):
        py = y + 0.5 - h / 2
        qy = abs(py) - (half_h - radius)
        qy_shadow = abs(py - drop) - (half_h - radius)
        # Rows well inside the shape are all the same, so draw each only once.
        key = (max(qy, -1.0), max(qy_shadow, -float(shadow) - 1))
        if key not in rows:
            pixels = []
            for x in range(w):
                qx = abs(x + 0.5 - w / 2) - (half_w - radius)
                cover = min(1.0, max(0.0, 0.5 - edge_distance(qx, qy)))
                dark = 0.0
                if shadow:
                    dark = 0.10 * min(1.0, max(0.0, 1 - edge_distance(qx, qy_shadow) / shadow)) ** 2
                pixels.append("#%02x%02x%02x" % tuple(
                    round((b + (i - b) * dark) * (1 - cover) + f * cover)
                    for b, i, f in zip(bg_rgb, ink_rgb, fill_rgb)))
            rows[key] = " ".join(pixels)
        out.append(rows[key])
    return out


def _run_window(run: _Run, initial_status: str) -> "_Run | None":
    """Phase 2 — show the progress splash until the app is ready. If setup fails,
    the same window turns into the error screen, and Retry starts a new _Run.
    Returns the last _Run, or None if there's no way to show a window (no Tk, or
    no usable display), in which case the caller just waits with the output
    still streaming to stdout."""
    try:
        import tkinter as tk
        import tkinter.font as tkfont
        root = tk.Tk()
    except Exception as exc:
        _dbg(f"no window available: {exc}")
        return None

    root.title("Fragmenta")
    root.configure(bg=_BG)
    root.resizable(False, False)
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    canvas = tk.Canvas(root, bg=_BG, highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)

    try:
        root.attributes("-topmost", True)
        # The launcher should add no Dock icon (the real app window is a separate
        # process). Tk forces a Dock-showing policy at startup, so override it back
        # to accessory now and again shortly after the window maps (Tk can
        # re-assert it on show). Agent windows don't auto-front, so lift() too.
        root.lift()
        _macos_hide_dock_icon()
        root.after(200, _macos_hide_dock_icon)
    except Exception:
        pass

    margin, shadow, padding = 18, 12, 28
    wordmark_font = tkfont.Font(family=_FONT, size=26, weight="bold")
    title_font = tkfont.Font(family=_FONT, size=20, weight="bold")
    body_font = tkfont.Font(family=_FONT, size=13)
    small_font = tkfont.Font(family=_FONT, size=11)
    button_font = tkfont.Font(family=_FONT, size=12, weight="bold")

    images: "dict[tuple, tk.PhotoImage]" = {}

    def shape(w: int, h: int, radius: float, fill: str, bg: str, blur: int = 0) -> "tk.PhotoImage":
        key = (w, h, radius, fill, bg, blur)
        if key not in images:
            image = tk.PhotoImage(master=root, width=w, height=h)
            image.put(" ".join("{" + row + "}" for row in _shape_rows(w, h, radius, fill, bg, blur)))
            images[key] = image
        return images[key]

    def place_window(w: int, h: int) -> None:
        canvas.configure(width=w, height=h)
        root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 3}")
        # The card sits under everything drawn so far.
        card = shape(w - 2 * margin + 2 * shadow, h - 2 * margin + 2 * shadow, 24, _CARD, _BG, shadow)
        canvas.tag_lower(canvas.create_image(margin - shadow, margin - shadow, image=card, anchor="nw"))

    # macOS Aqua Tk needs a real mainloop() to stay responsive (a bare
    # root.update() loop leaves the window frozen). The dismiss races the app's
    # webview window stealing focus — once this launcher is in the background
    # macOS App Nap throttles its Tk timers and a late dismiss stalls. So poll
    # tightly and tear the window down the instant we see readiness, while this
    # process is still frontmost. start.py also yields briefly after printing the
    # readiness sentinel (packaged mode) to guarantee that head start.
    state = {"run": run, "done": False, "status": initial_status, "showing": None, "started": time.monotonic()}

    def finish(reason: str) -> None:
        if state["done"]:
            return
        state["done"] = True
        _dbg(f"window dismissing ({reason})")
        # Take the window off-screen and flush that to the window server *while the
        # event loop is still spinning* (we're inside a mainloop() callback, and
        # start.py keeps us frontmost for ~0.6s here). A bare root.destroy() leaves
        # a ghost on macOS: mainloop() returns and the process immediately blocks
        # in the join with no run loop left to clear the pixels, so the dead
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

    def set_status(text: str) -> None:
        state["status"] = text
        canvas.itemconfigure("status", text=text)

    def animate() -> None:
        if state["done"] or state["showing"] != "progress":
            return
        elapsed = time.monotonic() - state["started"]
        phase = (1 - math.cos(elapsed * math.pi)) / 2
        left, top, travel = state["bar"]
        canvas.coords("glider", left + travel * phase, top)
        root.after(16, animate)

    def draw_progress() -> None:
        canvas.delete("all")
        w = max(460, body_font.measure(_LONGEST_STATUS) + 2 * (margin + padding))
        canvas.create_text(w // 2, 76, text="Fragmenta", font=wordmark_font, fill=_FG)
        canvas.create_text(w // 2, 118, text=state["status"], font=body_font, fill=_SUBFG, tags="status")
        track_w, glider_w, bar_h = w - 2 * (margin + padding), 96, 8
        left, top = (w - track_w) // 2, 146
        canvas.create_image(left, top, anchor="nw", image=shape(track_w, bar_h, bar_h / 2, _CONTROL, _CARD))
        canvas.create_image(left, top, anchor="nw", image=shape(glider_w, bar_h, bar_h / 2, _ACCENT, _CONTROL),
                            tags="glider")
        state["bar"] = (left, top, track_w - glider_w)
        h = 196
        warnings = state["run"].warnings
        if warnings:
            item = canvas.create_text(left, top + bar_h + 18, anchor="nw", text="\n".join(warnings), width=track_w,
                                      font=small_font, fill=_SUBFG)
            h = canvas.bbox(item)[3] + 34
        place_window(w, h)

    def show_progress(status: str) -> None:
        state["status"] = status
        root.unbind("<Return>")
        root.unbind("<Escape>")
        canvas.configure(cursor="")
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass
        draw_progress()
        was_animating = state["showing"] == "progress"
        state["showing"] = "progress"
        if not was_animating:
            animate()

    def button_width(label: str) -> int:
        return max(92, button_font.measure(label) + 40)

    def add_button(x: int, y: int, label: str, command, primary: bool = False, enabled: bool = True) -> int:
        """A pill drawn at x, y; returns where the next one can start."""
        w, h = button_width(label), 38
        fill, hover = (_ACCENT, _ACCENT_HOVER) if primary else (_CONTROL, _CONTROL_HOVER)
        rest_image, hover_image = shape(w, h, h / 2, fill, _CARD), shape(w, h, h / 2, hover, _CARD)
        tag = "button-" + label.split()[0].lower()
        pill = canvas.create_image(x, y, anchor="nw", image=rest_image, tags=tag)
        canvas.create_text(x + w // 2, y + h // 2, text=label, font=button_font, tags=(tag, tag + "-label"),
                           fill=_FG if enabled else _DISABLED)
        if enabled:
            def enter(_event) -> None:
                canvas.itemconfigure(pill, image=hover_image)
                canvas.configure(cursor="hand2")

            def leave(_event) -> None:
                canvas.itemconfigure(pill, image=rest_image)
                canvas.configure(cursor="")

            canvas.tag_bind(tag, "<Enter>", enter)
            canvas.tag_bind(tag, "<Leave>", leave)
            # After the click's own binding returns: Quit and Retry remove this
            # very item, and destroying the window from inside its binding crashes Tk.
            canvas.tag_bind(tag, "<ButtonRelease-1>", lambda _event: root.after(0, command))
        return x + w + 8

    def show_error() -> None:
        failure = state["run"].failure or {}
        state["showing"] = "error"
        canvas.delete("all")
        canvas.configure(cursor="")
        inner = margin + padding
        labels = ("Retry", "Open log", "Copy diagnostics", "Quit")
        width = max(620, sum(button_width(label) + 8 for label in labels) - 8 + 2 * inner)
        canvas.create_text(inner, inner + 2, anchor="nw", text="Fragmenta couldn't start",
                           font=title_font, fill=_FG)
        message = canvas.create_text(inner, inner + 46, anchor="nw", width=width - 2 * inner,
                                     text=failure.get("message", _GENERIC_FAILURE), font=body_font, fill=_SUBFG)
        y = canvas.bbox(message)[3] + 22
        x = add_button(inner, y, "Retry", retry, primary=True)
        x = add_button(x, y, "Open log", open_log, enabled=bool(failure.get("log")))
        x = add_button(x, y, "Copy diagnostics", copy_diagnostics)
        add_button(x, y, "Quit", lambda: finish("quit"))
        place_window(width, y + 38 + inner)
        root.bind("<Return>", lambda _event: retry())
        root.bind("<Escape>", lambda _event: finish("quit"))
        try:
            # Stay out of the way of the log it can open.
            root.attributes("-topmost", False)
        except tk.TclError:
            pass
        _dbg(f"error shown: {failure.get('code')}")

    def poll() -> None:
        current = state["run"]
        try:
            while True:
                kind, payload = current.q.get_nowait()
                if kind == "ready":
                    finish("ready")
                    return
                if kind == "exit":
                    finish("child exited")
                    return
                if kind == "failed":
                    show_error()
                    return
                if kind == "warning":
                    draw_progress()
                if kind == "line":
                    mapped = _status_from_line(payload)
                    if mapped:
                        set_status(mapped)
        except queue.Empty:
            pass
        if not state["done"]:
            root.after(30, poll)

    def retry() -> None:
        state["run"] = _Run(state["run"].bundle)
        show_progress("Starting Fragmenta...")
        root.after(0, poll)

    def open_log() -> None:
        log = (state["run"].failure or {}).get("log")
        if log:
            _open_path(log)

    def copy_diagnostics() -> None:
        root.clipboard_clear()
        root.clipboard_append(_diagnostics(state["run"]))
        root.update()
        canvas.itemconfigure("button-copy-label", text="Copied")
        root.after(1500, lambda: canvas.itemconfigure("button-copy-label", text="Copy diagnostics"))

    if run.failure:
        show_error()
    else:
        show_progress(initial_status)
        root.after(0, poll)
    _dbg("window shown")
    root.mainloop()
    _dbg("window closed")
    return state["run"]


def main() -> int:
    run = _Run(_bundle_root())

    # Phase 1: warm launches reach readiness fast — don't flash a splash.
    kind, status = _drain_until_ready(run.q, SPLASH_AFTER_SECONDS)

    # Phase 2: slow / first run, or it already failed.
    if kind in ("timeout", "failed"):
        run = _run_window(run, status) or run

    run.thread.join()
    if run.failure:
        _dbg(f"failed: {run.failure['message']}")
        _dbg(f"log: {run.failure['log']}")
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
