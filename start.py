import argparse
import os
import requests
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
BACKEND_URL = "http://127.0.0.1:5001"
HEALTH_ENDPOINT = f"{BACKEND_URL}/api/health"
APP_WM_CLASS = "Fragmenta"
APP_ICON_PATH = PROJECT_ROOT / "app" / "frontend" / "public" / "fragmenta_icon_1024.png"

WINDOWS_ICON_PATH = PROJECT_ROOT / "app" / "frontend" / "public" / "fragmenta.ico"
WINDOWS_APP_ID = "Fragmenta.Desktop.1"
WINDOWS_WINDOW_TITLE = "Fragmenta"
CHROMIUM_CANDIDATES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "brave-browser",
)
CHROMIUM_USER_DATA_DIR = Path.home() / ".cache" / "fragmenta-chrome-profile"
DESKTOP_ENTRY_PATH = (
    Path.home() / ".local" / "share" / "applications" / "fragmenta.desktop"
)

def wait_for_backend(timeout_seconds: int = 60) -> bool:
    """Wait until the Flask backend becomes reachable."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=1.5)
            if response.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.4)
    return False

def backend_process_running() -> bool:
    """Return True if something is already listening on backend port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", 5001)) == 0

def start_backend_subprocess() -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("FRAGMENTA_LOG_LEVEL", "INFO")
    return subprocess.Popen(
        [sys.executable, "-m", "app.backend.app"],
        cwd=str(PROJECT_ROOT),
        env=env,
    )

def run_browser_mode() -> int:
    backend_process = None
    try:
        if not backend_process_running():
            backend_process = start_backend_subprocess()
            if not wait_for_backend():
                print("Backend failed to start in time.")
                return 1

        webbrowser.open(BACKEND_URL)
        print(f"Fragmenta is running at {BACKEND_URL}")
        print("Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return 0
    finally:
        if backend_process and backend_process.poll() is None:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                backend_process.kill()

def _chromium_install_paths() -> list[str]:
    """Absolute install locations to probe on macOS/Windows (where the browser
    isn't on PATH), in priority order: real Chrome/Chromium before Edge/Brave."""
    if sys.platform == "darwin":
        home = Path.home()
        return [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            str(home / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        ]
    if sys.platform == "win32":
        pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
        pfx86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
        local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return [
            rf"{pf}\Google\Chrome\Application\chrome.exe",
            rf"{pfx86}\Google\Chrome\Application\chrome.exe",
            rf"{local}\Google\Chrome\Application\chrome.exe",
            rf"{pf}\Chromium\Application\chrome.exe",
            rf"{pf}\Microsoft\Edge\Application\msedge.exe",
            rf"{pfx86}\Microsoft\Edge\Application\msedge.exe",
            rf"{pf}\BraveSoftware\Brave-Browser\Application\brave.exe",
            rf"{pfx86}\BraveSoftware\Brave-Browser\Application\brave.exe",
        ]
    return []  # Linux resolves via PATH (below)


def find_chromium() -> str | None:
    # 1) PATH names — primarily Linux, but also mac/win if a browser is on PATH.
    #    Skip snap-confined browsers (they misbehave in --app mode).
    for name in CHROMIUM_CANDIDATES:
        path = shutil.which(name)
        if not path:
            continue
        resolved = os.path.realpath(path)
        if (
            path.startswith("/snap/")
            or resolved.startswith("/snap/")
            or resolved == "/usr/bin/snap"
        ):
            continue
        return path
    # 2) Absolute install locations (macOS / Windows, where Chrome isn't on PATH).
    for candidate in _chromium_install_paths():
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def ensure_desktop_entry() -> None:
    """Install a .desktop file so the WM picks up Fragmenta's name and icon."""
    if DESKTOP_ENTRY_PATH.exists():
        return
    DESKTOP_ENTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    contents = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Fragmenta\n"
        "Comment=Fine-tune and use text-to-audio models\n"
        f"Icon={APP_ICON_PATH}\n"
        "Categories=AudioVideo;Audio;\n"
        f"StartupWMClass={APP_WM_CLASS}\n"
        "NoDisplay=true\n"
    )
    DESKTOP_ENTRY_PATH.write_text(contents)


def run_chromium_app_mode(chromium_path: str) -> int:
    backend_process = None
    chromium_process = None
    try:
        if not backend_process_running():
            backend_process = start_backend_subprocess()
            if not wait_for_backend():
                print("Backend failed to start in time.")
                return 1

        if sys.platform == "linux":
            ensure_desktop_entry()
        CHROMIUM_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

        launch_args_base = [
            chromium_path,
            f"--app={BACKEND_URL}",
            "--window-size=1400,850",
            "--no-first-run",
            "--no-default-browser-check",
            # Quiet the browser's own console spam (Chromium INFO/WARNING/ERROR,
            # e.g. Brave's P3A telemetry). FATAL-only.
            "--log-level=3",
            "--disable-logging",
        ]
        if sys.platform == "linux":
            # X11/Wayland WM class so the launcher/taskbar group under Fragmenta.
            # macOS/Windows ignore (or warn about) it, so it's Linux-only.
            launch_args_base.insert(2, f"--class={APP_WM_CLASS}")

        attempts = [
            (
                "custom profile",
                [
                    *launch_args_base,
                    f"--user-data-dir={CHROMIUM_USER_DATA_DIR}",
                ],
            ),
            ("default profile", launch_args_base),
        ]

        # Silence the browser subprocess entirely — both Chromium's own logging
        # and library spam printed straight to stderr (e.g. the Mesa
        # "MESA-LOADER: failed to open dri ... Permission denied" GL fallback
        # notice) flow through here. Our Python backend logs from a separate
        # process and is unaffected.
        quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        last_exit_code = 1
        for index, (label, args) in enumerate(attempts):
            chromium_process = subprocess.Popen(args, cwd=str(PROJECT_ROOT), **quiet)
            chromium_process.wait()
            last_exit_code = chromium_process.returncode or 0
            if last_exit_code == 0:
                return 0

            if index == 0:
                print(
                    "Chromium app mode failed with "
                    f"{label} (exit code {last_exit_code}); retrying with default profile."
                )
                continue

        return last_exit_code
    except KeyboardInterrupt:
        return 0
    finally:
        if chromium_process and chromium_process.poll() is None:
            chromium_process.terminate()
            try:
                chromium_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                chromium_process.kill()
        if backend_process and backend_process.poll() is None:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                backend_process.kill()

def check_linux_webview_deps() -> tuple[bool, str]:
    """Check if Linux has required GTK/WebKit dependencies for pywebview."""
    if sys.platform != "linux":
        return True, ""
    try:
        import gi
        gi.require_version("Gtk", "3.0")
        from gi.repository import Gtk
        gi.require_version("WebKit2", "4.1")
        from gi.repository import WebKit2
        return True, ""
    except ImportError as e:
        return False, f"Missing GI bindings: {e}"
    except ValueError as e:
        return False, f"Missing GTK/WebKit: {e}"

def _ensure_windows_icon() -> bool:
    """Generate a multi-resolution .ico from the source PNG if it's missing.

    Cached on disk after first run; relies on Pillow, which is already part of
    the project's dependency footprint. Returns False if generation fails.
    """
    if WINDOWS_ICON_PATH.exists():
        return True
    try:
        from PIL import Image
        img = Image.open(APP_ICON_PATH)
        WINDOWS_ICON_PATH.parent.mkdir(parents=True, exist_ok=True)
        img.save(
            WINDOWS_ICON_PATH,
            format="ICO",
            sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )
        return True
    except Exception as exc:
        print(f"Could not generate Windows icon: {exc}")
        return False

def _windows_set_app_id() -> None:
    """Give the host process its own AppUserModelID. Without this, Windows
    looks up the taskbar icon by the host executable (python.exe) and groups
    Fragmenta under "Python" in the taskbar/alt-tab list."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except Exception as exc:
        print(f"Could not set AppUserModelID: {exc}")

def _macos_set_app_metadata() -> None:
    """Make the macOS menu-bar app name read 'Fragmenta' instead of 'Python'.

    Running a bare interpreter, the app menu's bold title comes from the
    process bundle's CFBundleName — which is the Python interpreter. Patch the
    in-memory Info dictionary before pywebview builds the Cocoa menu. (The
    permanent fix is shipping a real .app bundle — see distribution.md.)"""
    if sys.platform != "darwin":
        return
    try:
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None:
            info["CFBundleName"] = "Fragmenta"
    except Exception as exc:
        print(f"Could not set macOS app name: {exc}")


def _macos_apply_dock_icon() -> None:
    """Set the Dock icon to Fragmenta's at runtime (no .app bundle needed).
    Bound to the window 'shown' event so NSApplication is already up."""
    if sys.platform != "darwin":
        return
    try:
        from AppKit import NSApplication, NSImage
        image = NSImage.alloc().initByReferencingFile_(str(APP_ICON_PATH))
        if image is not None:
            NSApplication.sharedApplication().setApplicationIconImage_(image)
    except Exception as exc:
        print(f"Could not set macOS dock icon: {exc}")


def _linux_set_app_metadata() -> None:
    """Linux analogue of `_windows_set_app_id` + bundled icon setup.

    Sets the GTK program name / WM_CLASS and registers Fragmenta's PNG as the
    default icon for every Gtk.Window created after this call. pywebview's
    GTK backend constructs its WebKit window inside `webview.start()`, so as
    long as this runs first the dock picks up the icon and associates the
    window with our identity.
    """
    if sys.platform != "linux":
        return
    if not APP_ICON_PATH.exists():
        return
    try:
        import gi
        gi.require_version("Gtk", "3.0")
        from gi.repository import GLib, Gtk
        # WM_CLASS res_name (first string).
        GLib.set_prgname(APP_WM_CLASS)
        GLib.set_application_name("Fragmenta")
        # WM_CLASS res_class (second string) — what GNOME/KDE compare against
        # StartupWMClass when resolving a window to a .desktop entry.
        try:
            gi.require_version("Gdk", "3.0")
            from gi.repository import Gdk
            Gdk.set_program_class(APP_WM_CLASS)
        except Exception:
            pass
        Gtk.Window.set_default_icon_from_file(str(APP_ICON_PATH))
    except Exception as exc:
        print(f"Could not set Linux app metadata: {exc}")


def _linux_apply_window_icon() -> None:
    """Belt-and-suspenders: after pywebview shows the window, reach into the
    GTK backend and reapply the window icon directly. Safe no-op if the
    internals differ from the version we expect.

    NOTE: we deliberately do NOT reapply WM_CLASS here. set_wmclass() is
    deprecated in GTK3 and a no-op once the window is realized (the WM reads
    WM_CLASS only at map time) — calling it from the 'shown' event just emits
    a DeprecationWarning + Gtk-WARNING. The class is set correctly before the
    window exists, in _linux_set_app_metadata() via Gdk.set_program_class().
    """
    if sys.platform != "linux":
        return
    if not APP_ICON_PATH.exists():
        return
    try:
        from webview.platforms.gtk import BrowserView
        for inst in BrowserView.instances.values():
            gtk_window = getattr(inst, "window", None)
            if gtk_window is None:
                continue
            try:
                gtk_window.set_icon_from_file(str(APP_ICON_PATH))
            except Exception:
                pass
    except Exception as exc:
        print(f"Could not set Linux window icon: {exc}")


def _windows_apply_window_icon() -> None:
    """Replace the pywebview window's titlebar/taskbar icon with Fragmenta's.
    Called from the window's `shown` event so the HWND is guaranteed to exist.
    """
    if sys.platform != "win32":
        return
    if not _ensure_windows_icon():
        return
    try:
        import ctypes
        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1
        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x00000010

        hwnd = ctypes.windll.user32.FindWindowW(None, WINDOWS_WINDOW_TITLE)
        if not hwnd:
            return
        icon_path = str(WINDOWS_ICON_PATH)
        h_small = ctypes.windll.user32.LoadImageW(
            None, icon_path, IMAGE_ICON, 16, 16, LR_LOADFROMFILE
        )
        h_big = ctypes.windll.user32.LoadImageW(
            None, icon_path, IMAGE_ICON, 32, 32, LR_LOADFROMFILE
        )
        if h_small:
            ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, h_small)
        if h_big:
            ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, h_big)
    except Exception as exc:
        print(f"Could not set Windows window icon: {exc}")


def run_pywebview_mode() -> int:
    try:
        import webview
    except ImportError:
        print("pywebview is unavailable; falling back to browser mode.")
        return run_browser_mode()

    _windows_set_app_id()
    _linux_set_app_metadata()
    _macos_set_app_metadata()

    if sys.platform == "linux":
        deps_ok, deps_error = check_linux_webview_deps()
        if not deps_ok:
            print("=" * 60)
            print("DESKTOP MODE UNAVAILABLE")
            print("=" * 60)
            print(f"Reason: {deps_error}")
            print()
            print("To use the desktop window, install system dependencies:")
            print()
            print("  Ubuntu/Debian:")
            print("    sudo apt install python3-gi python3-gi-cairo gir1.2-webkit2-4.1")
            print()
            print("  Fedora:")
            print("    sudo dnf install python3-gobject webkit2gtk4.1")
            print()
            print("  Arch:")
            print("    sudo pacman -S python-gobject webkit2gtk")
            print()
            print("Falling back to browser mode...")
            print("=" * 60)
            return run_browser_mode()

    backend_process = None
    try:
        if not backend_process_running():
            backend_process = start_backend_subprocess()
            if not wait_for_backend():
                print("Backend failed to start in time.")
                return 1

        try:
            window = webview.create_window(
                title=WINDOWS_WINDOW_TITLE,
                url=BACKEND_URL,
                width=1400,
                height=850,
                min_size=(1000, 700),
                background_color="#0D1117",
            )
            if sys.platform == "win32":
                window.events.shown += _windows_apply_window_icon
            elif sys.platform == "linux":
                window.events.shown += _linux_apply_window_icon
            elif sys.platform == "darwin":
                window.events.shown += _macos_apply_dock_icon
            webview.start()
            return 0 if window else 1
        except Exception as exc:
            print(
                "pywebview initialization failed "
                f"({exc}); falling back to browser mode."
            )
            print(
                "Tip: ensure required GTK/WebKit system packages are installed."
            )
            return run_browser_mode()
    finally:
        if backend_process and backend_process.poll() is None:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                backend_process.kill()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Fragmenta desktop app")
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Use the system browser instead of the pywebview window",
    )
    parser.add_argument(
        "--chromium",
        action="store_true",
        help=(
            "Launch in a real Chromium browser (--app mode) instead of the "
            "native window. Only needed for cue output-device routing "
            "(AudioContext.setSinkId), which the OS WebViews lack on mac/linux."
        ),
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if args.browser:
        return run_browser_mode()
    # Default window engine per OS:
    #   macOS/Windows → the native pywebview window (WKWebView/WebView2),
    #     branded (Fragmenta icon + name) and rendering reliably.
    #   Linux → Chromium --app mode. WebKitGTK frequently renders the app as a
    #     blank/dark window (the page loads but never paints), so the native
    #     window isn't dependable here; a real Chromium engine is. Falls back
    #     to pywebview if no usable (non-snap) Chromium/Brave is found.
    # `--chromium` forces Chromium mode on any OS; `--browser` opens the system
    # browser. MIDI is native (rtmidi); core audio + master recording work in
    # every engine. The only thing the OS WebViews lack on mac/linux is setSinkId
    # (routing the cue to a *separate* output device).
    prefer_chromium = args.chromium or sys.platform == "linux"
    if prefer_chromium:
        chromium_path = find_chromium()
        if chromium_path:
            chromium_exit_code = run_chromium_app_mode(chromium_path)
            if chromium_exit_code == 0:
                return 0
            print(
                "Chromium app mode failed "
                f"(exit code {chromium_exit_code}); falling back to pywebview."
            )
    return run_pywebview_mode()

if __name__ == "__main__":
    raise SystemExit(main())
