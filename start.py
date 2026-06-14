import argparse
import os
import requests
import shutil
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 5001
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
HEALTH_ENDPOINT = f"{BACKEND_URL}/api/health"
APP_WM_CLASS = "Fragmenta"
APP_ICON_PATH = PROJECT_ROOT / "app" / "frontend" / "public" / "fragmenta_icon_1024.png"

try:
    ABOUT_VERSION = (PROJECT_ROOT / "VERSION").read_text().strip() or "1.0.0"
except Exception:
    ABOUT_VERSION = "1.0.0"
ABOUT_COPYRIGHT = "© 2026 Misagh Azimi · www.misaghazimi.com"
ABOUT_CREDITS = (
    "Local-first text-to-audio: prepare datasets, train, generate and perform "
    "with diffusion models.\n\nMade by the composer and researcher Misagh Azimi."
)

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
CHROMIUM_MIN_SESSION_SECONDS = 15.0
WEBVIEW_STORAGE_DIR = Path.home() / ".cache" / "fragmenta-webview-profile"
DESKTOP_ENTRY_PATH = (
    Path.home() / ".local" / "share" / "applications" / "fragmenta.desktop"
)

def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name != "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((BACKEND_HOST, port))
            return True
        except OSError:
            return False


def _pick_backend_port(preferred: int = 5001) -> int:
    for candidate in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((BACKEND_HOST, candidate))
                return probe.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("no free TCP port available for the Fragmenta backend")


def _is_fragmenta_backend(port: int) -> bool:
    try:
        resp = requests.get(
            f"http://{BACKEND_HOST}:{port}/api/health", timeout=1.5
        )
        return resp.status_code == 200 and "components_ready" in resp.json()
    except Exception:
        return False


def _install_sigterm_handler() -> None:
    def _terminate(_signum, _frame):
        raise SystemExit(143)
    try:
        signal.signal(signal.SIGTERM, _terminate)
    except (ValueError, OSError, AttributeError):
        pass


def _terminate_backend_on_port(port: int) -> bool:
    try:
        import psutil
    except ImportError:
        return False
    victim = None
    for proc in psutil.process_iter():
        try:
            for conn in proc.net_connections(kind="tcp"):
                if (conn.status == psutil.CONN_LISTEN and conn.laddr
                        and conn.laddr.port == port):
                    victim = proc
                    break
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        if victim is not None:
            break
    if victim is None:
        return False
    try:
        victim.terminate()
        try:
            victim.wait(timeout=10)
        except psutil.TimeoutExpired:
            victim.kill()
            victim.wait(timeout=5)
    except psutil.NoSuchProcess:
        pass
    except Exception:
        return False

    for _ in range(20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((BACKEND_HOST, port)) != 0:
                return True
        time.sleep(0.25)
    return False


def _maybe_replace_stale_backend(port: int) -> None:
    try:
        resp = requests.get(
            f"http://{BACKEND_HOST}:{port}/api/health", timeout=1.5
        )
        running_version = (resp.json() or {}).get("version")
    except Exception:
        return
    if running_version == ABOUT_VERSION:
        return
    print(
        f"The Fragmenta backend already running on port {port} is version "
        f"{running_version or 'unknown (older build)'}, but this installation "
        f"is {ABOUT_VERSION}. Restarting it so the app matches the installed code."
    )
    if not _terminate_backend_on_port(port):
        print(
            "Could not stop the old backend automatically — if the app "
            "misbehaves, close the other Fragmenta instance and relaunch."
        )


def configure_backend_endpoint() -> None:
    global BACKEND_PORT, BACKEND_URL, HEALTH_ENDPOINT
    if _port_available(5001):
        BACKEND_PORT = 5001
    elif _is_fragmenta_backend(5001):
        BACKEND_PORT = 5001
        _maybe_replace_stale_backend(5001)
    else:
        BACKEND_PORT = _pick_backend_port()
    BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
    HEALTH_ENDPOINT = f"{BACKEND_URL}/api/health"
    if BACKEND_PORT != 5001:
        print(
            f"Port 5001 is busy with another program; using {BACKEND_URL} "
            "instead. Note: presets, MIDI mappings and fragments saved on 5001 "
            "won't be visible here (browser storage is per-port)."
        )


def announce_ui_ready() -> None:
    print("__FRAGMENTA_UI_READY__", flush=True)
    if os.environ.get("FRAGMENTA_PACKAGED") == "1":
        time.sleep(0.6)


def wait_for_backend(timeout_seconds: int = 60) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _is_fragmenta_backend(BACKEND_PORT):
            return True
        time.sleep(0.4)
    return False

def backend_process_running() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((BACKEND_HOST, BACKEND_PORT)) == 0

def start_backend_subprocess() -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("FRAGMENTA_LOG_LEVEL", "INFO")
    env["FLASK_HOST"] = BACKEND_HOST
    env["FLASK_PORT"] = str(BACKEND_PORT)
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
        announce_ui_ready()

        webbrowser.open(BACKEND_URL)
        print(f"Fragmenta is running at {BACKEND_URL}")
        print("Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            if backend_process is not None:
                rc = backend_process.poll()
                if rc is not None:
                    print(f"Backend exited (code {rc}); shutting down.")
                    return 1 if rc else 0
            elif not backend_process_running():
                print("The reused backend is no longer running; shutting down.")
                return 1
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
    return []


def find_chromium() -> str | None:
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
    for candidate in _chromium_install_paths():
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def ensure_desktop_entry() -> None:
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
        announce_ui_ready()

        if sys.platform == "linux":
            ensure_desktop_entry()
        CHROMIUM_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

        launch_args_base = [
            chromium_path,
            f"--app={BACKEND_URL}",
            "--window-size=1400,850",
            "--no-first-run",
            "--no-default-browser-check",
            "--log-level=3",
            "--disable-logging",
        ]
        if sys.platform == "linux":
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

        quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        last_exit_code = 1
        for index, (label, args) in enumerate(attempts):
            launched_at = time.monotonic()
            chromium_process = subprocess.Popen(args, cwd=str(PROJECT_ROOT), **quiet)
            chromium_process.wait()
            elapsed = time.monotonic() - launched_at
            last_exit_code = chromium_process.returncode or 0
            if last_exit_code == 0:
                return 0
            if elapsed >= CHROMIUM_MIN_SESSION_SECONDS:
                print(
                    f"Chromium exited with code {last_exit_code} after "
                    f"{elapsed:.0f}s — treating it as the end of the session."
                )
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
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except Exception as exc:
        print(f"Could not set AppUserModelID: {exc}")

def _macos_set_app_metadata() -> None:
    if sys.platform != "darwin":
        return
    try:
        from Foundation import NSBundle, NSProcessInfo
        try:
            NSProcessInfo.processInfo().setProcessName_("Fragmenta")
        except Exception:
            pass
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None:
            info["CFBundleName"] = "Fragmenta"
            info["CFBundleDisplayName"] = "Fragmenta"
    except Exception as exc:
        print(f"Could not set macOS app name: {exc}")


def _macos_set_webview_app_name() -> None:
    if sys.platform != "darwin":
        return
    try:
        from webview.platforms import cocoa as _cocoa
        if getattr(_cocoa, "info", None) is not None:
            _cocoa.info["CFBundleName"] = "Fragmenta"
            _cocoa.info["CFBundleDisplayName"] = "Fragmenta"
            _cocoa.info["CFBundleShortVersionString"] = ABOUT_VERSION
            _cocoa.info["NSHumanReadableCopyright"] = ABOUT_COPYRIGHT
            try:
                from AppKit import NSAttributedString
                _cocoa.info["Credits"] = NSAttributedString.alloc().initWithString_(
                    ABOUT_CREDITS
                )
            except Exception:
                pass
    except Exception as exc:
        print(f"Could not set webview app name: {exc}")


def _macos_apply_dock_icon() -> None:
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
    if sys.platform != "linux":
        return
    if not APP_ICON_PATH.exists():
        return
    try:
        import gi
        gi.require_version("Gtk", "3.0")
        from gi.repository import GLib, Gtk
        GLib.set_prgname(APP_WM_CLASS)
        GLib.set_application_name("Fragmenta")
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
    _macos_set_app_metadata()
    try:
        import webview
    except ImportError:
        print("pywebview is unavailable; falling back to browser mode.")
        return run_browser_mode()

    _macos_set_webview_app_name()
    _windows_set_app_id()
    _linux_set_app_metadata()

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
        announce_ui_ready()

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

            WEBVIEW_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            webview.start(private_mode=False, storage_path=str(WEBVIEW_STORAGE_DIR))
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
    _install_sigterm_handler()
    configure_backend_endpoint()
    if args.browser:
        return run_browser_mode()
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
