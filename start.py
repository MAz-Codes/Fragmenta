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


def find_chromium() -> str | None:
    """Return path to a Chromium-family browser, skipping snap-confined ones.

    Snap'd browsers can't write to --user-data-dir outside ~/snap/, which
    breaks --app mode with a custom profile path.
    """
    for name in CHROMIUM_CANDIDATES:
        path = shutil.which(name)
        if not path:
            continue
        resolved = os.path.realpath(path)
        # Snap launchers often appear as /snap/bin/<browser> but resolve to
        # /usr/bin/snap, so check both forms before selecting the browser.
        if (
            path.startswith("/snap/")
            or resolved.startswith("/snap/")
            or resolved == "/usr/bin/snap"
        ):
            continue
        return path
    return None


def ensure_desktop_entry() -> None:
    """Install a .desktop file so the WM picks up Fragmenta's name and icon."""
    if DESKTOP_ENTRY_PATH.exists():
        return
    DESKTOP_ENTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    contents = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Fragmenta Desktop\n"
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

        ensure_desktop_entry()
        CHROMIUM_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

        launch_args_base = [
            chromium_path,
            f"--app={BACKEND_URL}",
            f"--class={APP_WM_CLASS}",
            "--window-size=1200,800",
            "--no-first-run",
            "--no-default-browser-check",
        ]

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

        last_exit_code = 1
        for index, (label, args) in enumerate(attempts):
            chromium_process = subprocess.Popen(args, cwd=str(PROJECT_ROOT))
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


def run_pywebview_mode() -> int:
    try:
        import webview
    except ImportError:
        print("pywebview is unavailable; falling back to browser mode.")
        return run_browser_mode()

    # Pre-check Linux dependencies for better error messages
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
                title="Fragmenta Desktop",
                url=BACKEND_URL,
                width=1200,
                height=800,
                min_size=(1000, 700),
                background_color="#0D1117",
            )
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
        help="Use the system browser instead of pywebview window",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.browser:
        return run_browser_mode()
    if sys.platform == "linux":
        chromium_path = find_chromium()
        if chromium_path:
            chromium_exit_code = run_chromium_app_mode(chromium_path)
            if chromium_exit_code == 0:
                return 0
            print(
                "Chromium app mode failed "
                f"(exit code {chromium_exit_code}); falling back to pywebview/browser mode."
            )
    return run_pywebview_mode()


if __name__ == "__main__":
    raise SystemExit(main())
