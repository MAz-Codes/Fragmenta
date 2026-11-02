"""Settings → About → Uninstall Fragmenta: the installed app removing itself.

Pure stdlib + the project venv; no pytest, GPU or model needed. Runs the real
delete script against a stand-in for the running app. Run:
    venv/bin/python tests/test_uninstall.py
"""
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.backend import uninstall  # noqa: E402

PACKAGED = {"FRAGMENTA_PACKAGED": "1", "FRAGMENTA_DOCKER": "0", "FRAGMENTA_HF_SPACE": "0", "SPACE_ID": ""}


def write(path, size=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


class Installed:
    """An unpacked Linux install, its data folder and a few desktop entries."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.app = tmp / "Fragmenta-2.0.0-Linux-x86_64"
        (self.app / "runtime" / "python").mkdir(parents=True)
        write(self.app / "payload" / "install.py")
        write(self.app / "AppRun")
        self.data = tmp / "share" / "FragmentaDesktop"
        cached = write(self.data / "cache" / "uv" / "torch.so", 1000)
        os.link(cached, self.data / "cache" / "uv" / "torch-link.so")
        (self.data / "venv" / "lib").mkdir(parents=True)
        os.link(cached, self.data / "venv" / "lib" / "torch.so")  # uv hard-links the venv into its cache
        write(self.data / "models" / "pretrained" / "model.safetensors", 300)
        write(self.data / "output" / "clip.wav", 40)
        write(self.data / "preflight.json", 2)
        self.entries = (write(tmp / "applications" / "fragmenta.desktop"), write(tmp / "icons" / "fragmenta.png"))
        config = SimpleNamespace(user_data_dir=self.data, project_root=self.app / "payload")
        self._patches = [mock.patch.object(uninstall, "get_config", return_value=config),
                         mock.patch.object(uninstall, "LINUX_ENTRIES", self.entries),
                         mock.patch.dict(os.environ, PACKAGED),
                         mock.patch.object(uninstall.sys, "platform", "linux")]
        os.environ.pop("APPIMAGE", None)
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()


def run_script(text, timeout=20):
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(text)
    proc = subprocess.run(["/bin/sh", f.name], timeout=timeout)
    return proc.returncode, Path(f.name)


def test_source_installs_and_deployments_cant_uninstall():
    with Installed():
        with mock.patch.dict(os.environ, {"FRAGMENTA_PACKAGED": "0"}):
            assert uninstall.plan() == {"available": False, "reason": "Only the installed app can uninstall itself."}
        with mock.patch.dict(os.environ, {"SPACE_ID": "x"}):
            assert not uninstall.plan()["available"]
            assert uninstall.start(keep_data=True, busy=None)[1] == 403


def test_nothing_happens_unless_it_looks_exactly_like_an_installed_app():
    with Installed() as i:
        (i.app / "AppRun").unlink()
        assert uninstall.plan()["reason"] == "Couldn't find the installed app to remove."
    with Installed() as i:
        with mock.patch.object(uninstall, "get_config", return_value=SimpleNamespace(
                user_data_dir=i.data.parent, project_root=i.app / "payload")):
            assert not uninstall.plan()["available"]


def test_the_appimage_itself_is_what_gets_removed():
    with Installed() as i:
        image = write(i.app.parent / "Fragmenta.AppImage")
        with mock.patch.dict(os.environ, {"APPIMAGE": str(image)}):
            assert uninstall._app() == image


def test_mac_and_windows_find_their_app():
    with Installed() as i:
        app = i.app.parent / "Fragmenta.app"
        bundle = app / "Contents" / "Resources" / "fragmenta"
        (bundle / "runtime" / "python").mkdir(parents=True)
        write(bundle / "payload" / "install.py")
        write(app / "Contents" / "Info.plist")
        config = SimpleNamespace(user_data_dir=i.data, project_root=bundle / "payload")
        with mock.patch.object(uninstall, "get_config", return_value=config), \
                mock.patch.object(uninstall.sys, "platform", "darwin"):
            assert uninstall._app() == app
        write(i.app / "fragmenta.exe")
        with mock.patch.object(uninstall.sys, "platform", "win32"):
            assert uninstall._app() is None  # no uninstaller beside it
            write(i.app / "unins000.exe")
            assert uninstall._app() == i.app


def test_sizes_count_hard_linked_packages_once():
    with Installed():
        result = uninstall.plan(sizes=True)
        assert result["available"] and result["packages_bytes"] == 1000 and result["yours_bytes"] == 342, result
        assert "packages_bytes" not in uninstall.plan()


def test_busy_app_refuses():
    with Installed():
        with mock.patch.object(uninstall.subprocess, "Popen") as popen:
            assert uninstall.start(keep_data=True, busy="A training run is going.") == (
                {"error": "A training run is going."}, 409)
        popen.assert_not_called()


def started_script(keep_data):
    with mock.patch.object(uninstall.subprocess, "Popen") as popen:
        body, code = uninstall.start(keep_data=keep_data, busy=None)
    assert code == 202 and body["keep_data"] is keep_data
    cmd = popen.call_args.args[0]
    assert cmd[0] == "/bin/sh" and popen.call_args.kwargs["start_new_session"]
    return Path(cmd[1])


def test_keeping_data_removes_only_the_packages_and_the_app():
    with Installed() as i:
        script = started_script(keep_data=True)
        text = script.read_text()
        script.unlink()
        # Waits for this process (the backend) before deleting anything.
        assert f"kill -0 {os.getpid()}" in text
        with mock.patch.object(uninstall.os, "getpid", return_value=999999), \
                mock.patch.object(uninstall.os, "getppid", return_value=999998):
            script = started_script(keep_data=True)
        code, _ = run_script(script.read_text())
        assert code == 0
        assert not (i.data / "venv").exists() and not (i.data / "cache").exists()
        assert (i.data / "models" / "pretrained" / "model.safetensors").exists() and (i.data / "output").exists()
        assert not i.app.exists() and not any(e.exists() for e in i.entries)


def test_removing_everything_takes_the_whole_data_folder():
    with Installed() as i:
        with mock.patch.object(uninstall.os, "getpid", return_value=999999), \
                mock.patch.object(uninstall.os, "getppid", return_value=999998):
            script = started_script(keep_data=False)
        code, ran = run_script(script.read_text())
        assert code == 0 and not i.data.exists() and i.data.parent.exists() and not i.app.exists()
        assert not ran.exists()  # the script removes itself


def test_script_waits_for_the_app_to_quit_first():
    with Installed() as i:
        app = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
        # Reaped as soon as it ends, as start.py and install.py reap the app's processes.
        threading.Thread(target=app.wait, daemon=True).start()
        started = time.monotonic()
        code, _ = run_script(uninstall._posix_script([i.data], i.app, [app.pid]))
        assert code == 0 and time.monotonic() - started >= 2 and not i.data.exists()


def test_paths_with_quotes_and_spaces_are_quoted():
    with Installed() as i:
        odd = i.data.parent / "it's here" / "FragmentaDesktop"
        write(odd / "venv" / "x")
        code, _ = run_script(uninstall._posix_script([odd / "venv"], i.app, []))
        assert code == 0 and not (odd / "venv").exists() and odd.exists()


def test_mac_moves_the_app_to_the_trash_without_finder():
    with Installed() as i, mock.patch.object(uninstall.sys, "platform", "darwin"):
        text = uninstall._posix_script([i.data], Path("/Applications/Frag's.app"), [1])
    assert "/usr/bin/osascript -l JavaScript" in text and "trashItemAtURL" in text and '"/Applications/Frag' in text
    assert "Finder" not in text and "rm -rf -- '/Applications" not in text


def test_windows_script_waits_deletes_then_runs_the_uninstaller():
    with Installed() as i:
        app = Path("C:/Users/O'Neil/AppData/Local/Programs/Fragmenta")
        text = uninstall._windows_script([i.data / "venv", i.data / "cache"], app)
    assert "O''Neil" in text  # PowerShell quoting
    assert text.index("Get-Process") < text.index("Remove-Item -LiteralPath $target") < text.index("unins*.exe")
    assert "'/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART'" in text and "$PSCommandPath" in text


def test_start_py_closes_the_window_only_for_an_uninstall():
    import start
    for code, closes in ((start.UNINSTALL_EXIT_CODE, True), (0, False), (1, False)):
        closed = mock.Mock()
        backend = subprocess.Popen([sys.executable, "-c", f"raise SystemExit({code})"])
        start._close_on_uninstall(backend, closed)
        backend.wait()
        time.sleep(0.3)
        assert closed.called is closes, code
    assert start.UNINSTALL_EXIT_CODE == uninstall.EXIT_CODE


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
