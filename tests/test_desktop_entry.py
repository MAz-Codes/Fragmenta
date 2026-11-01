"""The Linux window-icon entry survives the code moving, as it does at every
AppImage launch.

Pure stdlib + the project venv; no pytest required. Run:
    venv/bin/python tests/test_desktop_entry.py
"""
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import start  # noqa: E402


class Home:
    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.entry = self.dir / "applications" / "fragmenta.desktop"
        self.icon = self.dir / "icons" / "fragmenta.png"
        self.source = self.dir / "mount_a" / "icon.png"
        self.source.parent.mkdir()
        self.source.write_bytes(b"icon one")
        self._patches = [mock.patch.object(start, "DESKTOP_ENTRY_PATH", self.entry),
                         mock.patch.object(start, "ICON_COPY_PATH", self.icon),
                         mock.patch.object(start, "APP_ICON_PATH", self.source)]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()


def test_the_entry_points_at_a_copy_not_the_code():
    with Home() as h:
        start.ensure_desktop_entry()
        assert f"Icon={h.icon}\n" in h.entry.read_text()
        assert h.icon.read_bytes() == b"icon one"
        assert "mount_a" not in h.entry.read_text()


def test_the_entry_stays_the_same_when_the_code_moves():
    with Home() as h:
        start.ensure_desktop_entry()
        before = h.entry.read_text()
        moved = h.dir / "mount_b" / "icon.png"
        moved.parent.mkdir()
        moved.write_bytes(b"icon one")
        with mock.patch.object(start, "APP_ICON_PATH", moved):
            start.ensure_desktop_entry()
        assert h.entry.read_text() == before


def test_a_new_icon_replaces_the_copy():
    with Home() as h:
        start.ensure_desktop_entry()
        h.source.write_bytes(b"icon two")
        start.ensure_desktop_entry()
        assert h.icon.read_bytes() == b"icon two"


def test_an_entry_from_an_older_version_is_rewritten():
    with Home() as h:
        h.entry.parent.mkdir()
        h.entry.write_text("[Desktop Entry]\nIcon=/tmp/.mount_old/icon.png\n")
        start.ensure_desktop_entry()
        assert f"Icon={h.icon}\n" in h.entry.read_text()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
