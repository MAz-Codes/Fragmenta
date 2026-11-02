"""The Performance page's state as files in the data folder, Import of it, and
the installed app's first launch bringing over a source install's window.

Pure stdlib + the project venv; no pytest, GPU or model needed. Run:
    venv/bin/python tests/test_performance_state.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_TMP = Path(tempfile.mkdtemp(prefix="fragmenta-perf-"))
os.environ["FRAGMENTA_PROJECTS_DIR"] = str(_TMP / "projects")
os.environ["FRAGMENTA_UPLOADS_DIR"] = str(_TMP / "uploads")
for name in ("FRAGMENTA_DOCKER", "FRAGMENTA_HF_SPACE", "SPACE_ID"):
    os.environ.pop(name, None)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.backend.app as A  # noqa: E402
import start  # noqa: E402
from app.backend import data_import as di  # noqa: E402
from app.backend import performance_state as ps  # noqa: E402

WAV = b"RIFF....WAVEfmt fake"


class Data:
    """A throwaway data folder for performance_state."""
    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "performance"
        self._patch = mock.patch.object(ps, "root", return_value=self.dir)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        self._tmp.cleanup()


def test_nothing_saved_reads_as_missing():
    with Data():
        assert ps.load() == {"session": None, "midi": None, "presets": None}


def test_state_and_presets_round_trip():
    with Data() as d:
        ps.save_state("session", {"bpm": 128})
        ps.save_state("midi", {"mappings": []})
        ps.save_preset("Live set", {"bpm": 90})
        ps.save_preset("a/b..c", {"bpm": 100})
        assert ps.load() == {"session": {"bpm": 128}, "midi": {"mappings": []},
                             "presets": {"Live set": {"bpm": 90}, "a/b..c": {"bpm": 100}}}
        assert sorted(p.name for p in (d.dir / "presets").iterdir()) == ["Live%20set.json", "a%2Fb%2E%2Ec.json"]
        mask = os.umask(0)
        os.umask(mask)
        assert (d.dir / "session.json").stat().st_mode & 0o777 == 0o666 & ~mask, "file not made like the rest"
        ps.delete_preset("Live set")
        ps.delete_state("session")
        assert ps.load() == {"session": None, "midi": {"mappings": []}, "presets": {"a/b..c": {"bpm": 100}}}


def test_names_cannot_leave_the_folder():
    with Data() as d:
        for bad in ("..", ".", "../x", "", "x" * 300):
            try:
                ps.put_fragment(bad, "1", WAV, "audio/wav")
            except ValueError:
                continue
            assert all(d.dir in p.resolve().parents for p in d.dir.rglob("*")), bad
        try:
            ps.save_state("../session", {})
            raise AssertionError("unknown state name accepted")
        except ValueError:
            pass


def test_fragments_put_get_copy_clear():
    with Data():
        ps.put_fragment("session-ch0", "1730_0", WAV, "audio/wav")
        ps.put_fragment("session-ch0", "1730_1", b"ID3mp3", "audio/mpeg")
        path = ps.get_fragment("session-ch0", "1730_0")
        assert path.read_bytes() == WAV and ps.content_type(path) == "audio/wav"
        assert ps.content_type(ps.get_fragment("session-ch0", "1730_1")) == "audio/mpeg"
        ps.put_fragment("session-ch0", "1730_1", WAV, "audio/wav")  # same id, new type
        assert ps.get_fragment("session-ch0", "1730_1").suffix == ".wav"
        assert len(list(ps.get_fragment("session-ch0", "1730_1").parent.iterdir())) == 2
        scope = ps.preset_scope("My set", 0)
        assert scope == "preset-My%20set-ch0"
        assert ps.copy_scope("session-ch0", scope) == 2
        assert ps.get_fragment(scope, "1730_0").read_bytes() == WAV
        ps.clear_scope("session-ch0")
        assert ps.get_fragment("session-ch0", "1730_0") is None
        assert ps.list_fragments() == [{"scope": scope, "id": "1730_0"}, {"scope": scope, "id": "1730_1"}]
        ps.delete_fragment(scope, "1730_0")
        assert ps.list_fragments() == [{"scope": scope, "id": "1730_1"}]


def test_routes():
    with Data():
        c = A.app.test_client()
        r = c.get("/api/performance/state").get_json()
        assert r == {"available": True, "session": None, "midi": None, "presets": None}, r
        assert c.put("/api/performance/state/session", json={"bpm": 99}).status_code == 200
        assert c.put("/api/performance/presets/My%20set", json={"bpm": 1}).status_code == 200
        assert c.put("/api/performance/state/other", json={}).status_code == 400
        r = c.get("/api/performance/state").get_json()
        assert r["session"] == {"bpm": 99} and r["presets"] == {"My set": {"bpm": 1}}
        url = "/api/performance/fragments/" + "preset-My%2520set-ch0" + "/1730_0"
        assert c.put(url, data=WAV, content_type="audio/wav").status_code == 200
        got = c.get(url)
        assert got.status_code == 200 and got.data == WAV and got.mimetype == "audio/wav"
        assert c.get(url + "x").status_code == 404
        assert c.get("/api/performance/fragments").get_json() == {
            "fragments": [{"scope": "preset-My%20set-ch0", "id": "1730_0"}]}
        r = c.post("/api/performance/fragments/preset-My%2520set-ch0/copy", json={"to": "session-ch1"})
        assert r.get_json() == {"copied": 1}
        assert c.delete("/api/performance/fragments/session-ch1").status_code == 200
        assert c.get("/api/performance/fragments/session-ch1/1730_0").status_code == 404
        assert c.delete("/api/performance/state/session").status_code == 200
        assert c.get("/api/performance/state").get_json()["session"] is None


def test_off_on_the_space():
    with Data(), mock.patch.dict(os.environ, {"FRAGMENTA_HF_SPACE": "1"}):
        c = A.app.test_client()
        assert c.get("/api/performance/state").get_json() == {"available": False}
        assert c.put("/api/performance/state/session", json={}).status_code == 404
        assert c.get("/api/performance/fragments").status_code == 404


# --- Settings > Import -------------------------------------------------------------

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())


class Import:
    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.old, self.here = base / "old", base / "here"
        old = self.old / "performance"
        write(old / "session.json", {"bpm": 140})
        write(old / "midi.json", {"mappings": [1]})
        write(old / "presets" / "Live%20set.json", {"bpm": 90})
        write(old / "presets" / "Ambient.json", {"bpm": 70})
        write(old / "fragments" / "session-ch0" / "1_0.wav", b"session audio")
        write(old / "fragments" / "session-ch2" / "2_0.wav", b"session audio 2")
        write(old / "fragments" / "preset-Live%2520set-ch1" / "3_0.wav", b"preset audio")
        write(old / "fragments" / "preset-Ambient-ch0" / "4_0.wav", b"ambient audio")
        paths = {"models_pretrained": self.here / "models/pretrained",
                 "models_fine_tuned": self.here / "models/fine_tuned",
                 "output": self.here / "output", "performance": self.here / "performance"}
        config = SimpleNamespace(get_path=paths.__getitem__, user_data_dir=self.here)
        self.here.mkdir()
        self.patches = [mock.patch.object(di, "get_config", return_value=config),
                        mock.patch.object(di, "get_projects_dir", return_value=self.here / "projects"),
                        mock.patch.object(ps, "root", return_value=self.here / "performance")]
        for p in self.patches:
            p.start()
        di._job.clear()
        di._job.update(state="idle")
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()
        self._tmp.cleanup()

    def item(self):
        body, code = di.scan(str(self.old))
        assert code == 200, body
        return next(i for i in body["items"] if i["id"] == "performance")

    def run(self):
        body, code = di.start(str(self.old), ["performance"], True, SimpleNamespace(list_jobs=lambda: []))
        assert code == 200, body
        while di.status()["state"] == "running":
            pass
        assert di.status()["state"] == "done", di.status()


def test_import_into_a_fresh_install_brings_everything():
    with Import() as e:
        assert e.item()["count"] == 4
        e.run()
        state = ps.load()
        assert state["midi"] == {"mappings": [1]}
        assert state["presets"] == {"Live set": {"bpm": 90}, "Ambient": {"bpm": 70},
                                    "Imported session": {"bpm": 140}}, state["presets"]
        assert state["session"] is None, "this install's session was touched"
        imported = ps.preset_scope("Imported session", 0)[:-1]
        assert ps.get_fragment(f"{imported}0", "1_0").read_bytes() == b"session audio"
        assert ps.get_fragment(f"{imported}2", "2_0").read_bytes() == b"session audio 2"
        assert ps.get_fragment("preset-Live%20set-ch1", "3_0").read_bytes() == b"preset audio"
        assert ps.get_fragment("preset-Ambient-ch0", "4_0").read_bytes() == b"ambient audio"
        assert (e.old / "performance" / "session.json").exists(), "kept the old folder"


def test_import_never_overwrites():
    with Import() as e:
        ps.save_state("session", {"bpm": 1})
        ps.save_state("midi", {"mappings": ["mine"]})
        ps.save_preset("Ambient", {"bpm": "mine"})
        ps.save_preset("Imported session", {"bpm": "earlier import"})
        ps.put_fragment("preset-Ambient-ch0", "mine", b"mine", "audio/wav")
        item = e.item()
        assert item["count"] == 2 and item["already"] == 2, item
        e.run()
        state = ps.load()
        assert state["session"] == {"bpm": 1} and state["midi"] == {"mappings": ["mine"]}
        assert state["presets"]["Ambient"] == {"bpm": "mine"}
        assert state["presets"]["Imported session"] == {"bpm": "earlier import"}
        assert state["presets"]["Imported session 2"] == {"bpm": 140}
        assert ps.get_fragment("preset-Ambient-ch0", "4_0") is None, "audio of a preset not imported came over"
        assert ps.get_fragment(ps.preset_scope("Imported session 2", 0), "1_0") is not None


# --- first launch of the installed app -----------------------------------------------

def test_first_launch_brings_over_the_source_window():
    with tempfile.TemporaryDirectory() as tmp:
        old, new = Path(tmp) / "old", Path(tmp) / "new"
        write(old / "EBWebView" / "Default" / "Local Storage" / "leveldb" / "000003.log", b"state")
        write(old / "EBWebView" / "Default" / "IndexedDB" / "x.indexeddb.blob" / "1", b"audio")
        write(old / "EBWebView" / "Default" / "Cache" / "data_0", b"cache")
        write(old / "EBWebView" / "Default" / "Code Cache" / "js" / "a", b"cache")
        write(old / "EBWebView" / "SingletonLock", b"")
        with mock.patch.dict(os.environ, {"FRAGMENTA_PACKAGED": "0"}):
            start.bring_over_window_storage(old, new)
        assert not new.exists(), "a source install copied"
        with mock.patch.dict(os.environ, {"FRAGMENTA_PACKAGED": "1"}), mock.patch.object(start.sys, "platform", "win32"):
            start.bring_over_window_storage(old, new)
            got = sorted(str(p.relative_to(new)) for p in new.rglob("*") if p.is_file())
            assert got == ["EBWebView/Default/IndexedDB/x.indexeddb.blob/1",
                           "EBWebView/Default/Local Storage/leveldb/000003.log"], got
            # Only while the app has none of its own.
            write(old / "EBWebView" / "Default" / "Local Storage" / "leveldb" / "000003.log", b"newer")
            start.bring_over_window_storage(old, new)
            assert (new / "EBWebView/Default/Local Storage/leveldb/000003.log").read_bytes() == b"state"
        with mock.patch.dict(os.environ, {"FRAGMENTA_PACKAGED": "1"}), mock.patch.object(start.sys, "platform", "darwin"):
            start.bring_over_window_storage(old, Path(tmp) / "mac")
            assert not (Path(tmp) / "mac").exists()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
