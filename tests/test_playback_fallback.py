"""Bug-report fixes on the backend: the WAV playback fallback and missing MIDI.

Pure stdlib + the project venv; no pytest required. Run:
    venv/bin/python tests/test_playback_fallback.py
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

_TMP = Path(tempfile.mkdtemp(prefix="fragmenta-playback-"))
os.environ["FRAGMENTA_PROJECTS_DIR"] = str(_TMP / "projects")
os.environ["FRAGMENTA_UPLOADS_DIR"] = str(_TMP / "uploads")
os.environ.pop("FRAGMENTA_DOCKER", None)
(_TMP / "projects").mkdir(parents=True, exist_ok=True)
(_TMP / "uploads").mkdir(parents=True, exist_ok=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import app.backend.app as A  # noqa: E402
from app.core.audio import decode as D  # noqa: E402
from app.core.audio import midi_input as M  # noqa: E402


def make_clip(name="clip.flac", seconds=0.5, sr=44100):
    path = _TMP / name
    tone = 0.25 * np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, int(sr * seconds)))
    sf.write(str(path), np.stack([tone, tone], axis=1), sr)
    return path


def test_flac_converts_to_a_same_length_wav():
    src = make_clip()
    out = D.playable_wav(src)
    info = sf.info(str(out))
    assert out.suffix == ".wav" and info.format == "WAV" and info.subtype == "PCM_16"
    assert info.frames == sf.info(str(src)).frames and info.channels == 2


def test_conversion_is_cached_and_only_the_latest_stays():
    a, b = make_clip("a.flac"), make_clip("b.flac")
    first = D.playable_wav(a)
    stamp = first.stat().st_mtime_ns
    assert D.playable_wav(a).stat().st_mtime_ns == stamp, "reconverted a cached file"
    second = D.playable_wav(b)
    assert second != first and not first.exists() and second.exists()


def _route(path):
    session = mock.Mock()
    session.lock = mock.MagicMock()
    session.clips = {path.name: mock.Mock(path=str(path))}
    return mock.patch("app.backend.data.projects.get_session_handle", return_value=session)


def test_route_serves_wav_with_ranges_on_request():
    src = make_clip("route.flac")
    client = A.app.test_client()
    with _route(src):
        r = client.get(f"/api/projects/p/clip/{src.name}/audio?format=wav")
        assert r.status_code == 200 and r.mimetype == "audio/wav", (r.status_code, r.mimetype)
        ranged = client.get(f"/api/projects/p/clip/{src.name}/audio?format=wav", headers={"Range": "bytes=0-99"})
        assert ranged.status_code == 206 and len(ranged.data) == 100
        plain = client.get(f"/api/projects/p/clip/{src.name}/audio")
        assert plain.status_code == 200 and plain.data == src.read_bytes(), "the normal path changed"


def test_route_reports_undecodable_audio():
    bad = _TMP / "bad.flac"
    bad.write_bytes(b"not audio")
    with _route(bad):
        r = A.app.test_client().get(f"/api/projects/p/clip/{bad.name}/audio?format=wav")
    assert r.status_code == 415, r.status_code


class _FakeMidiIn:
    """rtmidi.MidiIn stand-in: `ports` are plugged in, `busy` won't open."""
    ports = ["Twister", "X-Touch Mini", "Busy"]
    busy = {"Busy"}

    def get_ports(self):
        return list(self.ports)

    def open_port(self, index):
        if self.ports[index] in self.busy:
            raise RuntimeError("in use")
        self.port = self.ports[index]

    def ignore_types(self, **_):
        pass

    def set_callback(self, fn, data=None):
        self.callback = (fn, data)

    def cancel_callback(self):
        pass

    def close_port(self):
        pass

    def delete(self):
        pass


def _fake_rtmidi():
    return mock.patch.multiple(M, rtmidi=mock.Mock(MidiIn=_FakeMidiIn), _RTMIDI_OK=True)


def test_several_midi_devices_open_at_once():
    client = A.app.test_client()
    with _fake_rtmidi():
        r = client.post("/api/midi/select", json={"port_ids": ["Twister", "X-Touch Mini"]})
        assert r.status_code == 200 and r.get_json()["open"] == ["Twister", "X-Touch Mini"], r.get_data()
        r = client.post("/api/midi/select", json={"port_ids": ["X-Touch Mini"]})
        assert r.get_json()["open"] == ["X-Touch Mini"], "the device switched off stayed open"
        client.post("/api/midi/select", json={"port_ids": []})
        assert M.current_ports() == []


def test_messages_carry_the_port_they_came_from():
    q = M.subscribe()
    try:
        with _fake_rtmidi():
            M.open_inputs(["Twister"])
            fn, port = M._inputs["Twister"].callback
            fn(([0xB1, 0, 127], 0.0), port)
            assert q.get_nowait() == {"data": [0xB1, 0, 127], "port": "Twister"}
            M.open_inputs([])
    finally:
        M.unsubscribe(q)


def test_missing_midi_device_is_not_an_error():
    client = A.app.test_client()
    with _fake_rtmidi():
        r = client.post("/api/midi/select", json={"port_ids": ["Gone", "Twister"]})
        client.post("/api/midi/select", json={"port_ids": []})
    assert r.status_code == 200, r.status_code
    assert r.get_json() == {"open": ["Twister"], "missing": ["Gone"], "failed": []}, r.get_data()


def test_a_device_that_wont_open_doesnt_stop_the_others():
    client = A.app.test_client()
    with _fake_rtmidi():
        r = client.post("/api/midi/select", json={"port_ids": ["Busy", "Twister"]})
        client.post("/api/midi/select", json={"port_ids": []})
    assert r.get_json() == {"open": ["Twister"], "missing": [], "failed": ["Busy"]}, r.get_data()


def test_old_single_port_requests_still_work():
    client = A.app.test_client()
    with _fake_rtmidi():
        r = client.post("/api/midi/select", json={"port_id": "Twister"})
        client.post("/api/midi/select", json={"port_ids": []})
    assert r.get_json()["open"] == ["Twister"], r.get_data()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
