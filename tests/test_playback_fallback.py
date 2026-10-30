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


def test_missing_midi_device_is_not_an_error():
    client = A.app.test_client()
    with mock.patch.object(M, "open_input", return_value=False), \
            mock.patch.object(M, "list_inputs", return_value=[]):
        r = client.post("/api/midi/select", json={"port_id": "Gone"})
    assert r.status_code == 200 and r.get_json() == {"current": None, "missing": True}, r.get_data()


def test_a_present_midi_device_that_fails_is_still_an_error():
    client = A.app.test_client()
    with mock.patch.object(M, "open_input", return_value=False), \
            mock.patch.object(M, "list_inputs", return_value=[{"id": "Busy", "name": "Busy", "index": 0}]):
        r = client.post("/api/midi/select", json={"port_id": "Busy"})
    assert r.status_code == 400, r.status_code


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")
