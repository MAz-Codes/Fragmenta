"""Signed AI-output marking: what signing does to a file, and that it never
blocks an export.

No pytest, GPU, model or network needed. Run:
    venv/bin/python tests/test_content_credentials.py
"""
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import c2pa  # noqa: E402

from app.core.generation import content_credentials as cc  # noqa: E402
from app.core.generation import provenance as pv  # noqa: E402

UNTRUSTED = "signingCredential.untrusted"


class Env:
    """A scratch credentials folder."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.creds = self.dir / "credentials"
        config = mock.Mock()
        config.get_path.return_value = self.creds
        self._patch = mock.patch.object(cc, "get_config", return_value=config)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        self._tmp.cleanup()

    def wav(self, name="a.wav", tags=None):
        path = self.dir / name
        pcm = (np.random.default_rng(1).normal(size=(44100, 2)) * 3000).astype("int16")
        pv.write_tagged_wav(path, pcm, 44100, tags or pv.generation_tags("small-music", []))
        return path, pcm


def store(path):
    with open(path, "rb") as f, c2pa.Reader("audio/wav", f) as reader:
        return json.loads(reader.json())


def failures(path):
    results = store(path)["validation_results"]["activeManifest"]
    return {e["code"] for e in results.get("failure") or []}


def action(path):
    data = store(path)
    manifest = data["manifests"][data["active_manifest"]]
    actions = next(a for a in manifest["assertions"] if a["label"].startswith("c2pa.actions"))
    return actions["data"]["actions"][0]


def flip_byte(path, offset):
    raw = bytearray(path.read_bytes())
    raw[offset] ^= 1
    path.write_bytes(raw)


def test_signing_leaves_the_audio_and_tags_alone():
    with Env() as env:
        path, pcm = env.wav()
        assert cc.sign_wav(path, pv.generation_manifest("small-music"))
        got, sr = sf.read(path, dtype="int16")
        assert sr == 44100 and np.array_equal(got, pcm)
        with sf.SoundFile(path) as f:
            assert f.title == "AI-generated audio" and "digital_source_type=" in f.comment
        assert not list(env.dir.glob(".*signing"))


def test_a_signed_file_verifies_with_only_the_unknown_signer_noted():
    with Env() as env:
        path, _ = env.wav()
        cc.sign_wav(path, pv.generation_manifest("small-music"))
        assert store(path)["validation_state"] == "Valid"
        assert failures(path) == {UNTRUSTED}


def test_manifests_say_what_the_file_is():
    with Env() as env:
        path, _ = env.wav()
        adapter = "/x/my_lora/checkpoints/1000.safetensors"
        cc.sign_wav(path, pv.generation_manifest("small-music", [adapter]))
        got = action(path)
        assert got["digitalSourceType"] == pv.SOURCE_GENERATED
        assert got["parameters"] == {"model": "small-music", "adapters": ["my_lora/1000"]}

        rec, _ = env.wav("rec.wav", pv.recording_tags())
        cc.sign_wav(rec, pv.recording_manifest())
        assert action(rec)["digitalSourceType"] == pv.SOURCE_COMPOSITE


def test_changing_the_audio_or_tags_is_detected():
    with Env() as env:
        audio, _ = env.wav("audio.wav")
        cc.sign_wav(audio, pv.generation_manifest("small-music"))
        flip_byte(audio, audio.stat().st_size // 2)
        assert store(audio)["validation_state"] == "Invalid"
        assert "assertion.dataHash.mismatch" in failures(audio)

        tags, _ = env.wav("tags.wav")
        cc.sign_wav(tags, pv.generation_manifest("small-music"))
        flip_byte(tags, tags.read_bytes().find(b"INAM") + 9)
        assert store(tags)["validation_state"] == "Invalid"


def test_rewriting_a_signed_file_drops_the_record_rather_than_faking_it():
    # A plain re-save (any editor or DAW) leaves an unsigned file, not a valid one.
    with Env() as env:
        path, pcm = env.wav()
        cc.sign_wav(path, pv.generation_manifest("small-music"))
        sf.write(path, pcm, 44100, subtype="PCM_16")
        try:
            store(path)
        except c2pa.C2paError:
            return
        raise AssertionError("expected no manifest after a re-save")


def test_the_key_is_kept_private_and_reused():
    with Env() as env:
        first, _ = env.wav("1.wav")
        second, _ = env.wav("2.wav")
        cc.sign_wav(first, pv.generation_manifest("m"))
        key_file = env.creds / cc._KEY_FILE
        before = key_file.read_bytes()
        cc.sign_wav(second, pv.generation_manifest("m"))
        assert key_file.read_bytes() == before
        if os.name == "posix":
            assert stat.S_IMODE(key_file.stat().st_mode) == 0o600

        def serial(path):
            data = store(path)
            return data["manifests"][data["active_manifest"]]["signature_info"]["cert_serial_number"]
        assert serial(first) == serial(second)


def test_the_certificate_names_no_person_or_machine():
    chain, _ = cc._new_credentials()
    from cryptography import x509
    for cert in x509.load_pem_x509_certificates(chain):
        subject = cert.subject.rfc4514_string()
        assert "unverified" in subject
        for private in filter(None, (os.environ.get("USER"), os.environ.get("HOSTNAME"))):
            assert private not in subject


def test_a_damaged_or_mismatched_key_is_replaced():
    with Env() as env:
        path, _ = env.wav()
        cc._load_credentials()
        (env.creds / cc._KEY_FILE).write_bytes(b"not a key")
        assert cc.sign_wav(path, pv.generation_manifest("m"))

        # A key that doesn't belong to the certificate is just as unusable.
        _, other_key = cc._new_credentials()
        (env.creds / cc._KEY_FILE).write_bytes(other_key)
        second, _ = env.wav("2.wav")
        assert cc.sign_wav(second, pv.generation_manifest("m"))
        assert store(second)["validation_state"] == "Valid"


def test_a_certificate_near_expiry_is_replaced():
    with Env() as env:
        chain, key = cc._load_credentials()
        with mock.patch.object(cc, "_RENEW_BEFORE", cc.dt.timedelta(days=365 * 20)):
            assert not cc._usable(chain, key)
        assert cc._usable(chain, key)


def test_signing_asks_no_server_for_a_time_stamp():
    with Env() as env:
        path, _ = env.wav()
        with mock.patch.object(c2pa, "C2paSignerInfo", wraps=c2pa.C2paSignerInfo) as info:
            assert cc.sign_wav(path, pv.generation_manifest("m"))
        assert info.call_args.kwargs["ta_url"] is None
        data = store(path)
        assert "time" not in data["manifests"][data["active_manifest"]]["signature_info"]


def test_signing_failures_leave_the_tagged_file_untouched():
    with Env() as env:
        path, _ = env.wav()
        before = path.read_bytes()

        with mock.patch.dict(sys.modules, {"c2pa": None}):  # library not installed
            assert cc.sign_wav(path, pv.generation_manifest("m")) is False
        with mock.patch.object(cc, "_load_credentials", side_effect=OSError("read-only")):
            assert cc.sign_wav(path, pv.generation_manifest("m")) is False
        assert cc.sign_wav(path, {"assertions": "nonsense"}) is False

        assert path.read_bytes() == before
        assert not list(env.dir.glob(".*signing"))


def test_generator_and_recording_paths_sign_after_tagging():
    source = (ROOT / "app/core/generation/audio_generator.py").read_text()
    assert source.index("write_tagged_wav(out_path") < source.index("sign_wav(out_path")
    source = (ROOT / "app/backend/app.py").read_text()
    assert "if tag_existing_wav(dest, recording_tags()):\n        sign_wav(dest, recording_manifest())" in source


def test_verify_command_reports_each_case():
    with Env() as env:
        script = str(ROOT / "utils/verify_credentials.py")

        def run(*paths):
            return subprocess.run([sys.executable, script, *map(str, paths)],
                                  capture_output=True, text=True)

        intact, _ = env.wav("intact.wav")
        cc.sign_wav(intact, pv.generation_manifest("small-music"))
        result = run(intact)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "intact" in result.stdout and "trainedAlgorithmicMedia" in result.stdout
        assert "small-music" in result.stdout and "unverified" in result.stdout

        changed, _ = env.wav("changed.wav")
        cc.sign_wav(changed, pv.generation_manifest("m"))
        flip_byte(changed, changed.stat().st_size // 2)
        result = run(changed)
        assert result.returncode == 1 and "CHANGED" in result.stdout

        unsigned, _ = env.wav("unsigned.wav")
        result = run(unsigned)
        assert result.returncode == 2 and "no Content Credentials" in result.stdout

        assert run(changed, unsigned).returncode == 1
        assert run(unsigned, changed).returncode == 1


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
