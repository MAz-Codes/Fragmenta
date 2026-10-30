"""Audio file I/O without torchaudio's backends (M1.2, D2).

Covers the shared decoder (vendored stable_audio_3/data/audio_io.py, used by
the app through app/core/audio/decode.py): every accepted format decodes,
including m4a/AAC via PyAV; early stop; writing; load_mono vs librosa.load;
the training scan picks up m4a; the dataset's bounded retries.

Run:  PYTHONPATH=vendor/stable-audio-3 python tests/test_audio_io.py
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vendor" / "stable-audio-3"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from app.core.audio.decode import AUDIO_EXTENSIONS, load_mono, read_audio  # noqa: E402
from stable_audio_3.data import audio_io  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


SR = 44100


def tone(seconds=3.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return np.stack([0.4 * np.sin(2 * np.pi * 440 * t), 0.4 * np.sin(2 * np.pi * 660 * t)]).astype(np.float32)


def write_with_av(path, data, sr, codec, fmt=None):
    import av
    with av.open(str(path), "w", format=fmt) as out:
        st = out.add_stream(codec, rate=sr, layout="stereo")
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(data), format="fltp", layout="stereo")
        frame.sample_rate = sr
        for packet in st.encode(frame):
            out.mux(packet)
        for packet in st.encode(None):
            out.mux(packet)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="fragmenta-audio-io-"))
    data = tone()

    print("=== the app and the vendored dataset share one list ===")
    check("same extensions in app and vendor", tuple(AUDIO_EXTENSIONS) == tuple(audio_io.AUDIO_EXTENSIONS))
    for ext in (".m4a", ".aac", ".aiff"):
        check(f"{ext} accepted", ext in AUDIO_EXTENSIONS)

    print("=== every format decodes ===")
    files = {}
    for ext, kw in ((".wav", {}), (".flac", {}), (".ogg", {"format": "OGG", "subtype": "VORBIS"}), (".aiff", {})):
        files[ext] = tmp / f"tone{ext}"
        sf.write(str(files[ext]), data.T, SR, **kw)
    files[".m4a"] = tmp / "tone.m4a"
    write_with_av(files[".m4a"], data, SR, "aac", "mp4")
    files[".aac"] = tmp / "tone.aac"
    write_with_av(files[".aac"], data, SR, "aac", "adts")
    files[".opus"] = tmp / "tone.opus"
    write_with_av(files[".opus"], tone(sr=48000), 48000, "libopus", "ogg")
    for ext, f in files.items():
        audio, sr = read_audio(f)
        secs = audio.shape[1] / sr
        ok = audio.dtype == torch.float32 and audio.shape[0] == 2 and abs(secs - 3.0) < 0.06 and 0.2 < float(audio.pow(2).mean().sqrt()) < 0.35
        check(f"{ext}: stereo float32, ~3 s, at level", ok, f"{sr} Hz, {secs:.3f} s")
        cut, _ = read_audio(f, duration=1.0)
        check(f"{ext}: duration=1 stops after 1 s", abs(cut.shape[1] / sr - 1.0) < 1e-3, f"{cut.shape[1] / sr:.4f} s")

    print("=== writing (what torchaudio.save did) ===")
    pcm = (torch.from_numpy(data) * 32767).to(torch.int16)
    audio_io.write_audio(tmp / "pcm.wav", pcm, SR)
    info = sf.info(str(tmp / "pcm.wav"))
    back, _ = read_audio(tmp / "pcm.wav")
    check("int16 → 16-bit PCM WAV at the right level", info.subtype == "PCM_16" and np.allclose(back.numpy(), data, atol=1e-3), info.subtype)
    audio_io.write_audio(tmp / "float.wav", torch.from_numpy(data), SR)
    check("float → 32-bit float WAV, exact", sf.info(str(tmp / "float.wav")).subtype == "FLOAT"
          and np.array_equal(read_audio(tmp / "float.wav")[0].numpy(), data))

    print("=== load_mono matches librosa.load ===")
    import librosa
    # librosa.load's own steps (it can't be called directly where audioread's
    # GStreamer backend is broken, which is part of why it's gone).
    raw, raw_sr = sf.read(str(files[".wav"]), dtype="float32", always_2d=True)
    ref = librosa.resample(librosa.to_mono(raw.T), orig_sr=raw_sr, target_sr=22050, res_type="soxr_hq")
    got, sr = load_mono(files[".wav"], 22050)
    check("same samples at 22.05 kHz", sr == 22050 and got.shape == ref.shape and np.allclose(got, ref, atol=1e-5),
          f"max diff {np.abs(got - ref).max():.2e}")
    m4a, sr = load_mono(files[".m4a"], 8000, duration=2.0)
    check("m4a → mono, resampled, cut", sr == 8000 and abs(len(m4a) / 8000 - 2.0) < 1e-3)

    print("=== duration probe + slicing (health check, Suggest, ingest) ===")
    from app.core.audio.decode import audio_duration
    # Raw ADTS .aac has no length field: FFmpeg estimates it from the bitrate.
    for ext, tol in ((".wav", 0.01), (".m4a", 0.1), (".aac", 0.3), (".opus", 0.01)):
        d = audio_duration(files[ext])
        check(f"{ext}: duration from the header", d is not None and abs(d - 3.0) < tol, f"{d}")
    check("unreadable file → None", audio_duration(tmp / "missing.wav") is None)
    from app.backend.data.slicing import SlicePlan, write_slices
    sliced = tmp / "sliced"
    sliced.mkdir()
    plans = [SlicePlan(index=1, start_sec=0.0, end_sec=1.5), SlicePlan(index=2, start_sec=1.5, end_sec=3.0)]
    for ext in (".wav", ".m4a"):
        out = write_slices(files[ext], plans, sliced, f"s{ext[1:]}")
        lens = [sf.info(str(p)).frames / SR for p in out]
        check(f"{ext} source slices into WAVs", len(out) == 2 and all(abs(x - 1.5) < 1e-3 for x in lens), str(lens))

    print("=== long files: read only the crop window ===")
    off, _ = audio_io.read_audio(files[".wav"], offset=1.0, duration=1.0)
    check("offset+duration reads the right slice", np.allclose(off.numpy(), data[:, SR:2 * SR], atol=1e-3))
    off_m4a, _ = audio_io.read_audio(files[".m4a"], offset=1.0, duration=1.0)
    check("offset works through PyAV too", abs(off_m4a.shape[1] - SR) < 2048, str(off_m4a.shape))
    from stable_audio_3.data.dataset import LocalDatasetConfig, SampleDataset
    longdir = tmp / "long"
    longdir.mkdir()
    sf.write(str(longdir / "long.wav"), tone(seconds=12.0).T, SR)
    ds_long = SampleDataset([LocalDatasetConfig(id="l", path=str(longdir))], sample_size=2 * SR, sample_rate=SR, volume_norm=True, pad=False)
    a, inf = ds_long[0]
    check("long file → one window, timing against the whole file",
          a.shape[-1] == 2 * SR and inf["seconds_total"] == 12 and 0 <= inf["seconds_start"] <= 10,
          f"{a.shape} {inf['seconds_start']}/{inf['seconds_total']}")

    print("=== training scan + bounded retries ===")
    from stable_audio_3.data.dataset import LocalDatasetConfig, SampleDataset, get_audio_filenames
    proj = tmp / "proj"
    proj.mkdir()
    for ext in (".wav", ".m4a", ".flac"):
        (proj / f"clip{ext}").write_bytes(files[ext].read_bytes())
    found = sorted(Path(f).suffix for f in get_audio_filenames(str(proj)))
    check("the training scan includes m4a", found == [".flac", ".m4a", ".wav"], str(found))
    ds = SampleDataset([LocalDatasetConfig(id="t", path=str(proj))], sample_size=SR, sample_rate=SR)
    audio, info = ds[0]
    check("a dataset item loads (m4a included)", audio.shape[-1] == SR and "path" in info)

    bad = tmp / "bad"
    bad.mkdir()
    for i in range(3):
        (bad / f"broken{i}.wav").write_bytes(b"not audio at all")
    ds_bad = SampleDataset([LocalDatasetConfig(id="b", path=str(bad))], sample_size=SR, sample_rate=SR)
    try:
        ds_bad[0]
        check("all-unreadable dataset stops with a clear error", False, "returned an item")
    except RuntimeError as e:
        check("all-unreadable dataset stops with a clear error", "No usable sample" in str(e), str(e)[:80])
    except RecursionError:
        check("all-unreadable dataset stops with a clear error", False, "recursed without limit")

    print()
    if fails:
        print(f"FAILED: {fails}")
        return 1
    print("All audio I/O checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
