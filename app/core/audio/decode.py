"""Reading audio files, the same way everywhere.

One decoder for the whole app and the vendored training code: soundfile for
WAV/FLAC/Ogg/Opus/MP3/AIFF, PyAV (FFmpeg) for m4a/AAC — torchaudio's file
I/O is gone in torchaudio 2.9+, and librosa's audioread fallback goes in
librosa 1.0. The implementation lives in the vendored package
(stable_audio_3/data/audio_io.py) so training's dataset loader shares it; it
is imported on first use, so importing this module stays cheap (no torch).
"""

import sys
from pathlib import Path

# Every format Fragmenta accepts, everywhere (ingest, training, edit uploads).
# Keep in step with stable_audio_3/data/audio_io.py.
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".aif", ".aiff", ".m4a", ".aac")

_SA3_VENDOR = Path(__file__).resolve().parents[3] / "vendor" / "stable-audio-3"


def read_audio(path, duration=None):
    """(float32 tensor [channels, samples], sample_rate) — see audio_io."""
    if str(_SA3_VENDOR) not in sys.path:
        sys.path.insert(0, str(_SA3_VENDOR))
    from stable_audio_3.data.audio_io import read_audio as _read
    return _read(path, duration=duration)


def load_mono(path, sr=None, duration=None):
    """Decode to mono float32 numpy at `sr` (None keeps the file's rate).

    Drop-in for ``librosa.load(path, sr=sr, mono=True, duration=duration)``,
    including its default soxr_hq resampling. Returns (samples, sample_rate).
    """
    audio, in_sr = read_audio(path, duration=duration)
    y = audio.mean(dim=0).numpy()
    if sr and in_sr != sr:
        import librosa
        y = librosa.resample(y, orig_sr=in_sr, target_sr=sr, res_type="soxr_hq")
        return y, sr
    return y, in_sr


def audio_duration(path):
    """Length in seconds from the file header (no decode), or None."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return float(info.frames / info.samplerate) if info.samplerate > 0 else None
    except Exception:
        pass
    try:  # m4a/AAC: soundfile can't open these; ask FFmpeg's demuxer
        import av
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is not None and stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
            if container.duration:
                return float(container.duration / av.time_base)
    except Exception:
        pass
    return None
