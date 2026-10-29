# Fragmenta: audio file I/O without torchaudio's backends.
#
# torchaudio 2.9 moved load/save out to TorchCodec. This reads with soundfile
# (libsndfile: WAV, FLAC, Ogg/Opus, MP3, AIFF) and falls back to PyAV (FFmpeg)
# for what libsndfile can't decode, e.g. m4a/AAC. Fragmenta's app code uses
# the same functions, so training and the app decode files identically.
import numpy as np
import soundfile as sf
import torch

# Every format Fragmenta accepts. Keep in step with the app's list
# (app/core/audio/decode.py).
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".aif", ".aiff", ".m4a", ".aac")


def read_audio(path, duration=None, offset=0.0):
    """Decode `path` to (float32 tensor [channels, samples], sample_rate).

    `offset` (seconds) starts later and `duration` (seconds) stops early, so a
    long file isn't decoded in full when only a window of it is needed.
    """
    try:
        with sf.SoundFile(str(path)) as f:
            sr = f.samplerate
            if offset:
                f.seek(int(offset * sr))
            frames = -1 if duration is None else int(duration * sr)
            data = f.read(frames=frames, dtype="float32", always_2d=True)   # [samples, channels]
        return torch.from_numpy(np.ascontiguousarray(data.T)), sr
    except (sf.LibsndfileError, RuntimeError):
        return _read_with_av(path, duration, offset)


def audio_length(path):
    """(frames, sample_rate) from the header, or (None, None)."""
    try:
        info = sf.info(str(path))
        return info.frames, info.samplerate
    except (sf.LibsndfileError, RuntimeError):
        pass
    try:
        import av
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None or not container.duration:
                return None, None
            sr = stream.codec_context.sample_rate
            return int(container.duration / 1_000_000 * sr), sr
    except Exception:
        return None, None


def _read_with_av(path, duration=None, offset=0.0):
    import av

    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError(f"No audio stream in {path}")
        sr = stream.codec_context.sample_rate
        # Planar float32 at the source rate and layout: frames come out as
        # [channels, samples] arrays.
        resampler = av.AudioResampler(format="fltp", layout=stream.codec_context.layout, rate=sr)
        skip = int(offset * sr)
        limit = None if duration is None else int(duration * sr) + skip
        chunks, total = [], 0
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
                total += chunks[-1].shape[1]
            if limit is not None and total >= limit:
                break
        else:
            chunks.extend(out.to_ndarray() for out in resampler.resample(None))
    if not chunks:
        return torch.zeros((stream.codec_context.channels or 1, 0)), sr
    data = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    if limit is not None:
        data = data[:, :limit]
    if skip:
        data = data[:, skip:]
    return torch.from_numpy(np.ascontiguousarray(data)), sr


def write_audio(path, audio, sample_rate):
    """Write a [channels, samples] tensor or array, as torchaudio.save did:
    int16 data as 16-bit PCM, anything else as 32-bit float."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio)
    if audio.dtype == np.int16:
        sf.write(str(path), audio.T, int(sample_rate), subtype="PCM_16")
    else:
        sf.write(str(path), audio.astype(np.float32).T, int(sample_rate), subtype="FLOAT")
