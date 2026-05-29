"""Shared test fixture generation.

The mini-project fixture (a handful of short audio clips + caption sidecars) is
generated on demand rather than committed, so the repo stays free of binary
*.wav blobs (which are globally gitignored anyway). Idempotent: if the clips
already exist it's a no-op.
"""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

SR = 44100

# (filename, caption, tone hz) — distinct simple tones so a2a/inpaint have
# something to chew on without shipping real audio.
_CLIPS = [
    ("kick.wav", "a deep techno kick drum, 120 bpm", 60.0),
    ("hat.wav", "a crisp closed hi-hat tick", 8000.0),
    ("pad.wav", "a warm analog synth pad in C minor", 220.0),
    ("bass.wav", "a rolling sub bass line, dark", 80.0),
]


def ensure_mini_project(root: Path, duration_sec: float = 1.5) -> Path:
    """Create `root` with 4 short stereo 44.1 kHz clips + `.txt` captions.

    Returns `root`. Safe to call repeatedly.
    """
    root.mkdir(parents=True, exist_ok=True)
    n = int(SR * duration_sec)
    for name, caption, freq in _CLIPS:
        wav_path = root / name
        if not wav_path.exists():
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(SR)
                frames = bytearray()
                for i in range(n):
                    env = min(1.0, i / 1000) * max(0.0, 1.0 - i / n)
                    s = int(12000 * env * math.sin(2 * math.pi * freq * i / SR))
                    frames += struct.pack("<hh", s, s)
                w.writeframes(bytes(frames))
        (root / name).with_suffix(".txt").write_text(caption + "\n")
    return root
