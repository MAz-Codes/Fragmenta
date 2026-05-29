#!/usr/bin/env python
"""SA3 generation smoke test (W7) — CPU, model-gated.

Loads `sa3-small-music` on CPU and exercises the three SA3 inference modes:
text-to-audio, audio-to-audio (init_audio), and inpainting (single +
multi-region). Asserts the output is a 44.1 kHz stereo WAV above the silence
floor, and that a2a/inpaint actually change the audio.

Self-skips (exit 0) when the model isn't downloaded, so it's safe in CI
without checkpoints. Run:

    FRAGMENTA_FORCE_DEVICE=cpu PYTHONPATH=vendor/stable-audio-3 \
        python tests/smoke_test_sa3.py
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vendor" / "stable-audio-3"))

# Device: honour FRAGMENTA_FORCE_DEVICE if the caller set it (CI sets cpu),
# else auto-detect (cuda/mps/cpu). NB: SA3 routes to flash-attn ops whenever
# flash_attn is importable — so forcing cpu on a box that *has* flash-attn
# installed will fail. CI runners have no flash-attn, so cpu is correct there.

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from app.core.config import get_config  # noqa: E402
from app.core.model_manager import ModelManager  # noqa: E402

MODEL = "sa3-small-music"
failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def rms(path):
    audio, sr = sf.read(str(path))
    return float(np.sqrt(np.mean(np.square(audio)))), sr, (audio.shape[1] if audio.ndim > 1 else 1)


def correlation(a_path, b_path):
    a, _ = sf.read(str(a_path)); b, _ = sf.read(str(b_path))
    a = a.mean(axis=1) if a.ndim > 1 else a
    b = b.mean(axis=1) if b.ndim > 1 else b
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def main():
    cfg = get_config()
    if not ModelManager(cfg).is_model_downloaded(MODEL):
        print(f"SKIP: {MODEL} not downloaded — run the Checkpoint Manager to enable this test.")
        return 0

    print(f"SA3 generation smoke test (model={MODEL}, device={os.environ.get('FRAGMENTA_FORCE_DEVICE', 'auto')})\n")
    from app.core.generation.audio_generator import AudioGenerator
    gen = AudioGenerator(cfg)

    # 1. Text-to-audio
    out = gen.generate_audio("a techno kick drum, 120 bpm", model_id=MODEL,
                             duration=4.0, steps=8, seed=42)
    r, sr, ch = rms(out)
    check("text-to-audio produces audio", out.exists())
    check("output is 44.1 kHz", sr == 44100, f"sr={sr}")
    check("output is stereo", ch == 2, f"channels={ch}")
    check("output is above silence floor", r > 1e-4, f"rms={r:.5f}")

    # 2. Audio-to-audio: high noise level so it's clearly different but related.
    a2a = gen.generate_audio("a distorted techno kick", model_id=MODEL,
                             duration=4.0, steps=8, seed=7,
                             init_audio_path=str(out), init_noise_level=0.85)
    corr = correlation(out, a2a)
    check("audio-to-audio produces audio", a2a.exists())
    check("a2a output differs from source", corr < 0.999, f"corr={corr:.4f}")

    # 3. Inpaint single region
    single = gen.generate_audio("a snare fill", model_id=MODEL,
                                duration=4.0, steps=8, seed=11,
                                inpaint_audio_path=str(out),
                                inpaint_starts=[1.0], inpaint_ends=[2.0])
    check("single-region inpaint produces audio", single.exists())

    # 4. Inpaint multiple regions
    multi = gen.generate_audio("scattered percussion hits", model_id=MODEL,
                               duration=4.0, steps=8, seed=13,
                               inpaint_audio_path=str(out),
                               inpaint_starts=[0.5, 2.5], inpaint_ends=[1.0, 3.0])
    check("multi-region inpaint produces audio", multi.exists())

    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("All SA3 generation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
