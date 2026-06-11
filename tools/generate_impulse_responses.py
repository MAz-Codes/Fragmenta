#!/usr/bin/env python3
"""Generate Fragmenta's convolution-reverb impulse responses from scratch.

These three IRs are *synthesised* — early reflections plus a multiband
exponentially-decaying diffuse tail — so they are 100% original works with no
third-party licensing. They replace the previously-bundled Voxengo IRs (whose
"no charge for distribution" terms are incompatible with Fragmenta's AGPL-3.0
license). Output is written straight into the frontend's public IR folder.

Run:  python tools/generate_impulse_responses.py
Deterministic (fixed seeds) so re-running reproduces byte-identical files.

Output: app/frontend/public/ir/{hall,room,narrow}.wav
        (stereo, 44.1 kHz, 16-bit — matched to the Web Audio ConvolverNode,
        which normalizes the buffer, so absolute level only sets headroom.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt
import soundfile as sf

SR = 44_100
OUT_DIR = Path(__file__).resolve().parent.parent / "app" / "frontend" / "public" / "ir"

# ln(1000) ≈ 6.9078 — multiplier that makes exp(-K * t / rt60) reach -60 dB at t = rt60.
_K60 = 6.907_755


def _band_sos(kind: str, *cut):
    # Lowpass/highpass take a scalar cutoff; bandpass takes a [lo, hi] pair.
    wn = [c / (SR / 2) for c in cut]
    if len(wn) == 1:
        wn = wn[0]
    return butter(4, wn, btype=kind, output="sos")


def _diffuse_tail(n: int, rng: np.random.Generator,
                  rt60_low: float, rt60_mid: float, rt60_high: float) -> np.ndarray:
    """Decorrelated noise split into 3 bands, each decaying at its own RT60.

    Highs decay fastest (air absorption), lows ring longest — the natural
    'darkening' of a real room's tail.
    """
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)

    low = sosfilt(_band_sos("low", 500.0), noise) * np.exp(-_K60 * t / rt60_low)
    mid = sosfilt(_band_sos("band", 500.0, 4_000.0), noise) * np.exp(-_K60 * t / rt60_mid)
    high = sosfilt(_band_sos("high", 4_000.0), noise) * np.exp(-_K60 * t / rt60_high)
    return low + mid + high


def _early_reflections(n: int, taps_ms, gains, rng: np.random.Generator) -> np.ndarray:
    """Sparse discrete taps — the room's geometry/character before the tail."""
    out = np.zeros(n)
    for ms, g in zip(taps_ms, gains):
        i = int(ms * 1e-3 * SR)
        if 0 <= i < n:
            # tiny per-tap jitter so L/R taps aren't perfectly aligned (width)
            out[i] += g * (1.0 + 0.04 * rng.standard_normal())
    return out


def make_ir(*, length_s: float, predelay_ms: float,
            rt60_low: float, rt60_mid: float, rt60_high: float,
            taps_ms, tap_gains, er_level: float, seed: int) -> np.ndarray:
    """Build one stereo IR. L and R use independent noise → a wide, natural field."""
    n = int(length_s * SR)
    pre = int(predelay_ms * 1e-3 * SR)
    chans = []
    for ch in range(2):
        rng = np.random.default_rng(seed + ch)
        sig = np.zeros(n)

        # Early reflections start almost immediately.
        sig += er_level * _early_reflections(n, taps_ms, tap_gains, rng)

        # Diffuse tail begins after the pre-delay gap.
        tail = _diffuse_tail(n - pre, rng, rt60_low, rt60_mid, rt60_high)
        sig[pre:] += tail

        # Short raised-cosine fade-out so the truncated tail doesn't click.
        fade = min(int(0.030 * SR), n // 4)
        if fade > 0:
            sig[-fade:] *= np.cos(np.linspace(0, np.pi / 2, fade)) ** 2
        chans.append(sig)

    stereo = np.stack(chans, axis=1)
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo *= 0.89 / peak       # leave headroom; convolver renormalizes anyway
    return stereo.astype(np.float32)


# Three voices. Tap times are coprime-ish ms values so early reflections don't
# stack into a periodic 'metallic' comb.
SPECS = {
    # Big, lush, long — replaces "Scala Milan Opera Hall".
    "hall": dict(
        length_s=2.4, predelay_ms=28.0,
        rt60_low=2.6, rt60_mid=2.0, rt60_high=1.1,
        taps_ms=[11, 19, 29, 41, 53, 67, 83],
        tap_gains=[0.6, 0.5, 0.42, 0.34, 0.28, 0.22, 0.18],
        er_level=0.5, seed=1101,
    ),
    # Tight, punchy, brighter — replaces "Nice Drum Room".
    "room": dict(
        length_s=1.1, predelay_ms=9.0,
        rt60_low=0.8, rt60_mid=0.6, rt60_high=0.35,
        taps_ms=[5, 9, 14, 21, 31, 43],
        tap_gains=[0.8, 0.62, 0.48, 0.36, 0.27, 0.2],
        er_level=0.62, seed=2202,
    ),
    # Small + flutter-y ("bumpy") — replaces "Narrow Bumpy Space".
    "narrow": dict(
        length_s=0.9, predelay_ms=3.0,
        rt60_low=0.34, rt60_mid=0.28, rt60_high=0.16,
        taps_ms=[3, 7, 13, 17, 23, 27, 33, 39],   # dense slap/flutter
        tap_gains=[0.9, 0.78, 0.66, 0.56, 0.47, 0.39, 0.32, 0.26],
        er_level=0.72, seed=3303,
    ),
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in SPECS.items():
        ir = make_ir(**spec)
        path = OUT_DIR / f"{name}.wav"
        sf.write(str(path), ir, SR, subtype="PCM_16")
        dur = ir.shape[0] / SR
        print(f"wrote {path}  ({dur:.2f}s, stereo, {SR} Hz, 16-bit)")


if __name__ == "__main__":
    main()
