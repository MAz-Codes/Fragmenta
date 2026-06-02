"""Per-segment classification: transient-dominant vs sustained.

This is the branching point that lets the quantizer handle drums AND
melodic/harmonic content correctly (task_1.md §4). The chosen warp for
each inter-onset segment depends on this label:

  * ``"transient"``  → slice-and-place with linear-interp filler.
                        Cheap, sample-exact at the anchor, smears
                        sustained tones if misclassified.
  * ``"sustained"``  → WSOLA / Rubber Band time-stretch.
                        Preserves pitch, more expensive, has phase
                        artefacts on sharp transients.

The metric is spectral flatness, the geometric-mean / arithmetic-mean
ratio of the magnitude spectrum. Flatness ≈ 1 → broadband / noisy
(transient, percussive). Flatness ≈ 0 → narrow / harmonic (sustained
tonal). Threshold ``0.30`` separates clear cases; borderline material
defaults to ``"transient"`` (the cheaper, sample-exact path).
"""

from __future__ import annotations

from typing import Literal

import numpy as np


SegmentClass = Literal["transient", "sustained"]

DEFAULT_FLATNESS_THRESHOLD = 0.30
DEFAULT_N_FFT = 2048
DEFAULT_HOP = 512


def spectral_flatness(
    mono: np.ndarray,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
) -> float:
    """Mean spectral flatness across the audio, in [0, 1].

    Returns 1.0 for white noise, ~0 for a pure tone. Empty / sub-frame
    input returns 1.0 (treated as "no harmonic content to preserve").
    """
    if mono.ndim != 1:
        raise ValueError("spectral_flatness expects mono input")
    if mono.size < n_fft:
        return 1.0

    window = np.hanning(n_fft).astype(np.float32)
    n_frames = (mono.size - n_fft) // hop + 1
    flat_vals = np.empty(n_frames, dtype=np.float64)
    eps = 1e-12

    for i in range(n_frames):
        start = i * hop
        frame = mono[start : start + n_fft] * window
        spec = np.abs(np.fft.rfft(frame))
        spec = np.maximum(spec, eps)
        geo = float(np.exp(np.mean(np.log(spec))))
        arith = float(np.mean(spec))
        flat_vals[i] = geo / arith if arith > 0 else 1.0

    return float(np.mean(flat_vals))


def classify_segment(
    mono: np.ndarray,
    *,
    threshold: float = DEFAULT_FLATNESS_THRESHOLD,
) -> SegmentClass:
    """Label one inter-onset segment as ``"transient"`` or ``"sustained"``.

    Short segments (below the FFT window) and silent segments classify
    as ``"transient"`` — the cheaper path that doesn't risk smearing
    something that isn't there.
    """
    if mono.size < DEFAULT_N_FFT:
        return "transient"
    peak = float(np.abs(mono).max())
    if peak < 1e-4:
        return "transient"
    return "sustained" if spectral_flatness(mono) < threshold else "transient"
