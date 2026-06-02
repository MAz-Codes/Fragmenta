"""Sample-accurate onset refinement.

Onset detectors (madmom / Essentia / aubio / energy-flux placeholder) report
frame-coarse positions (typically ~10 ms hop). Phase coherence between
layers requires sample-accurate alignment, so each detected onset is
refined to its rising edge within a small window — usually ~15 ms — by
finding the steepest rise in the rectified amplitude envelope.

This is a pure-numpy port of ``_refine_to_transient`` from the legacy
``audio_post_process`` module (see AUDIT.md §9c).
"""

from __future__ import annotations

import numpy as np


def refine_to_transient(
    mono: np.ndarray,
    approx: int,
    sample_rate: int,
    *,
    window_sec: float = 0.015,
    rise_ratio: float = 0.15,
) -> int:
    """Snap an approximate onset position to its rising edge.

    Returns a refined sample index in ``[0, len(mono) - 1]``. Within a
    ``±window_sec`` window centered on ``approx``, finds the first sample
    whose amplitude crosses ``rise_ratio * window_peak``; if no such sample
    exists (very low contrast), returns the local amplitude peak instead.
    """
    if mono.ndim != 1:
        raise ValueError("refine_to_transient expects mono input")
    n = mono.size
    if n == 0:
        return 0

    radius = max(1, int(round(window_sec * sample_rate)))
    lo = max(0, int(approx) - radius)
    hi = min(n, int(approx) + radius + 1)
    if hi <= lo:
        return int(approx)

    window = np.abs(mono[lo:hi])
    peak = window.max()
    if peak <= 0.0:
        return int(approx)

    threshold = rise_ratio * peak
    crossings = np.where(window >= threshold)[0]
    if crossings.size > 0:
        return int(lo + crossings[0])
    return int(lo + np.argmax(window))
