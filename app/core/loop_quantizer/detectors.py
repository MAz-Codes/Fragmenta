"""Onset detection — swappable interface.

``task_1.md`` §2 mandates an external onset detector here: SuperFlux via
Essentia, madmom, or aubio. **librosa is explicitly excluded** from this
module ("Keep librosa only for the annotation tier, not here.").

Phase 1 (this commit) only ships ``EnergyFluxDetector``, a pure-numpy
placeholder good enough for synthetic test signals where transients are
sharp and well-separated. Phase 2 will add ``MadmomDetector`` (BSD,
RNN-based) and optionally ``EssentiaSuperFluxDetector`` /
``AubioDetector`` behind the same interface. Use ``EnergyFluxDetector``
for tests and CLI demos only — not for real audio.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class OnsetDetector(Protocol):
    """A pure function from (mono audio, sample_rate) to onset sample indices.

    Must be deterministic: same input → same output, every call. Output is a
    1-D int64 array in strictly increasing order.
    """

    def __call__(self, mono: np.ndarray, sample_rate: int) -> np.ndarray: ...


class EnergyFluxDetector:
    """Pure-numpy energy-flux onset detector. Phase 1 placeholder.

    Computes log-compressed frame energy, takes the rectified first-order
    difference, picks local maxima above ``threshold`` separated by at
    least ``min_gap_sec``. Suitable for synthetic test signals — NOT for
    real music. Replace with madmom / Essentia / aubio in Phase 2.
    """

    def __init__(
        self,
        *,
        hop: int = 128,
        win: int = 512,
        threshold: float = 0.30,
        min_gap_sec: float = 0.050,
    ) -> None:
        # Window/hop sizing math: a sliding-window energy detector reports
        # the START of the first frame that contains a transient, which is
        # up to `win` samples before the transient itself. Refinement is
        # bounded by `_V2_REFINE_WIN_SEC` (15 ms ≈ 661 samp at 44.1 kHz), so
        # we MUST keep `win <= refine_window_samples` or refine cannot
        # recover the true position. 512/128 stays inside that envelope
        # (worst-case anticipation 11.6 ms) and lands sample-accurate on
        # sharp clicks. This is a Phase 1 placeholder; Phase 2's spectral
        # flux / madmom detectors report close to the rising edge directly.
        self.hop = hop
        self.win = win
        self.threshold = threshold
        self.min_gap_sec = min_gap_sec

    def __call__(self, mono: np.ndarray, sample_rate: int) -> np.ndarray:
        if mono.ndim != 1:
            raise ValueError("EnergyFluxDetector expects mono input")
        n = mono.size
        if n < self.win:
            return np.zeros(0, dtype=np.int64)

        frames = np.lib.stride_tricks.sliding_window_view(mono, self.win)[:: self.hop]
        energy = (frames.astype(np.float64) ** 2).sum(axis=1)
        log_energy = np.log1p(energy * 1000.0)
        # prepend=0 so a transient inside frame 0 still registers as flux —
        # `prepend=log_energy[0]` would zero out frame 0 unconditionally and
        # silently miss a click at the head of the buffer.
        flux = np.diff(log_energy, prepend=0.0)
        np.maximum(flux, 0.0, out=flux)
        peak = flux.max()
        if peak <= 0.0:
            return np.zeros(0, dtype=np.int64)
        flux /= peak

        min_gap_frames = max(1, int(round(self.min_gap_sec * sample_rate / self.hop)))
        peaks: list[int] = []
        last = -min_gap_frames
        for i in range(1, flux.size - 1):
            if (
                flux[i] >= self.threshold
                and flux[i] >= flux[i - 1]
                and flux[i] >= flux[i + 1]
                and (i - last) >= min_gap_frames
            ):
                peaks.append(i * self.hop)
                last = i
        return np.asarray(peaks, dtype=np.int64)


_default_detector: OnsetDetector = EnergyFluxDetector()


def default_detector() -> OnsetDetector:
    """The detector used when no explicit detector is passed to the quantizer.

    Returns the Phase 1 placeholder. Phase 2 will switch this to a real
    detector (madmom-first) once it lands.
    """
    return _default_detector
