"""Onset detection — swappable interface.

``task_1.md`` §2 mandates an external onset detector here: SuperFlux via
Essentia, madmom, or aubio. **librosa is explicitly excluded** from this
module ("Keep librosa only for the annotation tier, not here.").

Phase 2 ships ``AubioDetector`` (GPL-3, fast C bindings via the ``aubio``
package — AGPL-3-compatible). madmom is currently broken on Python 3.11
(its Cython 0.27 build step fails — long-standing upstream issue);
Essentia is install-heavy and deferred. SuperFlux specifically is not
exposed through ``aubio.onset()`` (it lives under ``aubio.specdesc`` and
needs its own peak-picker), so this wrapper defaults to ``specflux`` —
the spectral-flux algorithm SuperFlux extends. Good enough for real
audio; a follow-up can wire true SuperFlux via ``specdesc`` if needed.

``EnergyFluxDetector`` (pure numpy) remains as a no-dep fallback for
tests and for environments where aubio cannot be installed.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

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
        # Frame 0 must be a candidate: a click at sample 0 produces a real
        # flux peak there because we prepend 0 (silence before the buffer).
        # Skipping it silently drops transients at the buffer head.
        if flux.size >= 2 and flux[0] >= self.threshold and flux[0] >= flux[1]:
            peaks.append(0)
            last = 0
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


class AubioDetector:
    """Onset detector backed by aubio.

    aubio is GPL-3.0; compatible with this project's AGPL-3.0 license. The
    library wraps libaubio (C) — fast, allocation-light, deterministic.
    Configure attribution in ``NOTICE.md``.

    Default method is ``specflux`` (spectral flux). aubio's onset reporter
    fires at the FIRST frame whose flux exceeds the threshold, which for
    a sharp transient lands a few hops BEFORE the actual rising edge —
    typically 15–20 ms early. The quantizer's ``refine_to_transient``
    pass (default ±25 ms) brings each onset back to sample accuracy.

    Methods passed verbatim to ``aubio.onset(...)``: ``energy``, ``hfc``,
    ``complex``, ``phase``, ``specdiff``, ``kl``, ``mkl``, ``specflux``.
    True ``superflux`` is not currently reachable through aubio.onset.
    """

    def __init__(
        self,
        *,
        method: str = "specflux",
        buf_size: int = 1024,
        hop_size: int = 512,
        threshold: float = 0.30,
        min_ioi_ms: float = 30.0,
    ) -> None:
        self.method = method
        self.buf_size = int(buf_size)
        self.hop_size = int(hop_size)
        self.threshold = float(threshold)
        self.min_ioi_ms = float(min_ioi_ms)

    def __call__(self, mono: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            import aubio  # noqa: WPS433 — lazy: aubio is an optional dep
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "AubioDetector requires the 'aubio' package; install with "
                "`pip install aubio` (GPL-3, see NOTICE.md)."
            ) from exc
        if mono.ndim != 1:
            raise ValueError("AubioDetector expects mono input")
        if mono.dtype != np.float32:
            mono = mono.astype(np.float32)

        onset = aubio.onset(self.method, self.buf_size, self.hop_size, int(sample_rate))
        onset.set_threshold(self.threshold)
        onset.set_minioi_ms(self.min_ioi_ms)

        positions: list[int] = []
        n = mono.size
        hop = self.hop_size
        end = n - hop + 1
        for i in range(0, end, hop):
            frame = np.ascontiguousarray(mono[i : i + hop])
            if onset(frame):
                positions.append(int(onset.get_last()))
        return np.asarray(positions, dtype=np.int64)


def _try_import_aubio() -> Optional["AubioDetector"]:
    try:
        import aubio  # noqa: F401
    except ImportError:
        return None
    return AubioDetector()


_default_detector: OnsetDetector = _try_import_aubio() or EnergyFluxDetector()


def default_detector() -> OnsetDetector:
    """The detector used when no explicit detector is passed to the quantizer.

    Prefers ``AubioDetector`` when aubio is installed (the Phase 2
    production path). Falls back to ``EnergyFluxDetector`` — the
    pure-numpy placeholder — when aubio is unavailable.
    """
    return _default_detector
