"""Time-stretch — swappable interface for sustained-segment warp.

``task_1.md`` §6 mandates a non-phase-vocoder time-stretch for sustained
content: "Never phase-vocode drums." The two approved options are
Rubber Band (GPL-v2+, quality default, requires the ``rubberband`` CLI
binary in ``PATH``) and pytsmod WSOLA (MIT, pure Python, lighter
install, good for small ratios).

Phase 3 ships ``WSOLAStretcher`` (pytsmod). ``RubberBandStretcher``
slot remains for Phase 3b — same interface, opt in by passing it to the
quantizer.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Stretcher(Protocol):
    """Time-stretch a mono or stereo segment by ``rate`` (output / input).

    Implementations MUST be deterministic: same input + rate → byte-
    identical output. Output length should be close to
    ``round(len(audio) * rate)`` but may be off by a few samples; the
    caller crops or zero-pads to the exact target.
    """

    def __call__(self, audio: np.ndarray, rate: float) -> np.ndarray: ...


class WSOLAStretcher:
    """WSOLA time-stretch via pytsmod (MIT). Pitch-preserving.

    For multi-channel input, each channel is stretched independently —
    pytsmod's API is mono. WSOLA's window-similarity criterion is
    deterministic, so per-channel runs produce repeatable results, but
    slight L/R phase drift can occur on tonal content. Acceptable for
    Phase 3; Rubber Band's coupled stereo mode is the future upgrade.
    """

    def __init__(self, *, win_size: int = 1024, syn_hop: int = 512) -> None:
        self.win_size = int(win_size)
        self.syn_hop = int(syn_hop)

    def __call__(self, audio: np.ndarray, rate: float) -> np.ndarray:
        try:
            import pytsmod  # noqa: WPS433 — lazy: optional dep
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "WSOLAStretcher requires the 'pytsmod' package; install "
                "with `pip install pytsmod` (MIT, see NOTICE.md)."
            ) from exc
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if audio.ndim == 1:
            return self._stretch_mono(pytsmod, audio.astype(np.float32, copy=False), rate)
        if audio.ndim != 2:
            raise ValueError(f"audio must be 1-D or 2-D, got shape {audio.shape}")
        # Stereo / multichannel: per-channel WSOLA.
        n_ch = audio.shape[1]
        cols = [
            self._stretch_mono(pytsmod, audio[:, c].astype(np.float32, copy=False), rate)
            for c in range(n_ch)
        ]
        # Channels may differ by 1–2 samples; pad to the longest.
        max_len = max(c.shape[0] for c in cols)
        out = np.zeros((max_len, n_ch), dtype=np.float32)
        for c, col in enumerate(cols):
            out[: col.shape[0], c] = col
        return out

    def _stretch_mono(self, pytsmod, mono: np.ndarray, rate: float) -> np.ndarray:
        if mono.size == 0:
            return mono
        # pytsmod.wsola(x, s): s is the duration multiplier (output / input).
        result = pytsmod.wsola(mono, rate, win_type="hann", win_size=self.win_size,
                               syn_hop_size=self.syn_hop, tolerance=512)
        return np.asarray(result, dtype=np.float32)


def _try_import_pytsmod() -> Optional["WSOLAStretcher"]:
    try:
        import pytsmod  # noqa: F401
    except ImportError:
        return None
    return WSOLAStretcher()


_default_stretcher: Optional[Stretcher] = _try_import_pytsmod()


def default_stretcher() -> Optional[Stretcher]:
    """Default sustained-segment stretcher, or ``None`` if no backend
    is installed (caller then falls back to linear-interp).
    """
    return _default_stretcher
