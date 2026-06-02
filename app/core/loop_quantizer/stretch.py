"""Time-stretch — swappable interface for sustained-segment warp.

``task_1.md`` §6 mandates a non-phase-vocoder time-stretch for sustained
content: "Never phase-vocode drums." The two approved options are
Rubber Band (GPL-v2+, quality default, requires the ``rubberband`` CLI
binary in ``PATH``) and pytsmod WSOLA (MIT, pure Python, lighter
install, good for small ratios).

Phases 3 + 3b both ship: ``RubberBandStretcher`` is the default when
``pyrubberband`` is installed AND the ``rubberband`` CLI is on PATH,
otherwise ``WSOLAStretcher`` is used. Callers can still pass an
explicit instance to override.
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


class RubberBandStretcher:
    """Rubber Band time-stretch via ``pyrubberband`` + the ``rubberband``
    CLI (GPL-v2+, bundle the binary in desktop packagers).

    Higher quality than WSOLA on sustained tonal content — preserves
    formants and transients better, has true coupled-stereo phase
    handling. ``task_1.md`` §6 calls it out as the preferred option for
    drum-adjacent content where WSOLA's per-channel mono treatment can
    cause subtle L/R drift.

    pyrubberband shells out to the CLI on each call, so per-segment
    overhead is ~50-150 ms (process startup + WAV roundtrip). For the
    typical 4-bar Performance Bars loop with ~16 segments this adds
    ~1 s to alignment — acceptable for the quality gain on tonal
    content; not used at all on transient-classified segments.
    """

    def __init__(self, sample_rate: int = 44100, *, crispness: int = 5) -> None:
        # crispness 0-6: trade transient preservation vs. smoothness.
        # 5 (default) keeps transients sharp; 6 is most percussive,
        # 3-4 is smoother / better for pads. The classifier already
        # routes transient segments to linear-interp, so the segments
        # this stretcher sees ARE the sustained ones — crispness=5 is
        # a good general-purpose default.
        self.sample_rate = int(sample_rate)
        self.crispness = int(crispness)

    def __call__(self, audio: np.ndarray, rate: float) -> np.ndarray:
        try:
            import pyrubberband  # noqa: WPS433 — lazy: optional dep
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "RubberBandStretcher requires the 'pyrubberband' package "
                "and the 'rubberband' CLI on PATH. Install with "
                "`pip install pyrubberband` and "
                "`apt install rubberband-cli` (Linux) / "
                "`brew install rubberband` (macOS)."
            ) from exc
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if audio.size == 0:
            return audio.astype(np.float32, copy=False)
        if audio.ndim not in (1, 2):
            raise ValueError(f"audio must be 1-D or 2-D, got shape {audio.shape}")
        # pyrubberband.time_stretch takes mono or (N, C) and uses
        # `rate = duration_in / duration_out`, opposite of our `rate =
        # out/in` convention. So input_rate := 1.0 / our_rate.
        in_rate = 1.0 / rate
        rbargs = {"--crispness": str(self.crispness)}
        result = pyrubberband.time_stretch(
            audio.astype(np.float32, copy=False),
            self.sample_rate,
            in_rate,
            rbargs=rbargs,
        )
        return np.asarray(result, dtype=np.float32)


def _try_default_stretcher() -> Optional["Stretcher"]:
    """Pick the best stretcher available on this machine.

    Order: RubberBand (when ``pyrubberband`` and the CLI are present)
    → WSOLA via pytsmod → ``None`` (caller falls back to linear
    interpolation for sustained segments).

    RubberBand is the production default — user A/B-confirmed it sounds
    clearly better on tonal/sustained content (pads, synth bass,
    textures). The +2 ms diagnostic regression vs WSOLA is the metric
    counting RubberBand's better-preserved secondary transients as
    "off-grid"; the perceptual reality is the opposite. Override with
    ``FRAGMENTA_LOOP_QUANTIZER_NO_RUBBERBAND=1`` to force WSOLA.
    """
    import os
    import shutil

    no_rubberband = os.environ.get(
        "FRAGMENTA_LOOP_QUANTIZER_NO_RUBBERBAND", "0"
    ).strip().lower() in ("1", "true", "yes", "on")

    if not no_rubberband:
        try:
            import pyrubberband  # noqa: F401
            if shutil.which("rubberband") is not None:
                return RubberBandStretcher()
        except ImportError:
            pass

    try:
        import pytsmod  # noqa: F401
    except ImportError:
        return None
    return WSOLAStretcher()


_default_stretcher: Optional[Stretcher] = _try_default_stretcher()


def default_stretcher() -> Optional[Stretcher]:
    """Default sustained-segment stretcher, or ``None`` if no backend
    is installed (caller then falls back to linear-interp).
    """
    return _default_stretcher
