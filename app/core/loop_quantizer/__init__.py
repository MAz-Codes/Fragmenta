"""Loop quantizer — replaces ``app.core.generation.audio_post_process``.

See ``task_1.md`` and ``AUDIT.md`` for the migration plan. This module
is wired into the runtime path **only** for Performance-tab Bars-mode
generations — gated behind ``FRAGMENTA_LOOP_QUANTIZER`` (default OFF).
When the flag is unset, the legacy / v2 ``align_to_grid`` /
``align_for_loop`` path executes (selectable in turn by
``FRAGMENTA_BEATSYNC_V2``). When the flag is set, the new module
replaces both legacy entries.

Scope (do NOT widen without confirming with the maintainer):
  * Performance tab → Bars mode + looping → ``loop_stitch="inpaint"``
    path in ``audio_generator._align_baseline_for_loop``.
  * Performance tab → Bars mode + non-looping → ``align_to_grid``
    call in ``app/backend/app.py``.

NOT scoped (these paths never set ``align_bars`` / ``align_bpm`` /
``loop_stitch``, so they cannot reach either alignment seam by
construction):
  * Generation tab (any mode).
  * Performance tab → Sec (seconds) mode.

Public API:
    quantize_to_loop(audio, bpm, bars, *, grid, time_sig, sample_rate, ...) -> ndarray
    quantize_batch(clips, bpm, bars, *, ...) -> list[ndarray]
    canonical_grid(bpm, bars, *, grid, time_sig, sample_rate) -> CanonicalGrid
    loop_quantizer_enabled() -> bool   # FRAGMENTA_LOOP_QUANTIZER flag
"""

import os

from .classify import SegmentClass, classify_segment, spectral_flatness
from .detectors import AubioDetector, EnergyFluxDetector, OnsetDetector, default_detector
from .grid import CanonicalGrid, canonical_grid, snap_to_grid
from .quantizer import NO_STRETCHER, quantize_batch, quantize_to_loop, quantize_wav_file
from .refine import refine_to_transient
from .stretch import Stretcher, WSOLAStretcher, default_stretcher


def loop_quantizer_enabled() -> bool:
    """Runtime gate for the Phase 5 integration.

    Default OFF: legacy / v2 path stays the active runtime until the new
    module is verified on real SA3 output. Set ``FRAGMENTA_LOOP_QUANTIZER=1``
    (or ``true`` / ``yes`` / ``on``) to route Performance Bars-mode
    alignment through this module. The flag has priority over
    ``FRAGMENTA_BEATSYNC_V2``; when both are set, the new module wins.
    """
    return os.environ.get("FRAGMENTA_LOOP_QUANTIZER", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


__all__ = [
    "AubioDetector",
    "CanonicalGrid",
    "EnergyFluxDetector",
    "NO_STRETCHER",
    "OnsetDetector",
    "SegmentClass",
    "Stretcher",
    "WSOLAStretcher",
    "canonical_grid",
    "classify_segment",
    "default_detector",
    "default_stretcher",
    "loop_quantizer_enabled",
    "quantize_batch",
    "quantize_to_loop",
    "quantize_wav_file",
    "refine_to_transient",
    "snap_to_grid",
    "spectral_flatness",
]
