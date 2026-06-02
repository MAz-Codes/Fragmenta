"""Loop quantizer — replaces ``app.core.generation.audio_post_process``.

See ``task_1.md`` and ``AUDIT.md`` for the migration plan. This module is
under active development; **do not wire it into production paths yet**.
The legacy ``align_to_grid`` / ``align_for_loop`` path remains the active
runtime until acceptance.

Public API (Phase 1):
    quantize_to_loop(audio, bpm, bars, *, grid, time_sig, sample_rate, ...) -> ndarray
    quantize_batch(clips, bpm, bars, *, ...) -> list[ndarray]
    canonical_grid(bpm, bars, *, grid, time_sig, sample_rate) -> CanonicalGrid
"""

from .detectors import AubioDetector, EnergyFluxDetector, OnsetDetector, default_detector
from .grid import CanonicalGrid, canonical_grid, snap_to_grid
from .quantizer import quantize_batch, quantize_to_loop
from .refine import refine_to_transient

__all__ = [
    "AubioDetector",
    "CanonicalGrid",
    "EnergyFluxDetector",
    "OnsetDetector",
    "canonical_grid",
    "default_detector",
    "quantize_batch",
    "quantize_to_loop",
    "refine_to_transient",
    "snap_to_grid",
]
