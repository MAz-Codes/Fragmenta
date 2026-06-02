"""Canonical musical grid for loop quantization.

A grid is fully determined by ``(bpm, bars, grid, time_sig, sample_rate)``.
This module is **pure** — same inputs produce byte-identical outputs across
runs and machines. That property is the foundation of the determinism
guarantee in ``quantize_batch``: the grid is computed ONCE and shared across
clips, so two clips processed at the same parameters land their anchored
transients on the exact same samples (the "no flamming" property).

The rounding convention is applied uniformly:

  total_samples       = round(bars * beats_per_bar * 60/bpm * sample_rate)
  grid_lines[i]       = round(i * total_samples / total_divisions)

Both endpoints are enforced: ``grid_lines[0] == 0`` and
``grid_lines[-1] == total_samples``. Beat positions are the subset of grid
lines at every ``grid/4`` step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CanonicalGrid:
    """Immutable description of one musical loop grid.

    ``grid_lines`` is monotonic, includes both endpoints, and has length
    ``total_divisions + 1``. ``beat_samples`` is a strict subset at every
    ``grid // 4`` step (assuming quarter-note beat).
    """

    bpm: float
    bars: int
    grid: int
    time_sig: Tuple[int, int]
    sample_rate: int
    total_samples: int
    grid_lines: np.ndarray
    beat_samples: np.ndarray
    metrical_levels: np.ndarray  # int32, same length as grid_lines.
    #   For each grid line: the COARSEST subdivision it belongs to.
    #   4 = quarter (downbeat / strong beat), 8 = eighth, 16 = sixteenth,
    #   32 = thirty-second. Used by hierarchical snap to prefer strong
    #   metrical positions when an onset is within tolerance of multiple
    #   levels.

    @property
    def total_divisions(self) -> int:
        return int(self.grid_lines.size - 1)

    @property
    def samples_per_division(self) -> float:
        return self.total_samples / self.total_divisions

    @property
    def samples_per_beat(self) -> float:
        return self.total_samples / (self.bars * self.time_sig[0])


def canonical_grid(
    bpm: float,
    bars: int,
    *,
    grid: int = 16,
    time_sig: Tuple[int, int] = (4, 4),
    sample_rate: int = 44100,
) -> CanonicalGrid:
    if bpm <= 0:
        raise ValueError(f"bpm must be positive, got {bpm}")
    if bars <= 0:
        raise ValueError(f"bars must be positive, got {bars}")
    if grid not in (4, 8, 16, 32):
        raise ValueError(
            f"grid must be one of (4, 8, 16, 32) — quarter / eighth / sixteenth / "
            f"thirty-second; got {grid}"
        )
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    beats_per_bar, beat_unit = time_sig
    if beats_per_bar <= 0 or beat_unit <= 0:
        raise ValueError(f"invalid time_sig {time_sig}")

    # Beat is the quarter note (60/bpm seconds). For 4/4, beats_per_bar
    # beats fit per bar; for 3/4, three; etc. The denominator only changes
    # which note value gets the beat, not the seconds-per-beat math.
    seconds_per_beat = 60.0 / float(bpm)
    total_beats = bars * beats_per_bar
    total_samples = int(round(total_beats * seconds_per_beat * sample_rate))

    # grid=16 → 4 divisions per beat (sixteenths); grid=8 → 2 (eighths).
    divisions_per_beat = grid // 4
    total_divisions = total_beats * divisions_per_beat

    # Single, uniform rounding: each grid line position is
    # round(i * total_samples / total_divisions). Endpoints are exact.
    indices = np.arange(total_divisions + 1, dtype=np.float64)
    grid_lines = np.rint(indices * (total_samples / total_divisions)).astype(np.int64)
    grid_lines[0] = 0
    grid_lines[-1] = total_samples

    beat_samples = grid_lines[::divisions_per_beat].copy()

    # Metrical level per line: the COARSEST subdivision the line belongs
    # to. Walk levels from coarsest to finest; each line's level is the
    # first one whose stride divides the line's index.
    metrical_levels = np.full(total_divisions + 1, grid, dtype=np.int32)
    for level in (4, 8, 16, 32):
        if level > grid:
            break
        stride = grid // level
        candidate_indices = np.arange(0, total_divisions + 1, stride)
        # Only overwrite if current level is finer (larger numeric value).
        mask = metrical_levels[candidate_indices] > level
        metrical_levels[candidate_indices[mask]] = level

    grid_lines.setflags(write=False)
    beat_samples.setflags(write=False)
    metrical_levels.setflags(write=False)

    return CanonicalGrid(
        bpm=float(bpm),
        bars=int(bars),
        grid=int(grid),
        time_sig=(int(beats_per_bar), int(beat_unit)),
        sample_rate=int(sample_rate),
        total_samples=int(total_samples),
        grid_lines=grid_lines,
        beat_samples=beat_samples,
        metrical_levels=metrical_levels,
    )


def snap_to_grid(positions: np.ndarray, grid: CanonicalGrid) -> np.ndarray:
    """Snap each position to its nearest grid line. Output is int64, same
    shape as input. Positions outside [0, total_samples] are clamped.
    """
    positions = np.asarray(positions, dtype=np.int64)
    lines = grid.grid_lines
    # np.searchsorted gives the insertion index; pick whichever of the two
    # neighbours is closer.
    idx = np.searchsorted(lines, positions)
    idx_left = np.clip(idx - 1, 0, lines.size - 1)
    idx_right = np.clip(idx, 0, lines.size - 1)
    left = lines[idx_left]
    right = lines[idx_right]
    pick_right = (positions - left) > (right - positions)
    return np.where(pick_right, right, left)
