"""Loop quantizer — public entry points.

``quantize_to_loop`` aligns one clip to a canonical musical grid.
``quantize_batch`` aligns N clips at the same parameters with a single,
shared grid — that's the determinism guarantee that prevents flamming
when multiple aligned clips are layered.

This is the **Phase 1** implementation (see ``task_1.md`` and the module
README). It implements:

  * The §1 canonical grid (pure, deterministic).
  * The §3 sample-accurate refinement (port of ``_refine_to_transient``).
  * The §5 grid-assignment guards: snap, strict monotonicity, ratio clamp,
    boundary anchors.
  * The §6 transient-dominant slice-and-place path (whole-segment linear
    interpolation as filler stretch — proper attack-preserving warp lands
    in Phase 3).

What's NOT in Phase 1, deferred per the project plan:

  * Real onset detector (madmom / SuperFlux / aubio). Currently uses the
    pure-numpy ``EnergyFluxDetector`` placeholder, which works for
    synthetic test signals but should not be trusted on real music yet.
  * Segment classification (transient-dominant vs sustained).
  * Sustained-segment stretch (Rubber Band / pytsmod WSOLA).
  * Loop-wrap overhang fold-back.
  * Parallel ``quantize_batch`` execution (current impl is sequential —
    correctness first, parallelism is a Phase 4 optimization).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

from .classify import DEFAULT_FLATNESS_THRESHOLD, classify_segment
from .detectors import OnsetDetector, default_detector
from .grid import CanonicalGrid, canonical_grid
from .refine import refine_to_transient
from .stretch import Stretcher, default_stretcher


DEFAULT_RATIO_MIN = 0.80
DEFAULT_RATIO_MAX = 1.25
# ±25 ms refine window. v2 used ±30 ms; the Phase 1 EnergyFluxDetector
# works at ±15 ms because its frame error is small. Phase 2's AubioDetector
# (specflux) fires up to ~20 ms before the actual rising edge, so we open
# the window a bit. 25 ms still can't reach a neighbour at any musically
# plausible tempo (32nd notes at 240 BPM = 62.5 ms apart).
DEFAULT_REFINE_WINDOW_SEC = 0.025
# Sentinel: pass this to quantize_*` to opt out of WSOLA stretching even
# when pytsmod is installed (e.g. when measuring pure slice-and-place
# behaviour). Distinguishable from ``None`` which means "use default".
NO_STRETCHER: object = object()


def quantize_to_loop(
    audio: np.ndarray,
    bpm: float,
    bars: int,
    *,
    grid: int = 16,
    time_sig: Tuple[int, int] = (4, 4),
    sample_rate: int = 44100,
    detector: Optional[OnsetDetector] = None,
    stretcher: object = None,
    ratio_min: float = DEFAULT_RATIO_MIN,
    ratio_max: float = DEFAULT_RATIO_MAX,
    refine_window_sec: float = DEFAULT_REFINE_WINDOW_SEC,
    flatness_threshold: float = DEFAULT_FLATNESS_THRESHOLD,
) -> np.ndarray:
    """Quantize one clip to the canonical grid for ``(bpm, bars, grid, …)``.

    Returns a float32 array of shape ``(total_samples, channels)`` — or 1-D
    if the input was 1-D. ``total_samples`` is fixed by the canonical grid
    (see ``grid.canonical_grid``), independent of the input length.

    Pass ``stretcher=NO_STRETCHER`` to disable WSOLA and force linear-interp
    for all segments (Phase 1 behaviour, useful for measurement). ``None``
    resolves to ``default_stretcher()`` (WSOLA if pytsmod is installed).
    """
    cg = canonical_grid(
        bpm=bpm, bars=bars, grid=grid, time_sig=time_sig, sample_rate=sample_rate
    )
    return _quantize_one(
        audio,
        cg,
        detector=detector,
        stretcher=_resolve_stretcher(stretcher),
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        refine_window_sec=refine_window_sec,
        flatness_threshold=flatness_threshold,
    )


def quantize_batch(
    clips: Iterable[np.ndarray],
    bpm: float,
    bars: int,
    *,
    grid: int = 16,
    time_sig: Tuple[int, int] = (4, 4),
    sample_rate: int = 44100,
    detector: Optional[OnsetDetector] = None,
    stretcher: object = None,
    ratio_min: float = DEFAULT_RATIO_MIN,
    ratio_max: float = DEFAULT_RATIO_MAX,
    refine_window_sec: float = DEFAULT_REFINE_WINDOW_SEC,
    flatness_threshold: float = DEFAULT_FLATNESS_THRESHOLD,
) -> List[np.ndarray]:
    """Quantize N clips against a single shared canonical grid.

    The grid is computed ONCE — that's the property that lets independent
    clips post-quantization stack on top of each other without flamming.
    Each clip is processed independently; same params + same input bytes
    produce byte-identical output.
    """
    cg = canonical_grid(
        bpm=bpm, bars=bars, grid=grid, time_sig=time_sig, sample_rate=sample_rate
    )
    resolved_stretcher = _resolve_stretcher(stretcher)
    return [
        _quantize_one(
            clip,
            cg,
            detector=detector,
            stretcher=resolved_stretcher,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            refine_window_sec=refine_window_sec,
            flatness_threshold=flatness_threshold,
        )
        for clip in clips
    ]


def _resolve_stretcher(stretcher: object) -> Optional[Stretcher]:
    if stretcher is NO_STRETCHER:
        return None
    if stretcher is None:
        return default_stretcher()
    return stretcher  # type: ignore[return-value]


# --- internals -------------------------------------------------------------


def _quantize_one(
    audio: np.ndarray,
    cg: CanonicalGrid,
    *,
    detector: Optional[OnsetDetector],
    stretcher: Optional[Stretcher],
    ratio_min: float,
    ratio_max: float,
    refine_window_sec: float,
    flatness_threshold: float,
) -> np.ndarray:
    audio_2d, was_1d = _to_2d(audio)
    src_length = audio_2d.shape[0]
    if src_length == 0:
        out = np.zeros((cg.total_samples, audio_2d.shape[1]), dtype=np.float32)
        return out[:, 0] if was_1d else out

    mono = audio_2d.mean(axis=1) if audio_2d.shape[1] > 1 else audio_2d[:, 0]
    det = detector if detector is not None else default_detector()
    raw_onsets = np.asarray(det(mono, cg.sample_rate), dtype=np.int64)
    refined = _refine_all(mono, raw_onsets, cg.sample_rate, window_sec=refine_window_sec)

    # Head-trim so the first strong onset lands at sample 0. Without this
    # the (0, 0) boundary anchor pins SILENCE to the downbeat, leaving the
    # first musical event stranded inside an unanchored stretched segment.
    # Equivalent to the v2 path's `beats[0]`-to-zero step (AUDIT.md §9c).
    if refined.size > 0 and int(refined[0]) > 0:
        head = int(refined[0])
        audio_2d = audio_2d[head:]
        mono = mono[head:]
        refined = refined - head
        src_length = audio_2d.shape[0]
        if src_length == 0:
            out = np.zeros((cg.total_samples, audio_2d.shape[1]), dtype=np.float32)
            return out[:, 0] if was_1d else out

    anchors = _assign_anchors(
        refined,
        cg.grid_lines,
        src_length=src_length,
        total_samples=cg.total_samples,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
    )
    out = _warp_segments(
        audio_2d,
        anchors,
        cg.total_samples,
        stretcher=stretcher,
        flatness_threshold=flatness_threshold,
    )
    return out[:, 0] if was_1d else out


def _to_2d(audio: np.ndarray) -> Tuple[np.ndarray, bool]:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)[:, None], True
    if audio.ndim == 2:
        return np.ascontiguousarray(audio, dtype=np.float32), False
    raise ValueError(f"audio must be 1-D or 2-D, got shape {audio.shape}")


def _refine_all(
    mono: np.ndarray,
    onsets: np.ndarray,
    sample_rate: int,
    *,
    window_sec: float,
) -> np.ndarray:
    if onsets.size == 0:
        return onsets
    refined = np.fromiter(
        (
            refine_to_transient(mono, int(o), sample_rate, window_sec=window_sec)
            for o in onsets
        ),
        dtype=np.int64,
        count=onsets.size,
    )
    # Deduplicate after refinement (two coarse onsets can refine to the
    # same sample) and ensure strictly increasing order.
    refined = np.unique(refined)
    return refined


def _assign_anchors(
    refined_onsets: np.ndarray,
    grid_lines: np.ndarray,
    *,
    src_length: int,
    total_samples: int,
    ratio_min: float,
    ratio_max: float,
) -> List[Tuple[int, int]]:
    """Build ``[(src, dst), …]`` anchor pairs with the §5 guards applied.

    Always includes the two boundary anchors ``(0, 0)`` and
    ``(src_length, total_samples)``. Onsets that would force a per-segment
    stretch outside ``[ratio_min, ratio_max]`` are dropped — they ride
    along with their surrounding segment instead of warping it violently.
    Strict monotonicity is enforced by bumping collisions to the next free
    grid line (and dropping the onset if no free line remains).
    """
    anchors: List[Tuple[int, int]] = [(0, 0)]
    last_src, last_dst = 0, 0
    n_lines = grid_lines.size
    # Final endpoint is grid_lines[-1] = total_samples — reserve it for the
    # closing boundary anchor.
    last_assignable = n_lines - 2

    for src in refined_onsets:
        src = int(src)
        if src <= last_src or src >= src_length:
            continue
        # Snap to nearest grid line, then bump for monotonicity if needed.
        idx = int(np.searchsorted(grid_lines, src))
        left = max(0, idx - 1)
        right = min(n_lines - 1, idx)
        cand = left if (src - grid_lines[left]) <= (grid_lines[right] - src) else right
        while cand <= last_assignable and grid_lines[cand] <= last_dst:
            cand += 1
        if cand > last_assignable:
            break  # no free grid lines remain before the closing boundary
        dst = int(grid_lines[cand])

        src_seg = src - last_src
        dst_seg = dst - last_dst
        if src_seg <= 0 or dst_seg <= 0:
            continue
        ratio = dst_seg / src_seg
        if ratio < ratio_min or ratio > ratio_max:
            continue  # leave this onset un-anchored; neighbors interpolate

        anchors.append((src, dst))
        last_src, last_dst = src, dst

    anchors.append((src_length, total_samples))
    return anchors


def _warp_segments(
    audio_2d: np.ndarray,
    anchors: List[Tuple[int, int]],
    total_samples: int,
    *,
    stretcher: Optional[Stretcher],
    flatness_threshold: float,
) -> np.ndarray:
    """Route each inter-anchor segment to its appropriate warp.

    * Identity (src_len == dst_len): direct copy.
    * Transient-dominant: linear-interp filler stretch. Cheap, preserves
      attack at the anchor since both endpoints are pinned.
    * Sustained (low spectral flatness, ``stretcher`` available): WSOLA
      time-stretch. Preserves pitch.
    * Sustained but no stretcher: falls back to linear-interp (will pitch-
      shift but stays correct on timing — graceful degradation).

    Phase 4 will add equal-power crossfade at the seams where segments
    meet. For Phase 3 there is no overlap zone; segments are placed
    butt-jointed at their assigned grid lines.
    """
    n_channels = audio_2d.shape[1]
    src_n = audio_2d.shape[0]
    out = np.zeros((total_samples, n_channels), dtype=np.float32)

    for (src0, dst0), (src1, dst1) in zip(anchors, anchors[1:]):
        src0 = max(0, int(src0))
        src1 = min(src_n, int(src1))
        dst0 = max(0, int(dst0))
        dst1 = min(total_samples, int(dst1))
        src_len = src1 - src0
        dst_len = dst1 - dst0
        if src_len <= 0 or dst_len <= 0:
            continue

        segment = audio_2d[src0:src1]
        if src_len == dst_len:
            out[dst0 : dst0 + dst_len] = segment
            continue

        if stretcher is not None:
            mono = segment.mean(axis=1) if n_channels > 1 else segment[:, 0]
            klass = classify_segment(mono, threshold=flatness_threshold)
            if klass == "sustained":
                _place_stretched(out, segment, dst0, dst_len, src_len, stretcher)
                continue

        _place_linear_interp(out, segment, dst0, dst_len, src_len, n_channels)

    return out


def _place_linear_interp(
    out: np.ndarray,
    segment: np.ndarray,
    dst0: int,
    dst_len: int,
    src_len: int,
    n_channels: int,
) -> None:
    x_old = np.arange(src_len, dtype=np.float64)
    x_new = np.linspace(0.0, src_len - 1, dst_len, dtype=np.float64)
    for c in range(n_channels):
        out[dst0 : dst0 + dst_len, c] = np.interp(
            x_new, x_old, segment[:, c].astype(np.float64)
        ).astype(np.float32)


def _place_stretched(
    out: np.ndarray,
    segment: np.ndarray,
    dst0: int,
    dst_len: int,
    src_len: int,
    stretcher: Stretcher,
) -> None:
    rate = dst_len / src_len
    warped = stretcher(segment, rate)
    if warped.ndim == 1:
        warped = warped[:, None]
    n_w = warped.shape[0]
    if n_w >= dst_len:
        out[dst0 : dst0 + dst_len] = warped[:dst_len]
    else:
        out[dst0 : dst0 + n_w] = warped
