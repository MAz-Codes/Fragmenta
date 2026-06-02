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

import os
from concurrent.futures import ThreadPoolExecutor
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
# Loop-wrap crossfade: tail-head equal-power blend so the wrap from
# out[total-1] → out[0] doesn't click on content that wasn't generated to
# loop seamlessly. ~5 ms is enough to mask the discontinuity without
# noticeably smearing the loop boundary. 0 disables.
DEFAULT_LOOP_WRAP_CROSSFADE_MS = 5.0
# Hierarchical snap: when enabled, each onset prefers the COARSEST
# metrical level (quarter > eighth > sixteenth) within the tolerance
# below. Lets strong beats lock on the quarter even when the finest
# grid is 16th, instead of snapping to whichever 16th line happens to
# be closest. 30 ms is generous enough for human drumming variation
# while still discriminating between adjacent 8ths at 120 BPM (where
# the 8th period is 250 ms — 12% of one 8th).
DEFAULT_HIERARCHICAL_TOLERANCE_MS = 30.0
DEFAULT_HIERARCHY = (4, 8, 16)
# Global tempo conform: if the source's measured BPM differs from the
# target by more than the deadband, apply a single uniform WSOLA stretch
# BEFORE per-onset snapping. This is the v2 recipe — it gets every onset
# within tolerance of its target grid line so the snap pass actually has
# something to do, instead of dropping most onsets by the ratio clamp.
DEFAULT_TEMPO_CONFORM_DEADBAND = 0.005   # 0.5 % — below this, skip stretch
DEFAULT_TEMPO_CONFORM_MIN_RATIO = 0.70   # safe range for the uniform stretch:
DEFAULT_TEMPO_CONFORM_MAX_RATIO = 1.50   # rates outside are octave errors or
                                          # tempo-detector failures; bail.
# Beat-track mode: instead of detecting every transient and snapping each
# to a grid line (most get dropped by the ratio clamp because they're
# off-tempo), use aubio.tempo to find the periodic PULSE — that produces
# ~1 anchor per quarter note, all musically meaningful. The tracker has a
# ~1.5 s warmup (first 3 beats at 120 BPM are missed); we extrapolate the
# missing early beats backwards from (first_detected_beat - k*period_samples)
# so the loop downbeat gets anchored too. Beats are then snapped to the
# nearest QUARTER-note line (metrical_levels <= 4) with a wider tolerance
# than per-onset snap, since beats are spaced ~500 ms apart at 120 BPM.
DEFAULT_BEAT_TRACK_TOLERANCE_MS = 80.0
DEFAULT_BEAT_TRACK_RATIO_MIN = 0.75
DEFAULT_BEAT_TRACK_RATIO_MAX = 1.33
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
    loop_wrap_crossfade_ms: float = DEFAULT_LOOP_WRAP_CROSSFADE_MS,
    hierarchical: bool = False,
    hierarchy: Tuple[int, ...] = DEFAULT_HIERARCHY,
    hierarchical_tolerance_ms: float = DEFAULT_HIERARCHICAL_TOLERANCE_MS,
    tempo_only: bool = False,
    tempo_conform: bool = True,
    beat_track: bool = True,
    beat_track_tolerance_ms: float = DEFAULT_BEAT_TRACK_TOLERANCE_MS,
) -> np.ndarray:
    """Quantize one clip to the canonical grid for ``(bpm, bars, grid, …)``.

    Returns a float32 array of shape ``(total_samples, channels)`` — or 1-D
    if the input was 1-D. ``total_samples`` is fixed by the canonical grid
    (see ``grid.canonical_grid``), independent of the input length.

    Pass ``stretcher=NO_STRETCHER`` to disable WSOLA and force linear-interp
    for all segments (Phase 1 behaviour, useful for measurement). ``None``
    resolves to ``default_stretcher()`` (WSOLA if pytsmod is installed).
    Set ``loop_wrap_crossfade_ms=0`` to disable the tail-head crossfade
    (e.g., when the source is known to loop cleanly already).

    Set ``hierarchical=True`` to enable the coarsest-within-tolerance
    snap: each refined onset tries to lock to a quarter line first, then
    an eighth, then a sixteenth — only falling through when out of
    tolerance. Preserves metrical hierarchy (strong beats stay on
    quarters even when the finest grid is 16). Default tolerance is
    ±30 ms; tighten for stricter timing, loosen for laid-back grooves.

    Set ``tempo_only=True`` to skip per-onset snapping entirely — only
    the head-trim + boundary anchors run, so the loop is just stretched
    to fit the target length with the first transient at sample 0.
    This is the v2-style minimum behaviour; useful as a baseline.

    ``tempo_conform`` (default True) applies a single uniform WSOLA
    stretch BEFORE per-onset snapping when the source's measured BPM
    differs from the target. Without it, sources that drifted from the
    target tempo end up with most onsets dropped by the ratio clamp —
    the snap step has nothing to anchor and the loop falls back to
    boundary-stretch only. With it, every onset arrives close to its
    target grid line and the snap actually fires. Disable for the rare
    cases when source tempo is known to be exact or when you want to
    measure the snap step in isolation.
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
        loop_wrap_crossfade_ms=loop_wrap_crossfade_ms,
        hierarchical=hierarchical,
        hierarchy=hierarchy,
        hierarchical_tolerance_ms=hierarchical_tolerance_ms,
        tempo_only=tempo_only,
        tempo_conform=tempo_conform,
        beat_track=beat_track,
        beat_track_tolerance_ms=beat_track_tolerance_ms,
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
    loop_wrap_crossfade_ms: float = DEFAULT_LOOP_WRAP_CROSSFADE_MS,
    hierarchical: bool = False,
    hierarchy: Tuple[int, ...] = DEFAULT_HIERARCHY,
    hierarchical_tolerance_ms: float = DEFAULT_HIERARCHICAL_TOLERANCE_MS,
    tempo_only: bool = False,
    tempo_conform: bool = True,
    beat_track: bool = True,
    beat_track_tolerance_ms: float = DEFAULT_BEAT_TRACK_TOLERANCE_MS,
    workers: Optional[int] = None,
) -> List[np.ndarray]:
    """Quantize N clips against a single shared canonical grid.

    The grid is computed ONCE — that's the property that lets independent
    clips post-quantization stack on top of each other without flamming.
    Each clip is processed independently; same params + same input bytes
    produce byte-identical output, in the same order as ``clips``.

    ``workers`` controls parallelism: ``None`` (default) auto-picks
    ``min(len(clips), os.cpu_count())`` and runs a ``ThreadPoolExecutor``;
    pass ``1`` to force sequential. NumPy and aubio release the GIL
    during heavy work, so threading helps; pytsmod (pure Python) holds it,
    so the speedup is partial. Results are byte-identical regardless of
    worker count.
    """
    cg = canonical_grid(
        bpm=bpm, bars=bars, grid=grid, time_sig=time_sig, sample_rate=sample_rate
    )
    resolved_stretcher = _resolve_stretcher(stretcher)
    clip_list = list(clips)

    def _one(clip: np.ndarray) -> np.ndarray:
        return _quantize_one(
            clip,
            cg,
            detector=detector,
            stretcher=resolved_stretcher,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            refine_window_sec=refine_window_sec,
            flatness_threshold=flatness_threshold,
            loop_wrap_crossfade_ms=loop_wrap_crossfade_ms,
            hierarchical=hierarchical,
            hierarchy=hierarchy,
            hierarchical_tolerance_ms=hierarchical_tolerance_ms,
            tempo_only=tempo_only,
            tempo_conform=tempo_conform,
            beat_track=beat_track,
            beat_track_tolerance_ms=beat_track_tolerance_ms,
        )

    n_workers = workers if workers is not None else min(
        len(clip_list), os.cpu_count() or 1
    )
    if n_workers <= 1 or len(clip_list) < 2:
        return [_one(c) for c in clip_list]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(_one, clip_list))


def _resolve_stretcher(stretcher: object) -> Optional[Stretcher]:
    if stretcher is NO_STRETCHER:
        return None
    if stretcher is None:
        return default_stretcher()
    return stretcher  # type: ignore[return-value]


def quantize_wav_file(
    path,
    *,
    bpm: float,
    bars: int,
    grid: int = 16,
    time_sig: Tuple[int, int] = (4, 4),
    **kwargs,
):
    """Convenience: read a WAV, quantize, write it back at the same SR.

    Used by ``app/backend/app.py`` to replace the legacy
    ``align_to_grid()`` file-roundtrip behaviour without duplicating
    soundfile I/O at the call site. In-memory callers should use
    ``quantize_to_loop`` directly to avoid disk hops.
    """
    import soundfile as sf  # lazy — only this entry needs the import

    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.astype(np.float32, copy=False)
    out = quantize_to_loop(
        audio,
        bpm=bpm,
        bars=bars,
        grid=grid,
        time_sig=time_sig,
        sample_rate=sr,
        **kwargs,
    )
    sf.write(str(path), out, sr, subtype="PCM_16")
    return path


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
    loop_wrap_crossfade_ms: float,
    hierarchical: bool = False,
    hierarchy: Tuple[int, ...] = DEFAULT_HIERARCHY,
    hierarchical_tolerance_ms: float = DEFAULT_HIERARCHICAL_TOLERANCE_MS,
    tempo_only: bool = False,
    tempo_conform: bool = True,
    beat_track: bool = True,
    beat_track_tolerance_ms: float = DEFAULT_BEAT_TRACK_TOLERANCE_MS,
) -> np.ndarray:
    audio_2d, was_1d = _to_2d(audio)
    src_length = audio_2d.shape[0]
    if src_length == 0:
        out = np.zeros((cg.total_samples, audio_2d.shape[1]), dtype=np.float32)
        return out[:, 0] if was_1d else out

    mono = audio_2d.mean(axis=1) if audio_2d.shape[1] > 1 else audio_2d[:, 0]
    det = detector if detector is not None else default_detector()

    # Global tempo conform pre-pass — see v2 recipe. Estimates source BPM
    # via aubio.tempo (a periodic-pulse tracker, not an onset detector),
    # then applies a single uniform WSOLA stretch so the snap step has
    # something to anchor. Without this most onsets fall outside the
    # ratio clamp and the snap silently degenerates to "stretch only".
    if tempo_conform and stretcher is not None:
        source_bpm = _estimate_source_bpm(mono, cg.sample_rate)
        if source_bpm is not None and source_bpm > 0:
            rate = float(source_bpm) / float(cg.bpm)
            if (
                DEFAULT_TEMPO_CONFORM_MIN_RATIO <= rate <= DEFAULT_TEMPO_CONFORM_MAX_RATIO
                and abs(rate - 1.0) > DEFAULT_TEMPO_CONFORM_DEADBAND
            ):
                stretched = stretcher(audio_2d, rate)
                if stretched.ndim == 1:
                    stretched = stretched[:, None]
                audio_2d = stretched.astype(np.float32, copy=False)
                src_length = audio_2d.shape[0]
                mono = (
                    audio_2d.mean(axis=1)
                    if audio_2d.shape[1] > 1
                    else audio_2d[:, 0]
                )

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

    if beat_track:
        # Use aubio.tempo to find PULSE positions instead of every transient.
        # The tracker locks onto the periodic beat (~1 anchor per quarter
        # note) and produces musically meaningful pulses, not just loud
        # events. Missing early beats (warmup gap) get extrapolated
        # backwards from (first_beat - k * period_samples). aubio.tempo
        # reports beat times with a phase lag (~5–20 ms behind the actual
        # rising edge), so we refine each beat to its local energy peak —
        # same routine used for onsets — before snapping to the grid.
        beats = _extract_beats(
            mono, cg.sample_rate, expected_bpm=cg.bpm, bpm_tolerance=0.10,
        )
        if beats.size > 0:
            beats = _refine_all(
                mono, beats, cg.sample_rate, window_sec=refine_window_sec
            )
            tolerance_samp = int(
                round(beat_track_tolerance_ms * 0.001 * cg.sample_rate)
            )
            anchors = _assign_anchors_beats(
                beats,
                cg.grid_lines,
                cg.metrical_levels,
                src_length=src_length,
                total_samples=cg.total_samples,
                ratio_min=DEFAULT_BEAT_TRACK_RATIO_MIN,
                ratio_max=DEFAULT_BEAT_TRACK_RATIO_MAX,
                tolerance_samples=tolerance_samp,
            )
        else:
            # aubio.tempo failed the BPM sanity check (locked onto wrong
            # subdivision, octave-error, or low confidence). Fall back to
            # hierarchical onset snap using the already-computed ``refined``
            # array — safer than committing to a confidently-wrong pulse.
            tolerance_samp = int(
                round(hierarchical_tolerance_ms * 0.001 * cg.sample_rate)
            )
            anchors = _assign_anchors_hierarchical(
                refined,
                cg.grid_lines,
                cg.metrical_levels,
                src_length=src_length,
                total_samples=cg.total_samples,
                ratio_min=ratio_min,
                ratio_max=ratio_max,
                tolerance_samples=tolerance_samp,
                hierarchy=hierarchy,
            )
    elif tempo_only:
        # Boundary anchors only — head-trim already happened above, so
        # the first transient is at sample 0; the closing anchor crops/
        # stretches to exact target_samples. v2-style minimum behaviour.
        anchors = [(0, 0), (src_length, cg.total_samples)]
    elif hierarchical:
        tolerance_samp = int(round(hierarchical_tolerance_ms * 0.001 * cg.sample_rate))
        anchors = _assign_anchors_hierarchical(
            refined,
            cg.grid_lines,
            cg.metrical_levels,
            src_length=src_length,
            total_samples=cg.total_samples,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            tolerance_samples=tolerance_samp,
            hierarchy=hierarchy,
        )
    else:
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
    if loop_wrap_crossfade_ms > 0:
        fade_samples = int(round(loop_wrap_crossfade_ms * 0.001 * cg.sample_rate))
        _loop_wrap_crossfade(out, fade_samples)
    return out[:, 0] if was_1d else out


def _extract_beats(
    mono: np.ndarray,
    sample_rate: int,
    *,
    expected_bpm: Optional[float] = None,
    bpm_tolerance: float = 0.10,
    min_confidence: float = 0.5,
) -> np.ndarray:
    """Detect beat positions (in samples) via aubio's periodic-pulse tracker.

    Unlike onset detection, ``aubio.tempo`` infers the underlying pulse and
    only fires at musically meaningful beat positions — typically one per
    quarter note. ``get_last()`` returns the causal sample index of each
    detected beat with ~5 ms accuracy on clean material. The tracker has
    a ~1.5 s warmup, so the first few beats of a 120 BPM loop are missed;
    this routine extrapolates them backwards from ``(first_beat - k*period)``
    so the loop downbeat gets an anchor too.

    ``expected_bpm`` enables a sanity check: if the tracker's BPM
    estimate disagrees with the target by more than ``bpm_tolerance``
    (default ±10 %), the result is rejected (returns empty). On dense or
    polyrhythmic content aubio.tempo can lock onto a wrong subdivision
    (e.g. 155 BPM on a 120 BPM kick pattern), and using those beats as
    anchors makes timing worse. Letting the caller fall back to onset
    detection is safer than committing to a confidently-wrong pulse.

    Returns an empty array if aubio isn't installed, the audio is too
    short, no beats detected, or the detected BPM fails the sanity check.
    """
    try:
        import aubio  # noqa: WPS433 — optional dep
    except ImportError:  # pragma: no cover
        return np.empty(0, dtype=np.int64)
    if mono.size < 4096:
        return np.empty(0, dtype=np.int64)
    try:
        hop = 512
        win = 1024
        tempo = aubio.tempo("default", win, hop, int(sample_rate))
        mono32 = np.ascontiguousarray(mono, dtype=np.float32)
        end = mono32.size - hop + 1
        beats: List[int] = []
        confidence_sum = 0.0
        confidence_n = 0
        for i in range(0, end, hop):
            if tempo(mono32[i : i + hop]):
                beats.append(int(tempo.get_last()))
                confidence_sum += float(tempo.get_confidence())
                confidence_n += 1
        bpm = float(tempo.get_bpm())
    except Exception:  # pragma: no cover
        return np.empty(0, dtype=np.int64)
    if not beats or bpm <= 0:
        return np.empty(0, dtype=np.int64)
    if expected_bpm is not None and expected_bpm > 0:
        rel_err = abs(bpm - expected_bpm) / expected_bpm
        if rel_err > bpm_tolerance:
            return np.empty(0, dtype=np.int64)
    mean_conf = confidence_sum / max(1, confidence_n)
    if mean_conf < min_confidence:
        return np.empty(0, dtype=np.int64)

    # Extrapolate backwards from beats[0] to cover the warmup gap.
    period_samples = int(round(60.0 / bpm * sample_rate))
    first = beats[0]
    prefix: List[int] = []
    k = 1
    while True:
        cand = first - k * period_samples
        if cand <= 0:
            # Include sample 0 if reasonably close to a beat phase.
            if cand > -period_samples // 4:
                prefix.append(0)
            break
        prefix.append(cand)
        k += 1
    prefix.reverse()
    all_beats = np.asarray(prefix + beats, dtype=np.int64)
    return np.unique(all_beats)


def _assign_anchors_beats(
    beats: np.ndarray,
    grid_lines: np.ndarray,
    metrical_levels: np.ndarray,
    *,
    src_length: int,
    total_samples: int,
    ratio_min: float,
    ratio_max: float,
    tolerance_samples: int,
) -> List[Tuple[int, int]]:
    """Snap each detected beat to its nearest QUARTER-note grid line.

    Beats are inherently pulse events (1 per quarter note). Eligible
    destinations are grid lines whose metrical level is ≤ 4 (quarters and
    coarser). Same §5 monotonicity + ratio-clamp + boundary anchors as
    the onset paths, but with a wider tolerance (default ±80 ms) since
    beats are ~500 ms apart at 120 BPM and a beat tracker's phase error
    is typically larger than an onset detector's frame error.
    """
    anchors: List[Tuple[int, int]] = [(0, 0)]
    last_src, last_dst = 0, 0
    n_lines = grid_lines.size
    last_assignable = n_lines - 2

    eligible = np.where(metrical_levels <= 4)[0]
    if eligible.size == 0 or beats.size == 0:
        anchors.append((src_length, total_samples))
        return anchors
    eligible_samples = grid_lines[eligible]

    for src in beats:
        src = int(src)
        if src <= last_src or src >= src_length:
            continue
        j = int(np.argmin(np.abs(eligible_samples - src)))
        line_idx = int(eligible[j])
        if abs(int(grid_lines[line_idx]) - src) > tolerance_samples:
            continue
        # Monotonicity: bump to next free QUARTER line on collision.
        while line_idx <= last_assignable and int(grid_lines[line_idx]) <= last_dst:
            # advance to next eligible (quarter-or-coarser) line
            next_eligible = eligible[eligible > line_idx]
            if next_eligible.size == 0:
                line_idx = last_assignable + 1
                break
            line_idx = int(next_eligible[0])
        if line_idx > last_assignable:
            break
        dst = int(grid_lines[line_idx])

        src_seg = src - last_src
        dst_seg = dst - last_dst
        if src_seg <= 0 or dst_seg <= 0:
            continue
        ratio = dst_seg / src_seg
        if ratio < ratio_min or ratio > ratio_max:
            continue

        anchors.append((src, dst))
        last_src, last_dst = src, dst

    anchors.append((src_length, total_samples))
    return anchors


def _estimate_source_bpm(mono: np.ndarray, sample_rate: int) -> Optional[float]:
    """Estimate the audio's tempo via aubio's periodic-pulse tracker.

    Returns ``None`` if aubio isn't importable, the audio is too short
    for the tracker, or no stable tempo is detected. Unlike onset
    detection (which fires on every transient), ``aubio.tempo`` infers
    the underlying beat rate, which is what we need for global tempo
    conform — the right knob to ask "how fast is the source playing
    relative to target" is "beat rate", not "transient frequency".
    """
    try:
        import aubio  # noqa: WPS433 — optional dep
    except ImportError:  # pragma: no cover
        return None
    if mono.size < 4096:
        return None
    try:
        hop = 512
        win = 1024
        tempo = aubio.tempo("default", win, hop, int(sample_rate))
        mono32 = np.ascontiguousarray(mono, dtype=np.float32)
        end = mono32.size - hop + 1
        for i in range(0, end, hop):
            tempo(mono32[i : i + hop])
        bpm = float(tempo.get_bpm())
    except Exception:  # pragma: no cover
        return None
    if bpm <= 0:
        return None
    return bpm


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


def _assign_anchors_hierarchical(
    refined_onsets: np.ndarray,
    grid_lines: np.ndarray,
    metrical_levels: np.ndarray,
    *,
    src_length: int,
    total_samples: int,
    ratio_min: float,
    ratio_max: float,
    tolerance_samples: int,
    hierarchy: Tuple[int, ...],
) -> List[Tuple[int, int]]:
    """Coarsest-within-tolerance anchor assignment.

    For each onset, walk ``hierarchy`` coarsest → finest. At each level,
    take the nearest grid line whose ``metrical_levels[i] <= level``. If
    that line is within ``tolerance_samples``, snap there and stop;
    otherwise try the next level. Onsets that don't fit any level (i.e.
    are too far from every line of any level in the hierarchy) are left
    unanchored and ride the surrounding stretch.

    Same §5 monotonicity + ratio-clamp + boundary anchors as
    ``_assign_anchors``.
    """
    anchors: List[Tuple[int, int]] = [(0, 0)]
    last_src, last_dst = 0, 0
    n_lines = grid_lines.size
    last_assignable = n_lines - 2  # reserve last index for closing boundary

    for src in refined_onsets:
        src = int(src)
        if src <= last_src or src >= src_length:
            continue

        # Walk hierarchy: coarsest first. At each level, the eligible
        # lines are those with metrical_levels <= level.
        snapped_idx: Optional[int] = None
        for target_level in hierarchy:
            eligible = np.where(metrical_levels <= target_level)[0]
            if eligible.size == 0:
                continue
            eligible_samples = grid_lines[eligible]
            j = int(np.argmin(np.abs(eligible_samples - src)))
            line_idx = int(eligible[j])
            if abs(grid_lines[line_idx] - src) <= tolerance_samples:
                snapped_idx = line_idx
                break

        if snapped_idx is None:
            continue  # onset rides the surrounding stretch

        # Monotonicity: bump to next free line (any level) if we'd collide.
        while snapped_idx <= last_assignable and grid_lines[snapped_idx] <= last_dst:
            snapped_idx += 1
        if snapped_idx > last_assignable:
            break
        dst = int(grid_lines[snapped_idx])

        src_seg = src - last_src
        dst_seg = dst - last_dst
        if src_seg <= 0 or dst_seg <= 0:
            continue
        ratio = dst_seg / src_seg
        if ratio < ratio_min or ratio > ratio_max:
            continue

        anchors.append((src, dst))
        last_src, last_dst = src, dst

    anchors.append((src_length, total_samples))
    return anchors


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


def _loop_wrap_crossfade(out: np.ndarray, fade_samples: int) -> None:
    """Equal-power tail-head crossfade to mask the loop boundary click.

    Replaces ``out[-K:]`` with ``out[-K:] * fade_out + out[:K] * fade_in``
    where ``fade_in = sin²`` and ``fade_out = cos²`` over ``[0, π/2]`` —
    so ``fade_in + fade_out = 1`` (constant-power). The HEAD is unchanged
    so the loop start stays sharp; the modified TAIL ends close to
    ``out[0]`` so the wrap from ``out[-1] → out[0]`` is continuous.

    This is the practical alternative to task_1.md §7's "render past
    loop_end, fold it back" — that requires a source that overgenerates,
    which Fragmenta's SA3 path does not. Tail-head crossfade is the
    standard loop-cleanup technique and lands the same musical result on
    content that doesn't overgenerate.
    """
    if fade_samples <= 0:
        return
    n = out.shape[0]
    k = min(fade_samples, n // 4)
    if k <= 0:
        return
    angle = np.linspace(0.0, np.pi / 2.0, k, dtype=np.float32)
    fade_in = np.sin(angle) ** 2  # 0 → 1
    fade_out = np.cos(angle) ** 2  # 1 → 0
    head_copy = out[:k].copy()
    if out.ndim == 2:
        out[-k:] = out[-k:] * fade_out[:, None] + head_copy * fade_in[:, None]
    else:
        out[-k:] = out[-k:] * fade_out + head_copy * fade_in


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
