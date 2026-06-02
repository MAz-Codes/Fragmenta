#!/usr/bin/env python
"""Phase 1+2 acceptance tests for ``app.core.loop_quantizer``.

These tests prove the determinism property — the headline of task_1.md:
*multiple clips processed independently at the same BPM/bars/grid must
align sample-exactly*. Phase 1 covers the canonical grid, slice-and-place
math, and §5 guards using an ``EnergyFluxDetector`` placeholder; Phase 2
adds tests against ``AubioDetector`` (the production detector) on
realistic kick-burst signals. The DSP-quality pieces still pending —
segment classification, Rubber Band stretch, loop-wrap fold-back — land
in Phases 3/4.

Run::

    python tests/test_loop_quantizer.py

Exit code is non-zero if any assertion fails; CI-safe (no network, no
external state, no fixtures).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.loop_quantizer import (  # noqa: E402
    NO_STRETCHER,
    CanonicalGrid,
    EnergyFluxDetector,
    canonical_grid,
    classify_segment,
    quantize_batch,
    quantize_to_loop,
)
from app.core.loop_quantizer.refine import refine_to_transient  # noqa: E402


SAMPLE_RATE = 44100
BPM = 120.0
BARS = 4
GRID = 16
TIME_SIG = (4, 4)
TOLERANCE_SAMPLES = 8  # ~0.18 ms at 44.1 kHz

# Total samples for the canonical grid above:
#   round(4 bars * 4 beats * 60/120 * 44100) = 352800
EXPECTED_TOTAL_SAMPLES = 352_800


# --- synthetic signal helpers ---------------------------------------------


def make_click(
    n_samples: int,
    positions: np.ndarray,
    *,
    decay_ms: float = 2.0,
    amplitude: float = 0.9,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Place a short decaying click at each sample in ``positions``.

    Used as ground truth: each click has a sharp rising edge at its
    position, so a sample-accurate onset detector + refiner should
    return exactly those positions.
    """
    out = np.zeros(n_samples, dtype=np.float32)
    decay = max(2, int(decay_ms * sr / 1000.0))
    env = (amplitude * np.exp(-np.arange(decay) / (decay * 0.3))).astype(np.float32)
    for p in positions:
        p = int(p)
        if p < 0 or p >= n_samples:
            continue
        end = min(n_samples, p + decay)
        out[p:end] += env[: end - p]
    return out


# --- the actual tests -----------------------------------------------------


def test_canonical_grid_shape_and_endpoints() -> None:
    cg = canonical_grid(
        bpm=BPM, bars=BARS, grid=GRID, time_sig=TIME_SIG, sample_rate=SAMPLE_RATE
    )
    assert isinstance(cg, CanonicalGrid)
    assert cg.total_samples == EXPECTED_TOTAL_SAMPLES, (
        f"total_samples mismatch: {cg.total_samples} != {EXPECTED_TOTAL_SAMPLES}"
    )
    assert cg.grid_lines[0] == 0
    assert cg.grid_lines[-1] == cg.total_samples
    # Sixteenth grid in 4/4 → 4 divisions per beat → 16 beats → 64 divisions.
    # grid_lines length = 65 (inclusive of both endpoints).
    assert cg.grid_lines.size == 65, f"unexpected grid line count {cg.grid_lines.size}"
    assert cg.beat_samples.size == 17, f"unexpected beat count {cg.beat_samples.size}"
    # Strict monotonicity.
    diffs = np.diff(cg.grid_lines)
    assert np.all(diffs > 0), "grid_lines must be strictly increasing"
    # Beat positions are every 4 grid lines (sixteenth → quarter).
    np.testing.assert_array_equal(cg.beat_samples, cg.grid_lines[::4])
    print("  ✓ canonical_grid shape and endpoints")


def test_canonical_grid_deterministic() -> None:
    cg_a = canonical_grid(bpm=BPM, bars=BARS, grid=GRID, sample_rate=SAMPLE_RATE)
    cg_b = canonical_grid(bpm=BPM, bars=BARS, grid=GRID, sample_rate=SAMPLE_RATE)
    np.testing.assert_array_equal(cg_a.grid_lines, cg_b.grid_lines)
    np.testing.assert_array_equal(cg_a.beat_samples, cg_b.beat_samples)
    assert cg_a.total_samples == cg_b.total_samples
    print("  ✓ canonical_grid is bit-identical across calls")


def test_quantize_length_exact() -> None:
    """Output length must equal canonical total_samples regardless of input."""
    rng = np.random.default_rng(0)
    n = SAMPLE_RATE * 9  # 9 s of noise, longer than the 8 s canonical loop
    noise = (rng.standard_normal(n).astype(np.float32) * 0.1)
    out = quantize_to_loop(
        noise,
        bpm=BPM,
        bars=BARS,
        grid=GRID,
        sample_rate=SAMPLE_RATE,
        detector=EnergyFluxDetector(),
        loop_wrap_crossfade_ms=0.0,
    )
    assert out.shape[0] == EXPECTED_TOTAL_SAMPLES, (
        f"length mismatch: {out.shape[0]} != {EXPECTED_TOTAL_SAMPLES}"
    )
    assert out.dtype == np.float32
    print(f"  ✓ length exactness: {out.shape[0]} samples == canonical")


def test_quantize_byte_identical_across_runs() -> None:
    """Same input + params → byte-identical output across two calls."""
    rng = np.random.default_rng(7)
    n = SAMPLE_RATE * 9
    noise = (rng.standard_normal(n).astype(np.float32) * 0.1)
    det = EnergyFluxDetector()
    kwargs = dict(
        bpm=BPM, bars=BARS, grid=GRID, sample_rate=SAMPLE_RATE,
        detector=det, loop_wrap_crossfade_ms=0.0,
    )
    a = quantize_to_loop(noise, **kwargs)
    b = quantize_to_loop(noise, **kwargs)
    assert np.array_equal(a, b), "quantize_to_loop must be deterministic"
    print("  ✓ byte-identical determinism (single-clip)")


def test_multi_layer_alignment() -> None:
    """Two layers with deliberately off-grid onsets, batch-quantized:

    * Each output's refined onsets land within TOLERANCE of some canonical
      grid line.
    * For grid lines where both layers have a hit, the refined positions
      match across layers within TOLERANCE.

    This is the no-flamming property: the 4-channel sampler can stack
    these layers and they'll lock to the same musical positions.
    """
    cg = canonical_grid(
        bpm=BPM, bars=BARS, grid=GRID, time_sig=TIME_SIG, sample_rate=SAMPLE_RATE
    )
    grid_lines = np.asarray(cg.grid_lines)
    beat_samples = np.asarray(cg.beat_samples)

    # Source is slightly longer than canonical so the boundary segment
    # stays within ratio bounds. 100 ms tail.
    src_len = cg.total_samples + int(0.1 * SAMPLE_RATE)

    rng = np.random.default_rng(42)

    # Layer 1: kick on every beat (17 hits incl. endpoints, but skip the
    # closing endpoint to avoid colliding with the boundary anchor).
    kick_jitter = rng.integers(-220, 221, size=beat_samples.size - 1)
    kick_positions = beat_samples[:-1] + kick_jitter
    layer1 = make_click(src_len, kick_positions, decay_ms=3.0)

    # Layer 2: hat on every eighth (every 2 sixteenths).
    eighth_lines = grid_lines[:-1:2]
    hat_jitter = rng.integers(-300, 301, size=eighth_lines.size)
    hat_positions = eighth_lines + hat_jitter
    layer2 = make_click(src_len, hat_positions, decay_ms=1.5, amplitude=0.6)

    outs = quantize_batch(
        [layer1, layer2],
        bpm=BPM,
        bars=BARS,
        grid=GRID,
        time_sig=TIME_SIG,
        sample_rate=SAMPLE_RATE,
        detector=EnergyFluxDetector(),
        loop_wrap_crossfade_ms=0.0,  # crossfade smears the test's onset reading
        tempo_conform=False,  # synthetic click trains have no real BPM
    )
    out1, out2 = outs
    assert out1.shape[0] == cg.total_samples
    assert out2.shape[0] == cg.total_samples

    refined1 = _detect_and_refine(out1, cg.sample_rate)
    refined2 = _detect_and_refine(out2, cg.sample_rate)
    assert refined1.size > 0, "no onsets detected in layer 1 output"
    assert refined2.size > 0, "no onsets detected in layer 2 output"

    # Every refined onset must be within TOLERANCE of *some* grid line.
    max_dev1 = _max_distance_to_grid(refined1, grid_lines)
    max_dev2 = _max_distance_to_grid(refined2, grid_lines)
    assert max_dev1 <= TOLERANCE_SAMPLES, (
        f"layer 1 max grid deviation {max_dev1} > tolerance {TOLERANCE_SAMPLES}"
    )
    assert max_dev2 <= TOLERANCE_SAMPLES, (
        f"layer 2 max grid deviation {max_dev2} > tolerance {TOLERANCE_SAMPLES}"
    )

    # For each grid line that is anchored in BOTH layers, the refined
    # positions must agree within TOLERANCE.
    shared_lines, cross_dev = _shared_grid_dev(refined1, refined2, grid_lines)
    assert shared_lines >= 8, (
        f"expected ≥8 shared grid lines for assertion, got {shared_lines}"
    )
    assert cross_dev <= TOLERANCE_SAMPLES, (
        f"cross-layer deviation {cross_dev} > tolerance {TOLERANCE_SAMPLES} "
        f"on {shared_lines} shared grid lines"
    )
    print(
        f"  ✓ multi-layer alignment: {refined1.size}+{refined2.size} onsets, "
        f"max grid dev={max(max_dev1, max_dev2)} samp, "
        f"cross-layer dev={cross_dev} samp on {shared_lines} shared lines"
    )


def test_batch_byte_identical() -> None:
    """``quantize_batch`` must produce byte-identical output across runs
    given the same inputs.
    """
    rng = np.random.default_rng(11)
    src_len = SAMPLE_RATE * 9
    a_in = rng.standard_normal(src_len).astype(np.float32) * 0.1
    b_in = rng.standard_normal(src_len).astype(np.float32) * 0.1
    det = EnergyFluxDetector()
    kwargs = dict(
        bpm=BPM, bars=BARS, grid=GRID, sample_rate=SAMPLE_RATE, detector=det,
    )
    run1 = quantize_batch([a_in, b_in], **kwargs)
    run2 = quantize_batch([a_in, b_in], **kwargs)
    assert np.array_equal(run1[0], run2[0]), "batch run 1 clip 0 not deterministic"
    assert np.array_equal(run1[1], run2[1]), "batch run 1 clip 1 not deterministic"
    print("  ✓ byte-identical determinism (batch)")


# --- Phase 2: real detector (aubio) ---------------------------------------


def _aubio_available() -> bool:
    try:
        import aubio  # noqa: F401
        return True
    except ImportError:
        return False


def make_kick_burst(
    n_samples: int,
    positions: np.ndarray,
    *,
    freq_hz: float = 80.0,
    duration_ms: float = 100.0,
    decay_rate: float = 30.0,
    amplitude: float = 0.9,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Realistic kick-like sine burst — aubio's spectral detectors actually
    fire on these, unlike the 3 ms impulses used by the EnergyFlux tests.
    """
    out = np.zeros(n_samples, dtype=np.float32)
    duration = int(duration_ms * sr / 1000.0)
    t = np.arange(duration) / sr
    env = (amplitude * np.exp(-t * decay_rate)).astype(np.float32)
    sine = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    kick = sine * env
    for p in positions:
        p = int(p)
        if p < 0 or p >= n_samples:
            continue
        end = min(n_samples, p + duration)
        out[p:end] += kick[: end - p]
    return out


def test_aubio_multi_layer_alignment() -> None:
    """Real production detector (AubioDetector / specflux) on kick-burst
    layers. Tighter tolerance is impossible with spectral flux because the
    detector itself fires up to ~20 ms before the rising edge; the refiner
    drags it back but cannot beat its own window. We hold the bar at 64
    samples (~1.5 ms) — well under one 16th-note for any musical tempo.
    """
    if not _aubio_available():
        print("  ↷ aubio not installed; skipping AubioDetector test")
        return
    from app.core.loop_quantizer import AubioDetector

    AUBIO_TOLERANCE = 64  # samples (~1.5 ms at 44.1 kHz)

    cg = canonical_grid(
        bpm=BPM, bars=BARS, grid=GRID, time_sig=TIME_SIG, sample_rate=SAMPLE_RATE
    )
    grid_lines = np.asarray(cg.grid_lines)
    beat_samples = np.asarray(cg.beat_samples)
    src_len = cg.total_samples + int(0.2 * SAMPLE_RATE)

    rng = np.random.default_rng(2026)
    kick_jitter = rng.integers(-220, 221, size=beat_samples.size - 1)
    kick_positions = beat_samples[:-1] + kick_jitter
    layer1 = make_kick_burst(src_len, kick_positions, freq_hz=80.0)

    eighth_lines = grid_lines[:-1:2]
    hat_jitter = rng.integers(-300, 301, size=eighth_lines.size)
    hat_positions = eighth_lines + hat_jitter
    # Hi-hat-ish: noise burst, short. Use kick_burst with high freq + short
    # decay as a stand-in.
    layer2 = make_kick_burst(
        src_len, hat_positions, freq_hz=6000.0, duration_ms=30.0, decay_rate=120.0, amplitude=0.5
    )

    det = AubioDetector()
    outs = quantize_batch(
        [layer1, layer2],
        bpm=BPM,
        bars=BARS,
        grid=GRID,
        time_sig=TIME_SIG,
        sample_rate=SAMPLE_RATE,
        detector=det,
        tempo_conform=False,  # synthetic burst signals confuse aubio.tempo
        beat_track=False,     # this test measures the per-onset snap path
    )
    out1, out2 = outs
    assert out1.shape[0] == cg.total_samples
    assert out2.shape[0] == cg.total_samples

    refined1 = _detect_and_refine_with(out1, cg.sample_rate, det)
    refined2 = _detect_and_refine_with(out2, cg.sample_rate, det)
    assert refined1.size > 0, "AubioDetector found no onsets in layer 1 output"
    assert refined2.size > 0, "AubioDetector found no onsets in layer 2 output"

    max_dev1 = _max_distance_to_grid(refined1, grid_lines)
    max_dev2 = _max_distance_to_grid(refined2, grid_lines)
    assert max_dev1 <= AUBIO_TOLERANCE, (
        f"aubio layer 1 max grid dev {max_dev1} > {AUBIO_TOLERANCE}"
    )
    assert max_dev2 <= AUBIO_TOLERANCE, (
        f"aubio layer 2 max grid dev {max_dev2} > {AUBIO_TOLERANCE}"
    )
    print(
        f"  ✓ aubio multi-layer alignment: {refined1.size}+{refined2.size} onsets, "
        f"max grid dev={max(max_dev1, max_dev2)} samp "
        f"(tolerance {AUBIO_TOLERANCE})"
    )


def test_aubio_byte_identical() -> None:
    """AubioDetector must be deterministic — the C library has no
    randomness, so two runs on the same input produce identical output.
    """
    if not _aubio_available():
        print("  ↷ aubio not installed; skipping AubioDetector determinism")
        return
    from app.core.loop_quantizer import AubioDetector

    src_len = SAMPLE_RATE * 9
    rng = np.random.default_rng(99)
    positions = (np.arange(8) * (SAMPLE_RATE // 2) + 100).astype(np.int64)
    audio = make_kick_burst(src_len, positions)

    det = AubioDetector()
    a = quantize_to_loop(
        audio, bpm=BPM, bars=BARS, grid=GRID, sample_rate=SAMPLE_RATE, detector=det
    )
    b = quantize_to_loop(
        audio, bpm=BPM, bars=BARS, grid=GRID, sample_rate=SAMPLE_RATE, detector=det
    )
    assert np.array_equal(a, b), "AubioDetector path must be deterministic"
    print("  ✓ aubio byte-identical determinism")


# --- Phase 3: classifier + WSOLA --------------------------------------------


def _pytsmod_available() -> bool:
    try:
        import pytsmod  # noqa: F401
        return True
    except ImportError:
        return False


def test_classifier_distinguishes_pad_from_noise() -> None:
    """Spectral-flatness classifier separates a tonal pad from white noise."""
    sr = SAMPLE_RATE
    n = sr * 2
    t = np.arange(n) / sr
    pad = (0.4 * np.sin(2 * np.pi * 220 * t)
           + 0.3 * np.sin(2 * np.pi * 440 * t)
           + 0.2 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    rng = np.random.default_rng(13)
    noise = (rng.standard_normal(n).astype(np.float32) * 0.3)

    pad_class = classify_segment(pad)
    noise_class = classify_segment(noise)
    assert pad_class == "sustained", f"pad → {pad_class!r}, expected 'sustained'"
    assert noise_class == "transient", f"noise → {noise_class!r}, expected 'transient'"
    print(f"  ✓ classifier: pad → {pad_class}, noise → {noise_class}")


def test_sustained_content_preserves_pitch() -> None:
    """A pure tone quantized through the WSOLA path should keep its
    fundamental — linear-interp resampling would shift it by the same
    factor as the duration change.
    """
    if not _pytsmod_available():
        print("  ↷ pytsmod not installed; skipping WSOLA pitch test")
        return

    sr = SAMPLE_RATE
    src_seconds = 9.0  # > 8 s canonical, so the segment compresses by 8/9
    n = int(sr * src_seconds)
    t = np.arange(n) / sr
    target_hz = 440.0
    audio = (0.5 * np.sin(2 * np.pi * target_hz * t)).astype(np.float32)

    out_wsola = quantize_to_loop(
        audio,
        bpm=BPM,
        bars=BARS,
        grid=GRID,
        sample_rate=sr,
        detector=EnergyFluxDetector(),  # finds no onsets in a pure sine — by design
    )
    out_lin = quantize_to_loop(
        audio,
        bpm=BPM,
        bars=BARS,
        grid=GRID,
        sample_rate=sr,
        detector=EnergyFluxDetector(),
        stretcher=NO_STRETCHER,
    )

    peak_wsola = _peak_freq(out_wsola, sr)
    peak_lin = _peak_freq(out_lin, sr)

    # WSOLA should preserve pitch within an FFT bin.
    bin_hz = sr / out_wsola.shape[0]
    assert abs(peak_wsola - target_hz) < max(2.0, 2 * bin_hz), (
        f"WSOLA shifted pitch: {peak_wsola:.2f} Hz != {target_hz} Hz "
        f"(tolerance {max(2.0, 2 * bin_hz):.2f} Hz)"
    )
    # Linear-interp must visibly shift — the segment compresses 9 → 8,
    # so pitch rises by 9/8 = 1.125x. Expect ~495 Hz.
    expected_lin = target_hz * (src_seconds * sr) / out_lin.shape[0]
    assert abs(peak_lin - expected_lin) < 5.0, (
        f"linear-interp control: got {peak_lin:.2f}, expected {expected_lin:.2f}"
    )
    print(
        f"  ✓ sustained pitch preserved: WSOLA peak {peak_wsola:.2f} Hz "
        f"(target {target_hz}), linear-interp shifted to {peak_lin:.2f} Hz"
    )


def test_speed_budget() -> None:
    """A 4-bar loop must quantize in well under 1 second on a modern CPU
    (task_1.md speed budget).
    """
    sr = SAMPLE_RATE
    src_len = sr * 9
    positions = (np.arange(16) * (sr // 2) + 100).astype(np.int64)
    audio = make_kick_burst(src_len, positions)

    t0 = time.time()
    out = quantize_to_loop(
        audio, bpm=BPM, bars=BARS, grid=GRID, sample_rate=sr
    )
    elapsed = time.time() - t0
    assert out.shape[0] == EXPECTED_TOTAL_SAMPLES
    assert elapsed < 1.0, f"quantize_to_loop took {elapsed:.3f} s, budget 1.0 s"
    print(f"  ✓ speed budget: {elapsed * 1000:.1f} ms for one 4-bar loop")


# --- Phase 4: loop-wrap crossfade + parallel batch -------------------------


def test_loop_boundary_clickfree() -> None:
    """Tail-head equal-power crossfade must collapse the wrap-point
    discontinuity for content that doesn't loop cleanly on its own.

    Source: a linear ramp from -0.5 to +0.5 over 9 s. After any
    pitch-preserving stretch, ``out[0] ≈ -0.5`` and ``out[-1] ≈ +0.5``
    — a textbook discontinuity. With crossfade on, ``out[-1]`` should
    sit near ``out[0]`` so the loop boundary doesn't click.
    """
    sr = SAMPLE_RATE
    src_len = sr * 9
    audio = np.linspace(-0.5, 0.5, src_len, dtype=np.float32)

    raw = quantize_to_loop(
        audio,
        bpm=BPM,
        bars=BARS,
        grid=GRID,
        sample_rate=sr,
        detector=EnergyFluxDetector(),
        loop_wrap_crossfade_ms=0.0,
    )
    blended = quantize_to_loop(
        audio,
        bpm=BPM,
        bars=BARS,
        grid=GRID,
        sample_rate=sr,
        detector=EnergyFluxDetector(),
    )

    disc_raw = float(abs(raw[-1] - raw[0]))
    disc_blended = float(abs(blended[-1] - blended[0]))

    assert disc_raw > 0.4, (
        f"sanity: ramp -0.5→+0.5 should leave a visible boundary mismatch, got {disc_raw}"
    )
    assert disc_blended < 0.05, (
        f"crossfade should mask boundary; got |out[-1] - out[0]| = {disc_blended}"
    )
    # Sample 0 must be unchanged — the loop start should stay sharp.
    assert raw[0] == blended[0], "crossfade modified the head (it should only touch the tail)"
    print(
        f"  ✓ loop boundary: discontinuity {disc_raw:.4f} → {disc_blended:.4f} "
        f"(head unchanged: {blended[0]:.5f})"
    )


def test_batch_parallel_matches_sequential() -> None:
    """``quantize_batch`` with workers > 1 must produce byte-identical
    output to the sequential path. Threading is a perf knob, not a
    semantic change.
    """
    rng = np.random.default_rng(2026)
    src_len = SAMPLE_RATE * 9
    clips = [
        make_kick_burst(src_len, (np.arange(8) * (SAMPLE_RATE // 2) + 100 * i).astype(np.int64))
        for i in range(4)
    ]
    kwargs = dict(
        bpm=BPM, bars=BARS, grid=GRID, sample_rate=SAMPLE_RATE,
    )
    seq = quantize_batch(clips, workers=1, **kwargs)
    par = quantize_batch(clips, workers=4, **kwargs)
    assert len(seq) == len(par) == len(clips)
    for i, (s, p) in enumerate(zip(seq, par)):
        assert np.array_equal(s, p), f"clip {i}: parallel != sequential"
    print(f"  ✓ parallel batch byte-identical (4 clips, workers=4 vs workers=1)")


# --- hierarchical metrical snap --------------------------------------------


def test_metrical_levels_pattern() -> None:
    """For grid=16 in 4/4, the level pattern across a beat is
    [quarter, 16, 8, 16] — quarters at indices 0/4/8/…, eighths halfway
    between, sixteenths in the remaining slots.
    """
    cg = canonical_grid(bpm=BPM, bars=BARS, grid=16, sample_rate=SAMPLE_RATE)
    levels = cg.metrical_levels
    # First 8 lines (= 2 beats at grid=16): expected pattern repeats
    # every 4 indices.
    expected = np.array([4, 16, 8, 16, 4, 16, 8, 16], dtype=np.int32)
    np.testing.assert_array_equal(levels[:8], expected)
    # End-of-loop (index total_divisions) is on a downbeat — should be 4.
    assert int(levels[-1]) == 4
    print("  ✓ metrical levels: [4,16,8,16] pattern across each beat")


def test_hierarchical_snap_prefers_coarse() -> None:
    """An interior onset within tolerance of a quarter line must snap to
    the quarter — NOT to a nearby 16th line that happens to be equally
    close. Without hierarchy, the snap is purely distance-based and
    arbitrary.

    Fixture: click at sample 100 (acts as head-trim anchor — drops out at
    the boundary) + click at beat 3 + 5 ms (the test subject). After
    head-trim by ~100, the test click is the first INTERIOR onset and
    must lock to the quarter line at ``beat_samples[3]``.
    """
    from app.core.loop_quantizer import EnergyFluxDetector, NO_STRETCHER

    sr = SAMPLE_RATE
    cg = canonical_grid(bpm=BPM, bars=BARS, grid=16, sample_rate=sr)
    beat_samples = np.asarray(cg.beat_samples)

    src_len = cg.total_samples + int(0.2 * sr)
    head_anchor_pos = 100
    target_beat = int(beat_samples[3])  # beat 3 = sample 66150
    test_click_pos = target_beat + int(0.005 * sr)  # +5 ms
    audio = make_click(src_len, np.array([head_anchor_pos, test_click_pos]))

    out = quantize_to_loop(
        audio,
        bpm=BPM,
        bars=BARS,
        grid=16,
        sample_rate=sr,
        detector=EnergyFluxDetector(),
        stretcher=NO_STRETCHER,
        hierarchical=True,
        loop_wrap_crossfade_ms=0.0,
    )
    refined = _detect_and_refine(out, sr)
    interior = refined[refined > 1000]
    assert interior.size >= 1, (
        f"no interior onset detected; got {refined.tolist()}"
    )
    nearest = int(interior[0])
    dist_to_quarter = abs(nearest - target_beat)
    assert dist_to_quarter <= 8, (
        f"hierarchical snap landed at sample {nearest}, expected near "
        f"quarter line {target_beat} (dist {dist_to_quarter})"
    )
    print(
        f"  ✓ hierarchical snap: onset at +5 ms past beat 3 locked to "
        f"quarter (dist {dist_to_quarter} samp)"
    )


# --- Phase 7: beat-tracking anchors ----------------------------------------


def test_beat_track_locks_to_quarter_lines() -> None:
    """Phase 7: when ``beat_track=True`` the quantizer should land its
    detected pulses on the QUARTER-note grid lines.

    Fixture: a clean 4-bar @ 120 BPM kick pattern (one strong burst per
    quarter, 16 events total). aubio.tempo locks onto this within ~1.5 s
    and produces beat positions at the actual quarter samples. After the
    beat-track snap, the audio's interior onsets must coincide with the
    quarter-note grid lines (within ±32 samples / ~0.7 ms).
    """
    from app.core.loop_quantizer import AubioDetector

    sr = SAMPLE_RATE
    cg = canonical_grid(bpm=BPM, bars=BARS, grid=16, sample_rate=sr)
    beat_samples = np.asarray(cg.beat_samples)

    # Build 4 bars of quarter-note kicks at the exact beat positions but
    # offset everything by +25 ms so we can prove the snap closes the gap.
    src_len = cg.total_samples + int(0.2 * sr)
    audio = np.zeros(src_len, dtype=np.float32)
    decay = int(0.030 * sr)
    env = (0.9 * np.exp(-np.arange(decay) / (decay * 0.2))).astype(np.float32)
    tone = np.sin(np.linspace(0.0, np.pi * 40, decay)).astype(np.float32)
    kick = env * tone
    offset = int(0.025 * sr)  # +25 ms past every quarter
    for q in beat_samples:
        p = int(q) + offset
        if 0 <= p < src_len - decay:
            audio[p : p + decay] += kick

    out = quantize_to_loop(
        audio,
        bpm=BPM,
        bars=BARS,
        grid=16,
        sample_rate=sr,
        detector=AubioDetector(),
        beat_track=True,
        loop_wrap_crossfade_ms=0.0,
        tempo_conform=False,  # source is already at target BPM
    )
    detected = _detect_and_refine(out, sr)
    # Count how many of the 16 quarter lines have a detected onset within
    # ±32 samples (~0.7 ms). With beat-track snap we expect most quarters
    # to land tightly; without it (default ±30 ms tolerance falls outside
    # because per-onset snap drops the +25 ms offsets), they don't.
    hits = 0
    for q in beat_samples:
        nearest = _nearest(detected, int(q))
        if nearest is not None and abs(nearest - int(q)) <= 32:
            hits += 1
    assert hits >= 12, (
        f"beat-track expected ≥12/16 quarter hits, got {hits}; "
        f"detected={detected.tolist()[:8]}"
    )
    print(
        f"  ✓ beat-track snap: {hits}/16 quarter lines anchored "
        f"within ±32 samp (source pre-offset +25 ms)"
    )


# --- Phase 5: runtime integration ------------------------------------------


def test_loop_quantizer_flag() -> None:
    """``FRAGMENTA_LOOP_QUANTIZER`` gates the runtime integration."""
    import os
    from app.core.loop_quantizer import loop_quantizer_enabled

    saved = os.environ.get("FRAGMENTA_LOOP_QUANTIZER")
    cases = [
        ("0", False), ("1", True), ("true", True), ("True", True),
        ("yes", True), ("on", True), ("false", False), ("no", False),
    ]
    try:
        os.environ.pop("FRAGMENTA_LOOP_QUANTIZER", None)
        assert loop_quantizer_enabled() is False, "default must be OFF (unset)"
        for val, expected in cases:
            os.environ["FRAGMENTA_LOOP_QUANTIZER"] = val
            got = loop_quantizer_enabled()
            assert got is expected, f"value {val!r}: got {got}, expected {expected}"
    finally:
        if saved is None:
            os.environ.pop("FRAGMENTA_LOOP_QUANTIZER", None)
        else:
            os.environ["FRAGMENTA_LOOP_QUANTIZER"] = saved
    print("  ✓ FRAGMENTA_LOOP_QUANTIZER toggles correctly (default OFF)")


def test_quantize_wav_file_roundtrip() -> None:
    """``quantize_wav_file`` reads, quantizes, writes back at the source
    sample rate. Output WAV has the canonical length.
    """
    import os
    import tempfile

    import soundfile as sf
    from app.core.loop_quantizer import quantize_wav_file

    rng = np.random.default_rng(31)
    src_len = SAMPLE_RATE * 9
    audio = (rng.standard_normal(src_len).astype(np.float32) * 0.1)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        path = tf.name
    try:
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        quantize_wav_file(path, bpm=BPM, bars=BARS)
        out, out_sr = sf.read(path, always_2d=True)
        assert out_sr == SAMPLE_RATE, f"sample rate changed: {out_sr}"
        assert out.shape[0] == EXPECTED_TOTAL_SAMPLES, (
            f"unexpected length: {out.shape[0]} != {EXPECTED_TOTAL_SAMPLES}"
        )
    finally:
        os.unlink(path)
    print(f"  ✓ quantize_wav_file: WAV in-place roundtrip, {EXPECTED_TOTAL_SAMPLES} samp")


def _peak_freq(audio: np.ndarray, sample_rate: int) -> float:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    window = np.hanning(mono.size).astype(np.float32)
    spec = np.abs(np.fft.rfft(mono * window))
    freqs = np.fft.rfftfreq(mono.size, 1 / sample_rate)
    return float(freqs[int(np.argmax(spec))])


def _detect_and_refine_with(audio: np.ndarray, sample_rate: int, detector) -> np.ndarray:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    raw = detector(mono.astype(np.float32), sample_rate)
    if raw.size == 0:
        return raw
    refined = np.fromiter(
        (refine_to_transient(mono, int(o), sample_rate, window_sec=0.025) for o in raw),
        dtype=np.int64,
        count=raw.size,
    )
    return np.unique(refined)


# --- analysis helpers -----------------------------------------------------


def _detect_and_refine(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Detect onsets in ``audio`` and refine to sample accuracy.

    Independent of the quantizer's internal detector so the test
    measures the OUTPUT'S onset positions, not the input's.
    """
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    raw = EnergyFluxDetector()(mono.astype(np.float32), sample_rate)
    if raw.size == 0:
        return raw
    refined = np.fromiter(
        (refine_to_transient(mono, int(o), sample_rate) for o in raw),
        dtype=np.int64,
        count=raw.size,
    )
    return np.unique(refined)


def _max_distance_to_grid(positions: np.ndarray, grid_lines: np.ndarray) -> int:
    """Max distance (in samples) from any position to its nearest grid line."""
    if positions.size == 0:
        return 0
    idx = np.searchsorted(grid_lines, positions)
    idx_l = np.clip(idx - 1, 0, grid_lines.size - 1)
    idx_r = np.clip(idx, 0, grid_lines.size - 1)
    left = grid_lines[idx_l]
    right = grid_lines[idx_r]
    dist = np.minimum(np.abs(positions - left), np.abs(positions - right))
    return int(dist.max())


def _shared_grid_dev(
    refined_a: np.ndarray,
    refined_b: np.ndarray,
    grid_lines: np.ndarray,
) -> tuple[int, int]:
    """For each grid line that has the nearest onset in BOTH refined
    sets within TOLERANCE, compute |a_pos - b_pos|.

    Returns (shared_count, max_dev).
    """
    shared = 0
    max_dev = 0
    for line in grid_lines:
        line = int(line)
        a_near = _nearest(refined_a, line)
        b_near = _nearest(refined_b, line)
        if a_near is None or b_near is None:
            continue
        if abs(a_near - line) > TOLERANCE_SAMPLES:
            continue
        if abs(b_near - line) > TOLERANCE_SAMPLES:
            continue
        shared += 1
        max_dev = max(max_dev, abs(a_near - b_near))
    return shared, max_dev


def _nearest(positions: np.ndarray, target: int) -> int | None:
    if positions.size == 0:
        return None
    i = int(np.argmin(np.abs(positions - target)))
    return int(positions[i])


# --- driver ---------------------------------------------------------------


def main() -> int:
    tests = [
        test_canonical_grid_shape_and_endpoints,
        test_canonical_grid_deterministic,
        test_quantize_length_exact,
        test_quantize_byte_identical_across_runs,
        test_multi_layer_alignment,
        test_batch_byte_identical,
        test_aubio_multi_layer_alignment,
        test_aubio_byte_identical,
        test_classifier_distinguishes_pad_from_noise,
        test_sustained_content_preserves_pitch,
        test_speed_budget,
        test_loop_boundary_clickfree,
        test_batch_parallel_matches_sequential,
        test_loop_quantizer_flag,
        test_quantize_wav_file_roundtrip,
        test_metrical_levels_pattern,
        test_hierarchical_snap_prefers_coarse,
        test_beat_track_locks_to_quarter_lines,
    ]
    print(f"\nloop_quantizer Phase 1–5 acceptance — {len(tests)} tests\n")
    t0 = time.time()
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            print(f"  ✗ {t.__name__}: {exc}")
            failures += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {t.__name__}: {type(exc).__name__}: {exc}")
            failures += 1
    elapsed = time.time() - t0
    print(f"\n{len(tests) - failures}/{len(tests)} passed in {elapsed:.2f}s\n")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
