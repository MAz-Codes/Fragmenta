# `app/core/loop_quantizer/`

Replacement for the legacy `app/core/generation/audio_post_process.py`
beat-alignment subsystem. Implements the design in `task_1.md` and the
inventory plan in `AUDIT.md`. **Under active development — not yet on the
runtime path.** The legacy `align_to_grid` / `align_for_loop` path remains
the active production code until acceptance.

## Status

| Phase | Item | State |
|-------|------|-------|
| 1 | Canonical grid (pure, deterministic) — `grid.py` | ✅ |
| 1 | `quantize_to_loop` / `quantize_batch` entries — `quantizer.py` | ✅ (minimal) |
| 1 | Sample-accurate refinement — `refine.py` | ✅ |
| 1 | Slice-and-place with §5 guards | ✅ (linear-interp filler) |
| 1 | Determinism + multi-layer-alignment test | ✅ |
| 1 | CLI for manual testing | ✅ |
| 2 | Onset detector interface — `detectors.py` | ✅ |
| 2 | Real onset detector (aubio specflux) | ✅ |
| 2 | Real-detector multi-layer alignment test | ✅ |
| 3 | Segment classification — `classify.py` (spectral flatness) | ✅ |
| 3 | pytsmod WSOLA for sustained stretch — `stretch.py` | ✅ |
| 3 | Per-segment warp routing in quantizer | ✅ |
| 3 | Sustained-pitch-preservation + speed-budget tests | ✅ |
| 3b | Rubber Band stretcher (opt-in, system-binary dep) | ⏳ |
| 4 | Equal-power crossfade at segment seams | ⏳ |
| 4 | Loop-wrap overhang fold-back | ⏳ |
| 4 | Parallel `quantize_batch` execution | ⏳ |
| 5 | Wire into `/api/generate` and Phase 7 loop path | ⏳ |
| 5 | Full acceptance test suite | ⏳ |

### Phase 3 notes

Each inter-anchor segment is classified by ``classify_segment`` via mean
spectral flatness (geometric mean / arithmetic mean of the magnitude
spectrum). Threshold 0.30 — below = sustained / tonal; above = noisy /
transient. Sustained segments route to ``WSOLAStretcher`` (pytsmod);
transient and identity-ratio segments use linear interpolation (cheap,
sample-exact at the anchor points).

The test suite enforces this with a 440 Hz pure-tone fixture: WSOLA
holds the fundamental at 440 Hz exactly, while linear-interp on the same
8/9 compression ratio shifts the peak to 495 Hz (exactly 9/8 — confirms
the routing is exercised).

``RubberBandStretcher`` is reserved for Phase 3b (same ``Stretcher``
Protocol, opt-in via the ``stretcher=`` kwarg). It needs the
``rubberband`` CLI binary in ``PATH`` — light desktop work, no
production-blocker. Skipped in Phase 3 to keep the install story
pip-only.

Pass ``stretcher=NO_STRETCHER`` to opt out of WSOLA — useful for
measurement / comparing against the pure slice-and-place baseline.

### Phase 2 notes

`AubioDetector` uses aubio's `specflux` (spectral flux) by default. True
SuperFlux is not currently reachable through `aubio.onset()` — it lives
in `aubio.specdesc` and needs its own peak picker; a follow-up can wire
that if specflux proves inadequate on real SA3 output. madmom is the
other detector named in `task_1.md` §2 but cannot install on Python 3.11
(Cython build failure, long-standing upstream issue); we'll revisit if a
patched fork lands or we change Python versions. Essentia is also
allowed by §2 but install-heavy (no pip wheel on Linux); deferred.

The default refine window opened from ±15 ms (Phase 1) to ±25 ms because
aubio's spectral-flux peak fires up to ~20 ms before the rising edge.
±25 ms still can't reach a neighbour at any musically plausible tempo
(32nd notes at 240 BPM = 62.5 ms apart).

`default_detector()` resolves to `AubioDetector()` when `aubio` is
importable and `EnergyFluxDetector()` otherwise — production paths get
the real detector, environments that can't install aubio (CI smoke,
docker-without-aubio) keep working at degraded quality.

## Public API

```python
from app.core.loop_quantizer import (
    quantize_to_loop,
    quantize_batch,
    canonical_grid,
    CanonicalGrid,
    OnsetDetector,
    EnergyFluxDetector,    # Phase 1 placeholder only
)
```

`quantize_to_loop(audio, bpm, bars, *, grid=16, time_sig=(4,4), sample_rate=44100, ...)`
returns a float32 ndarray of shape `(canonical_total_samples, channels)`,
or 1-D if input was 1-D.

`quantize_batch(clips, bpm, bars, *, ...)` returns a list of such arrays
and **guarantees** that anchored transients across clips land at the same
canonical sample positions (the no-flamming property).

## Determinism guarantee

`canonical_grid(...)` is a pure function: same inputs → byte-identical
`CanonicalGrid.grid_lines` and `total_samples` across runs and machines.
`quantize_batch` computes the grid ONCE and reuses it. As long as the
configured detector is deterministic (the Phase 1 `EnergyFluxDetector`
and the planned madmom/Essentia/aubio detectors all are), running the
same input through the quantizer twice produces byte-identical output.

The included test (`tests/test_loop_quantizer.py`) enforces this with
multi-layer synthetic signals: two layers with deliberately off-grid
onsets, quantized independently via `quantize_batch`, whose post-quantize
anchored onsets land on the same canonical samples.

## Dependencies and licenses

Phase 1:

| Library | Version | License | Used for |
|---------|---------|---------|----------|
| numpy   | (existing pin) | BSD-3-Clause | array ops, sliding-window framing |
| soundfile | (existing pin) | BSD-3-Clause | CLI I/O only |

Phase 2:

| Library | Version | License | Used for |
|---------|---------|---------|----------|
| aubio   | `>=0.4.9,<0.5` | GPL-3.0+ | onset detection (`specflux`) |

aubio's GPL-3.0 is compatible with this project's AGPL-3.0. The combined
work is governed by AGPL-3.0; source remains available per AGPL §13.

Phase 3 (this commit):

| Library | Version | License | Used for |
|---------|---------|---------|----------|
| pytsmod | `>=0.3.8,<0.4` | MIT | WSOLA time-stretch for sustained segments |

Considered but not used:

| Library | License | Status |
|---------|---------|--------|
| madmom | BSD-3-Clause | broken on Python 3.11 (Cython 0.27 build) |
| Essentia | AGPL-3.0 | install-heavy; no Linux pip wheel; deferred |
| Rubber Band (via pyrubberband) | GPL-2+ | Phase 3b — needs `rubberband` CLI in PATH |

Planned for Phase 3:

| Library | License | Compatible? |
|---------|---------|-------------|
| pyrubberband (wraps Rubber Band CLI) | GPLv2+ | yes — license now permitted |
| pytsmod | MIT | yes |

Notes:
* Rubber Band requires the `rubberband` system binary in `PATH`. Linux
  dev: `apt install rubberband-cli`. macOS dev: `brew install rubberband`.
  Desktop dist must bundle the binary.
* **AGPL §13 reminder**: anyone hosting a modified Fragmenta over a
  network owes users the corresponding source. Keep this module
  buildable and attributions intact.
* **Do not target the Apple App Store** with builds that include any
  GPL/AGPL DSP code (Essentia, aubio, Rubber Band).

## CLI

```bash
python -m app.core.loop_quantizer \
    --bpm 120 --bars 4 --grid 16 \
    input.wav output.wav
```

Reads via soundfile, processes in float32 in memory, writes PCM_16. Use
for ad-hoc experiments; production paths should call the library
directly to avoid disk round-trips.
