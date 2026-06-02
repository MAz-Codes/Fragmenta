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
| 1 | Onset detector interface — `detectors.py` | ✅ (placeholder) |
| 1 | Slice-and-place with §5 guards | ✅ (linear-interp filler) |
| 1 | Determinism + multi-layer-alignment test | ✅ |
| 1 | CLI for manual testing | ✅ |
| 2 | Real onset detector (madmom / Essentia / aubio) | ⏳ |
| 3 | Segment classification (transient vs sustained) | ⏳ |
| 3 | Rubber Band / pytsmod WSOLA for sustained stretch | ⏳ |
| 4 | Loop-wrap overhang fold-back + equal-power seam | ⏳ |
| 4 | Parallel `quantize_batch` execution | ⏳ |
| 5 | Wire into `/api/generate` and Phase 7 loop path | ⏳ |
| 5 | Full acceptance test suite | ⏳ |

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

Phase 1 (this commit):

| Library | Version | License | Used for |
|---------|---------|---------|----------|
| numpy   | (existing pin) | BSD-3-Clause | array ops, sliding-window framing |
| soundfile | (existing pin) | BSD-3-Clause | CLI I/O only |

No new pins. The Phase 1 detector is pure numpy.

Planned for Phase 2 (subject to install-story verification on Python 3.11):

| Library | License | Compatible with AGPL-3.0 host? |
|---------|---------|-------------------------------|
| madmom | BSD-3-Clause | yes (permissive) |
| Essentia | AGPLv3 | yes (same license) |
| aubio | GPL-3.0 | yes (GPL is AGPL-3.0-compatible) |

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
