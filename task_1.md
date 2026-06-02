# Task: Rewrite Fragmenta's loop quantization / beat-alignment system

## Context

Fragmenta (`MAz-Codes/Fragmenta`, AGPL-3.0, Python 3.11, Flask backend + React frontend) generates audio with Stable Audio Open models. Its **Performance Mode** is a 4-channel sampler with a master BPM, launch quantization, and Ableton Link sync. A "bars-mode" feature currently claims to render clips that are "beat-aligned and tempo-locked" and "loop cleanly on the grid."

**That claim is inaccurate.** The existing alignment does not reliably work. This task replaces it with a correct, fast, transient-accurate loop quantizer.

The license is now AGPL-3.0, so GPL/AGPL DSP libraries (aubio, Essentia, Rubber Band) are usable. Two reminders that still apply:
- AGPL §13 network clause governs the combined work. Anyone hosting a modified Fragmenta over a network owes users the source. This is fine for the current Docker/HF/desktop distribution but must not be broken (keep source buildable and attributions intact).
- Do **not** target the Apple App Store with this build; GPL/AGPL code cannot ship there.

## Goal

Given a generated audio file plus a known musical grid (master BPM, bar count, grid resolution), produce a seamlessly loopable file whose transients are snapped to that grid. The headline requirement: **multiple clips processed independently at the same BPM/bars/grid must align sample-exactly**, so layering channels in the 4-channel sampler never produces flammed or fighting transients.

This is conceptually Ableton's auto-warp, but simplified by the fact that **the grid is fully known in advance** — we are forcing audio onto a canonical grid, not discovering tempo.

## Step 0 — Audit and mark the existing code stale

Before building anything, find and document the current implementation. Do not assume; verify in the actual source.
- Locate **all** existing bars-mode / beat-alignment / "tempo-lock" / "loop cleanly on the grid" code. It may be spread across `app/core`, `app/backend`, `utils`, and the frontend (any JS that assumes aligned output). List every file, function, and call site.
- Read `requirements.txt` and record which DSP libraries are already pinned (the README mentions librosa for the annotation tier) and which exist *only* to serve the old alignment path.
- Identify exactly where bars-mode clips enter and leave the alignment step, and how the master BPM and bar length reach that code.
- Write a short `AUDIT.md` that inventories everything above: each stale symbol, its file and line, what depends on it, and the integration seams the new module must plug into.

**Mark the old code stale (do not delete it yet).** As part of this step:
- Add a clear deprecation marker at the top of each stale file/function — e.g. a `# DEPRECATED: superseded by app/core/loop_quantizer, scheduled for removal (see task.md "Final step")` comment, and where the language allows, a runtime `DeprecationWarning`.
- Record the full stale inventory in `AUDIT.md` under a heading like **"Scheduled for removal"** so the final decommission step has an exact checklist to work from.
- The old path must keep functioning until the new module passes acceptance — staleness is a flag, not a removal. Do **not** delete anything in this step.

## Architecture

Build a standalone module (suggested: `app/core/loop_quantizer/`) with a clean function boundary so it can be unit-tested in isolation and called from Performance Mode.

Primary entry point:
```
quantize_to_loop(
    audio: np.ndarray | path,
    bpm: float,
    bars: int,
    grid: int,            # 8 or 16 (eighth/sixteenth)
    time_sig=(4, 4),
    sample_rate=44100,
) -> np.ndarray
```

Batch entry point that processes N clips concurrently and **guarantees an identical canonical grid across all of them** (compute the grid once, share it):
```
quantize_batch(clips: list, bpm, bars, grid, ...) -> list[np.ndarray]
```

## Required pipeline

**1. Canonical grid (deterministic).**
Compute exact sample positions of every grid line from BPM/bars/grid/SR, plus total loop length in samples. Use a single, fixed rounding rule so the grid is bit-identical across runs and across clips. Not every BPM/bar combination yields an integer sample count per beat — pick one rounding convention and apply it everywhere. This function must be pure and side-effect free.

**2. Onset detection.**
Detect transients. Now that the license permits it, prefer a strong detector:
- **SuperFlux via Essentia** (best for percussive material, suppresses false positives), or **madmom** (BSD, RNN-based), or **aubio** (fast C bindings).
Make the detector swappable behind an interface. Keep librosa only for the annotation tier, not here.

**3. Sample-accurate refinement.**
Detectors report frame-coarse positions (~10ms). For each detected onset, refine to sample accuracy by snapping to the local energy peak / steepest rise within a small window of the raw envelope. This refinement is what produces tight alignment; do not skip it.

**4. Segment classification.**
Classify each inter-onset segment as **transient-dominant** vs **sustained** (e.g. spectral flatness / harmonic content). This branching is what makes the quantizer work for *all instruments*, not just drums.

**5. Grid assignment.**
Snap each qualifying onset to its nearest grid line, with these guards:
- **Strength threshold** — only anchor onsets above a confidence threshold; let ghost/weak hits ride along via interpolation rather than forcing them onto lines.
- **Strict monotonicity** — assigned targets must be strictly increasing; resolve collisions (two onsets → one line) by bumping to the next free line or merging.
- **Ratio clamp** — clamp the per-segment stretch ratio to a sane range (e.g. ~0.8–1.25). A snap that would exceed this usually means the BPM is wrong or the onset is spurious; flag and skip rather than warp violently.
- **Boundary anchors** — always anchor `0→0` and `src_end→loop_end` to lock total length.

**6. Warp / placement (per segment).**
- **Transient-dominant** → **slice-and-place**: cut at onsets, place each slice's start exactly on its grid line, micro-stretch only the inter-onset filler with equal-power crossfades at the seams. No phase-vocoder. This is near-free and keeps transients pristine.
- **Sustained** → time-stretch only as needed. Options: **Rubber Band** (`--timemap` warp markers, `--crisp 6` for percussive crispness — now license-permitted) or **pytsmod WSOLA** (MIT, faster to integrate, good for small ratios). Never phase-vocode drums.
- Choose between Rubber Band and WSOLA on **speed vs quality**, not license — both are now allowed.

**7. Seamless loop wrap.**
Grid alignment alone does not guarantee a clean boundary. Render/keep a short overhang past `loop_end`, fold it back, and equal-power crossfade it into the head so ringing tails carry across the loop point without clicks. If the final length is not sample-exact, force it with a high-quality resample (soxr).

## Determinism requirement (critical)

`quantize_batch` must produce outputs where, for any two clips, the post-quantization sample positions of anchored transients are identical to within a tiny tolerance. The canonical grid must be computed **once** and reused. No per-clip randomness, no per-clip rounding drift. This is the property that lets the 4-channel sampler stack layers without flamming — verify it explicitly in tests.

## Speed budget

Files are needed almost immediately after generation. Targets:
- A 4-bar loop quantized in well under 1 second on a modern CPU.
- Keep audio in float32 numpy in memory; avoid disk round-trips between stages.
- Process batch clips in parallel (worker pool); independent until the shared grid is applied, so latency ≈ slowest single clip, not the sum.
- Favor slice-and-place (effectively memcpy + tiny crossfades) over stretching wherever classification allows; reserve stretching for sustained segments.

## Integration

- Wire the new module into Performance Mode's bars-mode generation path, driven by the master BPM and selected bar length.
- Route bars-mode through the new module while leaving the old (now stale-flagged) code in place but no longer on the active path. Removal happens in the final step, only after acceptance.
- Update `requirements.txt` with the new (AGPL-safe) dependencies and update `NOTICE.md` attributions accordingly. Do **not** yet remove libraries that only the stale path uses — that happens at decommission.
- Correct any README wording that overstates the old behavior.

## Tests / acceptance criteria

Write automated tests covering:
1. **Multi-layer alignment** — generate ≥2 synthetic layers (e.g. a kick pattern and a hat pattern with deliberately off-grid onsets), quantize each independently via `quantize_batch`, and assert their anchored onset sample positions match within tolerance. Summed, they must show no audible flamming.
2. **Sustained material** — a pad/tone loop quantizes with no audible phase-vocoder smearing.
3. **Loop boundary** — assert the wrap point is click-free (no discontinuity / no energy spike at `loop_end`).
4. **Length exactness** — output sample count equals the canonical loop length exactly.
5. **Determinism** — same input + params produces byte-identical output across runs.
6. **Speed** — 4-bar loop meets the sub-second target.

## Deliverables

- `app/core/loop_quantizer/` module with `quantize_to_loop` and `quantize_batch`.
- A thin CLI wrapper for manual testing.
- `AUDIT.md` (from Step 0).
- Test suite as above.
- Updated `requirements.txt`, `NOTICE.md`, and corrected README wording.
- A short section in the module README documenting the license of every new dependency and confirming AGPL compatibility.

## Open questions to resolve early (don't build blind)

- Does Stable Audio Open render near the target BPM already? If clips arrive close to grid, required stretch ratios stay small and slice-and-place alone may suffice — confirm empirically on real generations before committing to a heavy stretch path.
- How does the existing master BPM / bar length actually reach the generation code? Match that seam rather than inventing a new one.

## Final step — Decommission the stale code (gated)

**Do not start this step until all acceptance tests pass and the new module has been confirmed working on real generations.** This is a separate, last task by design: the old path stays as a safety net until the new one is proven.

When the gate is met:
- Work through the **"Scheduled for removal"** checklist in `AUDIT.md` and delete every stale symbol, file, and call site listed there — including any frontend code that only existed to compensate for the old alignment.
- Remove dependencies from `requirements.txt` that were used *only* by the stale path, and update `NOTICE.md` to drop their attributions.
- Remove the deprecation markers and `DeprecationWarning`s (they go away with the code they annotated).
- Run the full test suite again after removal to confirm nothing on the new path depended on the deleted code.
- Note the removal in the changelog / commit message referencing this task, so the history shows the old bars-mode was intentionally retired, not lost.

If the new approach does **not** work out, do nothing here: leave the stale code in place and the deprecation markers as a record, and report what failed instead of removing the fallback.
