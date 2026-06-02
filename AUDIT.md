# AUDIT — Bars-mode / beat-alignment / loop-quantization subsystem

Inventory taken **2026-06-02** on branch `dev/sa3` at HEAD `b1b605a`, in preparation for replacement by `app/core/loop_quantizer/` (see `task_1.md`).

This file is the Step 0 deliverable. It catalogues every symbol, call site, dependency, and downstream assumption that the new module must either subsume or remain compatible with. **Nothing is deleted by this audit** — staleness is a flag, not a removal.

---

## TL;DR

- All Python alignment lives in **one file**: [app/core/generation/audio_post_process.py](app/core/generation/audio_post_process.py) (804 lines).
- Two entry points are called from outside: `align_to_grid()` (file-on-disk, called from `/api/generate`) and `align_for_loop()` (in-memory, called from the Phase 7 loop pipeline).
- A hardened **Stage A v2** path (`_stage_a_v2`) already exists in the same file, gated behind `FRAGMENTA_BEATSYNC_V2` (default **OFF**). Both entry points delegate to it when the flag is on; legacy path runs otherwise.
- Frontend code in [app/frontend/src/utils/](app/frontend/src/utils/) **assumes** backend invariants (downbeat at sample 0, sample-exact length, 44.1 kHz). It does not itself align audio, but breaks visibly if the backend stops honouring those invariants.
- DSP stack: **librosa** does everything heavy (beat tracking, onset detection, phase-vocoder stretch). soundfile handles WAV I/O. No aubio / madmom / Essentia / Rubber Band / pytsmod is currently used.
- Only one Python file (`audio_post_process.py`) and three call sites need rewiring. The blast radius is narrow.

---

## 1. Backend — legacy alignment path (v1)

All in [app/core/generation/audio_post_process.py](app/core/generation/audio_post_process.py).

| Symbol | Lines | What it does |
|---|---|---|
| `align_to_grid()` | [412–528](app/core/generation/audio_post_process.py#L412-L528) | **Public entry**, file-based. Reads WAV → `_detect_grid` → `_best_stretch_rate` → `_time_stretch_multichannel` → `_detect_first_onset_sample` (head-trim) → `_snap_to_beat` (end) → `_trailing_audio_end` (tail trim) → `_apply_fade`. Writes back to disk. Delegates to `_stage_a_v2` when `beatsync_v2_enabled()` is true (branch at L423). |
| `align_for_loop()` | [533–643](app/core/generation/audio_post_process.py#L533-L643) | **Public entry**, in-memory. Same logic as `align_to_grid` but returns numpy, returns exactly `target_samples`, skips tail trim. Delegates to `_stage_a_v2` when v2 flag is on (branch at L566). |
| `_detect_grid()` | [776–798](app/core/generation/audio_post_process.py#L776-L798) | librosa `beat_track` with optional BPM prior; returns `(bpm, beat_samples_array)`. Validates BPM in [40, 240]. |
| `_detect_first_onset_sample()` | [759–773](app/core/generation/audio_post_process.py#L759-L773) | librosa `onset_detect` fallback in first 1 s when beat tracking fails. |
| `_best_stretch_rate()` | [723–756](app/core/generation/audio_post_process.py#L723-L756) | Detected→target BPM rate mapping with octave-correction fallback (×0.5, ×2.0). |
| `_time_stretch_multichannel()` | [801–804](app/core/generation/audio_post_process.py#L801-L804) | Per-channel librosa phase-vocoder time-stretch. |
| `_snap_to_beat()` | [678–698](app/core/generation/audio_post_process.py#L678-L698) | Nearest detected beat within ±½ beat of target; falls back to target. |
| `_trailing_audio_end()` | [648–675](app/core/generation/audio_post_process.py#L648-L675) | Walks RMS windows backward to find last audible sample; legacy tail trim. |
| `_apply_fade()` | [701–712](app/core/generation/audio_post_process.py#L701-L712) | In-place equal-power head or tail fade. |
| `_equal_power_ramp()` | [715–720](app/core/generation/audio_post_process.py#L715-L720) | Cosine equal-power ramp helper. |

## 2. Backend — Stage A v2 alignment path

Also all in [audio_post_process.py](app/core/generation/audio_post_process.py). Activated when `FRAGMENTA_BEATSYNC_V2` is set; called via the same two public entries.

| Symbol | Lines | What it does |
|---|---|---|
| `_stage_a_v2()` | [140–220](app/core/generation/audio_post_process.py#L140-L220) | Hardened pipeline: `_grid_confidence` gates whether to trust detected BPM; on high confidence + high drift + `FRAGMENTA_BEATSYNC_WARP` opt-in, runs `_beat_sync_warp`; otherwise `_conform_stretch` (global tempo nudge). Anchors `beats[0]` refined by `_refine_to_transient` to sample 0. Crops to exact length via `_exact_len`. |
| `_exact_len()` | [223–236](app/core/generation/audio_post_process.py#L223-L236) | Crop to exactly `target_samples` (INV#2/#3); zero-pad only as logged last resort. |
| `_grid_drift_samples()` | [239–248](app/core/generation/audio_post_process.py#L239-L248) | Std-dev of beat residuals vs uniform least-squares grid; measures tempo wobble. |
| `_refine_to_transient()` | [251–271](app/core/generation/audio_post_process.py#L251-L271) | Snaps an approximate beat to the rising edge within ~15 ms (INV#4: sample-accurate downbeat). |
| `_beat_sync_warp()` | [274–299](app/core/generation/audio_post_process.py#L274-L299) | Per-beat phase-vocoder warp (Ableton "Beats" style). **OFF BY DEFAULT** — only when `FRAGMENTA_BEATSYNC_WARP=1` AND drift > 15 ms AND ≥ 6 beats. Detector-fragile; retained for experiments only. |
| `_grid_confidence()` | [302–335](app/core/generation/audio_post_process.py#L302-L335) | Combines beat-interval CV and onset-envelope autocorrelation into [0, 1]. Threshold `_GRID_CONFIDENCE_MIN = 0.65` (L117) gates the decision tree in `_stage_a_v2`. |
| `_first_strong_transient()` | [338–388](app/core/generation/audio_post_process.py#L338-L388) | Two-stage onset → rising-edge refinement. **Currently unreferenced inside `_stage_a_v2`** (verify before deleting; may be used elsewhere or kept for future). |
| `_conform_stretch()` | [391–409](app/core/generation/audio_post_process.py#L391-L409) | librosa phase-vocoder tempo conform, bounded to rate ∈ [0.85, 1.15]. INV#5 "justified equivalent" — downbeat is positioned by trim, not by this stretch. |
| `beatsync_v2_enabled()` | [39–50](app/core/generation/audio_post_process.py#L39-L50) | Reads `FRAGMENTA_BEATSYNC_V2`. |
| `_warp_enabled()` | [53–62](app/core/generation/audio_post_process.py#L53-L62) | Reads `FRAGMENTA_BEATSYNC_WARP`. |
| Constants `_V2_*`, `_GRID_CONFIDENCE_MIN`, `_WARP_*` | [101–126](app/core/generation/audio_post_process.py#L101-L126) | Tunable thresholds for the v2 path. |

## 3. Call sites & integration seams

### 3a. `/api/generate` endpoint — bars-mode file path

[app/backend/app.py](app/backend/app.py):

- **Request schema doc** (L325–326, L337–339): `align_bars` (1–64) and `align_bpm` are the user-facing knobs. Bars-mode = both present.
- **Parsing & validation** (L400–408): `do_align = align_bars is not None and align_bpm is not None`.
- **Loop-stitch validation** (L459–473): `loop_stitch ∈ {"inpaint", "crossfade", None}`; requires `do_align` when set.
- **SA3 generate call** (L564–573): passes `loop_mode=do_align`, `loop_stitch=…`, `loop_bars`, `loop_bpm`.
- **Post-gen call to `align_to_grid`** at [L582–585](app/backend/app.py#L582-L585):
  ```python
  if do_align and not loop_stitch:
      from app.core.generation.audio_post_process import align_to_grid
      align_to_grid(...)
  ```
  → **This is the v1 / v2 entry seam for the file-based bars-mode path.** When `loop_stitch` is set, the in-memory Phase 7 path has already aligned the baseline, so this branch is skipped.

### 3b. Phase 7 in-memory loop pipeline

[app/core/generation/audio_generator.py](app/core/generation/audio_generator.py):

- `generate()` at [L362–530](app/core/generation/audio_generator.py#L362-L530) — accepts `loop_stitch`, `loop_bars`, `loop_bpm`.
- **Pre-inpaint alignment** at [L490–510](app/core/generation/audio_generator.py#L490-L510): when `loop_stitch == "inpaint"`, calls `self._align_baseline_for_loop(...)` BEFORE `self._make_loop_inpaint(...)`. Comment notes the inpaint pass smooths only the seam, not the phase.
- **Crossfade branch** at L527: `loop_stitch == "crossfade"` skips alignment, calls `_crossfade_seam` only.
- `_align_baseline_for_loop()` at [L537–588](app/core/generation/audio_generator.py#L537-L588) — torch→numpy conversion, calls `align_for_loop` ([L559–562](app/core/generation/audio_generator.py#L559-L562)), torch→ back. Catches all exceptions and falls back to plain crop/pad ([L577](app/core/generation/audio_generator.py#L577)).
- `_make_loop_inpaint()` at [L590–678](app/core/generation/audio_generator.py#L590-L678) — wrap-by-N/2 + SA3 inpaint pass for seam smoothing. **Not part of alignment**, but its correctness depends on the baseline being sample-exact and downbeat-at-0.
- `_crossfade_seam()` at [L681–708](app/core/generation/audio_generator.py#L681-L708) — equal-power crossfade fallback. **Not part of alignment**, kept as-is.

### 3c. How master BPM / bar length reach the aligner

**The seam to preserve.** Both paths take `align_bars` and `align_bpm` from the JSON request payload (`/api/generate`). There is no global state, no config file, no model side-channel — the new module should match this signature.

- File path: `app.py` request JSON → `align_bars`, `align_bpm` locals → `align_to_grid(path, target_samples, target_bpm)`.
- Phase 7 path: same JSON → `loop_bars`, `loop_bpm` kwargs into `SA3.generate()` → `_align_baseline_for_loop(audio_tensor, target_samples, target_bpm)` → `align_for_loop(audio_np, target_samples, target_bpm)`.

## 4. Frontend — code that depends on alignment invariants

No frontend code performs alignment, but several files **assume** the backend honours INV#2 (sample-exact length), INV#4 (downbeat at sample 0), and INV#6 (44.1 kHz). Breakage in those invariants surfaces as missed-grid launches and audible flamming.

| File | Symbol | Lines | Assumption |
|---|---|---|---|
| [app/frontend/src/utils/phaseLock.js](app/frontend/src/utils/phaseLock.js) | `phaseOffsetSec()` | [12–18](app/frontend/src/utils/phaseLock.js#L12-L18) | Pure function; computes Link-phase → loop-position offset. Assumes loop length is musically exact and clip starts on the "1". |
| [app/frontend/src/utils/performanceAudio.js](app/frontend/src/utils/performanceAudio.js) | `ENGINE_SAMPLE_RATE = 44100`, `getAudioContext()` | [9–35](app/frontend/src/utils/performanceAudio.js#L9-L35) | Pins Web Audio context to 44.1 kHz; warns if the browser refuses. Prevents silent resampling that would change loop sample counts. |
| [app/frontend/src/utils/performanceAudio.js](app/frontend/src/utils/performanceAudio.js) | `ChannelStrip.play()` | ~L223–251 | Schedules launches at quantized boundaries; assumes head = downbeat. |
| [app/frontend/src/components/PerformancePanel.js](app/frontend/src/components/PerformancePanel.js), `PerformanceChannel.js`, `usePerformanceSession.js` | (verify) | — | UI wiring around quantum, BPM, Link. **Verify** whether any hardcoded offsets exist that compensate for legacy misalignment. |
| [app/frontend/src/components/AudioWaveform.js](app/frontend/src/components/AudioWaveform.js), `GenerationWaveform.js` | (verify) | — | Reference `44100` / `sampleRate`; likely canvas-rendering only, but **verify** before deleting. |

[app/core/audio/link_sync.py](app/core/audio/link_sync.py) is the Ableton Link bridge. It does **not** align audio — it only exposes BPM/beat over HTTP — but it is the BPM source-of-truth for the Performance UI and therefore for any clip the user re-generates with bars-mode. Out of scope for replacement; in scope for verifying the new module accepts the same BPM contract.

## 5. Feature flags

| Flag | Default | Read at | Effect |
|---|---|---|---|
| `FRAGMENTA_BEATSYNC_V2` | `"0"` (OFF) | [audio_post_process.py:39–50](app/core/generation/audio_post_process.py#L39-L50), branches at L423 and L566 | Routes `align_to_grid` / `align_for_loop` to `_stage_a_v2`. |
| `FRAGMENTA_BEATSYNC_WARP` | `"0"` (OFF) | [audio_post_process.py:53–62](app/core/generation/audio_post_process.py#L53-L62), checked at L194 | Enables per-beat warp inside `_stage_a_v2` when drift and beat-count conditions are met. Off by default because librosa beat detection mis-places beats on real audio. |

## 6. Dependencies — [requirements.txt](requirements.txt)

| Line | Pin | Used by alignment? |
|---|---|---|
| 7–9 | torch / torchaudio / torchvision | **Yes** — Phase 7 path uses torch tensors; torchaudio for I/O. Shared with the whole stack. |
| 21 | `librosa>=0.11.0,<0.12` | **Yes, core** — beat_track, onset_detect, time_stretch, autocorrelate. Used pervasively. **Shared** with other generation code. |
| 22 | `soundfile>=0.13.1` | **Yes** — `align_to_grid` reads/writes WAVs via `sf.read` / `sf.write`. **Shared** with the rest of the backend. |
| 24 | `scipy>=1.13,<2` | Indirect (librosa internal). Not directly imported by `audio_post_process.py`. **Shared.** |
| 25 | `numba>=0.60,<1` | Indirect (librosa internal). **Shared.** |
| 26 | `numpy>=2.2.6,<3` | **Yes** — pervasive. **Shared.** |
| 31 | `torchlibrosa>=0.1.0` | **No** in alignment; used by SA3 model code. |
| 11–19, 23, 28–48 | Flask, transformers, pytorch_lightning, einops, pandas, ftfy, pywebview, etc. | **No** — unrelated to alignment. |

**Conclusion:** there is **no dependency to remove** when the legacy path is decommissioned. Every DSP library currently used (librosa, soundfile, numpy, scipy) is also needed elsewhere or by librosa itself. New AGPL-safe libs (Essentia / madmom / aubio / Rubber Band / pytsmod) are **not currently pinned** — they will be additions in a later step, not replacements.

[NOTICE.md](NOTICE.md) (172 lines) attributes the existing licenses. No alignment-specific attributions need removal at decommission; **new** attributions will be added when new DSP libs are introduced.

## 7. Tests

| Path | Coverage |
|---|---|
| [tests/smoke_test_beatsync.py](tests/smoke_test_beatsync.py) | Stage A v2 acceptance: synthetic click trains, drifty signals, real-fixture measurement. Imports `align_to_grid`, `align_for_loop`, `_conform_stretch`, `_grid_confidence`, `_detect_grid`, `beatsync_v2_enabled`. Runs under `FRAGMENTA_BEATSYNC_V2=1`. |
| [tests/smoke_test_phaselock.mjs](tests/smoke_test_phaselock.mjs) | Stage B `phaseOffsetSec` pure-function tests. |
| [scripts/smoke_test_loop.py](scripts/smoke_test_loop.py) | End-to-end Phase 7 integration via live `/api/generate`. Checks seam discontinuity, RMS within ±3 dB. Not CI-safe without a running backend. |
| [tests/smoke_test_sa3.py](tests/smoke_test_sa3.py), [tests/smoke_test_sa3_contract.py](tests/smoke_test_sa3_contract.py), [tests/smoke_test_sa3_lora.py](tests/smoke_test_sa3_lora.py), [tests/smoke_test_pre_encode.py](tests/smoke_test_pre_encode.py) | SA3 / training contracts. Not directly testing alignment; consume the same model outputs. |

`tests/` is `.gitignore`d with surgical `!exception` lines for the smoke tests above.

## 8. README claims to correct

[README.md](README.md) — overstates the *legacy* path (flag-OFF behaviour). All current claims hold under v2; only the flag-OFF default differs:

- [L35](README.md#L35) — `- **Seamless loops** — bars-mode clips are tempo-locked and the loop seam is smoothed by inpainting, not a crossfade`
- [L40](README.md#L40) — `  - **Bars-mode generation** — request clips by bar count; output is beat-aligned and tempo-locked to the master BPM`
- [L209](README.md#L209) — `- Master tempo (BPM) — drives bars-mode generation and launch quantization`
- [L213](README.md#L213) — `**Bars-mode generation:**`
- [L214](README.md#L214) — `Switch a channel from \`sec\` to \`bars\` and pick a bar length (1–16). The clip is rendered to that bar count at the master BPM, then automatically beat-aligned and tempo-locked so it loops cleanly on the grid. The first beat lands at the start of the file; subsequent loops stay on the bar.`

These are the lines the task spec calls out as overstatements. Once the new module is the only path, the wording is *accurate*; until then, it overpromises the legacy/flag-OFF behaviour.

## 9. Scheduled for removal — final-step checklist

Every entry below: **stale-flag now, delete after the new `app/core/loop_quantizer/` module passes acceptance.** Helpers shared with code outside the alignment subsystem (none currently) would need migration, not deletion — there are no such cases.

### 9a. Public entry points (call-site rewires)
- [ ] `align_to_grid()` — [audio_post_process.py:412–528](app/core/generation/audio_post_process.py#L412-L528)
- [ ] `align_for_loop()` — [audio_post_process.py:533–643](app/core/generation/audio_post_process.py#L533-L643)
- [ ] Call site in `/api/generate`: `from app.core.generation.audio_post_process import align_to_grid` at [app.py:584–585](app/backend/app.py#L584-L585)
- [ ] Call site in Phase 7: `from app.core.generation.audio_post_process import align_for_loop` at [audio_generator.py:559–562](app/core/generation/audio_generator.py#L559-L562) inside `_align_baseline_for_loop`

### 9b. Legacy (v1) internals — delete outright
- [ ] `_detect_first_onset_sample()` — [L759–773](app/core/generation/audio_post_process.py#L759-L773)
- [ ] `_snap_to_beat()` — [L678–698](app/core/generation/audio_post_process.py#L678-L698)
- [ ] `_trailing_audio_end()` — [L648–675](app/core/generation/audio_post_process.py#L648-L675)
- [ ] `_best_stretch_rate()` — [L723–756](app/core/generation/audio_post_process.py#L723-L756)
- [ ] `_time_stretch_multichannel()` — [L801–804](app/core/generation/audio_post_process.py#L801-L804) (delete if new module replaces librosa stretch entirely; new module may reuse the same one-liner)
- [ ] `_apply_fade()`, `_equal_power_ramp()` — [L701–720](app/core/generation/audio_post_process.py#L701-L720) (delete if new module's crossfade lives in its own helpers; trivially re-implementable)

### 9c. Stage A v2 internals — port into the new module (do not "delete" wholesale)
The v2 path is the closest thing in-tree to what the new module must do. Treat these as **reference implementations to lift, refactor, and replace** — not pure deletions. Once the new module subsumes them, they can be removed from `audio_post_process.py`:
- [ ] `_stage_a_v2()` — [L140–220](app/core/generation/audio_post_process.py#L140-L220)
- [ ] `_exact_len()` — [L223–236](app/core/generation/audio_post_process.py#L223-L236)
- [ ] `_grid_drift_samples()` — [L239–248](app/core/generation/audio_post_process.py#L239-L248)
- [ ] `_refine_to_transient()` — [L251–271](app/core/generation/audio_post_process.py#L251-L271)
- [ ] `_beat_sync_warp()` — [L274–299](app/core/generation/audio_post_process.py#L274-L299) (low priority — disabled by default, detector-fragile; consider dropping rather than porting)
- [ ] `_grid_confidence()` — [L302–335](app/core/generation/audio_post_process.py#L302-L335)
- [ ] `_first_strong_transient()` — [L338–388](app/core/generation/audio_post_process.py#L338-L388) (**verify if dead code first**; currently not called from `_stage_a_v2`)
- [ ] `_conform_stretch()` — [L391–409](app/core/generation/audio_post_process.py#L391-L409)
- [ ] `_detect_grid()` — [L776–798](app/core/generation/audio_post_process.py#L776-L798) (port or replace with chosen onset detector)

### 9d. Feature-flag scaffolding
- [ ] `beatsync_v2_enabled()` and every callsite — [L39–50](app/core/generation/audio_post_process.py#L39-L50), L423, L566. Once new module is the only path, the flag becomes meaningless.
- [ ] `_warp_enabled()` — [L53–62](app/core/generation/audio_post_process.py#L53-L62). Remove with the warp code unless it lives on in the new module.
- [ ] Constants block at [L101–126](app/core/generation/audio_post_process.py#L101-L126).

### 9e. Test rewires
- [ ] [tests/smoke_test_beatsync.py](tests/smoke_test_beatsync.py) — imports must move from `app.core.generation.audio_post_process` to `app.core.loop_quantizer`. Cases stay valid (the invariants are the same).
- [ ] [scripts/smoke_test_loop.py](scripts/smoke_test_loop.py) — no symbol imports from the alignment module; verify wrapper still passes against the new pipeline.

### 9f. Whole-file removal (only after every symbol above is gone)
- [ ] Delete [app/core/generation/audio_post_process.py](app/core/generation/audio_post_process.py) entirely. **Grep for any remaining `from app.core.generation.audio_post_process import …` first.**

### 9g. README + docs
- [ ] Update L35, L40, L209, L213, L214 of [README.md](README.md) once new behaviour is the default — either tighten the wording to honest claims, or leave as-is if the new module fully delivers what the old wording promised.

### 9h. Dependencies
- [ ] No removal needed (see §6). Possibly *additions* — Essentia/madmom/aubio for onset detection, pytsmod/Rubber Band for stretching. Update [requirements.txt](requirements.txt) and [NOTICE.md](NOTICE.md) when those land.

## 10. Open questions to manually verify before the new module commits

1. **Does Stable Audio Open generate near-grid at the requested BPM?** Empirically measure required stretch ratios on real SA3 outputs at 100/110/120/128/140 BPM. If ratios stay close to 1.0, slice-and-place alone may suffice and the heavy stretch path can be deferred.
2. **Is `_first_strong_transient()` dead code?** It's defined in `_stage_a_v2`'s neighbourhood but appears unreferenced in the active v2 path. Confirm before porting or deleting.
3. **`AudioWaveform.js` / `GenerationWaveform.js` references to 44100** — likely canvas-rendering only, but confirm none of it compensates for backend misalignment.
4. **PerformancePanel / PerformanceChannel / usePerformanceSession** — read these end-to-end to confirm no hardcoded offsets compensate for legacy alignment quirks.
5. **`/api/generate` JSON schema is the BPM/bars seam.** Confirm no other endpoint or websocket path delivers bars-mode payloads to the model.
6. **Link bridge contract** — [link_sync.py](app/core/audio/link_sync.py) returns `(bpm, beat)`. New module's batch entry should accept the same BPM source-of-truth without changes.
7. **Tail behaviour difference between paths** — legacy `align_to_grid` *trims trailing silence then snaps*; legacy `align_for_loop` *returns exact length*; v2 always returns exact length. Confirm new module unifies on exact-length output (the task spec requires it).
