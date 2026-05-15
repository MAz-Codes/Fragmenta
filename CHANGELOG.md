# Changelog

All notable changes to Fragmenta Desktop are documented here. Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] — 2026-05-15

Significant feature release on the 0.1 line. Linux-validated; cross-platform (Windows / macOS) validation pending — a future 0.2.0 will mark that milestone.

### Added

- **LoRA training and inference end-to-end.** Train a small (~60 MB) adapter on top of Stable Audio Open 1.0, then apply it at inference time on the Generation tab or in Performance Mode. Adapter rank, alpha, dropout, and inference multiplier are all configurable.
- **Per-checkpoint LoRA picker.** Each saved checkpoint of a LoRA (e.g. step 200, 400, … 2000) is selectable for inference, so you can A/B early-vs-late training states.
- **"Suggest hyperparameters" button** on the Training tab. Reads the dataset (file count + total duration via `ffprobe`) and detected VRAM, then proposes batch size, learning rate, rank, alpha, and epochs based on a small-vs-medium-vs-large heuristic. Rationale is shown in a collapsible "Why these values?" panel.
- **Audio Setup menu** in the Performance tab toolbar. Lists OS-enumerated output devices and binds the engine's `AudioContext` to the chosen device.
- **Main / Cue output channel-pair routing** (Stage 2 plumbing). `ChannelSplitter → ChannelMerger → destination` graph in both the engine and cue contexts; pair selectors populate from `destination.maxChannelCount`. Functional on macOS / Windows via CoreAudio / WASAPI; on Linux / PipeWire, Chromium's WebAudio backend caps `maxChannelCount` at 2, so the dropdowns only show Ch 1-2 there.
- **CSV + audio folder dataset import.** Bulk-load a CSV of `file_name,prompt` rows plus a folder of audio with conflict-status preview (new / replace / skip).
- **Per-item delete buttons** on the Generation tab and Performance tab dropdowns. Removes the entire fine-tuned model or LoRA directory after a confirmation prompt.

### Changed

- **Relicensed from Apache 2.0 to GNU AGPL v3.0.** Strong copyleft + network clause: forks, derivatives, and hosted services must publish source under the same terms. Alternative commercial licensing available on request from the copyright holder.
- **Stability AI Community License compliance.** Added a verbatim copy of the SACL at `vendor/STABLE_AUDIO.md`, the required attribution string in `NOTICE.md`, and "Powered by Stability AI" in the README and the in-app About dialog.
- **Third-party code consolidated under `vendor/`.** `stable-audio-tools/`, `loraw_vendor/`, and the SACL all moved out of the repo root.
- **Vendored LoRAW patched with 200-step linear LR warmup** in `configure_optimizers`. Stabilizes Adam's early-run moment estimates and makes higher alpha / LR experiments safe.
- **Performance tab redesigned.** Model + checkpoint + LoRA pickers moved from the top strip to the bottom bar (left), with stable-width slots so picking a LoRA no longer reflows the layout. Compact typography throughout (BPM input, STEPS, SEED, AUTO BPM, all dropdowns).
- **About dialog rebrand.** Fragmenta logo + brand-font title at the top; required SACL attribution as small footer text.
- **README:** added LoRA section, training mode comparison table, CSV import to the dataset workflow; dropped Stable Audio Open Small from training (kept as inference-only — distilled models mode-collapse under standard diffusion fine-tuning).

### Fixed

- LoRA checkpoint discovery now correctly counts checkpoints saved under the nested `<save_dir>/<wandb_id>/checkpoints/` layout (previously reported 0 checkpoints despite files on disk).
- LoRA inference no longer fails with `KeyError: 'model/model/transformer/...'` — the engine now unwraps `torch.compile`'s `OptimizedModule` before passing the model to LoRAW's `scan_model`.
- `custom_metadata.py` lookup paths updated for the new `vendor/stable-audio-tools/` location (previously trained on empty prompts after the vendor move).
- Inference steps slider restores to 250 when leaving the distilled small model (previously stuck at 8, producing noise from the full base model).
- `LICENSE-stable-audio.md` and other dotfile state in `data/` are now wiped by "Start Fresh".

### Performance / Internal

- Removed the SAO model_config demo-audio callback from LoRA runs (was burning ~3 minutes per run on demo generation users never listen to).
- `data/.duration_cache.json` caches `ffprobe` duration sums between Suggest-hyperparameters clicks; invalidates automatically on dataset change.
- LoRA model picker filters by `base_model` from `training_metadata.json` — only LoRAs trained against the currently-selected base appear.
- `/api/models` now excludes directories with `mode: "lora"` in their metadata, so LoRAs don't show up as fine-tuned models in the main dropdown.

### Known Limitations

- Multichannel output routing is capped at 2 channels on Linux + PipeWire due to a limitation in Chromium's WebAudio backend. Windows (WASAPI) and macOS (CoreAudio) expose the device's real channel count and the existing code paths work without modification. Verified on Linux; pending validation on the other two platforms.
- Diffusion training loss is per-timestep variance-dominated; the chart's smoothed line is honest but rarely shows clear convergence for LoRA at this scale. The validation signal is auditory — generate same-prompt-same-seed with and without the LoRA.

## [0.1.1]

Initial public release line. See `git log v0.1.1` (and earlier) for history.
