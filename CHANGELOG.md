# Changelog

All notable changes to Fragmenta are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [1.3.0] - 2026-11-02

### Added
- New look across the whole app, with a sidebar, a top toolbar and a new colour
  palette. The window scales to fit your screen.
- Apple Silicon: Small - Music, Small - SFX and Medium run on MLX, with no
  account needed and far less memory than PyTorch. Medium now works on a Mac.
  LoRAs trained on a Base model work on them; training still uses PyTorch.
- First run: with no model installed, Fragmenta offers one to download with no
  account. Read the licenses, tick to agree, then download with pause and resume.
- Settings → Import: bring models, LoRAs, projects, generated audio and your
  CLAP vocabulary over from an older Fragmenta folder. Nothing gets overwritten.
- New Settings window with Audio, MIDI and About pages.
- Pitch knob on every Performance channel: transpose a clip up to an octave
  either way without changing its length. Double-click to reset.
- New on Mac: choose the output device in Settings → Audio, and send Main and Cue to
  different channels of an audio interface.
- Generated audio and performance recordings now carry signed metadata (C2PA
  Content Credentials), made on your computer. See `docs/AI_TRANSPARENCY.md`.
- Training monitor shows time left and speed.

### Changed
- Runs on Python 3.14 with PyTorch 2.11 (was 3.11 / 2.7). The launchers look for
  3.14, and the first start after updating rebuilds `venv/`.
- Every dependency is pinned with its checksum in `locks/`. Each install writes
  a log to `logs/install-*.log`.
- Checkpoint Manager: a wider window with a card for every model. Hover over an
  installed model to delete it, and click Storage to see what's using your disk.
- Ableton Link: loops stay on Live's beat over long sessions, and launches are
  steadier.
- Link fine-tune reaches −200…+500 ms, with ±1 ms buttons.
- Linux: lower audio output latency (about 26 ms, was 35 ms).
- Performance: model, LoRA, steps and seed moved into one pill in the top bar.
  The fragment list shows the newest on top.
- Base models start at 50 steps, not 250.
- Tabs and the top toolbar lock while training runs to avoid crashes.
- Stable Audio 3 code updated to Stability's October release. Adapter strengths
  between bypassed and full sound slightly different than in 1.2.

### Fixed
- The backend only answers requests addressed to localhost and refuses requests
  from other websites. Project routes refuse names that aren't plain folder names.
- Pre-encoding is much faster and lighter. Medium now pre-encodes without an
  NVIDIA GPU, in about 7 GB of memory.
- Clips without a prompt are left out of pre-encoding and training. They used to
  be swapped for another clip or trained with an empty prompt.
- m4a/AAC and AIFF clips are now used in training. They used to show in the
  Dataset tab and then get skipped. Long m4a/AAC clips can be sliced too.
- Edit → Style transfer: base models no longer distort, and the strength slider
  works across its whole range. It starts at 0.5.
- Edit → Extend: the new part keeps playing instead of fading out or coming out
  silent.
- Edit → Inpaint: the region starts at nearly the whole clip, so you hear the
  prompt. Drag it smaller once you have the sound.
- Training stops with a clear error when dataset files can't be read, instead of
  retrying forever. A failed run shows the real error, and a finished run says
  how it ended.
- Training and pre-encoding no longer die at the start when the terminal that
  launched Fragmenta has been closed.
- Mac: the toolbar no longer says "Not detected · CPU mode". It shows free unified
  memory, and Free GPU memory works there too.
- LAION-CLAP only shows as installed once its download has finished. Its real size
  shows in the Checkpoint Manager, and Delete and Cancel remove it from disk.
- Cancel stops a Checkpoint Manager download right away.
- Every page's player follows the output picked in Settings.

## [1.2.0] - 2026-10-27

### Added
- Cue point for Performance channels: set where a launch or retrigger enters
  the clip, while repeats return to the loop start. Sharper waveforms and a
  playhead too.
- Edit existing is back on the Generation page: style transfer, inpaint and
  extend. Drag a fragment's name onto it to use it as the source.
- Tighter Ableton Link sync: Fragmenta now plays early by its own output
  latency, so launches land on your peers' beat. Fine-tune per output, or
  calibrate by ear against Live's metronome.
- Base models download and run without a Hugging Face account; the Checkpoint
  Manager marks them "No account needed" and shows correct download sizes.
- Update notices: Fragmenta checks GitHub once a day and tells you when a new
  version is out. Nothing personal is sent; turn it off in About.
- Generated Fragments: click or drag a waveform to scrub, with a smooth
  playhead.
- Pre-encoding shows what it's doing: loading the encoder, which clip, time
  elapsed and time left, on both the Training and Dataset pages. FLAC
  and other formats are counted too, not just WAV.
- Model licence texts and notices ship with the app, and About and the welcome
  screen show "Powered by Stability AI".
- AI transparency note (`docs/AI_TRANSPARENCY.md`): how outputs are marked
  and how to check a file.

### Fixed
- The backend no longer crashes when the terminal that launched it is closed.
- The Dataset tab no longer shows "Project not found" when the last project
  you opened is gone; it opens empty instead.
- Download progress and the Checkpoint Manager's storage sizes no longer
  count files twice (the bar could run past 100%).
- Pre-encoding for the medium model works after a cancelled or small-model
  run; the Training page only shows "Pre-encoded" for latents that match the
  chosen base.
- Pre-encoding works with only the base models downloaded (no Hugging Face
  account); it used to look only in the account-only models for the encoder.
- The app window opens at a sensible size in your screen's shape on every
  platform: it was oversized on Macs and on scaled Windows displays.
- The launcher restarts a backend left running from another Fragmenta folder
  or from before an update, instead of reusing its old code.
- A failed pre-encode shows the encoder's actual error, on the Training page
  too.

## [1.1.1] - 2026-10-22

### Added
- Generated audio and performance recordings are now marked as AI-generated
  in their metadata (WAV INFO tags with the IPTC digital source type), as the
  EU AI Act's transparency rules (Art. 50) require. Check with `ffprobe <file>`.

[1.3.0]: https://github.com/MAz-Codes/Fragmenta/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/MAz-Codes/Fragmenta/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/MAz-Codes/Fragmenta/compare/v1.1.0...v1.1.1
