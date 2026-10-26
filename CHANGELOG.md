# Changelog

All notable changes to Fragmenta are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [1.2.0] - Unreleased

### Added
- Cue point for Performance channels: set where a launch or retrigger enters
  the clip, while repeats return to the loop start. Sharper waveforms and a
  playhead too.
- Tighter Ableton Link sync: Fragmenta now plays early by its own output
  latency, so launches land on your peers' beat. Fine-tune per output, or
  calibrate by ear against Live's metronome.
- Base models download and run without a Hugging Face account; the Checkpoint
  Manager marks them "No account needed" and shows correct download sizes.
- Update notices: Fragmenta checks GitHub once a day and tells you when a new
  version is out. Nothing personal is sent; turn it off in About.
- Model licence texts and notices ship with the app, and About and the welcome
  screen show "Powered by Stability AI".
- AI transparency note (`docs/AI_TRANSPARENCY.md`): how outputs are marked
  and how to check a file.
- Edit existing is back on the Generation page: style transfer, inpaint and
  extend. Drag a fragment's name onto it to use it as the source.
- Generated Fragments: click or drag a waveform to scrub, with a smooth
  playhead.
- Pre-encoding shows what it's doing: loading the encoder, which clip, time
  elapsed and time left, on both the Training and Dataset pages. FLAC
  and other formats are counted too, not just WAV.

### Fixed
- The backend no longer crashes when the terminal that launched it is closed.
- The Dataset tab no longer shows "Project not found" when the last project
  you opened is gone; it opens empty instead.
- Download progress and the Checkpoint Manager's storage sizes no longer
  count files twice (the bar could run past 100%).
- Pre-encoding for the medium model works after a cancelled or small-model
  run; the Training page only shows "Pre-encoded" for latents that match the
  chosen base.
- Windows: the taskbar shows the Fragmenta icon instead of Python's.
- The launcher restarts a backend left running from another Fragmenta folder
  or from before an update, instead of reusing its old code.
- A failed pre-encode shows the encoder's actual error, on the Training page
  too.

## [1.1.1] - 2026-10-22

### Added
- Generated audio and performance recordings are now marked as AI-generated
  in their metadata (WAV INFO tags with the IPTC digital source type), as the
  EU AI Act's transparency rules (Art. 50) require. Check with `ffprobe <file>`.

[1.2.0]: https://github.com/MAz-Codes/Fragmenta/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/MAz-Codes/Fragmenta/compare/v1.1.0...v1.1.1
