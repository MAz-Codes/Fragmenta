# Changelog

All notable changes to Fragmenta are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Pitch knob next to Generate on every Performance channel: transpose a clip
  up to an octave up or down in semitones without changing its length, so
  loops, cue points and Link timing stay put. Only what you hear changes,
  never the file. It rests at 0 st when you pass it; double-click to reset.
- Mac: choose the output device in Settings → Audio, like on Windows. Fragmenta
  makes it the Mac's sound output while it runs, so Main and Cue can go to
  different channel pairs of an audio interface; the previous output comes
  back when Fragmenta closes.

### Changed
- Ableton Link: loops stay on Live's beat over long sessions. Fragmenta
  corrects the slow drift between the sound card and Link with tiny speed
  adjustments you can't hear, and reads Link's timing from Link's own clock,
  so launches are steadier too.
- Linux: lower audio output latency (about 26 ms instead of 35 ms).
- Link fine-tune now reaches −200…+500 ms (enough for Bluetooth headphones),
  with ±1 ms buttons for precise steps.
- Stable Audio 3 code updated to Stability's October release. Adapters sound
  the same at full strength and when bypassed; strengths in between now also
  scale DoRA magnitudes, so they sound slightly different than in 1.2.
- Audio files are read the same way everywhere (datasets, training, edit
  uploads), without torchaudio's file loaders. m4a/AAC and AIFF clips are now
  used in training; before, they showed in the Dataset tab but were skipped.
- Runs on Python 3.14 with PyTorch 2.11 (was 3.11 / 2.7). The launchers look
  for 3.14 and the first start after updating rebuilds `venv/` by itself.

### Fixed
- Pre-encoding is much faster and lighter: each clip is encoded at its own
  length instead of the model's full window. Medium (SAME-L) pre-encoding
  now works without an NVIDIA GPU (CPU, Mac) in about 7 GB of memory;
  before, it could run out of memory.
- Clips without a prompt are left out of pre-encoding and training. Before,
  each one was replaced by a copy of another clip (so some clips counted
  twice) or trained with an empty prompt.
- Edit → Style transfer: base models no longer distort (at strength 0.9 they
  pushed tens of thousands of samples into clipping), and the strength slider
  now works across its whole range. Before, anything above about 0.6 ignored
  the source; now it moves evenly from keeping the source (0) to following
  the prompt (1), and starts halfway, at 0.5.
- m4a/AAC clips: the dataset health check now sees their length, and long
  ones can be sliced.
- Training on a dataset whose files can't be read stops with a clear error
  instead of retrying forever.

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

[1.2.0]: https://github.com/MAz-Codes/Fragmenta/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/MAz-Codes/Fragmenta/compare/v1.1.0...v1.1.1
