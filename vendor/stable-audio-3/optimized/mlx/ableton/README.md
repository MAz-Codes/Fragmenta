# Ableton Live integration (Experimental!)

Insert `sa3` generations directly into Ableton Live at the current playhead position.

Two pieces work together:

| File | Where it runs |
|------|--------------|
| `AudioInserter/` | Inside Ableton — a MIDI Remote Script that listens on a local socket |
| `insert_audio.py` | On the command line — sends files to that socket |

## 1. Install the AudioInserter Remote Script

There are two places Ableton looks for Remote Scripts on Mac:

| Location | Notes |
|----------|-------|
| `~/Music/Ableton/User Library/Remote Scripts/` | **Recommended** — persists across Ableton updates |
| `/Applications/Ableton Live *.app/Contents/App-Resources/MIDI Remote Scripts/` | Works, but gets wiped when you update Ableton |

Install into the User Library (create the folder if it doesn't exist yet):

```bash
mkdir -p ~/Music/Ableton/User\ Library/Remote\ Scripts
cp -r AudioInserter ~/Music/Ableton/User\ Library/Remote\ Scripts/AudioInserter
```

Then in Ableton: **Preferences → MIDI → Control Surfaces** — add a new surface and choose **AudioInserter**. No MIDI port needed.

Verify it's running:

```bash
python3 insert_audio.py --ping
# ✓ Ableton is connected and ready
```

## 2. Insert audio

```bash
# Insert a specific file at the current playhead
python3 insert_audio.py /path/to/out.wav

# Insert whatever wav was most recently dropped on your Desktop
python3 insert_audio.py

# Watch a folder and auto-insert each new file as it appears
python3 insert_audio.py --watch ~/Desktop
```

The track is named after the file by default. If the env var `_SA3_LAST_PROMPT` is set, that prompt text is used as the track name instead — handy if you wrap `sa3` in a shell alias.

## 3. Typical workflow with sa3

```bash
# Generate
./sa3 --prompt "driving techno loop" --dit medium --decoder same-l --out out.wav

# Insert
python3 ableton/insert_audio.py out.wav
```

Or use watch mode: start it once and every `out.wav` you generate lands in Ableton automatically:

```bash
python3 ableton/insert_audio.py --watch /path/to/sa3/output/dir
```
