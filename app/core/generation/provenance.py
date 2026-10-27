"""Machine-readable AI-output marking for exported audio.

EU AI Act Art. 50(2) requires synthetic audio to be marked "in a
machine-readable format and detectable as artificially generated". Every
file Fragmenta writes to output/ carries that marking in its metadata:
WAV LIST/INFO chunks (INAM / ICMT / ISFT via libsndfile), which ffprobe,
DAWs and tag editors read. The comment embeds the IPTC digital source type
URI so a machine can tell AI output apart from free text.

These tags are unsigned — a good-faith Art. 50(2) baseline, not the Code of
Practice's signed-metadata measure (see redesign/FRAGMENTA_2_0_MASTER_PLAN.md).
"""
import os
from pathlib import Path
from typing import Iterable

import soundfile as sf

from utils.logger import get_logger

logger = get_logger("Provenance")

# IPTC NewsCodes digital source types (the vocabulary C2PA uses too).
# A generation is pure model output; a performance recording is a live mix
# of generated fragments, so it's a composite that includes AI media.
SOURCE_GENERATED = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
SOURCE_COMPOSITE = "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"

# Read once at import, like app.py's _APP_VERSION: the tags describe the code
# this process is running.
try:
    _APP_VERSION = (Path(__file__).resolve().parents[3] / "VERSION").read_text().strip() or "unknown"
except Exception:
    _APP_VERSION = "unknown"

# Lossless dtype per PCM subtype when a file is rewritten to add tags.
_READ_DTYPE = {"PCM_16": "int16", "PCM_24": "int32", "PCM_32": "int32",
               "FLOAT": "float32", "DOUBLE": "float64"}


def _adapter_label(path: str) -> str:
    """Trained adapters live at <run>/checkpoints/<step>.safetensors — the run
    name is what the user recognises, the step tells checkpoints apart."""
    p = Path(path)
    if p.parent.name == "checkpoints":
        return f"{p.parent.parent.name}/{p.stem}"
    return p.stem


def generation_tags(model_id: str, adapters: Iterable[str] = ()) -> dict:
    """Tags for a file written straight from model output."""
    names = ", ".join(_adapter_label(a) for a in adapters)
    return {
        "title": "AI-generated audio",
        "comment": (f"AI-generated with Fragmenta {_APP_VERSION}; model={model_id}; "
                    f"adapters=[{names}]; digital_source_type={SOURCE_GENERATED}"),
        "software": f"Fragmenta {_APP_VERSION}",
    }


def recording_tags() -> dict:
    """Tags for a performance recording (a live mix of generated fragments)."""
    return {
        "title": "Performance recording with AI-generated audio",
        "comment": (f"Contains AI-generated audio; recorded with Fragmenta {_APP_VERSION}; "
                    f"digital_source_type={SOURCE_COMPOSITE}"),
        "software": f"Fragmenta {_APP_VERSION}",
    }


def _apply(f: sf.SoundFile, tags: dict) -> None:
    # Set before any audio is written — the safe order for libsndfile's
    # string chunks.
    for key, value in tags.items():
        setattr(f, key, value)


def write_tagged_wav(path: Path, pcm, samplerate: int, tags: dict,
                     subtype: str = "PCM_16") -> None:
    """sf.write() plus provenance tags. `pcm` is (samples, channels)."""
    channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    with sf.SoundFile(str(path), "w", samplerate=samplerate, channels=channels,
                      subtype=subtype, format="WAV") as f:
        _apply(f, tags)
        f.write(pcm)


def tag_existing_wav(path: Path, tags: dict) -> bool:
    """Rewrite an audio file in place with provenance tags, losslessly.

    For files that reach disk without passing through write_tagged_wav —
    performance recordings are encoded in the browser and uploaded as raw
    bytes. Streams in blocks (recordings can be long) into a temp file, then
    swaps it in atomically. Returns False (file left untouched) if libsndfile
    can't read it.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tagging")
    try:
        with sf.SoundFile(str(path)) as src:
            dtype = _READ_DTYPE.get(src.subtype, "float64")
            with sf.SoundFile(str(tmp), "w", samplerate=src.samplerate,
                              channels=src.channels, subtype=src.subtype,
                              format=src.format) as dst:
                _apply(dst, tags)
                for block in src.blocks(blocksize=65536, dtype=dtype, always_2d=True):
                    dst.write(block)
        os.replace(tmp, path)
        return True
    except Exception as exc:
        logger.warning(f"Could not add provenance tags to {path.name}: {exc}")
        try:
            tmp.unlink()
        except OSError:
            pass
        return False
