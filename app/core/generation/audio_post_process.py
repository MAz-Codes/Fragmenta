"""Beat-align and tempo-conform a generated WAV to a target BPM and bar count.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def align_to_grid(
    input_path: Path,
    target_bpm: float,
    target_bars: int,
    beats_per_bar: int = 4,
) -> Path:
    audio, sr = sf.read(str(input_path), always_2d=True)
    audio = audio.astype(np.float32, copy=False)
    target_samples = int(round(target_bars * beats_per_bar * 60.0 / target_bpm * sr))

    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]

    detected_bpm, first_beat = _detect_grid_anchor(mono, sr)

    head_offset = 0
    if first_beat is not None and 0 < first_beat < sr * 1.5:
        head_offset = first_beat
        logger.info(f"align_to_grid: trimmed {head_offset / sr * 1000:.1f} ms to first beat")
    elif first_beat is None:
        head_offset = _detect_first_onset_sample(mono, sr)
        if head_offset > 0:
            logger.info(f"align_to_grid: trimmed {head_offset / sr * 1000:.1f} ms (onset fallback)")

    if head_offset > 0:
        audio = audio[head_offset:]
        mono = mono[head_offset:]

    if detected_bpm is not None:
        rate = target_bpm / detected_bpm
        if 0.7 <= rate <= 1.4:
            audio = _time_stretch_multichannel(audio, rate)
            logger.info(
                f"align_to_grid: detected {detected_bpm:.2f} BPM, "
                f"stretched by {rate:.4f} to match target {target_bpm:.2f} BPM"
            )
        else:
            logger.info(
                f"align_to_grid: detected {detected_bpm:.2f} BPM out of safe stretch "
                f"range vs target {target_bpm:.2f}; skipping warp"
            )
    else:
        logger.info("align_to_grid: no usable tempo detected; skipping warp")

    if audio.shape[0] > target_samples:
        audio = audio[:target_samples]
    elif audio.shape[0] < target_samples:
        pad = np.zeros((target_samples - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
        audio = np.concatenate([audio, pad], axis=0)

    sf.write(str(input_path), audio, sr, subtype="PCM_16")
    return input_path


def _detect_first_onset_sample(mono: np.ndarray, sr: int) -> int:
    """Return the sample index of the first detected onset, or 0 if none found."""
    try:
        onsets = librosa.onset.onset_detect(
            y=mono, sr=sr, units="samples", backtrack=True
        )
    except Exception as exc:
        logger.warning(f"onset detection failed: {exc}")
        return 0
    if onsets is None or len(onsets) == 0:
        return 0
    first = int(onsets[0])
    if first > sr * 1.0:
        return 0
    return first


def _detect_grid_anchor(mono: np.ndarray, sr: int) -> Tuple[Optional[float], Optional[int]]:
    try:
        tempo, beats = librosa.beat.beat_track(y=mono, sr=sr, units="samples")
    except Exception as exc:
        logger.warning(f"beat tracking failed: {exc}")
        return None, None
    if beats is None or len(beats) < 4:
        return None, None
    bpm = float(np.atleast_1d(tempo).flatten()[0])
    if not (40.0 <= bpm <= 240.0):
        return None, None
    return bpm, int(beats[0])


def _time_stretch_multichannel(audio: np.ndarray, rate: float) -> np.ndarray:
    """Phase-vocoder time stretch, applied per channel and re-stacked."""
    stretched = librosa.effects.time_stretch(audio.T, rate=rate)
    return np.ascontiguousarray(stretched.T)
