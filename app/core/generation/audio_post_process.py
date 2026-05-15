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


# Safe range for phase-vocoder time-stretching. Wider than the previous
# [0.7, 1.4] so we actually warp in more cases — librosa's vocoder produces
# acceptable audio across this range for music, and the alternative
# (no warp at all) drifts off the grid completely on loop.
_STRETCH_SAFE_MIN = 0.6
_STRETCH_SAFE_MAX = 1.7


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

    # Pass target_bpm as a prior to librosa — biases the beat tracker away
    # from half-time / double-time interpretations of the same grid.
    detected_bpm, first_beat = _detect_grid_anchor(mono, sr, start_bpm=target_bpm)

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
        rate, effective_bpm = _best_stretch_rate(detected_bpm, target_bpm)
        if rate is not None:
            if abs(rate - 1.0) > 1e-3:
                audio = _time_stretch_multichannel(audio, rate)
            interp_note = (
                f" (interpreted as {effective_bpm:.2f} BPM, "
                f"octave={effective_bpm / detected_bpm:.2f}×)"
                if abs(effective_bpm - detected_bpm) > 1e-2
                else ""
            )
            logger.info(
                f"align_to_grid: detected {detected_bpm:.2f} BPM{interp_note}, "
                f"stretched by {rate:.4f} to match target {target_bpm:.2f} BPM"
            )
        else:
            logger.info(
                f"align_to_grid: detected {detected_bpm:.2f} BPM has no safe "
                f"interpretation vs target {target_bpm:.2f}; skipping warp"
            )
    else:
        logger.info("align_to_grid: no usable tempo detected; skipping warp")

    if audio.shape[0] > target_samples:
        audio = audio[:target_samples]
        # 8ms tail fade prevents the click at the loop boundary when the
        # truncation point lands mid-waveform.
        fade_samples = min(int(0.008 * sr), audio.shape[0])
        if fade_samples > 1:
            fade = np.linspace(1.0, 0.0, fade_samples, dtype=audio.dtype)
            audio[-fade_samples:] *= fade[:, np.newaxis] if audio.ndim > 1 else fade
    elif audio.shape[0] < target_samples:
        pad = np.zeros((target_samples - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
        audio = np.concatenate([audio, pad], axis=0)

    sf.write(str(input_path), audio, sr, subtype="PCM_16")
    return input_path


def _best_stretch_rate(
    detected_bpm: float,
    target_bpm: float,
) -> Tuple[Optional[float], float]:
    """Pick the time-stretch rate that maps detected → target, considering
    half-time and double-time interpretations of the detected tempo. Returns
    (rate, effective_bpm) where effective_bpm is the (possibly octave-
    corrected) interpretation that was chosen, or (None, detected_bpm) if
    nothing safe is available.

    Order of preference:
      1. Detected as-is, if it lands inside the safe stretch range.
      2. Octave-corrected (detected × 0.5 or × 2.0), only when the as-is
         interpretation is out of range. This is the librosa half-/double-
         time error recovery path.

    This biases the algorithm toward honesty: only re-interpret the
    detector's reading when it can't otherwise produce a usable stretch.
    """
    # First, try the detector's reading at face value.
    rate_asis = target_bpm / detected_bpm
    if _STRETCH_SAFE_MIN <= rate_asis <= _STRETCH_SAFE_MAX:
        return rate_asis, detected_bpm

    # As-is is out of safe range — almost certainly a librosa octave error.
    # Try the half-time and double-time reinterpretations and pick whichever
    # is closest to a no-op stretch.
    candidates = []
    for octave_factor in (0.5, 2.0):
        interpreted = detected_bpm * octave_factor
        rate = target_bpm / interpreted
        if _STRETCH_SAFE_MIN <= rate <= _STRETCH_SAFE_MAX:
            candidates.append((abs(rate - 1.0), rate, interpreted))
    if not candidates:
        return None, detected_bpm
    candidates.sort()
    _, best_rate, best_interp = candidates[0]
    return best_rate, best_interp


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


def _detect_grid_anchor(
    mono: np.ndarray,
    sr: int,
    start_bpm: Optional[float] = None,
) -> Tuple[Optional[float], Optional[int]]:
    """Run librosa beat tracking with the target tempo as a prior. Passing
    start_bpm reduces (but doesn't eliminate) half-time / double-time errors.
    The octave-correction in _best_stretch_rate handles whatever librosa
    still gets wrong."""
    try:
        kwargs = {"y": mono, "sr": sr, "units": "samples"}
        if start_bpm is not None and start_bpm > 0:
            kwargs["start_bpm"] = float(start_bpm)
        tempo, beats = librosa.beat.beat_track(**kwargs)
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
