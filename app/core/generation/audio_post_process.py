"""Beat-align and tempo-conform a generated WAV to a target BPM and bar count.

Pipeline (in order):
  1. Detect tempo + beat grid via librosa (with target BPM as prior).
  2. Head-trim to the first detected beat (or first onset as fallback),
     followed by a 3 ms equal-power fade-in to mask the trim seam.
  3. Tempo-conform via phase-vocoder time-stretch, but ONLY when the
     detected tempo is meaningfully off (>2 %) — skipping the stretch when
     already close preserves transients.
  4. End-anchored truncation: snap the cut to the nearest detected beat
     within ±½ beat of the mathematical target sample count, so loops
     don't end mid-note. Followed by an 8 ms equal-power fade-out so the
     loop seam doesn't click.
  5. Zero-pad if the audio came out shorter than the target.
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

# Don't bother time-stretching when the detected tempo is already within
# this fraction of target — the stretch artifacts (smeared transients)
# cost more than the sub-percent alignment gain.
_STRETCH_DEADBAND = 0.02

# Fade durations applied at trim points.
_HEAD_FADE_SEC = 0.003   # mask click at the trimmed head
_TAIL_FADE_SEC = 0.008   # mask click at the truncated tail / loop seam


def align_to_grid(
    input_path: Path,
    target_bpm: float,
    target_bars: int,
    beats_per_bar: int = 4,
) -> Path:
    audio, sr = sf.read(str(input_path), always_2d=True)
    audio = audio.astype(np.float32, copy=False)
    samples_per_beat = sr * 60.0 / float(target_bpm)
    target_samples = int(round(target_bars * beats_per_bar * samples_per_beat))

    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]

    detected_bpm, beat_samples = _detect_grid(mono, sr, start_bpm=target_bpm)

    # --- Head trim ---------------------------------------------------------
    head_offset = 0
    if beat_samples is not None and len(beat_samples) > 0:
        first_beat = int(beat_samples[0])
        if 0 < first_beat < sr * 1.5:
            head_offset = first_beat
            logger.info(f"align_to_grid: trimmed {head_offset / sr * 1000:.1f} ms to first beat")
    elif beat_samples is None:
        head_offset = _detect_first_onset_sample(mono, sr)
        if head_offset > 0:
            logger.info(f"align_to_grid: trimmed {head_offset / sr * 1000:.1f} ms (onset fallback)")

    if head_offset > 0:
        audio = audio[head_offset:]
        mono = mono[head_offset:]
        if beat_samples is not None:
            shifted = np.asarray(beat_samples, dtype=np.int64) - head_offset
            beat_samples = shifted[shifted > 0]
        # Head fade-in: 3 ms equal-power so the trim seam doesn't click.
        _apply_fade(audio, _HEAD_FADE_SEC, sr, fade_in=True)

    # --- Tempo conform -----------------------------------------------------
    if detected_bpm is not None:
        rate, effective_bpm = _best_stretch_rate(detected_bpm, target_bpm)
        if rate is not None and abs(rate - 1.0) > _STRETCH_DEADBAND:
            audio = _time_stretch_multichannel(audio, rate)
            # Beats have moved — re-detect from the warped audio so the
            # end-snap step below sees current beat positions.
            mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
            _, beat_samples = _detect_grid(mono, sr, start_bpm=target_bpm)
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
        elif rate is not None:
            logger.info(
                f"align_to_grid: detected {detected_bpm:.2f} BPM is within "
                f"{_STRETCH_DEADBAND * 100:.0f}% of target {target_bpm:.2f}; "
                f"skipping stretch to preserve transients"
            )
        else:
            logger.info(
                f"align_to_grid: detected {detected_bpm:.2f} BPM has no safe "
                f"interpretation vs target {target_bpm:.2f}; skipping warp"
            )
    else:
        logger.info("align_to_grid: no usable tempo detected; skipping warp")

    # --- End-anchored truncation ------------------------------------------
    if audio.shape[0] > target_samples:
        end = _snap_to_beat(target_samples, beat_samples, samples_per_beat, audio.shape[0])
        audio = audio[:end]
        _apply_fade(audio, _TAIL_FADE_SEC, sr, fade_in=False)
    elif audio.shape[0] < target_samples:
        pad = np.zeros((target_samples - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
        audio = np.concatenate([audio, pad], axis=0)

    sf.write(str(input_path), audio, sr, subtype="PCM_16")
    return input_path


# --- helpers ---------------------------------------------------------------

def _snap_to_beat(
    target_samples: int,
    beat_samples: Optional[np.ndarray],
    samples_per_beat: float,
    audio_len: int,
) -> int:
    """Return the cut point: the nearest detected beat within ±½ beat of
    target_samples, falling back to target_samples itself if no beat is in
    range. Never overshoots audio length."""
    fallback = min(target_samples, audio_len)
    if beat_samples is None or len(beat_samples) == 0:
        return fallback
    tol = samples_per_beat * 0.5
    valid = beat_samples[(beat_samples > 0) & (beat_samples <= audio_len)]
    if len(valid) == 0:
        return fallback
    diffs = np.abs(valid - target_samples)
    idx = int(np.argmin(diffs))
    if diffs[idx] <= tol:
        return int(valid[idx])
    return fallback


def _apply_fade(audio: np.ndarray, duration_sec: float, sr: int, *, fade_in: bool) -> None:
    """In-place equal-power fade on the head (fade_in=True) or tail."""
    n = min(int(duration_sec * sr), audio.shape[0])
    if n <= 1:
        return
    ramp = _equal_power_ramp(n, fade_in=fade_in, dtype=audio.dtype)
    if audio.ndim > 1:
        ramp = ramp[:, np.newaxis]
    if fade_in:
        audio[:n] *= ramp
    else:
        audio[-n:] *= ramp


def _equal_power_ramp(n: int, *, fade_in: bool, dtype) -> np.ndarray:
    """Cosine-shaped equal-power fade. Energy at the midpoint is preserved
    when summing fade-out + fade-in of complementary segments, avoiding the
    perceptible 'duck' that linear ramps produce at loop seams."""
    t = np.linspace(0.0, np.pi / 2.0, n).astype(dtype, copy=False)
    return np.sin(t) if fade_in else np.cos(t)


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
    """
    rate_asis = target_bpm / detected_bpm
    if _STRETCH_SAFE_MIN <= rate_asis <= _STRETCH_SAFE_MAX:
        return rate_asis, detected_bpm

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


def _detect_grid(
    mono: np.ndarray,
    sr: int,
    start_bpm: Optional[float] = None,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Run librosa beat tracking with the target tempo as a prior. Returns
    (bpm, beat_samples_array). Passing start_bpm reduces (but doesn't
    eliminate) half-time / double-time errors; the octave-correction in
    _best_stretch_rate handles whatever librosa still gets wrong."""
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
    return bpm, np.asarray(beats, dtype=np.int64)


def _time_stretch_multichannel(audio: np.ndarray, rate: float) -> np.ndarray:
    """Phase-vocoder time stretch, applied per channel and re-stacked."""
    stretched = librosa.effects.time_stretch(audio.T, rate=rate)
    return np.ascontiguousarray(stretched.T)
