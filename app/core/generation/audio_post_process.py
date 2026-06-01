"""Beat-align and tempo-conform a generated WAV to a target BPM and bar count.

SA3 generates at the exact requested duration via variable-length flow
matching, so the post-processor's role is **drift correction**, not length
control: it only nudges the audio when librosa detects that the realised
tempo has drifted from the target. The tempo-conform gate is intentionally
tight — `|rate - 1| > 5%` AND `rate in [0.85, 1.15]` — so we never warp
audibly when SA3 was already close.

Pipeline (in order):
  1. Detect tempo + beat grid via librosa (with target BPM as prior).
  2. Head-trim to the first detected beat (or first onset as fallback),
     followed by a 3 ms equal-power fade-in to mask the trim seam.
  3. Tempo-conform via phase-vocoder time-stretch, ONLY when the detected
     tempo drifts >5% from target AND the resulting stretch lies inside
     the safe range [0.85, 1.15]. Outside this window we leave the audio
     alone and let the user re-roll.
  4. End-anchored truncation: snap the cut to the nearest detected beat
     within ±½ beat of the mathematical target sample count, so loops
     don't end mid-note. Followed by an 8 ms equal-power fade-out so the
     loop seam doesn't click.
  5. Zero-pad if the audio came out shorter than the target.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def beatsync_v2_enabled() -> bool:
    """Feature gate for the hardened Stage A pipeline (sample-exact length,
    first-transient-to-zero alignment, transient-preserving stretch).

    Off by default: with the flag unset, every Stage A function takes its
    legacy code path, so Bars-mode output is byte-identical to pre-flag
    builds and Seconds mode (which never enters Stage A at all) is unaffected.
    Enable with ``FRAGMENTA_BEATSYNC_V2=1``.
    """
    return os.environ.get("FRAGMENTA_BEATSYNC_V2", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Liberal module-default range for `_best_stretch_rate`. Kept wide so any
# future force-warp caller has room; the bars-mode drift-correction path
# (`align_to_grid`) overrides with tighter bounds below.
_STRETCH_SAFE_MIN = 0.6
_STRETCH_SAFE_MAX = 1.7

# Bars-mode drift correction. SA3 hits the requested duration exactly via
# variable-length generation, so the post-processor only kicks in when the
# detected tempo of the generated audio drifts from the requested target.
# Tight gates avoid audible vocoder artifacts when SA3 was already close.
_BARS_MODE_STRETCH_MIN = 0.85
_BARS_MODE_STRETCH_MAX = 1.15
_BARS_MODE_DEADBAND = 0.05

# Loop-mode (Phase 7) is stricter — a 5% tempo slack compounds visibly when
# multiple loop channels run side-by-side, even though loop iteration
# lengths are sample-exact. 0.5% is below librosa's noise floor for beat
# detection on rhythmic content, so we won't be acting on noise, but we
# WILL correct anything detectable that the looser bars-mode would skip.
_LOOP_MODE_DEADBAND = 0.005

# Fade durations applied at trim points. Kept very short — the fade is
# click-prevention, not a perceptible ramp. Performance Mode loops these
# clips, and longer fades audibly "duck" the loop seam.
_HEAD_FADE_SEC = 0.003   # mask click at the trimmed head
_TAIL_FADE_SEC = 0.003   # mask click at a mid-note truncation; skipped on beats

# Trailing-silence detection. SA3 occasionally pads a generation with low-
# level tail; the post-processor used to keep that and fade over it, which
# produced perceptible "silence + duck" at the loop point.
_SILENCE_THRESHOLD_DB = -50.0          # anything below is silence
_SILENCE_WINDOW_SEC = 0.05             # RMS window granularity
_SILENCE_TAIL_KEEP_SEC = 0.010         # leave a tiny natural decay

# v2 first-transient search: a downbeat lands within the first bar or two of
# generated content, so we never hunt past this window for the musical "1".
_V2_TRANSIENT_SEARCH_SEC = 1.5
_V2_STRONG_RATIO = 0.30                 # candidate must reach 30% of peak
_V2_RISE_RATIO = 0.15                   # rising-edge threshold for refinement
_V2_REFINE_WIN_SEC = 0.03               # +/- window for sample-accurate refine


# === Stage A v2 (FRAGMENTA_BEATSYNC_V2) ====================================
# A single hardened core shared by both align entry points. It enforces the
# locked invariants directly instead of relying on librosa's beat[0] for
# phase and on end-snap/silence-trim for length:
#   * tempo conform with a transient-preserving stretch (rubberband, librosa
#     fallback) — gen-time warp only, no live tracking (decision: v1);
#   * align the first STRONG transient to sample 0 (rotate-free head trim) so
#     two independently-correct clips share a downbeat with zero per-clip code;
#   * crop to the exact target sample count — overgenerate-then-trim, never
#     zero-pad in the common path (pad only as a logged last resort).

def _stage_a_v2(
    audio: np.ndarray,
    sr: int,
    *,
    target_samples: int,
    target_bpm: float,
    deadband: float,
) -> np.ndarray:
    """Hardened Stage A core. Input/return: float32 ``[T, C]``.

    Order: tempo-conform -> first-strong-transient to sample 0 -> exact crop.
    """
    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    detected_bpm, _beats = _detect_grid(mono, sr, start_bpm=target_bpm)

    # --- tempo conform (transient-preserving, gen-time) -------------------
    if detected_bpm is not None:
        rate, eff = _best_stretch_rate(
            detected_bpm, target_bpm,
            safe_min=_BARS_MODE_STRETCH_MIN, safe_max=_BARS_MODE_STRETCH_MAX,
        )
        if rate is not None and abs(rate - 1.0) > deadband:
            audio = _transient_stretch(audio, rate, sr)
            mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
            logger.info(
                "stage_a_v2: detected %.2f BPM (eff %.2f), transient-stretched "
                "by %.4f to %.2f target", detected_bpm, eff, rate, target_bpm,
            )
        else:
            logger.info(
                "stage_a_v2: detected %.2f BPM within %.2f%% of %.2f target; "
                "no stretch", detected_bpm, deadband * 100, target_bpm,
            )
    else:
        logger.info("stage_a_v2: no usable tempo detected; skipping stretch")

    # --- first strong transient -> sample 0 (INV#4, enables INV#9) --------
    d = _first_strong_transient(mono, sr)
    if d > 0:
        audio = audio[d:]
        logger.info("stage_a_v2: trimmed %.1f ms to first strong transient",
                    d / sr * 1000)

    # --- exact length, no tail pad in the common path (INV#2, INV#3) ------
    if audio.shape[0] >= target_samples:
        audio = audio[:target_samples]
    else:
        pad = target_samples - audio.shape[0]
        logger.warning(
            "stage_a_v2: content short by %d samp (%.0f ms) after trim — "
            "padding as a last resort; raise generation headroom or re-roll",
            pad, pad / sr * 1000,
        )
        audio = np.concatenate(
            [audio, np.zeros((pad, audio.shape[1]), dtype=np.float32)], axis=0,
        )
    return np.ascontiguousarray(audio, dtype=np.float32)


def _first_strong_transient(mono: np.ndarray, sr: int) -> int:
    """Sample index of the first STRONG transient, refined to the rising edge.

    Two-stage so we neither latch onto low-level noise nor lose sample
    accuracy to librosa's 512-sample hop:
      1. librosa onset candidates; take the first whose local peak reaches
         ``_V2_STRONG_RATIO`` of the search-window peak;
      2. refine within a small window to the first sample crossing
         ``_V2_RISE_RATIO`` of that local peak — the attack's true start.
    Returns 0 when the clip is silent or no strong transient is found.
    """
    n = len(mono)
    search = min(n, int(sr * _V2_TRANSIENT_SEARCH_SEC))
    if search <= 0:
        return 0
    peak = float(np.max(np.abs(mono[:search])))
    if peak <= 1e-6:
        return 0

    try:
        onsets = librosa.onset.onset_detect(
            y=mono, sr=sr, units="samples", backtrack=True
        )
    except Exception as exc:
        logger.warning("v2 onset detection failed: %s", exc)
        onsets = None

    cand: Optional[int] = None
    if onsets is not None and len(onsets) > 0:
        look = int(sr * 0.05)
        for o in np.asarray(onsets, dtype=np.int64):
            if o >= search:
                break
            lo, hi = int(o), min(n, int(o) + look)
            if float(np.max(np.abs(mono[lo:hi]))) >= _V2_STRONG_RATIO * peak:
                cand = int(o)
                break

    if cand is None:
        # No qualifying onset — fall back to the first sample that crosses a
        # fraction of the window peak (handles smooth/pad content).
        idx = np.flatnonzero(np.abs(mono[:search]) >= _V2_STRONG_RATIO * peak)
        return int(idx[0]) if len(idx) else 0

    win = int(sr * _V2_REFINE_WIN_SEC)
    lo = max(0, cand - win)
    hi = min(n, cand + win)
    local_peak = float(np.max(np.abs(mono[lo:hi]))) or peak
    seg = np.abs(mono[lo:hi])
    above = np.flatnonzero(seg >= _V2_RISE_RATIO * local_peak)
    return int(lo + above[0]) if len(above) else cand


def _transient_stretch(audio: np.ndarray, rate: float, sr: int) -> np.ndarray:
    """Time-stretch preserving transients (INV#5).

    Prefers RubberBand in crisp-transient mode via pyrubberband; falls back to
    the librosa phase vocoder when the rubberband CLI / wrapper is unavailable,
    so hosts without the binary still work (just without transient mode)."""
    if abs(rate - 1.0) < 1e-9:
        return audio
    try:
        import pyrubberband as pyrb  # type: ignore
        out = pyrb.time_stretch(
            audio, sr, rate, rbargs={"--transients": "crisp"}
        )
        return np.ascontiguousarray(out.astype(np.float32))
    except Exception as exc:
        logger.info(
            "rubberband unavailable (%s); using librosa phase vocoder", exc,
        )
        return _time_stretch_multichannel(audio, rate)


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

    if beatsync_v2_enabled():
        out = _stage_a_v2(
            np.ascontiguousarray(audio), sr,
            target_samples=target_samples, target_bpm=float(target_bpm),
            deadband=_BARS_MODE_DEADBAND,
        )
        # 3 ms head fade-in masks any click at the new sample-0 transient.
        _apply_fade(out, _HEAD_FADE_SEC, sr, fade_in=True)
        sf.write(str(input_path), out, sr, subtype="PCM_16")
        logger.info("align_to_grid[v2]: %d samples (exact target %d)",
                    out.shape[0], target_samples)
        return input_path

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
        rate, effective_bpm = _best_stretch_rate(
            detected_bpm,
            target_bpm,
            safe_min=_BARS_MODE_STRETCH_MIN,
            safe_max=_BARS_MODE_STRETCH_MAX,
        )
        if rate is not None and abs(rate - 1.0) > _BARS_MODE_DEADBAND:
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
                f"{_BARS_MODE_DEADBAND * 100:.0f}% of target {target_bpm:.2f}; "
                f"skipping stretch to preserve transients"
            )
        else:
            logger.info(
                f"align_to_grid: detected {detected_bpm:.2f} BPM has no safe "
                f"interpretation vs target {target_bpm:.2f} within "
                f"[{_BARS_MODE_STRETCH_MIN:.2f}, {_BARS_MODE_STRETCH_MAX:.2f}]; "
                f"skipping warp (user re-roll recommended)"
            )
    else:
        logger.info("align_to_grid: no usable tempo detected; skipping warp")

    # --- Trim trailing silence --------------------------------------------
    # Done before end-snap so the snap operates on real audio, not on
    # beats that happen to fall inside a quiet tail.
    new_len = _trailing_audio_end(audio, sr)
    if new_len < audio.shape[0]:
        trimmed_ms = (audio.shape[0] - new_len) / sr * 1000
        logger.info(f"align_to_grid: trimmed {trimmed_ms:.0f} ms trailing silence")
        audio = audio[:new_len]
        if beat_samples is not None:
            beat_samples = beat_samples[beat_samples < new_len]

    # --- End-anchored truncation ------------------------------------------
    if audio.shape[0] > target_samples:
        end = _snap_to_beat(target_samples, beat_samples, samples_per_beat, audio.shape[0])
        cut_on_beat = beat_samples is not None and end in beat_samples.tolist()
        audio = audio[:end]
        if not cut_on_beat:
            # Mid-note cut — short fade hides the click. On a clean beat
            # boundary the cut is on a natural transient edge, so the fade
            # would only "duck" the start of the next beat at the loop
            # seam without preventing any audible click.
            _apply_fade(audio, _TAIL_FADE_SEC, sr, fade_in=False)
    # If we came in shorter than target, return the actual audio without
    # zero-padding. A 7.5-bar clip that loops cleanly beats an 8-bar clip
    # with 0.5 bars of silence at the loop seam.

    sf.write(str(input_path), audio, sr, subtype="PCM_16")
    return input_path


# --- Phase 7 loop alignment -----------------------------------------------

def align_for_loop(
    audio: np.ndarray,
    sr: int,
    *,
    target_samples: int,
    target_bpm: float,
) -> np.ndarray:
    """Align a baseline clip for seamless looping at an exact length.

    Pipeline (in-memory, no disk I/O):
      1. Detect tempo + beat grid via librosa.
      2. Time-stretch (uniformly) if detected BPM drifts past the bars-mode
         deadband AND the required rate is in the safe range. Drift
         beyond the safe range is left alone (caller can re-roll).
      3. Head-trim to the first detected beat (or first onset as fallback),
         within the first ~1.5 s. This is the phase-alignment step — it
         puts the loop's "downbeat" at sample 0 so multiple channels'
         beats coincide when launched on a bar boundary.
      4. Crop or zero-pad to exactly `target_samples`. No end-snap: the
         loop iteration length is sample-exact so it stays phase-locked
         to the master clock across iterations.

    Returns a `np.ndarray` of shape `(target_samples, channels)` (or 1-D
    if input was 1-D). The caller is expected to wrap-and-inpaint the
    output to smooth the seam — `align_for_loop` does no fade.
    """
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
        squeeze_out = True
    else:
        squeeze_out = False
    audio = np.ascontiguousarray(audio, dtype=np.float32)

    if beatsync_v2_enabled():
        out = _stage_a_v2(
            audio, sr,
            target_samples=target_samples, target_bpm=float(target_bpm),
            deadband=_LOOP_MODE_DEADBAND,
        )
        return out.squeeze(1) if squeeze_out else out

    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    detected_bpm, beat_samples = _detect_grid(mono, sr, start_bpm=target_bpm)

    # --- 1+2: tempo conform ---------------------------------------------
    if detected_bpm is not None:
        rate, effective_bpm = _best_stretch_rate(
            detected_bpm,
            target_bpm,
            safe_min=_BARS_MODE_STRETCH_MIN,
            safe_max=_BARS_MODE_STRETCH_MAX,
        )
        if rate is not None and abs(rate - 1.0) > _LOOP_MODE_DEADBAND:
            audio = _time_stretch_multichannel(audio, rate)
            mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
            _, beat_samples = _detect_grid(mono, sr, start_bpm=target_bpm)
            interp = (
                f" (interpreted as {effective_bpm:.2f} BPM)"
                if abs(effective_bpm - detected_bpm) > 1e-2 else ""
            )
            logger.info(
                "align_for_loop: detected %.2f BPM%s, stretched by %.4f to "
                "match %.2f target",
                detected_bpm, interp, rate, target_bpm,
            )
        elif rate is not None:
            logger.info(
                "align_for_loop: detected %.2f BPM within %.2f%% of %.2f target; "
                "no stretch",
                detected_bpm, _LOOP_MODE_DEADBAND * 100, target_bpm,
            )
        else:
            logger.info(
                "align_for_loop: detected %.2f BPM has no safe stretch to "
                "%.2f target within [%.2f, %.2f]; leaving tempo as-is",
                detected_bpm, target_bpm,
                _BARS_MODE_STRETCH_MIN, _BARS_MODE_STRETCH_MAX,
            )
    else:
        logger.info("align_for_loop: no usable tempo detected; skipping stretch")

    # --- 3: head-trim to first beat / onset (phase alignment) -----------
    head_offset = 0
    if beat_samples is not None and len(beat_samples) > 0:
        first_beat = int(beat_samples[0])
        if 0 < first_beat < sr * 1.5:
            head_offset = first_beat
    if head_offset == 0:
        # Onset fallback when beat tracking didn't lock — gives at least
        # a transient-aligned start instead of mid-attack on sample 0.
        head_offset = _detect_first_onset_sample(mono, sr)
        if head_offset >= sr * 1.5:
            head_offset = 0
    if head_offset > 0:
        audio = audio[head_offset:]
        logger.info(
            "align_for_loop: head-trimmed %.1f ms to first beat/onset",
            head_offset / sr * 1000,
        )

    # --- 4: crop or pad to exact target_samples -------------------------
    if audio.shape[0] > target_samples:
        audio = audio[:target_samples]
    elif audio.shape[0] < target_samples:
        pad = target_samples - audio.shape[0]
        audio = np.concatenate(
            [audio, np.zeros((pad, audio.shape[1]), dtype=audio.dtype)],
            axis=0,
        )

    return audio.squeeze(1) if squeeze_out else audio


# --- helpers ---------------------------------------------------------------

def _trailing_audio_end(audio: np.ndarray, sr: int) -> int:
    """Return the sample index just past the last audible content.

    Walks backwards in non-overlapping windows of `_SILENCE_WINDOW_SEC` and
    finds the last window whose RMS exceeds `_SILENCE_THRESHOLD_DB`. Returns
    the end of that window plus a small natural-decay tail.

    Falls back to the original audio length when the entire clip is below
    threshold (silent input) or shorter than one window.
    """
    n = audio.shape[0]
    window = int(sr * _SILENCE_WINDOW_SEC)
    if n <= window:
        return n
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    # Squared amplitudes — comparing to threshold² is equivalent to RMS vs
    # threshold but avoids a sqrt per window.
    sq = (mono ** 2)
    thresh_sq = (10.0 ** (_SILENCE_THRESHOLD_DB / 20.0)) ** 2
    tail_keep = int(sr * _SILENCE_TAIL_KEEP_SEC)
    end = n
    while end > 0:
        start = max(0, end - window)
        if float(sq[start:end].mean()) > thresh_sq:
            return min(n, end + tail_keep)
        end = start
    # Whole clip is below threshold — leave as-is rather than truncate to 0.
    return n


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
    *,
    safe_min: float = _STRETCH_SAFE_MIN,
    safe_max: float = _STRETCH_SAFE_MAX,
) -> Tuple[Optional[float], float]:
    """Pick the time-stretch rate that maps detected → target, considering
    half-time and double-time interpretations of the detected tempo. Returns
    (rate, effective_bpm) where effective_bpm is the (possibly octave-
    corrected) interpretation that was chosen, or (None, detected_bpm) if
    nothing safe is available.

    Order of preference:
      1. Detected as-is, if it lands inside [safe_min, safe_max].
      2. Octave-corrected (detected × 0.5 or × 2.0), only when the as-is
         interpretation is out of range. This is the librosa half-/double-
         time error recovery path.
    """
    rate_asis = target_bpm / detected_bpm
    if safe_min <= rate_asis <= safe_max:
        return rate_asis, detected_bpm

    candidates = []
    for octave_factor in (0.5, 2.0):
        interpreted = detected_bpm * octave_factor
        rate = target_bpm / interpreted
        if safe_min <= rate <= safe_max:
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
