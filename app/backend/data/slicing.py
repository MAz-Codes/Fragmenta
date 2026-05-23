"""Audio slicing for the Dataset Workbench.

Splits one audio file into N children. Three strategies:

  hard       — uniform cuts every `target_duration` seconds.
  transient  — uniform anchor points, each snapped to the nearest onset
               (librosa.onset.onset_detect).
  silence    — uniform anchor points, each snapped to the nearest low-RMS
               window (cleanest splice between phrases).

All three honor `overlap_sec`, applied as a head-overlap on every child
after the first: child i starts at (end of child i-1) - overlap_sec.

Writes WAV regardless of source format (lossless, no codec deps). Parent
prompt is inherited verbatim; the user edits children individually after.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple

logger = logging.getLogger(__name__)

SliceStrategy = Literal["hard", "transient", "silence"]
VALID_STRATEGIES = ("hard", "transient", "silence")

# How far a snap is allowed to move from the uniform anchor. Beyond this we
# just take the anchor — better a tidy cut than a wildly off-target chunk.
SNAP_WINDOW_FRAC = 0.35


@dataclass
class SlicePlan:
    """One child's location inside the parent. Times are in seconds."""
    index: int          # 1-based
    start_sec: float
    end_sec: float


def _uniform_anchors(duration_sec: float, target_sec: float, overlap_sec: float) -> List[Tuple[float, float]]:
    """Return [(start, end), ...] for uniform cuts, before any snapping."""
    if target_sec <= 0:
        raise ValueError("target_duration must be positive")
    if overlap_sec < 0 or overlap_sec >= target_sec:
        raise ValueError("overlap_sec must be >= 0 and < target_duration")
    step = target_sec - overlap_sec
    anchors: List[Tuple[float, float]] = []
    start = 0.0
    while start < duration_sec - 0.05:  # don't emit a sub-50ms tail
        end = min(start + target_sec, duration_sec)
        anchors.append((start, end))
        if end >= duration_sec:
            break
        start += step
    return anchors


def _snap_to_onsets(anchors: List[Tuple[float, float]], y, sr: int, target_sec: float) -> List[Tuple[float, float]]:
    """Snap each cut boundary to the nearest detected onset within a window."""
    import librosa
    import numpy as np
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    if len(onsets) == 0:
        return anchors
    snap_window = target_sec * SNAP_WINDOW_FRAC
    out: List[Tuple[float, float]] = []
    for i, (s, e) in enumerate(anchors):
        if i > 0:
            # Snap the start (= previous end) to nearest onset within window.
            candidates = onsets[(onsets >= s - snap_window) & (onsets <= s + snap_window)]
            if len(candidates):
                s = float(min(candidates, key=lambda t: abs(t - s)))
        out.append((s, e))
    # Stitch ends to match next start so no gap/overlap drift creeps in.
    for i in range(len(out) - 1):
        s, _ = out[i]
        next_s, _ = out[i + 1]
        out[i] = (s, next_s + (target_sec * 0.0))  # next_s alone — overlap is in next_s already from caller
    return out


def _snap_to_silence(anchors: List[Tuple[float, float]], y, sr: int, target_sec: float) -> List[Tuple[float, float]]:
    """Snap each cut boundary to the lowest-RMS frame within a window."""
    import librosa
    import numpy as np
    # Frame-level RMS at ~20ms hop.
    hop = max(1, sr // 50)
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    if len(rms) == 0:
        return anchors
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    snap_window = target_sec * SNAP_WINDOW_FRAC
    out: List[Tuple[float, float]] = []
    for i, (s, e) in enumerate(anchors):
        if i > 0:
            mask = (frame_times >= s - snap_window) & (frame_times <= s + snap_window)
            if mask.any():
                local_idx = int(np.argmin(rms[mask]))
                # Map masked-index back to absolute time.
                masked_times = frame_times[mask]
                s = float(masked_times[local_idx])
        out.append((s, e))
    return out


def plan_slices(
    audio_path: Path,
    target_sec: float,
    overlap_sec: float,
    strategy: SliceStrategy,
) -> List[SlicePlan]:
    """Compute the (start, end) for each child without writing anything yet."""
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    import librosa
    # Use mono for boundary detection only; final write uses the original.
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = float(len(y) / sr) if len(y) else 0.0
    if duration <= 0:
        raise ValueError(f"{audio_path.name} has zero duration")
    if duration < target_sec:
        # Single child = the whole file. Skip the slice loop entirely.
        return [SlicePlan(index=1, start_sec=0.0, end_sec=duration)]

    anchors = _uniform_anchors(duration, target_sec, overlap_sec)
    if strategy == "transient":
        anchors = _snap_to_onsets(anchors, y, sr, target_sec)
    elif strategy == "silence":
        anchors = _snap_to_silence(anchors, y, sr, target_sec)

    return [
        SlicePlan(index=i + 1, start_sec=s, end_sec=e)
        for i, (s, e) in enumerate(anchors)
    ]


def write_slices(
    audio_path: Path,
    plans: List[SlicePlan],
    out_dir: Path,
    stem: str,
) -> List[Path]:
    """Write children as `<stem>__001.wav`, `<stem>__002.wav`, ... in `out_dir`.

    Uses soundfile for lossless WAV write at the source's native sample rate.
    Skips names that already exist on disk to avoid clobbering.
    """
    import soundfile as sf
    import numpy as np

    info = sf.info(str(audio_path))
    sr = info.samplerate
    total_frames = info.frames
    written: List[Path] = []
    width = max(3, len(str(len(plans))))

    with sf.SoundFile(str(audio_path)) as src:
        for plan in plans:
            start_frame = max(0, int(plan.start_sec * sr))
            end_frame = min(total_frames, int(plan.end_sec * sr))
            if end_frame <= start_frame:
                logger.warning("Skipping empty slice %s [%.2f-%.2f]", plan.index, plan.start_sec, plan.end_sec)
                continue
            src.seek(start_frame)
            data = src.read(end_frame - start_frame, dtype="float32", always_2d=True)

            child_name = f"{stem}__{plan.index:0{width}d}.wav"
            child_path = out_dir / child_name
            if child_path.exists():
                # Don't silently overwrite; bump the suffix until free.
                k = 2
                while True:
                    candidate = out_dir / f"{stem}__{plan.index:0{width}d}_{k}.wav"
                    if not candidate.exists():
                        child_path = candidate
                        break
                    k += 1
            sf.write(str(child_path), data, sr, subtype="PCM_16")
            written.append(child_path)
    return written
