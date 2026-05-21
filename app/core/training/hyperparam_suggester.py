"""Heuristic hyperparameter suggester for the Training tab's "Suggest" button.

Given the dataset on disk and the current hardware, returns a config that
trades off "small dataset, needs more updates per epoch" vs "big dataset,
batch up for throughput", plus the practical VRAM ceilings of the LoRA path
on Stable Audio Open 1.0. Returns the same shape the frontend `trainingConfig`
uses, so Apply can spread the result into state directly.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}

# Cache file for total-duration measurement. ffprobe across 500 files takes
# 10-30s; we don't want to pay that on every button click. Cache key is the
# (file_count, max_mtime_int) pair — invalidates automatically when files
# are added/removed/touched.
_DURATION_CACHE_NAME = ".duration_cache.json"


def _list_audio_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    return [
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]


def _measure_total_duration(audio_files: List[Path], cache_path: Path) -> float:
    if not audio_files:
        return 0.0

    file_count = len(audio_files)
    max_mtime = int(max(p.stat().st_mtime for p in audio_files))
    cache_key = f"{file_count}:{max_mtime}"

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("key") == cache_key:
                return float(cached["duration_sec"])
        except Exception:
            pass

    total = 0.0
    for f in audio_files:
        try:
            out = subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(f)],
                text=True, timeout=10,
            ).strip()
            total += float(out)
        except Exception:
            # Skip files ffprobe can't read; better to under-report than crash.
            continue

    try:
        cache_path.write_text(json.dumps({
            "key": cache_key,
            "duration_sec": total,
        }))
    except Exception:
        pass

    return total


def _detect_vram_gb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return None


def _bucket(file_count: int) -> str:
    if file_count < 20:
        return "tiny"
    if file_count < 100:
        return "small"
    if file_count < 500:
        return "medium"
    return "large"


def _heuristic(file_count: int, vram_gb: Optional[float], mode: str) -> Dict[str, Any]:
    """SA3 LoRA defaults — what train_lora.py wants directly.

    SA3 trains by total `--steps` (not by epochs), so this returns a step
    count tuned to the dataset bucket. Upstream's documented default is
    10 000 steps; we trim that for smaller datasets where the LoRA
    overfits well before then.
    """

    bucket = _bucket(file_count)
    has_vram = vram_gb is not None
    constrained = (has_vram and vram_gb < 12)

    # Target total weight updates per bucket. Tiny datasets overfit by
    # ~2–3K steps; large datasets benefit from full 10K+ runs.
    steps_by_bucket = {
        "tiny":   2000,
        "small":  5000,
        "medium": 10000,
        "large":  20000,
    }
    steps = steps_by_bucket[bucket]

    # Rank/LR/alpha — same shape as SA2 era. SA3's adapter families
    # (dora-rows etc.) keep the parameter-budget math equivalent.
    if bucket in ("tiny", "small"):
        rank, alpha, lr, dropout = 16, 16, 1e-4, 0.05
    else:
        rank, alpha, lr, dropout = 16, 16, 1e-4, 0.0

    if bucket == "tiny":
        batch = 1
    elif bucket == "small":
        batch = 1 if constrained else 2
    elif bucket == "medium":
        batch = 2 if constrained else 4
    else:
        batch = 4 if constrained else 8

    return {
        "steps": steps,
        "batchSize": batch,
        "learningRate": lr,
        "loraRank": rank,
        "loraAlpha": alpha,
        "loraDropout": dropout,
        "adapterType": "dora-rows",
        "precision": "bf16",
        "duration": 30.0,
        "checkpointSteps": max(250, steps // 10),
        "_meta": {
            "bucket": bucket,
            "target_steps": steps,
            "vram_constrained": constrained,
        },
    }


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def _compose_rationale(file_count: int, duration_sec: float, vram_gb: Optional[float],
                       mode: str, meta: Dict[str, Any]) -> List[str]:
    """Human-readable explanation, returned as a list of bullet strings."""
    bullets = []
    bullets.append(
        f"Dataset: {file_count} audio file{'s' if file_count != 1 else ''}, "
        f"total {_format_duration(duration_sec)} → "
        f"\"{meta['bucket']}\" bucket."
    )
    if vram_gb is not None:
        constraint = "VRAM-constrained" if meta["vram_constrained"] else "comfortable VRAM headroom"
        bullets.append(f"Detected GPU with {vram_gb:.1f} GB ({constraint}).")
    else:
        bullets.append("No GPU detected — assuming consumer-class constraints.")
    bullets.append(
        f"Targeting ~{meta['target_steps']} optimizer steps total (SA3 trains "
        f"by step count, not epochs)."
    )
    if meta["bucket"] in ("tiny", "small"):
        bullets.append(
            "Small dataset → conservative 1e-4 LR + rank=16 + dropout 0.05 to "
            "delay overfit. Default adapter is DoRA-rows (SA3 upstream default)."
        )
    else:
        bullets.append(
            "Larger dataset → standard 1e-4 LR + rank=16 + DoRA-rows. Rank "
            "has plenty of capacity for the prompt distribution this size implies."
        )
    return bullets


def suggest(data_dir: Path, mode: str = "lora") -> Dict[str, Any]:
    """Public entry point. Returns the suggestion + a rationale + raw stats."""
    audio_files = _list_audio_files(data_dir)
    file_count = len(audio_files)
    if file_count == 0:
        return {
            "ok": False,
            "error": f"No audio files found in {data_dir}",
        }

    cache_path = data_dir / _DURATION_CACHE_NAME
    duration_sec = _measure_total_duration(audio_files, cache_path)
    vram_gb = _detect_vram_gb()

    suggestion = _heuristic(file_count, vram_gb, mode)
    meta = suggestion.pop("_meta")
    rationale = _compose_rationale(file_count, duration_sec, vram_gb, mode, meta)

    return {
        "ok": True,
        "stats": {
            "file_count": file_count,
            "duration_sec": duration_sec,
            "duration_human": _format_duration(duration_sec),
            "vram_gb": round(vram_gb, 2) if vram_gb is not None else None,
            "bucket": meta["bucket"],
            "total_steps": meta["target_steps"],
        },
        "config": suggestion,
        "rationale": rationale,
    }
