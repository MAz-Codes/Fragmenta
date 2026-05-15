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
    """The rules-of-thumb. Same shape regardless of mode; the frontend ignores
    LoRA-specific keys when mode='full'."""

    bucket = _bucket(file_count)
    has_vram = vram_gb is not None
    constrained = (has_vram and vram_gb < 12)

    # Target total weight updates. Sublinear with dataset size so tiny sets
    # still get enough gradient steps, while large sets don't run forever.
    target_steps_by_bucket = {
        "tiny":   2500,
        "small":  2000,
        "medium": 1500,
        "large":  3000,
    }
    target_steps = target_steps_by_bucket[bucket]

    # Rank/LR/alpha scale with how much "capacity per data point" the run needs.
    # Small dataset trick: keep rank moderate (16) and conservative LR (1e-4 —
    # 2e-4 caused overshoot/flat loss in testing), but boost alpha so the
    # LoRA delta trains at higher effective voltage (scaling = alpha/rank).
    # This produces a stronger imprint without the parameter bloat of rank=32
    # or the instability of higher LR.
    if bucket in ("tiny", "small"):
        rank, alpha, lr = 16, 32, 1e-4
    else:
        rank, alpha, lr = 16, 16, 1e-4

    # Batch size: smaller on small datasets (more updates per epoch + better
    # gradient noise); larger on medium/large for throughput. VRAM caps the top.
    if bucket == "tiny":
        batch = 1 if constrained else 2
    elif bucket == "small":
        # Hold batch=2 even on roomy VRAM — the noise benefit on a small
        # dataset outweighs the throughput win, and it keeps the epoch
        # count to a reasonable display number.
        batch = 2
    elif bucket == "medium":
        batch = 2 if constrained else 4
    else:
        batch = 4 if constrained else 8

    steps_per_epoch = max(1, file_count // batch)
    epochs = max(20, round(target_steps / steps_per_epoch))

    return {
        "batchSize": batch,
        "learningRate": lr,
        "epochs": epochs,
        "loraRank": rank,
        "loraAlpha": alpha,
        "loraDropout": 0,
        "loraMultiplier": 1.0,
        "_meta": {
            "bucket": bucket,
            "target_steps": target_steps,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": steps_per_epoch * epochs,
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
        f"Targeting ~{meta['target_steps']} weight updates total; with batch_size "
        f"the dataset gives {meta['steps_per_epoch']} steps/epoch, so "
        f"{meta['total_steps']} steps over the recommended epoch count."
    )
    if meta["bucket"] in ("tiny", "small"):
        bullets.append(
            "Small dataset → conservative 1e-4 LR + rank=16 for stability, "
            "but alpha=32 (alpha/rank = 2.0) so the LoRA delta trains at "
            "double voltage. Stronger imprint without overshoot risk."
        )
    else:
        bullets.append(
            "Larger dataset → moderate batch + standard 1e-4 LR. Rank=16 has "
            "plenty of capacity for the prompt distribution this size implies."
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
            "steps_per_epoch": meta["steps_per_epoch"],
            "total_steps": meta["total_steps"],
        },
        "config": suggestion,
        "rationale": rationale,
    }
