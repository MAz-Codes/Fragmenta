"""SA3 LoRA hyperparameter suggester for the Training tab's "Suggest" button.

Reads a Dataset Workbench project directly — counts SA3-compatible audio
files, measures their durations via the same `soundfile.info()` header-only
probe used elsewhere in the app, factors in the user's picked base model
and detected GPU VRAM, and returns a config that:

  * matches the upstream SA3 LoRA docs as the starting point
    (see vendor/stable-audio-3/docs/workflows/lora.md)
  * sets `--include transformer.layers` and `--exclude seconds_total
    to_local_embed` by default (documented best practices, prevents the
    "conditioner hijacking" failure mode on small datasets)
  * picks a `-XS` adapter family when VRAM is tight for the chosen base
  * proposes a `duration` derived from the actual clip lengths in the
    project — not a hardcoded 30s
  * warns when the dataset is below SA3's documented minimum (~20 clips)
    or when clips are too short to learn from

Returns the same shape the frontend `trainingConfig` uses, so Apply can
spread the result into state directly.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.backend.data.projects import _clip_duration_sec
from app.core.training.sa3_lora_runner import SA3_AUDIO_EXTENSIONS, SA3_BASE_MODELS


# --- Discovery -------------------------------------------------------------


def _list_audio_files(data_dir: Path) -> List[Path]:
    """Files SA3's loader would actually train on. Mirrors the loader's filter."""
    if not data_dir.exists():
        return []
    return [
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SA3_AUDIO_EXTENSIONS
    ]


def _duration_stats(audio_files: List[Path]) -> Dict[str, Optional[float]]:
    """Header-only duration probe + summary stats. None-safe for unreadable files."""
    durations: List[float] = []
    for f in audio_files:
        d = _clip_duration_sec(f)
        if d is not None and d > 0:
            durations.append(d)
    if not durations:
        return {"count": 0, "total": 0.0, "median": None, "p95": None, "max": None, "min": None}
    durations.sort()
    n = len(durations)
    return {
        "count": n,
        "total": float(sum(durations)),
        "median": float(durations[n // 2]),
        "p95": float(durations[min(n - 1, int(math.ceil(0.95 * n)) - 1)]),
        "max": float(durations[-1]),
        "min": float(durations[0]),
    }


def _detect_vram_gb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return None


# --- Bucketing & sizing ----------------------------------------------------


def _bucket(file_count: int) -> str:
    if file_count < 20:
        return "tiny"
    if file_count < 100:
        return "small"
    if file_count < 500:
        return "medium"
    return "large"


# SA3's documented quick-start: --steps 1000, with no dataset-size caveat.
# (vendor/stable-audio-3/docs/workflows/lora.md, "Standard (recommended starting point)".)
# SA3 trains by *windows seen*, not epochs, so a 5h dataset doesn't need more
# steps than a 30min one — it just produces more diverse sampling per step.
# We keep the SA3 default for tiny/small, and bump modestly only when a
# dataset is large enough that 1000 steps won't see all unique windows.
_STEPS_BY_BUCKET: Dict[str, int] = {
    "tiny":   1000,
    "small":  1000,
    "medium": 2000,
    "large":  4000,
}


# Per-base-model VRAM table from SA3 docs. (standard_gb, xs_bf16_gb)
# Source: docs/workflows/lora.md memory table.
_VRAM_REQ: Dict[str, Tuple[float, float]] = {
    "sa3-small-music-base": (2.5, 2.0),
    "sa3-small-sfx-base":   (2.5, 2.0),
    "sa3-medium-base":      (6.5, 5.5),
}


def _pick_adapter(base_model: Optional[str], vram_gb: Optional[float]) -> Tuple[str, bool]:
    """Choose adapter family. Returns (adapter_type, vram_constrained_flag).

    SA3 docs recommend the `-xs` family + bf16 base precision for VRAM-limited
    hosts. Headroom rule: standard_gb + 4 GB activations is the comfort target;
    below that we pick the xs family.
    """
    default = "dora-rows"
    if base_model is None or vram_gb is None:
        return default, False
    std_gb, _xs_gb = _VRAM_REQ.get(base_model, (2.5, 2.0))
    comfort = std_gb + 4.0
    constrained = vram_gb < comfort
    return ("dora-rows-xs" if constrained else default), constrained


def _model_max_window_sec(base_model: Optional[str]) -> float:
    """SA3's native training length for the base, from its model config
    sample_size / sample_rate: medium-base ≈380s, small bases ≈120s. The
    `seconds_total` conditioner caps at 384s, so 380 is the safe medium ceiling.
    Longer windows aren't a model limit below these — they're VRAM/time bound.
    """
    if base_model and "medium" in base_model:
        return 380.0
    return 120.0


def _pick_duration(p95_clip_sec: Optional[float], base_model: Optional[str]) -> float:
    """Set training window from the project's actual p95 clip length.

    Floors at 5s; caps at — and defaults to — the model's native length
    (≈120s small / ≈380s medium) rather than an arbitrary 30s. SA3 random-crops
    longer files, so the only real limits are the model's sequence length and
    VRAM. Rounds up p95 with 2s headroom so the window isn't cropping the tails
    of typical clips. With no duration data, defaults to the model max.
    """
    model_max = _model_max_window_sec(base_model)
    if p95_clip_sec is None or p95_clip_sec <= 0:
        return model_max
    suggested = math.ceil(p95_clip_sec + 2.0)
    return float(max(5, min(model_max, suggested)))


def _pick_batch_size(bucket: str, vram_gb: Optional[float],
                     base_model: Optional[str] = None) -> int:
    """Batch size trades VRAM throughput against gradient-update frequency.

    For small datasets batch 1 is deliberate, not a VRAM limit: more optimizer
    updates per pass, which matters more than throughput for personalization
    (SA3's own examples all use batch 1, and tiny sets want the noisier updates).
    We only scale up once the dataset is large enough to benefit, and within the
    model's *real* per-sample cost — the small bases are cheap (~0.6 GB/sample on
    top of a ~2.5 GB base), the medium base is not (~6.5 GB + heavy bf16
    activations per sample), so they get very different VRAM floors.
    """
    # Tiny/small datasets: keep frequent updates regardless of spare VRAM.
    if bucket in ("tiny", "small"):
        return 1

    is_small = base_model is None or "small" in base_model

    if not is_small:
        # medium-base: only step up on workstation-class cards.
        if vram_gb is not None and vram_gb >= 24 and bucket in ("medium", "large"):
            return 2
        return 1

    # small bases: per-sample cost is low, so dataset size drives this with a
    # modest VRAM floor rather than the medium-calibrated 24 GB gate.
    if vram_gb is None:
        return 2  # unknown VRAM but dataset is 100+; 2 is safe on a small base
    if bucket == "large" and vram_gb >= 12:
        return 4
    if vram_gb >= 8:
        return 2
    return 1


# Filter pattern straight from SA3 docs:
#   --include transformer.layers --exclude seconds_total to_local_embed
# "Everything except local embedding and seconds_total conditioner" — prevents
# the conditioner-hijacking failure mode that bites small datasets hardest.
_INCLUDE_DEFAULT: List[str] = ["transformer.layers"]
_EXCLUDE_DEFAULT: List[str] = ["seconds_total", "to_local_embed"]


# --- Suggestion + rationale ------------------------------------------------


def _heuristic(
    file_count: int,
    dur_stats: Dict[str, Optional[float]],
    base_model: Optional[str],
    vram_gb: Optional[float],
) -> Dict[str, Any]:
    bucket = _bucket(file_count)
    steps = _STEPS_BY_BUCKET[bucket]
    adapter, constrained = _pick_adapter(base_model, vram_gb)
    duration = _pick_duration(dur_stats.get("p95"), base_model)
    batch = _pick_batch_size(bucket, vram_gb, base_model)

    # Mild dropout for tiny datasets only — extra regularization where overfit
    # is most likely. SA3 default is 0.0; we deviate intentionally.
    dropout = 0.05 if bucket == "tiny" else 0.0

    # Checkpoint cadence: ~10 checkpoints per run, but keep within sane bounds
    # so we don't write a checkpoint every 50 steps on tiny runs or sit on a
    # 2K-step gap on long ones.
    checkpoint_every = max(250, min(1000, steps // 10))

    return {
        "steps": steps,
        "batchSize": batch,
        "learningRate": 1e-4,
        "loraRank": 16,
        "loraAlpha": 16,
        "loraDropout": dropout,
        "adapterType": adapter,
        "precision": "bf16",
        "duration": duration,
        "checkpointSteps": checkpoint_every,
        "include": list(_INCLUDE_DEFAULT),
        "exclude": list(_EXCLUDE_DEFAULT),
        "_meta": {
            "bucket": bucket,
            "target_steps": steps,
            "vram_constrained": constrained,
            "picked_adapter_for_vram": constrained,
        },
    }


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def _compose_rationale(
    file_count: int,
    dur_stats: Dict[str, Optional[float]],
    base_model: Optional[str],
    vram_gb: Optional[float],
    config: Dict[str, Any],
    meta: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """Return (bullets, warnings). Warnings are surfaced separately in the UI."""
    bullets: List[str] = []
    warnings: List[str] = []

    total = dur_stats.get("total") or 0.0
    bullets.append(
        f"Dataset: {file_count} clip{'s' if file_count != 1 else ''}, "
        f"total {_format_duration(total)} → \"{meta['bucket']}\" bucket."
    )

    p95 = dur_stats.get("p95")
    median = dur_stats.get("median")
    if p95 is not None and median is not None:
        bullets.append(
            f"Clip durations: median {median:.1f}s, p95 {p95:.1f}s. "
            f"Training window set to {config['duration']:.0f}s."
        )

    if vram_gb is not None:
        bullets.append(
            f"Detected GPU: {vram_gb:.1f} GB"
            + (" (tight for the chosen base — switched adapter to a -XS variant)."
               if meta["vram_constrained"] else " (comfortable headroom).")
        )
    else:
        bullets.append("No CUDA GPU detected — adapter defaults to dora-rows; "
                       "training will run on CPU/MPS where supported.")

    if meta["target_steps"] == 1000:
        bullets.append(
            "Target 1 000 optimizer steps — SA3's documented quick-start. "
            "LoRAs typically overfit well before this; watch the loss curve."
        )
    else:
        bullets.append(
            f"Target {meta['target_steps']:,} optimizer steps — modest bump "
            f"above SA3's 1 000-step default for larger datasets to see more "
            "unique sampling windows."
        )

    bullets.append(
        f"Layer filter: include `{config['include'][0]}`, exclude "
        f"`{' '.join(config['exclude'])}`. "
        "Documented SA3 default — prevents conditioner-hijacking on small sets."
    )

    bullets.append(
        f"Adapter `{config['adapterType']}` · rank 16 · α 16 · "
        f"dropout {config['loraDropout']} · {config['precision']} base."
    )

    # --- Warnings (separate channel) ---------------------------------------

    if file_count < 20:
        warnings.append(
            f"{file_count} clips is below SA3's documented minimum of ~20. "
            "Expect heavy overfit and poor generalization — add more data if you can."
        )
    if median is not None and median < 2.0:
        warnings.append(
            f"Median clip is only {median:.1f}s — most of the training window "
            f"({config['duration']:.0f}s) will be silence-padded. "
            "Re-slice the source material to longer chunks for better signal."
        )
    if config["duration"] > 45:
        warnings.append(
            f"Training window is {config['duration']:.0f}s. Longer windows use "
            "markedly more VRAM and step time (DiT attention scales with length). "
            "If you hit OOM, lower the window or pre-encode the dataset first."
        )

    # VRAM × base model crosscheck
    if base_model in _VRAM_REQ:
        std_gb, xs_gb = _VRAM_REQ[base_model]
        if vram_gb is None:
            if base_model == "sa3-medium-base":
                warnings.append(
                    "No CUDA GPU detected, but you picked Medium-Base. "
                    "Medium-base needs CUDA + Flash-Attn 2 (Linux) and ≥5.5 GB VRAM. "
                    "Consider Small-Music-Base or Small-SFX-Base for CPU/MPS hosts."
                )
        elif vram_gb < xs_gb:
            warnings.append(
                f"GPU has {vram_gb:.1f} GB; even {base_model} with bf16+lora-xs needs "
                f"~{xs_gb:.1f} GB. Training will likely OOM. Pick a smaller base."
            )
        elif vram_gb < std_gb:
            warnings.append(
                f"GPU has {vram_gb:.1f} GB; {base_model} standard config needs "
                f"~{std_gb:.1f} GB. The -XS adapter (selected) brings it to ~{xs_gb:.1f} GB."
            )

    return bullets, warnings


def suggest(data_dir: Path, base_model: Optional[str] = None) -> Dict[str, Any]:
    """Public entry point. SA3 is LoRA-only; no `mode` switch."""
    audio_files = _list_audio_files(data_dir)
    file_count = len(audio_files)
    if file_count == 0:
        return {
            "ok": False,
            "error": (
                f"No SA3-compatible audio in {data_dir}. SA3's loader accepts "
                + ", ".join(SA3_AUDIO_EXTENSIONS) + "."
            ),
        }

    dur_stats = _duration_stats(audio_files)
    vram_gb = _detect_vram_gb()

    suggestion = _heuristic(file_count, dur_stats, base_model, vram_gb)
    meta = suggestion.pop("_meta")
    bullets, warnings = _compose_rationale(
        file_count, dur_stats, base_model, vram_gb, suggestion, meta
    )

    # Caption coverage: SA3 trains on audio + matching .txt sidecars, and
    # silently drops clips whose prompt is blank. Surface missing captions so
    # the user isn't unknowingly training on a fraction of the dataset.
    uncaptioned = sum(
        1 for p in audio_files
        if not (p.with_suffix(".txt").exists()
                and p.with_suffix(".txt").read_text(encoding="utf-8", errors="ignore").strip())
    )
    if uncaptioned:
        warnings.insert(0, (
            f"{uncaptioned} of {file_count} clip{'s' if file_count != 1 else ''} "
            "have no annotation. SA3 silently skips un-captioned clips at train "
            "time — annotate them first or they won't contribute to the LoRA."
        ))

    return {
        "ok": True,
        "stats": {
            "file_count": file_count,
            "duration_sec": dur_stats.get("total") or 0.0,
            "duration_human": _format_duration(dur_stats.get("total") or 0.0),
            "median_clip_sec": dur_stats.get("median"),
            "p95_clip_sec": dur_stats.get("p95"),
            "max_clip_sec": dur_stats.get("max"),
            "min_clip_sec": dur_stats.get("min"),
            "vram_gb": round(vram_gb, 2) if vram_gb is not None else None,
            "bucket": meta["bucket"],
            "total_steps": meta["target_steps"],
            "base_model": base_model,
        },
        "config": suggestion,
        "rationale": bullets,
        "warnings": warnings,
    }
