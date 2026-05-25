"""Helpers for the SA3 LoRA training pipeline.

Responsibilities:
  * Pre-stage the base model in an app-folder HF cache so the training
    subprocess finds it without falling back to ~/.cache/huggingface.
  * Build the train_lora.py subprocess command + env.
  * Convert PyTorch Lightning .ckpt LoRA outputs to SA3-native .safetensors
    with the base_model and run name embedded in the metadata header.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# SA3 model_id → (sa3_name passed to train_lora.py --model, HF repo id)
# Only `*-base` variants are valid LoRA targets — SA3 won't train against
# the post-trained / distilled checkpoints.
SA3_BASE_MODELS: Dict[str, Tuple[str, str]] = {
    "sa3-small-music-base": ("small-music-base", "stabilityai/stable-audio-3-small-music-base"),
    "sa3-small-sfx-base":   ("small-sfx-base",   "stabilityai/stable-audio-3-small-sfx-base"),
    "sa3-medium-base":      ("medium-base",      "stabilityai/stable-audio-3-medium-base"),
}


# Extensions SA3's training data loader actually accepts.
# Source: vendor/stable-audio-3/stable_audio_3/data/dataset.py:91.
# Single source of truth — both the health check and the hyperparam suggester
# use this so what we count matches what the loader will train on.
SA3_AUDIO_EXTENSIONS: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".aif", ".opus")


# --- Base model pre-staging -------------------------------------------------

def prestage_base_model(
    sa3_model_id: str,
    hub_dir: Path,
    token: Optional[str] = None,
    progress_callback: Optional[Any] = None,
) -> Path:
    """Ensure the base model is in `hub_dir` (HF-cache layout, inside app folder).

    train_lora.py calls `model_cfg.resolve()` which is hf_hub_download under
    the hood — it reads from the HF cache root. We point it at hub_dir via
    the HF_HUB_CACHE env var on the subprocess; for that to actually find
    files we need to download into hub_dir using snapshot_download with
    `cache_dir=hub_dir`.

    Idempotent: if the model is already cached there, returns the cached
    snapshot dir without re-downloading.
    """
    if sa3_model_id not in SA3_BASE_MODELS:
        raise ValueError(
            f"'{sa3_model_id}' is not a valid LoRA base. Pick one of "
            f"{list(SA3_BASE_MODELS)} (only *-base variants are CFG-aware)."
        )
    sa3_name, repo_id = SA3_BASE_MODELS[sa3_model_id]
    hub_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    allow_patterns = [
        "*.safetensors", "*.json", "*.txt", "*.model",
        "tokenizer*", "*.tiktoken",
    ]

    if progress_callback:
        progress_callback(5, f"Staging {sa3_name} base model in {hub_dir.name}/...")

    # Prefer cache. snapshot_download otherwise phones home on every run to
    # check the model's revision — wasteful and noisy when the user just
    # downloaded the weights through the Checkpoint Manager. If anything's
    # missing, fall back to an online fetch.
    try:
        local_snap = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(hub_dir),
            token=token,
            allow_patterns=allow_patterns,
            local_files_only=True,
        )
        if progress_callback:
            progress_callback(15, "Base model ready (from cache).")
    except Exception:
        if progress_callback:
            progress_callback(8, "Cache miss — fetching from HuggingFace…")
        local_snap = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(hub_dir),
            token=token,
            allow_patterns=allow_patterns,
        )
        if progress_callback:
            progress_callback(15, "Base model ready.")

    return Path(local_snap)


# --- Subprocess command builder ---------------------------------------------

def build_train_command(
    *,
    venv_python: str,
    sa3_vendor_dir: Path,
    sa3_model_name: str,
    data_dir: Path,
    save_dir: Path,
    rank: int = 16,
    lora_alpha: Optional[int] = None,
    adapter_type: str = "dora-rows",
    dropout: float = 0.0,
    lr: float = 1e-4,
    steps: int = 5000,
    batch_size: int = 1,
    duration: float = 30.0,
    base_precision: str = "bf16",
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    seed: int = 42,
    checkpoint_every: int = 500,
    log_every: int = 50,
    num_workers: int = 2,
    name: str = "fragmenta-lora",
) -> List[str]:
    """Construct the train_lora.py subprocess argv."""
    cmd = [
        venv_python,
        str(sa3_vendor_dir / "scripts" / "train_lora.py"),
        "--model", sa3_model_name,
        "--data_dir", str(data_dir),
        "--save_dir", str(save_dir),
        "--rank", str(int(rank)),
        "--adapter_type", adapter_type,
        "--dropout", str(float(dropout)),
        "--lr", str(float(lr)),
        "--steps", str(int(steps)),
        "--batch_size", str(int(batch_size)),
        "--duration", str(float(duration)),
        "--base_precision", base_precision,
        "--seed", str(int(seed)),
        "--checkpoint_every", str(int(checkpoint_every)),
        "--log_every", str(int(log_every)),
        "--num_workers", str(int(num_workers)),
        "--name", name,
        "--logger", "csv",
        # demo_every set to a very large number — Fragmenta's training
        # monitor doesn't surface demo audio, no need to spend cycles.
        "--demo_every", "1000000",
    ]
    if lora_alpha is not None:
        cmd += ["--lora_alpha", str(int(lora_alpha))]
    if include:
        cmd += ["--include", *include]
    if exclude:
        cmd += ["--exclude", *exclude]
    return cmd


# --- Checkpoint conversion (.ckpt → .safetensors with base_model metadata) ---

def convert_run_checkpoints_to_safetensors(
    run_dir: Path,
    base_model: str,
    model_name: Optional[str] = None,
    delete_originals: bool = True,
) -> List[Path]:
    """Convert PyTorch Lightning .ckpt files in a run's checkpoints/ directory
    to SA3's native .safetensors LoRA format, with `base_model` injected into
    the safetensors metadata header so /api/loras can filter by it.

    Why: SA3's `train_lora.py` writes Lightning .ckpt files. The inference
    LoRA picker (/api/loras) globs for *.safetensors only. Without this
    conversion, every trained LoRA is functionally orphaned — saved
    correctly to disk but invisible to the inference loader.

    Idempotent: skips any .ckpt whose .safetensors sibling already exists
    with a non-zero size.

    Returns the list of paths to the produced .safetensors files (sorted).
    """
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []

    # Imports deferred so this module can be imported without the SA3 vendor
    # being on sys.path (e.g., during pure orchestrator construction).
    from app.core.config import get_config
    sa3_vendor = get_config().get_path("stable_audio_3")
    pp = sys.path[:]
    if str(sa3_vendor) not in pp:
        sys.path.insert(0, str(sa3_vendor))
    try:
        from stable_audio_3.models.lora.utils import load_lora_checkpoint
        from safetensors.torch import save_file as st_save_file
    finally:
        # Don't permanently mutate sys.path from a helper call.
        if sys.path != pp:
            sys.path[:] = pp

    written: List[Path] = []
    for ckpt_path in sorted(ckpt_dir.glob("*.ckpt")):
        out_path = ckpt_path.with_suffix(".safetensors")
        if out_path.exists() and out_path.stat().st_size > 0:
            # Already converted (older artifact or a previous pass). Just
            # bookkeep so the caller sees it in the return list.
            written.append(out_path)
            continue
        try:
            state_dict, lora_config = load_lora_checkpoint(ckpt_path)
        except Exception:
            # Corrupt or truncated ckpt — skip rather than crash the
            # post-training pass.
            continue

        # Top-level metadata is what /api/loras' safetensors reader inspects
        # directly. We also keep the canonical `lora_config` JSON blob so
        # SA3's own load_lora_checkpoint() can parse the file as-is.
        metadata = {
            "lora_config": json.dumps(lora_config or {}),
            "base_model": base_model,
        }
        if model_name:
            metadata["model_name"] = model_name
        # Cast fp16 to keep file sizes consistent with SA3's standard format.
        fp16_dict = {k: (v.half() if v.is_floating_point() else v)
                     for k, v in state_dict.items()}
        st_save_file(fp16_dict, str(out_path), metadata=metadata)
        if delete_originals:
            try:
                ckpt_path.unlink()
            except OSError:
                pass
        written.append(out_path)
    return sorted(written)


def build_train_env(sa3_vendor_dir: Path, hub_dir: Path) -> Dict[str, str]:
    """Subprocess env: redirect HF cache into the app folder + silence WANDB."""
    env = os.environ.copy()
    # Make `import stable_audio_3` work without pip-installing the package.
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{sa3_vendor_dir}{os.pathsep}{pp}" if pp else str(sa3_vendor_dir)
    )
    # Pin the HF cache to our app-folder hub dir; otherwise train_lora.py's
    # model_cfg.resolve() would write into ~/.cache/huggingface/hub. Cover
    # the legacy + transformers env names too for defense-in-depth.
    env["HF_HUB_CACHE"] = str(hub_dir)
    env["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
    env["TRANSFORMERS_CACHE"] = str(hub_dir)
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["WANDB_DISABLED"] = "1"
    # Force the training subprocess into offline mode for HF — we already
    # pre-staged the base model in prestage_base_model(), so any remaining
    # network call from the SA3 internals would be a noisy revision check
    # against a cache we know is current.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env
