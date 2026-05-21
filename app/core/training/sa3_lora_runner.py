"""Helpers for the SA3 LoRA training pipeline.

Responsibilities:
  * Materialize <basename>.txt captions from Fragmenta's data/metadata.json
    (SA3's train_lora.py expects .txt sidecars; the rest of Fragmenta keeps
    using metadata.json as the editable source of truth).
  * Pre-stage the base model in an app-folder HF cache so the training
    subprocess finds it without falling back to ~/.cache/huggingface.
  * Build the train_lora.py subprocess command + env.
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


# --- Caption materializer ---------------------------------------------------

def materialize_captions(metadata_json_path: Path, data_dir: Path) -> Dict[str, Any]:
    """Write <basename>.txt sidecars next to every audio file in data_dir.

    Reads metadata.json (Fragmenta's editable source of truth) and emits one
    .txt per row. Idempotent — files are only touched when contents differ.
    Returns {"written": int, "skipped": int, "missing_audio": list}.
    """
    if not metadata_json_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {metadata_json_path}")
    try:
        rows = json.loads(metadata_json_path.read_text())
    except Exception as e:
        raise ValueError(f"metadata.json is not valid JSON: {e}")
    if not isinstance(rows, list):
        raise ValueError("metadata.json must be a list of {file_name, prompt} rows")

    written = 0
    skipped = 0
    missing_audio: List[str] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        file_name = row.get("file_name") or row.get("filename")
        prompt = (row.get("prompt") or "").strip()
        if not file_name or not prompt:
            continue
        audio_path = data_dir / file_name
        if not audio_path.exists():
            missing_audio.append(file_name)
            continue
        txt_path = audio_path.with_suffix(".txt")
        if txt_path.exists() and txt_path.read_text() == prompt:
            skipped += 1
            continue
        txt_path.write_text(prompt)
        written += 1

    return {
        "written": written,
        "skipped": skipped,
        "missing_audio": missing_audio,
    }


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

    if progress_callback:
        progress_callback(5, f"Staging {sa3_name} base model in {hub_dir.name}/...")

    local_snap = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(hub_dir),
        token=token,
        allow_patterns=[
            "*.safetensors", "*.json", "*.txt", "*.model",
            "tokenizer*", "*.tiktoken",
        ],
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


def build_train_env(sa3_vendor_dir: Path, hub_dir: Path) -> Dict[str, str]:
    """Subprocess env: redirect HF cache into the app folder + silence WANDB."""
    env = os.environ.copy()
    # Make `import stable_audio_3` work without pip-installing the package.
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{sa3_vendor_dir}{os.pathsep}{pp}" if pp else str(sa3_vendor_dir)
    )
    # Pin the HF cache to our app-folder hub dir; otherwise train_lora.py's
    # model_cfg.resolve() would write into ~/.cache/huggingface/hub.
    env["HF_HUB_CACHE"] = str(hub_dir)
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["WANDB_DISABLED"] = "1"
    return env
