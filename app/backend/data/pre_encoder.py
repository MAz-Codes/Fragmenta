"""SA3 pre-encoding job runner — Phase 6.

Encodes every audio clip in a Dataset Workbench project into SA3 latents
once, ahead of training, so the training subprocess can skip the SAME
autoencoder pass per step. Mirrors the shape of `_project_annotate_jobs`
in app.py (background thread, per-project state, cooperative cancel).

Latents land in `<project>/.latents/` — a hidden subdirectory inside the
project folder. Disk layout matches SA3's `pre_encode_dataset.py`:

  <project>/.latents/
    000000000000.npy     # latent tensor (shape (256, T_lat))
    000000000000.json    # {"prompt": "...", "padding_mask": [...], ...}
    000001000000.npy
    000001000000.json
    ...
    silence.npy          # padding latent (auto-generated)
    _meta.json           # Fragmenta-specific: AE used, source clip count

SA3's `train_lora.py --encoded_dir <project>/.latents` consumes this layout
directly. `SA3Trainer._stage_dataset` auto-detects the directory and feeds
`--encoded_dir` to the subprocess when latents are present.

Cache invalidation lives in projects.py — any project mutation that could
desync the latents (commit, delete_clip, slice_clip) wipes the directory.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from app.backend.data.projects import project_path
from app.core.config import get_config
from utils.logger import get_logger
from utils.process_control import graceful_stop

logger = get_logger("PreEncoder")


# --- Per-project job registry ----------------------------------------------

_pre_encode_jobs: Dict[str, Dict[str, Any]] = {}
_pre_encode_jobs_lock = threading.Lock()
_pre_encode_processes: Dict[str, subprocess.Popen] = {}


def get_pre_encode_job(project_name: str) -> Dict[str, Any]:
    """Snapshot of the current job state for a project. Always returns a
    well-formed dict so the frontend can render against it without guards."""
    with _pre_encode_jobs_lock:
        job = _pre_encode_jobs.get(project_name)
        if job is None:
            return _idle_job()
        return dict(job)


def count_running_pre_encode_jobs() -> int:
    """How many pre-encode jobs are queued or running across all projects.

    Used by the optional global concurrent-job cap in web deployments.
    """
    with _pre_encode_jobs_lock:
        return sum(1 for j in _pre_encode_jobs.values()
                   if j.get("state") in ("queued", "running"))


def _idle_job() -> Dict[str, Any]:
    return {
        "state": "idle",          # idle | queued | running | complete | failed | cancelled
        "current": 0,             # batch index (0-based)
        "total": 0,               # total batches (derived from clip count)
        "current_file": "",
        "error": None,
        "started_at": None,
        "finished_at": None,
        "autoencoder": None,
    }


# --- Autoencoder selection -------------------------------------------------

# Bind latents to a specific SA3 autoencoder. Latents from same-s only work
# with small-music / small-sfx DiTs; same-l latents only work with medium.
# For v1 we default to same-s (covers the most common base) and leave a
# manifest in .latents/_meta.json that training reads to verify
# compatibility. If a user trains against medium with same-s latents,
# SA3Trainer falls back to non-encoded training and logs a warning.
DEFAULT_AUTOENCODER = "same-s"

# Audio length (samples per channel) the dataset pads/crops to before
# encoding. SA3's pre_encode_dataset.py defaults to ~285s at 44.1 kHz, which
# covers any training-time --duration up to that limit (and SA3 small caps
# at 120s anyway). Longer clips in the project will be cropped to this
# length during encoding — a documented limitation for v1.
DEFAULT_SAMPLE_SIZE = 12_582_912


# --- Job lifecycle ---------------------------------------------------------

def latents_dir(project_name: str) -> Path:
    return project_path(project_name) / ".latents"


def latents_count(project_name: str) -> int:
    d = latents_dir(project_name)
    if not d.exists():
        return 0
    return sum(
        1 for p in d.glob("*.npy")
        if p.name != "silence.npy"
    )


def latents_meta(project_name: str) -> Optional[Dict[str, Any]]:
    """Read the manifest we drop alongside the .npy files."""
    p = latents_dir(project_name) / "_meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def latents_match_base(project_name: str, base_model: str) -> bool:
    """Whether the cached latents are compatible with the chosen base.

    same-s ↔ small-music / small-sfx (and their *-base variants).
    same-l ↔ medium (and medium-base).
    """
    meta = latents_meta(project_name)
    if not meta:
        return False
    ae = meta.get("autoencoder")
    if ae == "same-s":
        return base_model in ("sa3-small-music", "sa3-small-music-base",
                              "sa3-small-sfx", "sa3-small-sfx-base")
    if ae == "same-l":
        return base_model in ("sa3-medium", "sa3-medium-base")
    return False


def cancel_pre_encode(project_name: str) -> bool:
    """Stop an in-flight job. Returns True if it was actually cancelled.

    Order matters: signal the subprocess FIRST, flip the user-visible state
    only once that succeeded. The old code marked the job "cancelled" before
    signalling, so a failed signal (the exact Windows failure mode —
    send_signal(SIGINT) raises ValueError there) reported success while the
    encoder kept running. The `cancelled` flag is set up front though: the
    worker thread uses it to label the child's exit, and it would race us if
    set only after the process dies.
    """
    with _pre_encode_jobs_lock:
        job = _pre_encode_jobs.get(project_name)
        if not job or job.get("state") not in ("queued", "running"):
            return False
        job["cancelled"] = True

    proc = _pre_encode_processes.get(project_name)
    if proc is not None and proc.poll() is None:
        try:
            graceful_stop(proc, wait_timeout=5, kill_timeout=3)
        except Exception as exc:
            logger.warning("Failed to signal pre-encode subprocess: %s", exc)
            with _pre_encode_jobs_lock:
                live = _pre_encode_jobs.get(project_name)
                if live is not None:
                    live["cancelled"] = False
            return False

    with _pre_encode_jobs_lock:
        job = _pre_encode_jobs.get(project_name)
        if job is not None and job.get("state") in ("queued", "running"):
            job["state"] = "cancelled"
            job["finished_at"] = time.time()
    return True


def start_pre_encode(
    project_name: str,
    autoencoder: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Spawn the pre-encode subprocess in a background thread. Returns the
    job state — frontend polls /pre-encode/status thereafter.
    """
    proj_dir = project_path(project_name)
    if not proj_dir.exists():
        raise FileNotFoundError(f"project not found: {project_name}")

    ae = autoencoder or DEFAULT_AUTOENCODER
    if ae not in ("same-s", "same-l"):
        raise ValueError(f"autoencoder must be 'same-s' or 'same-l'; got {ae!r}")

    with _pre_encode_jobs_lock:
        existing = _pre_encode_jobs.get(project_name)
        if existing and existing.get("state") in ("queued", "running"):
            return dict(existing)

        # Count source clips (sidecars committed) so we know the denominator.
        sidecars = list(proj_dir.glob("*.txt"))
        clip_count = sum(
            1 for p in sidecars
            if p.read_text(encoding="utf-8").strip()
            and p.with_suffix(".wav").exists()  # cheap & accurate enough
        )

        job: Dict[str, Any] = {
            "state": "queued",
            "current": 0,
            "total": clip_count,
            "current_file": "",
            "error": None,
            "started_at": time.time(),
            "finished_at": None,
            "autoencoder": ae,
            "cancelled": False,
        }
        _pre_encode_jobs[project_name] = job

    thread = threading.Thread(
        target=_run_pre_encode,
        args=(project_name, ae, sample_size or DEFAULT_SAMPLE_SIZE),
        daemon=True,
        name=f"sa3-pre-encode:{project_name}",
    )
    thread.start()
    return get_pre_encode_job(project_name)


# --- Worker ----------------------------------------------------------------

def _update_job(project_name: str, **fields: Any) -> None:
    with _pre_encode_jobs_lock:
        job = _pre_encode_jobs.get(project_name)
        if job is None:
            return
        job.update(fields)


def _run_pre_encode(project_name: str, ae: str, sample_size: int) -> None:
    """Background-thread target. Spawns the SA3 pre_encode_dataset.py script,
    streams stdout for progress, writes a _meta.json manifest on success."""
    cfg = get_config()
    proj_dir = project_path(project_name)
    out_dir = latents_dir(project_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    sa3_vendor = cfg.get_path("stable_audio_3")
    venv_python = sys.executable

    cmd = [
        venv_python,
        str(sa3_vendor / "scripts" / "pre_encode_dataset.py"),
        "--model", ae,
        "--data_dir", str(proj_dir),
        "--output_path", str(out_dir),
        "--batch_size", "1",
        "--sample_size", str(int(sample_size)),
    ]

    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{sa3_vendor}{os.pathsep}{pp}" if pp else str(sa3_vendor)
    )
    hub_dir = cfg.get_path("models_pretrained") / "sa3" / "hub"
    env["HF_HUB_CACHE"] = str(hub_dir)
    env["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
    env["TRANSFORMERS_CACHE"] = str(hub_dir)
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # Match the utf-8 decode above — see the Popen call.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    # A cancel can land between start_pre_encode queueing the job and this
    # thread getting scheduled — honour it before spawning anything.
    snapshot = get_pre_encode_job(project_name)
    if snapshot and snapshot.get("cancelled"):
        _update_job(project_name, state="cancelled", finished_at=time.time())
        return

    _update_job(project_name, state="running")
    logger.info(
        "Pre-encoding started · project=%s · autoencoder=%s · clips=%d · sample_size=%d",
        project_name, ae, get_pre_encode_job(project_name)["total"], sample_size,
    )

    batch_pat = re.compile(r"Processing batch (\d+)")
    process: Optional[subprocess.Popen] = None
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(cfg.project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # Same class of bug as the trainer monitor: locale-default
            # decoding is cp1252 on Windows, and one UTF-8 glyph in the
            # child's output would kill this reader thread mid-encode.
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        _pre_encode_processes[project_name] = process

        if process.stdout is not None:
            for line in process.stdout:
                line = line.rstrip()
                m = batch_pat.search(line)
                if m:
                    # Subprocess prints "Processing batch N" once per batch
                    # (and batch_size=1 → one batch per clip). N starts at 0.
                    _update_job(project_name, current=int(m.group(1)) + 1)

        rc = process.wait() if process else 1

        # Check whether we got cancelled mid-flight.
        snapshot = get_pre_encode_job(project_name)
        if snapshot.get("cancelled"):
            _update_job(
                project_name,
                state="cancelled",
                finished_at=time.time(),
            )
            logger.info("Pre-encoding cancelled · project=%s", project_name)
            return

        if rc != 0:
            _update_job(
                project_name,
                state="failed",
                error=f"pre_encode_dataset.py exited with code {rc}",
                finished_at=time.time(),
            )
            logger.error(
                "Pre-encoding failed (exit %s) · project=%s",
                rc, project_name,
            )
            return

        # Success — write manifest so SA3Trainer can verify AE compatibility.
        manifest = {
            "autoencoder": ae,
            "sample_size": sample_size,
            "created_at": time.time(),
            "source_clip_count": snapshot.get("total", 0),
            "encoded_count": latents_count(project_name),
        }
        try:
            (out_dir / "_meta.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to write latents manifest: %s", exc)

        _update_job(
            project_name,
            state="complete",
            current=manifest["encoded_count"],
            total=manifest["encoded_count"] or snapshot.get("total", 0),
            finished_at=time.time(),
        )
        logger.info(
            "Pre-encoding complete · project=%s · %d latent(s) · ae=%s",
            project_name, manifest["encoded_count"], ae,
        )

    except Exception as exc:
        _update_job(
            project_name,
            state="failed",
            error=str(exc),
            finished_at=time.time(),
        )
        logger.exception("Pre-encoding crashed for project=%s", project_name)
    finally:
        _pre_encode_processes.pop(project_name, None)
