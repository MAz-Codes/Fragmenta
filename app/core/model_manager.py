"""Checkpoint Manager — SA3 catalog, HF downloads, license + auth.

Phase 2a in SA3_INTEGRATION_PLAN.md. Replaces the SA2-era SAO catalog.
Eight downloadable artifacts (3 post-trained + 3 base + 2 autoencoders);
each is fetched via `huggingface_hub.snapshot_download` with cooperative
cancel + progress reporting.

The Phase 2b frontend (CheckpointManagerWindow.js) consumes the JSON shapes
returned by the `/api/checkpoints/*` endpoints in `app/backend/app.py`.
"""
import json
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from huggingface_hub import get_token, snapshot_download, whoami
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError


# --- Catalog ------------------------------------------------------------------

# Approximate sizes; the frontend can refine these by hitting
# `huggingface_hub.HfApi().model_info(repo_id)` lazily. Numbers come from the
# HF model cards (paragraph parameter counts × bytes/param, rounded).
_SA3_CATALOG: Dict[str, Dict[str, Any]] = {
    # --- Generation models (post-trained) ----------------------------------
    "sa3-small-music": {
        "user_visible": True,
        "kind": "post-trained",
        "name": "Small - Music",
        "sa3_name": "small-music",
        "repo": "stabilityai/stable-audio-3-small-music",
        "size_bytes": 2_270_000_000,
        "hardware": "cpu",                       # CPU / MPS / CUDA all work
        "max_duration_sec": 120,
        "description": "Fast distilled music generation. Locked to 8 steps, cfg 1.0.",
    },
    "sa3-small-sfx": {
        "user_visible": True,
        "kind": "post-trained",
        "name": "Small - SFX",
        "sa3_name": "small-sfx",
        "repo": "stabilityai/stable-audio-3-small-sfx",
        "size_bytes": 2_270_000_000,
        "hardware": "cpu",
        "max_duration_sec": 120,
        "description": "Fast distilled SFX/foley generation. Locked to 8 steps, cfg 1.0.",
    },
    "sa3-medium": {
        "user_visible": True,
        "kind": "post-trained",
        "name": "Medium",
        "sa3_name": "medium",
        "repo": "stabilityai/stable-audio-3-medium",
        "size_bytes": 9_220_000_000,
        "hardware": "cuda+flash-attn",
        "max_duration_sec": 380,
        "description": "Fast distilled hi-fi generation, up to 380s. Locked to 8 steps, cfg 1.0.",
    },
    # --- Base checkpoints (full artist control) ----------------------------
    # These are the CFG-aware pre-distillation models. Slower (~50 steps,
    # cfg ~7), but the user controls cfg_scale, steps, and the inference
    # trajectory. Also the canonical targets for LoRA training.
    "sa3-small-music-base": {
        "user_visible": True,
        "kind": "base",
        "name": "Small - Music (Base)",
        "sa3_name": "small-music-base",
        "repo": "stabilityai/stable-audio-3-small-music-base",
        "size_bytes": 2_270_000_000,
        "hardware": "cpu",
        "max_duration_sec": 120,
        "description": "CFG-aware base. Full control over cfg_scale, steps. Slower than distilled.",
    },
    "sa3-small-sfx-base": {
        "user_visible": True,
        "kind": "base",
        "name": "Small - SFX (Base)",
        "sa3_name": "small-sfx-base",
        "repo": "stabilityai/stable-audio-3-small-sfx-base",
        "size_bytes": 2_270_000_000,
        "hardware": "cpu",
        "max_duration_sec": 120,
        "description": "CFG-aware base. Full control over cfg_scale, steps. Slower than distilled.",
    },
    "sa3-medium-base": {
        "user_visible": True,
        "kind": "base",
        "name": "Medium (Base)",
        "sa3_name": "medium-base",
        "repo": "stabilityai/stable-audio-3-medium-base",
        "size_bytes": 9_220_000_000,
        "hardware": "cuda+flash-attn",
        "max_duration_sec": 380,
        "description": "CFG-aware base. Full control over cfg_scale, steps. Slower than distilled.",
    },
    # Standalone autoencoders: the AE is bundled INSIDE each DiT repo
    # already (StableAudioModel.from_pretrained loads it from there), so
    # we don't surface SAME-S / SAME-L in the manager. They remain
    # downloadable via /api/checkpoints?include=all for advanced uses
    # (autoencoder-only workflows, pre-encoding datasets for training).
    "sa3-same-s": {
        "user_visible": False,
        "kind": "autoencoder",
        "name": "SAME-S",
        "sa3_name": "same-s",
        "repo": "stabilityai/SAME-S",
        "size_bytes": 530_000_000,
        "hardware": "cpu",
        "description": "Standalone autoencoder (266M). Already bundled with the small-* DiTs.",
    },
    "sa3-same-l": {
        "user_visible": False,
        "kind": "autoencoder",
        "name": "SAME-L",
        "sa3_name": "same-l",
        "repo": "stabilityai/SAME-L",
        "size_bytes": 3_400_000_000,
        "hardware": "cuda",
        "description": "Standalone autoencoder (1.7B). Already bundled with medium.",
    },
}

# --- Job state for in-flight downloads ----------------------------------------

@dataclass
class _DownloadJob:
    """In-memory record of one download attempt."""
    job_id: str
    model_id: str
    status: str = "queued"             # queued | running | complete | failed | cancelled
    downloaded_bytes: int = 0
    total_bytes: int = 0
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    _cancel_flag: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "status": self.status,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class _DownloadCancelled(Exception):
    """Raised inside the tqdm hook when a job's cancel flag fires."""


# --- ModelManager -------------------------------------------------------------

class ModelManager:
    """Owns the SA3 catalog and the on-disk pretrained directory."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.models_dir: Path = config.get_path("models_pretrained")
        self.models_dir.mkdir(exist_ok=True, parents=True)

        # Single canonical store: <app>/models/pretrained/sa3/hub/ in HF
        # cache layout. snapshot_download, hf_hub_download, training, and
        # inference all read/write here via HF_HUB_CACHE.
        self.hub_dir: Path = self.models_dir / "sa3" / "hub"
        self.hub_dir.mkdir(exist_ok=True, parents=True)
        # setdefault — let an external override win (e.g. Docker container
        # mounting a separate cache volume).
        os.environ.setdefault("HF_HUB_CACHE", str(self.hub_dir))

        # available_models is exposed for backwards compat with the existing
        # /api/models/available endpoint. New code should use get_catalog().
        self.available_models: Dict[str, Dict] = {
            mid: dict(meta) for mid, meta in _SA3_CATALOG.items()
        }

        self._jobs: Dict[str, _DownloadJob] = {}
        self._jobs_lock = threading.Lock()

    # --- Catalog --------------------------------------------------------------

    def get_catalog(self, include_hidden: bool = False) -> List[Dict[str, Any]]:
        """Checkpoint Manager catalog with per-item state.

        Default returns only user-visible entries (the three generation
        models). `include_hidden=True` also returns base + standalone-AE
        entries — used by the Phase 5 training subprocess to ensure the
        right base variant is on disk before kicking train_lora.py.
        """
        return [
            self._catalog_entry(mid)
            for mid, info in _SA3_CATALOG.items()
            if include_hidden or info.get("user_visible")
        ]

    def _catalog_entry(self, model_id: str) -> Dict[str, Any]:
        info = _SA3_CATALOG[model_id]
        downloaded = self.is_model_downloaded(model_id)
        bytes_total = 0
        if downloaded:
            for d in (self._hub_cache_dir_for(model_id), self._legacy_flat_dir_for(model_id)):
                if d.exists():
                    bytes_total += self._dir_size(d)
        return {
            "id": model_id,
            "kind": info.get("kind"),
            "name": info["name"],
            "sa3_name": info["sa3_name"],
            "repo": info["repo"],
            "size_bytes": info["size_bytes"],
            "hardware": info["hardware"],
            "max_duration_sec": info.get("max_duration_sec"),
            "description": info["description"],
            "user_visible": info.get("user_visible", False),
            "downloaded": downloaded,
            "downloaded_bytes": bytes_total,
        }

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        if model_id not in _SA3_CATALOG:
            return None
        return self._catalog_entry(model_id)

    # --- Filesystem layout ----------------------------------------------------

    def _hub_cache_dir_for(self, model_id: str) -> Path:
        """HF-cache-shaped directory inside the app folder."""
        info = _SA3_CATALOG.get(model_id)
        if info is None:
            return self.hub_dir / "_unknown"
        safe = "models--" + info["repo"].replace("/", "--")
        return self.hub_dir / safe

    def _legacy_flat_dir_for(self, model_id: str) -> Path:
        """Pre-unification per-model dir. Read-only fallback for migration."""
        return self.models_dir / "sa3" / model_id

    def _local_dir_for(self, model_id: str) -> Path:
        """Public: returns the canonical (HF cache) directory for a model."""
        return self._hub_cache_dir_for(model_id)

    def is_model_downloaded(self, model_id: str) -> bool:
        if model_id not in _SA3_CATALOG:
            return False
        # Canonical: HF cache layout under <app>/models/pretrained/sa3/hub/.
        hub = self._hub_cache_dir_for(model_id)
        if hub.is_dir():
            snaps = hub / "snapshots"
            if snaps.is_dir():
                for sub in snaps.iterdir():
                    if any(sub.rglob("*.safetensors")):
                        return True
        # Fallback: legacy flat layout (predates the unification). Counts as
        # downloaded for inference purposes; trainer will re-stage into hub.
        legacy = self._legacy_flat_dir_for(model_id)
        if legacy.is_dir() and any(legacy.rglob("*.safetensors")):
            return True
        return False

    # --- HF auth --------------------------------------------------------------

    @staticmethod
    def hf_auth_status() -> Dict[str, Any]:
        token = get_token()
        if not token:
            return {"signed_in": False, "username": None}
        try:
            user = whoami(token=token)
            return {"signed_in": True, "username": user.get("name") or user.get("fullname")}
        except Exception as err:
            return {"signed_in": False, "username": None, "error": str(err)}

    # --- Downloads ------------------------------------------------------------

    def start_download(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Spawn a background download job. Returns the job descriptor."""
        if model_id not in _SA3_CATALOG:
            return {"error": f"Unknown checkpoint: {model_id}"}

        job = _DownloadJob(
            job_id=str(uuid.uuid4()),
            model_id=model_id,
            total_bytes=_SA3_CATALOG[model_id]["size_bytes"],
        )
        with self._jobs_lock:
            self._jobs[job.job_id] = job

        thread = threading.Thread(
            target=self._run_download,
            args=(job, progress_callback),
            daemon=True,
            name=f"sa3-download:{model_id}",
        )
        job._thread = thread
        thread.start()
        return job.to_dict()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._jobs_lock:
            return [j.to_dict() for j in self._jobs.values()]

    def cancel_job(self, job_id: str) -> bool:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status not in ("queued", "running"):
            return False
        job._cancel_flag.set()
        return True

    def _run_download(
        self,
        job: _DownloadJob,
        progress_callback: Optional[Callable[[int, str], None]],
    ) -> None:
        info = _SA3_CATALOG[job.model_id]
        job.status = "running"
        job.started_at = datetime.now().isoformat()

        cache_dir = self._hub_cache_dir_for(job.model_id).parent  # = self.hub_dir
        cache_dir.mkdir(exist_ok=True, parents=True)
        target = self._hub_cache_dir_for(job.model_id)

        token = get_token()

        try:
            with _tqdm_progress_hook(job, progress_callback):
                # Write into hub/ in HF cache layout. snapshot_download in
                # hf-hub 1.x populates `<cache_dir>/models--<org>--<name>/`
                # with the blobs/refs/snapshots structure that
                # hf_hub_download() and StableAudioModel.from_pretrained()
                # both consume.
                snapshot_download(
                    repo_id=info["repo"],
                    cache_dir=str(cache_dir),
                    token=token,
                    allow_patterns=[
                        "*.safetensors", "*.json", "*.txt", "*.model",
                        "tokenizer*", "*.tiktoken",
                    ],
                )
            job.status = "complete"
            job.downloaded_bytes = self._dir_size(target)
            if progress_callback:
                progress_callback(100, f"Downloaded {info['name']}")
        except _DownloadCancelled:
            job.status = "cancelled"
            job.error = "Cancelled by user"
            shutil.rmtree(target, ignore_errors=True)
        except GatedRepoError as err:
            job.status = "failed"
            job.error = f"hf_auth_required: {err}"
        except RepositoryNotFoundError as err:
            job.status = "failed"
            job.error = f"Repository not found: {err}"
        except Exception as err:
            job.status = "failed"
            job.error = str(err)
        finally:
            job.finished_at = datetime.now().isoformat()

    # --- Delete ---------------------------------------------------------------

    def delete_model(self, model_id: str) -> bool:
        if model_id not in _SA3_CATALOG:
            return False
        # Remove both the canonical hub copy and the legacy flat copy if
        # they exist. Either being present is enough to consider the
        # model "downloaded", so both must be cleaned for the row to
        # flip back to "Get".
        hub = self._hub_cache_dir_for(model_id)
        legacy = self._legacy_flat_dir_for(model_id)
        any_existed = hub.exists() or legacy.exists()
        if hub.exists():
            shutil.rmtree(hub, ignore_errors=True)
        if legacy.exists():
            shutil.rmtree(legacy, ignore_errors=True)
        return any_existed and not (hub.exists() or legacy.exists())

    # --- Storage --------------------------------------------------------------

    def get_storage_info(self) -> Dict[str, Any]:
        per_model: List[Dict[str, Any]] = []
        total_used = 0
        for mid in _SA3_CATALOG:
            bytes_ = 0
            for d in (self._hub_cache_dir_for(mid), self._legacy_flat_dir_for(mid)):
                if d.exists():
                    bytes_ += self._dir_size(d)
            per_model.append({
                "id": mid,
                "downloaded": self.is_model_downloaded(mid),
                "bytes": bytes_,
            })
            total_used += bytes_
        return {
            "total_used_bytes": total_used,
            "total_free_bytes": shutil.disk_usage(self.models_dir).free,
            "per_model": per_model,
        }

    # --- Helpers --------------------------------------------------------------

    @staticmethod
    def _dir_size(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

# --- tqdm hook ----------------------------------------------------------------

import contextlib

@contextlib.contextmanager
def _tqdm_progress_hook(
    job: _DownloadJob,
    progress_callback: Optional[Callable[[int, str], None]],
):
    """Monkey-patch tqdm so snapshot_download updates flow into the job state.

    `snapshot_download` doesn't expose a progress callback. tqdm is its
    internal progress bar — we wrap `update` to update job state and raise
    `_DownloadCancelled` when the job's cancel flag fires.
    """
    from tqdm.auto import tqdm
    original_init = tqdm.__init__

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        original_update = self.update

        def new_update(n: int = 1) -> Any:
            if job._cancel_flag.is_set():
                raise _DownloadCancelled()
            result = original_update(n)
            if self.total:
                job.downloaded_bytes = max(job.downloaded_bytes, self.n)
                if job.total_bytes < self.total:
                    job.total_bytes = self.total
                if progress_callback:
                    pct = int(self.n / self.total * 100) if self.total else 0
                    mb_done = self.n / (1024 * 1024)
                    mb_total = self.total / (1024 * 1024)
                    progress_callback(pct, f"Downloading: {mb_done:.1f}MB / {mb_total:.1f}MB")
            return result

        self.update = new_update  # type: ignore[method-assign]

    tqdm.__init__ = patched_init  # type: ignore[method-assign]
    try:
        yield
    finally:
        tqdm.__init__ = original_init  # type: ignore[method-assign]
