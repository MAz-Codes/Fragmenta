"""Checkpoint Manager — SA3 catalog, HF downloads, license + auth.

Phase 2a in SA3_INTEGRATION_PLAN.md. Replaces the SA2-era SAO catalog.
Eight downloadable artifacts (3 post-trained + 3 base + 2 autoencoders);
each is fetched via `huggingface_hub.snapshot_download` with cooperative
cancel + progress reporting.

The Phase 2b frontend (CheckpointManagerWindow.js) consumes the JSON shapes
returned by the `/api/checkpoints/*` endpoints in `app/backend/app.py`.
"""
import json
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

# SA3 Community License revision — bumped if upstream LICENSE text changes.
_SA3_LICENSE_VERSION = "stability-community-2026"

# Approximate sizes; the frontend can refine these by hitting
# `huggingface_hub.HfApi().model_info(repo_id)` lazily. Numbers come from the
# HF model cards (paragraph parameter counts × bytes/param, rounded).
_SA3_CATALOG: Dict[str, Dict[str, Any]] = {
    # --- Generation models (post-trained) ----------------------------------
    "sa3-small-music": {
        "group": "post-trained",
        "name": "Small — Music",
        "sa3_name": "small-music",
        "repo": "stabilityai/stable-audio-3-small-music",
        "size_bytes": 1_280_000_000,
        "hardware": "cpu",                       # CPU / MPS / CUDA all work
        "max_duration_sec": 120,
        "requires_ae": "sa3-same-s",
        "license": _SA3_LICENSE_VERSION,
        "description": "Fast music generation on CPU or any GPU.",
        "best_for": "Laptops, quick ideation, HF Spaces.",
    },
    "sa3-small-sfx": {
        "group": "post-trained",
        "name": "Small — SFX",
        "sa3_name": "small-sfx",
        "repo": "stabilityai/stable-audio-3-small-sfx",
        "size_bytes": 1_280_000_000,
        "hardware": "cpu",
        "max_duration_sec": 120,
        "requires_ae": "sa3-same-s",
        "license": _SA3_LICENSE_VERSION,
        "description": "Foley, ambience, and one-shot sound effects.",
        "best_for": "SFX libraries, game audio, prototyping.",
    },
    "sa3-medium": {
        "group": "post-trained",
        "name": "Medium",
        "sa3_name": "medium",
        "repo": "stabilityai/stable-audio-3-medium",
        "size_bytes": 5_400_000_000,
        "hardware": "cuda+flash-attn",
        "max_duration_sec": 380,
        "requires_ae": "sa3-same-l",
        "license": _SA3_LICENSE_VERSION,
        "description": "High-fidelity music + SFX, long-form (up to 380s).",
        "best_for": "Professional use on a CUDA GPU with Flash Attention 2.",
    },
    # --- Base models (for LoRA training) -----------------------------------
    "sa3-small-music-base": {
        "group": "base-for-lora",
        "name": "Small Music — Base",
        "sa3_name": "small-music-base",
        "repo": "stabilityai/stable-audio-3-small-music-base",
        "size_bytes": 1_280_000_000,
        "hardware": "cpu",
        "max_duration_sec": 120,
        "requires_ae": "sa3-same-s",
        "license": _SA3_LICENSE_VERSION,
        "description": "Pre-post-train music checkpoint; CFG-aware. Use as a LoRA base.",
        "best_for": "Training small-music LoRAs.",
    },
    "sa3-small-sfx-base": {
        "group": "base-for-lora",
        "name": "Small SFX — Base",
        "sa3_name": "small-sfx-base",
        "repo": "stabilityai/stable-audio-3-small-sfx-base",
        "size_bytes": 1_280_000_000,
        "hardware": "cpu",
        "max_duration_sec": 120,
        "requires_ae": "sa3-same-s",
        "license": _SA3_LICENSE_VERSION,
        "description": "Pre-post-train SFX checkpoint; CFG-aware. Use as a LoRA base.",
        "best_for": "Training small-sfx LoRAs.",
    },
    "sa3-medium-base": {
        "group": "base-for-lora",
        "name": "Medium — Base",
        "sa3_name": "medium-base",
        "repo": "stabilityai/stable-audio-3-medium-base",
        "size_bytes": 5_400_000_000,
        "hardware": "cuda+flash-attn",
        "max_duration_sec": 380,
        "requires_ae": "sa3-same-l",
        "license": _SA3_LICENSE_VERSION,
        "description": "Pre-post-train medium checkpoint; CFG-aware. Use as a LoRA base.",
        "best_for": "Training high-fidelity LoRAs on a CUDA GPU.",
    },
    # --- Autoencoders ------------------------------------------------------
    "sa3-same-s": {
        "group": "autoencoder",
        "name": "SAME-S",
        "sa3_name": "same-s",
        "repo": "stabilityai/SAME-S",
        "size_bytes": 700_000_000,
        "hardware": "cpu",
        "license": _SA3_LICENSE_VERSION,
        "description": "Small autoencoder (266M). Pairs with the small DiTs.",
        "best_for": "Required by every small-* model.",
        "paired_with": [
            "sa3-small-music", "sa3-small-sfx",
            "sa3-small-music-base", "sa3-small-sfx-base",
        ],
    },
    "sa3-same-l": {
        "group": "autoencoder",
        "name": "SAME-L",
        "sa3_name": "same-l",
        "repo": "stabilityai/SAME-L",
        "size_bytes": 3_500_000_000,
        "hardware": "cuda",
        "license": _SA3_LICENSE_VERSION,
        "description": "Large autoencoder (1.7B). Pairs with medium.",
        "best_for": "Required by sa3-medium and sa3-medium-base.",
        "paired_with": ["sa3-medium", "sa3-medium-base"],
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

        # available_models is exposed for backwards compat with the existing
        # /api/models/available endpoint. New code should use get_catalog().
        self.available_models: Dict[str, Dict] = {
            mid: dict(meta) for mid, meta in _SA3_CATALOG.items()
        }

        self.terms_file = Path("config/terms_accepted.json")
        self.terms_file.parent.mkdir(exist_ok=True)

        self._jobs: Dict[str, _DownloadJob] = {}
        self._jobs_lock = threading.Lock()

    # --- Catalog --------------------------------------------------------------

    def get_catalog(self) -> List[Dict[str, Any]]:
        """Full Checkpoint Manager catalog with per-item state."""
        return [self._catalog_entry(mid) for mid in _SA3_CATALOG]

    def _catalog_entry(self, model_id: str) -> Dict[str, Any]:
        info = _SA3_CATALOG[model_id]
        local_dir = self._local_dir_for(model_id)
        downloaded = self.is_model_downloaded(model_id)
        return {
            "id": model_id,
            "group": info["group"],
            "name": info["name"],
            "sa3_name": info["sa3_name"],
            "repo": info["repo"],
            "size_bytes": info["size_bytes"],
            "hardware": info["hardware"],
            "max_duration_sec": info.get("max_duration_sec"),
            "requires_ae": info.get("requires_ae"),
            "paired_with": info.get("paired_with", []),
            "license": info["license"],
            "description": info["description"],
            "best_for": info["best_for"],
            "downloaded": downloaded,
            "downloaded_bytes": self._dir_size(local_dir) if downloaded else 0,
            "terms_accepted": self.is_terms_accepted(model_id),
        }

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        if model_id not in _SA3_CATALOG:
            return None
        return self._catalog_entry(model_id)

    # --- Filesystem layout ----------------------------------------------------

    def _local_dir_for(self, model_id: str) -> Path:
        """Per-model subdirectory under models/pretrained/sa3/."""
        return self.models_dir / "sa3" / model_id

    def is_model_downloaded(self, model_id: str) -> bool:
        if model_id not in _SA3_CATALOG:
            return False
        local = self._local_dir_for(model_id)
        if not local.is_dir():
            return False
        # Any *.safetensors weight presence is enough for a positive signal;
        # the loader will surface a config-missing error later if needed.
        return any(local.rglob("*.safetensors"))

    # --- Pairing --------------------------------------------------------------

    def required_companions(self, model_id: str) -> List[str]:
        """Return checkpoint IDs that should be downloaded alongside `model_id`."""
        info = _SA3_CATALOG.get(model_id, {})
        ae = info.get("requires_ae")
        return [ae] if ae and not self.is_model_downloaded(ae) else []

    # --- Terms / license ------------------------------------------------------

    def get_license_text(self, model_id: str) -> Optional[str]:
        """Return the Stability AI Community License text for the modal.

        The *weights* are governed by the Stability AI Community License (the
        SA3 repo's own LICENSE is MIT and only covers the Python source —
        showing that to the user would be misleading).

        Resolution order:
          1. config/stability_ai_community_license.txt (canonical text, if bundled).
          2. A short authoritative-link summary so the user can still review
             and accept knowingly before download.
        """
        if model_id not in _SA3_CATALOG:
            return None

        bundled = Path("config/stability_ai_community_license.txt")
        if bundled.exists():
            try:
                return bundled.read_text(encoding="utf-8")
            except OSError:
                pass

        return (
            "Stability AI Community License Agreement\n"
            "==========================================\n\n"
            "All Stable Audio 3 weights (and the SAME autoencoders) are\n"
            "released under the Stability AI Community License. By clicking\n"
            "Accept, you agree to the terms at:\n\n"
            "    https://stability.ai/license\n\n"
            "Key terms (this summary is non-binding — read the full license):\n\n"
            "  * Free for non-commercial use and for commercial use by\n"
            "    individuals or organizations with annual revenue under\n"
            "    USD 1 million.\n"
            "  * Above that threshold, a paid enterprise license is required.\n"
            "  * You must include attribution (\"Powered by Stability AI\")\n"
            "    and the license notice in products built on these models.\n"
            "  * The use policy prohibits generating illegal, harmful, or\n"
            "    deceptive content. See the license for the full policy.\n\n"
            "If config/stability_ai_community_license.txt exists in your\n"
            "Fragmenta install, the canonical text is shown there instead\n"
            "of this summary.\n"
        )

    def is_terms_accepted(self, model_id: str) -> bool:
        if not self.terms_file.exists():
            return False
        try:
            with open(self.terms_file, "r") as fh:
                terms = json.load(fh)
        except Exception:
            return False
        entry = terms.get(model_id) or {}
        # Re-prompt if the license version moved (Phase 0 catches text drift).
        if entry.get("license") != _SA3_CATALOG.get(model_id, {}).get("license"):
            return False
        return bool(entry.get("accepted"))

    def accept_terms(self, model_id: str) -> bool:
        if model_id not in _SA3_CATALOG:
            return False
        info = _SA3_CATALOG[model_id]
        terms: Dict[str, Any] = {}
        if self.terms_file.exists():
            try:
                with open(self.terms_file, "r") as fh:
                    terms = json.load(fh)
            except Exception:
                terms = {}
        terms[model_id] = {
            "accepted": True,
            "accepted_at": datetime.now().isoformat(),
            "model_name": info["name"],
            "license": info["license"],
        }
        try:
            with open(self.terms_file, "w") as fh:
                json.dump(terms, fh, indent=2)
            return True
        except OSError as err:
            print(f"Error saving terms acceptance: {err}")
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
        if not self.is_terms_accepted(model_id):
            return {"error": "Terms not accepted for this checkpoint."}

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

        local_dir = self._local_dir_for(job.model_id)
        local_dir.mkdir(exist_ok=True, parents=True)

        token = get_token()  # uses standard ~/.cache/huggingface/token

        try:
            with _tqdm_progress_hook(job, progress_callback):
                snapshot_download(
                    repo_id=info["repo"],
                    local_dir=str(local_dir),
                    token=token,
                    # Slim the download: weights + small configs/tokenizers, no docs.
                    allow_patterns=[
                        "*.safetensors", "*.json", "*.txt", "*.model",
                        "tokenizer*", "*.tiktoken",
                    ],
                )
            job.status = "complete"
            job.downloaded_bytes = self._dir_size(local_dir)
            if progress_callback:
                progress_callback(100, f"Downloaded {info['name']}")
        except _DownloadCancelled:
            job.status = "cancelled"
            job.error = "Cancelled by user"
            shutil.rmtree(local_dir, ignore_errors=True)
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
        local = self._local_dir_for(model_id)
        if not local.exists():
            return False
        shutil.rmtree(local, ignore_errors=True)
        return not local.exists()

    # --- Storage --------------------------------------------------------------

    def get_storage_info(self) -> Dict[str, Any]:
        per_model: List[Dict[str, Any]] = []
        total_used = 0
        for mid in _SA3_CATALOG:
            local = self._local_dir_for(mid)
            bytes_ = self._dir_size(local) if local.exists() else 0
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

    # --- Backwards-compat shims (used by legacy endpoints; removed in 2b) -----

    def get_available_models(self) -> List[Dict]:
        return self.get_catalog()

    def is_model_downloaded_legacy(self, model_id: str) -> bool:
        # Phase 2a callers may still pass SA2-era IDs (stable-audio-open-*).
        # They unconditionally return False — the catalog only knows SA3.
        return self.is_model_downloaded(model_id)

    def download_model(self, model_id: str, progress_callback=None) -> bool:
        """Synchronous-feeling shim around start_download for legacy callers."""
        result = self.start_download(model_id, progress_callback=progress_callback)
        if "error" in result:
            print(f"Download init failed: {result['error']}")
            return False
        # Wait for the spawned thread (legacy callers expect blocking behaviour).
        job = self._jobs[result["job_id"]]
        if job._thread:
            job._thread.join()
        return job.status == "complete"

    def get_download_progress(self, model_id: str) -> Dict:
        with self._jobs_lock:
            jobs = [j for j in self._jobs.values() if j.model_id == model_id]
        if not jobs:
            return {"model_id": model_id, "downloaded": self.is_model_downloaded(model_id)}
        latest = max(jobs, key=lambda j: j.started_at or "")
        return latest.to_dict()


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
