"""First-run starter model: the state behind the "Set up your first model" card.

Someone with no generation model gets offered one that needs no account. The
card keeps coming back until a model is on disk, or one is downloading; closing
it decides nothing. Everything here is derived from what's on disk, plus
``<user_data>/config/starter_model.json``, which remembers the agreement and
whether a download was under way when the app quit.
"""
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.core.config import get_config

STARTER_MODEL = "sa3-small-music-base"
# Date the texts in licenses/ were taken. Change it when they change, and
# everyone is asked to agree again.
TERMS_VERSION = "2026-10-22"
DISK_MARGIN_BYTES = 1_000_000_000
GENERATION_KINDS = ("post-trained", "base")

LICENSE_TEXTS: Dict[str, Dict[str, str]] = {
    "community": {
        "title": "Stability AI Community License",
        "tab": "Community License",
        "file": "STABILITY_AI_COMMUNITY_LICENSE.md",
        "url": "https://stability.ai/community-license-agreement",
    },
    "gemma": {
        "title": "Gemma Terms of Use",
        "tab": "Gemma Terms",
        "file": "GEMMA_TERMS_OF_USE.md",
        "url": "https://ai.google.dev/gemma/terms",
    },
    "stability-aup": {
        "title": "Stability AI Acceptable Use Policy",
        "tab": "Stability Use Policy",
        "file": "STABILITY_AI_ACCEPTABLE_USE_POLICY.md",
        "url": "https://stability.ai/use-policy",
    },
    "gemma-aup": {
        "title": "Gemma Prohibited Use Policy",
        "tab": "Gemma Use Policy",
        "file": "GEMMA_PROHIBITED_USE_POLICY.md",
        "url": "https://ai.google.dev/gemma/prohibited_use_policy",
    },
}

_OFFLINE_SIGNS = (
    "connecterror", "connection", "timed out", "timeout", "name resolution", "getaddrinfo",
    "offline", "proxy", "check your internet", "localentrynotfound",
)
_SERVER_SIGNS = (" 500", " 502", " 503", " 504", "server error", "service unavailable", "bad gateway")


def _state_path() -> Path:
    return get_config().user_data_dir / "config" / "starter_model.json"


def _load_state() -> Dict[str, Any]:
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(**changes: Any) -> Dict[str, Any]:
    state = {**_load_state(), **changes}
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass
    return state


def _plain_error(raw: Optional[str]) -> str:
    text = (raw or "").lower()
    # Offline first: hf-hub raises it as an OSError, which the job words as a disk problem.
    if any(sign in text for sign in _OFFLINE_SIGNS):
        return "Couldn't reach Hugging Face. Check your internet connection and try again."
    if any(sign in text for sign in _SERVER_SIGNS):
        return "Hugging Face isn't responding right now. Try again in a few minutes."
    # Disk problems already come worded for people (model_manager._friendly_disk_error).
    if "disk" in text or "permission denied" in text or "read-only" in text or "path was too long" in text:
        return raw or ""
    return f"The download stopped: {raw}" if raw else "The download stopped."


def _downloaded_so_far(manager: Any) -> int:
    target = manager._local_dir_for(STARTER_MODEL)
    if not target.exists():
        return 0
    return sum(p.stat().st_size for p in target.rglob("*")
               if p.is_file() and not p.is_symlink() and p.suffix != ".incomplete")


def _latest_job(manager: Any) -> Optional[Dict[str, Any]]:
    jobs = [j for j in manager.list_jobs() if j["model_id"] == STARTER_MODEL]
    return max(jobs, key=lambda j: j.get("started_at") or "") if jobs else None


def status(manager: Any) -> Dict[str, Any]:
    """Everything the card needs. ``phase`` is one of: idle, downloading, paused,
    failed, interrupted (the app quit mid-download, so the card resumes it)."""
    state = _load_state()
    catalog = [c for c in manager.get_catalog() if c.get("kind") in GENERATION_KINDS]
    installed = any(c["downloaded"] for c in catalog)
    other_downloading = any(c["active_job"] for c in catalog if c["id"] != STARTER_MODEL)
    entry = manager.get_model_info(STARTER_MODEL)
    job = _latest_job(manager)
    in_flight = bool(job and job["status"] in ("queued", "running"))

    if installed and not state.get("completed_at"):
        state = _save_state(status="complete", completed_at=datetime.now().isoformat())

    error = None
    if in_flight:
        phase = "downloading"
    elif installed:
        phase = "idle"
    elif state.get("status") == "paused":
        phase = "paused"
    elif job and job["status"] == "failed":
        phase = "failed"
        error = _plain_error(job.get("error"))
    elif state.get("status") == "downloading":
        phase = "interrupted"
    else:
        phase = "idle"

    deployed = (os.environ.get("FRAGMENTA_DOCKER", "0") == "1"
                or os.environ.get("FRAGMENTA_HF_SPACE", "0") == "1"
                or bool(os.environ.get("SPACE_ID")))
    size = entry["size_bytes"]
    so_far = job["downloaded_bytes"] if in_flight else _downloaded_so_far(manager)
    so_far = size if installed else min(so_far, size)
    free = shutil.disk_usage(manager.models_dir).free
    return {
        "show": not deployed and not installed and not other_downloading,
        "installed": installed,
        # They had a model once and deleted it: the short version of the card.
        "returning": bool(state.get("completed_at")) and not installed,
        "phase": phase,
        "error": error,
        "model_id": STARTER_MODEL,
        "model_name": entry["name"],
        "size_bytes": size,
        "downloaded_bytes": so_far,
        "free_bytes": free,
        "needed_bytes": max(0, size - so_far) + DISK_MARGIN_BYTES,
        "terms_version": TERMS_VERSION,
        "terms_accepted": state.get("accepted_terms_version") == TERMS_VERSION,
        "licenses": [{"id": k, "title": v["title"], "tab": v["tab"], "url": v["url"]}
                     for k, v in LICENSE_TEXTS.items()],
    }


def agreement_error(accept_terms_version: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Why a no-account model can't download yet, or None. The agreement is
    recorded here, when someone who ticked the box presses Download. There's only
    one: every no-account model comes with these same license texts."""
    state = _load_state()
    if accept_terms_version == TERMS_VERSION:
        state = _save_state(accepted_terms_version=TERMS_VERSION, accepted_at=datetime.now().isoformat())
    if state.get("accepted_terms_version") != TERMS_VERSION:
        return {"error": "Please agree to the license terms first.", "code": "terms"}
    return None


def start(manager: Any, accept_terms_version: Optional[str] = None) -> Tuple[Dict[str, Any], int]:
    """Start or resume the download. Returns (body, HTTP status)."""
    refused = agreement_error(accept_terms_version)
    if refused:
        return refused, 403
    state = _load_state()

    size = manager.get_model_info(STARTER_MODEL)["size_bytes"]
    needed = max(0, size - _downloaded_so_far(manager)) + DISK_MARGIN_BYTES
    free = shutil.disk_usage(manager.models_dir).free
    if free < needed:
        return {"error": f"Not enough space (need {needed / 1e9:.1f} GB, have {free / 1e9:.1f} GB).",
                "code": "disk"}, 409

    job = manager.start_download(STARTER_MODEL, keep_partial=True)
    if job.get("error"):
        return {"error": job["error"], "code": "busy"}, 409
    updates: Dict[str, Any] = {"status": "downloading"}
    if not state.get("started_at"):
        updates["started_at"] = datetime.now().isoformat()
    _save_state(**updates)
    return job, 200


def pause(manager: Any) -> Dict[str, Any]:
    paused = [j["job_id"] for j in manager.list_jobs()
              if j["model_id"] == STARTER_MODEL and j["status"] in ("queued", "running")
              and manager.pause_job(j["job_id"])]
    if paused:
        _save_state(status="paused")
    return {"paused": paused}


def cancel(manager: Any) -> Dict[str, Any]:
    state = _load_state()
    cancelled = [j["job_id"] for j in manager.list_jobs()
                 if j["model_id"] == STARTER_MODEL and j["status"] in ("queued", "running")
                 and manager.cancel_job(j["job_id"])]
    # A paused download has no job left to stop, so its files go here.
    if not cancelled and state.get("status") == "paused":
        shutil.rmtree(manager._local_dir_for(STARTER_MODEL), ignore_errors=True)
    _save_state(status="idle")
    return {"cancelled": cancelled}


def license_text(license_id: str) -> Optional[Dict[str, str]]:
    info = LICENSE_TEXTS.get(license_id)
    if not info:
        return None
    path = Path(get_config().project_root) / "licenses" / info["file"]
    try:
        return {"title": info["title"], "url": info["url"], "text": path.read_text(encoding="utf-8")}
    except OSError:
        return None
