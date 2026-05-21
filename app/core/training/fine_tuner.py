"""SA3 LoRA training orchestrator — stub.

This module is the Phase 5 rewrite target in SA3_INTEGRATION_PLAN.md.
For Phase 1 it preserves the public surface so the backend boots and
endpoints respond with structured "not yet wired" errors, instead of
500s from missing imports.

The Phase 5 implementation will be a thin subprocess dispatcher around
`vendor/stable-audio-3/scripts/train_lora.py`, with caption materializer
in `app/core/training/sa3_lora_runner.py` (writes `<basename>.txt`
sidecars from `data/metadata.json`).
"""
from typing import Any, Dict, Optional

from app.core.config import get_config


_NOT_WIRED = (
    "SA3 LoRA training not yet wired (Phase 5 of SA3_INTEGRATION_PLAN.md). "
    "Use git tag v0.1.x-legacy for the working SA2/LoRAW implementation."
)


def get_base_model_configs() -> Dict[str, Dict[str, str]]:
    # SA3-TODO(Phase 2): the Checkpoint Manager catalog is the new source of truth.
    return get_config().model_configs


class FineTuner:
    """Phase 5 will rewrite this class to dispatch SA3's train_lora.py."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.training_process = None
        self.training_status: Dict[str, Any] = {
            "is_training": False,
            "status": "idle",
            "message": _NOT_WIRED,
        }

    def start_training(self) -> Dict[str, Any]:
        return {"error": _NOT_WIRED}

    def get_status(self) -> Dict[str, Any]:
        return dict(self.training_status)

    def stop_training(self) -> Dict[str, Any]:
        return {"error": "Nothing to stop — SA3 training pipeline not yet wired."}


_trainer: Optional[FineTuner] = None


def get_trainer() -> Optional[FineTuner]:
    return _trainer


def start_training(config: Dict[str, Any]) -> Dict[str, Any]:
    return {"error": _NOT_WIRED}


def get_training_status() -> Dict[str, Any]:
    return {
        "is_training": False,
        "status": "idle",
        "message": _NOT_WIRED,
    }


def stop_training() -> Dict[str, Any]:
    return {"error": "Nothing to stop — SA3 training pipeline not yet wired."}


def preview_training_plan(config: Dict[str, Any]) -> Dict[str, Any]:
    return {"error": _NOT_WIRED}
