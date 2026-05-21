"""SA3 inference engine — stub.

This module is the Phase 3 rewrite target in SA3_INTEGRATION_PLAN.md.
For Phase 1 it provides a structurally-compatible stub: the backend boots
and the import surface is preserved, but any actual generation call raises
NotImplementedError so callers fail fast and loud.

The Phase 3 implementation will be a thin wrapper around
`stable_audio_3.StableAudioModel.from_pretrained(...)` (see Appendix A of
the plan), with multi-LoRA stacking via `stable_audio_3.models.lora`
(Phase 4) and bars-mode duration passthrough (Phase 6).
"""
from typing import Any


class GenerationStopped(Exception):
    """Raised when an in-flight generation is interrupted by a stop request."""


class AudioGenerator:
    """Phase 3 will rewrite this class against the SA3 API."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.model: Any = None
        self._stop_requested: bool = False

    def request_stop(self) -> bool:
        """Set the cooperative stop flag. Returns True if a stop was newly set."""
        if self._stop_requested:
            return False
        self._stop_requested = True
        return True

    def generate_audio(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError(
            "SA3 inference engine not yet wired (Phase 3 of SA3_INTEGRATION_PLAN.md). "
            "Use git tag v0.1.x-legacy for the working SA2 implementation."
        )
