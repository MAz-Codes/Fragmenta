"""What every generation engine shares: the request, and the engine's job.

An engine only loads models, encodes text, samples and decodes. Everything else
is written once, in AudioGenerator: validating and defaulting the request,
reading input audio, stop and progress, the output file and its AI-output tags.
So a feature added to the request reaches every engine, or fails the contract
test (tests/test_engine_contract.py) on the one that ignores it.

No torch or MLX here: this module has to import on any machine.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

# (sample rate, float32 [channels, samples]), always stereo.
AudioInput = Tuple[int, np.ndarray]
# Called per sampler step with at least {"i": step_index}. It raises to stop the run.
StepCallback = Callable[[Dict[str, Any]], None]


@dataclass
class GenerationRequest:
    prompt: str
    model_id: str
    duration: float
    steps: int
    cfg_scale: float
    seed: int
    negative_prompt: Optional[str] = None
    batch_size: int = 1
    chunked_decode: Optional[bool] = None
    init_audio: Optional[AudioInput] = None
    init_noise_level: float = 1.0
    inpaint_audio: Optional[AudioInput] = None
    inpaint_starts: Optional[List[float]] = None
    inpaint_ends: Optional[List[float]] = None


@dataclass(frozen=True)
class Capabilities:
    """What the UI may offer on an engine. It asks this, never the engine's name."""
    audio_to_audio: bool = True
    inpaint: bool = True
    multi_region_inpaint: bool = True
    loras: bool = True
    live_lora_strength: bool = True
    training: bool = False
    bending: bool = False


class Engine(ABC):
    name: str = ""
    capabilities: Capabilities = Capabilities()
    # GenerationRequest fields this engine acts on. A new field fails the contract
    # test until each engine lists it here.
    handled_fields: FrozenSet[str] = frozenset()

    def __init__(self, config: Any) -> None:
        self.config = config
        self.model: Any = None
        self.model_id: Optional[str] = None
        self.device: Optional[str] = None
        # [{"path": str, "strength": float}, ...] currently applied to the model.
        self.loaded_loras: List[Dict[str, Any]] = []

    @classmethod
    def available(cls) -> bool:
        """Whether this engine can run on this machine."""
        return True

    @abstractmethod
    def load(self, model_id: str, device: Optional[str] = None, half: bool = True) -> None:
        """Make this model ready; a no-op when it already is."""

    @abstractmethod
    def unload(self) -> None:
        """Drop the model and everything tied to it."""

    @abstractmethod
    def apply_loras(self, loras: List[Dict[str, Any]]) -> None:
        """Make the loaded model run with exactly this stack."""

    @abstractmethod
    def set_lora_strength(self, index: int, strength: float) -> bool:
        """Change one applied LoRA's strength. False if there's no such slot."""

    @abstractmethod
    def generate(self, request: GenerationRequest, on_step: StepCallback) -> np.ndarray:
        """Run the loaded model. Returns the first clip as [channels, samples] at
        44.1 kHz, unclipped: the shared output path limits it."""
