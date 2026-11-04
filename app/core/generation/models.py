"""The generation models Fragmenta knows, and which engine runs each."""
from typing import Dict, Tuple

# model_id -> (sa3_name passed to StableAudioModel.from_pretrained,
#              "user-visible or base" tag, max duration seconds).
# Kept in sync manually with _SA3_CATALOG in app/core/model_manager.py.
MODEL_INFO: Dict[str, Tuple[str, str, int]] = {
    "sa3-small-music":      ("small-music",      "post", 120),
    "sa3-small-sfx":        ("small-sfx",        "post", 120),
    "sa3-medium":           ("medium",           "post", 380),
    "sa3-small-music-base": ("small-music-base", "base", 120),
    "sa3-small-sfx-base":   ("small-sfx-base",   "base", 120),
    "sa3-medium-base":      ("medium-base",      "base", 380),
    # MLX: the first value is the DiT's name in mlx_engine.DITS.
    "sa3-mlx-small-music":  ("sm-music",         "post", 120),
    "sa3-mlx-small-sfx":    ("sm-sfx",           "post", 120),
    "sa3-mlx-medium":       ("medium",           "post", 380),
}


def engine_for(model_id: str) -> str:
    return "mlx" if model_id.startswith("sa3-mlx-") else "torch"


def lora_family(model_id: str) -> str:
    """Models whose LoRAs are interchangeable share this: a base model, its distilled
    sibling, and that one's MLX build all have the same backbone."""
    name = model_id.replace("sa3-mlx-", "sa3-")
    return name[:-5] if name.endswith("-base") else name
