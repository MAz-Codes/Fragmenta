"""Bendable-point registry for the SA3 pipeline.

The UI renders the model as a signal path:

    Prompt → Text encoder → cond embed → DiT blocks 1..N → sampler loop
           → VAE decode → Audio

Each stage below describes where bends can attach and how the session
resolves them to live modules. The registry is static-with-lookup: block
counts are read from the locally downloaded model_config.json when the
checkpoint exists, with conservative fallbacks otherwise, and every target
is re-verified against the live model at apply time (a missing target is a
warning, never a crash — vendor bumps must degrade gracefully).

Module paths are relative to the inner ConditionedDiffusionModelWrapper
(`StableAudioModel.model`):

    model.model                      DiffusionTransformer (the DiT)
    model.model.to_cond_embed        prompt-conditioning projection
    model.model.to_timestep_embed    timestep embedding projection
    model.model.transformer.layers.N TransformerBlock N
    pretransform.model.decoder       SAMEDecoder (VAE decode)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# model_id -> HF repo dirname fragment under models/pretrained/sa3/hub/
_HUB_DIRNAMES = {
    "sa3-small-music":      "models--stabilityai--stable-audio-3-small-music",
    "sa3-small-sfx":        "models--stabilityai--stable-audio-3-small-sfx",
    "sa3-medium":           "models--stabilityai--stable-audio-3-medium",
    "sa3-small-music-base": "models--stabilityai--stable-audio-3-small-music-base",
    "sa3-small-sfx-base":   "models--stabilityai--stable-audio-3-small-sfx-base",
    "sa3-medium-base":      "models--stabilityai--stable-audio-3-medium-base",
}

# Fallbacks for UI rendering before a checkpoint is downloaded. Small is
# verified from the shipped config (depth 20); medium's is an estimate and
# is corrected the moment its config is on disk.
_FALLBACK_DIT_DEPTH = {"small": 20, "medium": 24}
# The SA3 VAE (taae_v2) decodes in a single resampling stage (c_mults=[6]).
_FALLBACK_DECODER_BLOCKS = 1


def _local_model_config(model_id: str, pretrained_root: Path) -> Optional[dict]:
    dirname = _HUB_DIRNAMES.get(model_id)
    if not dirname:
        return None
    snaps = pretrained_root / "sa3" / "hub" / dirname / "snapshots"
    if not snaps.is_dir():
        return None
    for snap in sorted(snaps.iterdir()):
        cfg = snap / "model_config.json"
        if cfg.exists():
            try:
                with open(cfg) as fh:
                    return json.load(fh)
            except Exception:
                return None
    return None


def _dit_depth(model_id: str, cfg: Optional[dict]) -> int:
    if cfg:
        try:
            return int(cfg["model"]["diffusion"]["config"]["depth"])
        except Exception:
            pass
    return _FALLBACK_DIT_DEPTH["medium" if "medium" in model_id else "small"]


def _decoder_blocks(cfg: Optional[dict]) -> int:
    if cfg:
        try:
            c_mults = cfg["model"]["pretransform"]["config"]["decoder"]["config"]["c_mults"]
            return len(c_mults)
        except Exception:
            pass
    return _FALLBACK_DECODER_BLOCKS


def get_targets(model_id: str, pretrained_root: Path) -> Dict[str, Any]:
    """Full stage/target description for one model, for the Bend UI."""
    cfg = _local_model_config(model_id, pretrained_root)
    depth = _dit_depth(model_id, cfg)
    dec_blocks = _decoder_blocks(cfg)

    stages: List[Dict[str, Any]] = [
        {
            "stage": "cond",
            "label": "Cond embed",
            "kind": "module",
            "module": "model.model.to_cond_embed",
            "domains": ["activation", "weight"],
            "axes": {"feature": -1, "time": 1},
            "hint": ("Warps what the prompt means to the model — the words "
                     "stay, their meaning bends."),
            "size_hint": "small",
        },
        {
            "stage": "timestep",
            "label": "Timestep embed",
            "kind": "module",
            "module": "model.model.to_timestep_embed",
            "domains": ["activation", "weight"],
            "axes": {"feature": -1, "time": None},
            "hint": ("Confuses the model about where it is in denoising — "
                     "structural chaos."),
            "size_hint": "small",
        },
        {
            "stage": "dit",
            "label": "DiT blocks",
            "kind": "blocks",
            "module": "model.model.transformer.layers.{i}",
            "count": depth,
            "domains": ["activation", "weight", "structure"],
            "axes": {"feature": -1, "time": 1},
            "hint": ("The core playground. Early blocks shape structure and "
                     "composition; late blocks shape timbre and texture."),
            "size_hint": "large",
            # Weight sub-targets inside one block, resolved by subtree prefix.
            "weight_slots": [
                {"name": "self_attn", "label": "Attention"},
                {"name": "ff", "label": "Feed-forward"},
            ],
        },
        {
            "stage": "latent",
            "label": "Sampler latent",
            "kind": "latent",
            "domains": ["latent"],
            "axes": {"feature": 1, "time": -1},
            "hint": ("The evolving sound-in-progress, bent between denoising "
                     "steps. Use the step range: early = structure, late = "
                     "texture."),
            "size_hint": "small",
        },
        {
            "stage": "decoder",
            "label": "VAE decode",
            "kind": "blocks",
            "module": "pretransform.model.decoder.layers.{i}",
            # layers[0..2] are Transpose/Linear/Transpose plumbing; the
            # resampling blocks start at index 3.
            "block_offset": 3,
            "count": dec_blocks,
            "domains": ["activation", "weight"],
            "axes": {"feature": 1, "time": -1},
            "hint": ("Latent → audio. The closest stage to the loudspeaker — "
                     "bends here are the most immediately audible."),
            "size_hint": "medium",
        },
    ]

    return {
        "model_id": model_id,
        "config_found": cfg is not None,
        "dit_depth": depth,
        "decoder_blocks": dec_blocks,
        "stages": stages,
    }


def resolve_module_path(root, path: str):
    """Walk dotted-with-indices path ('a.b.3.c') from a module. Returns the
    module or None (missing targets degrade to warnings)."""
    node = root
    for part in path.split("."):
        if not part:
            continue
        if part.isdigit():
            try:
                node = node[int(part)]
            except Exception:
                return None
        else:
            node = getattr(node, part, None)
        if node is None:
            return None
    return node
