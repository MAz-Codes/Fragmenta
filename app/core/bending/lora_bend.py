"""Adapter-file bending & Model Blending (Phase 2).

Bends trained LoRA .safetensors files into NEW adapter files — this is the
"fix the bend into a persistent instrument" move that Kotowski & Font name
as the missing piece of their tool. Nothing is modified in place: the
source checkpoints stay pristine and the bent/blended result is written as
a new run under models/fine_tuned/<name>/checkpoints/, so it appears in
the existing LoRA picker with no extra plumbing.

Two entry points:

    bend_lora(src, ops, ...)      apply scale / noise / mask ops to the
                                  adapter's tensors (optionally filtered by
                                  key substring)
    blend_loras(a, b, k, ...)     Model Blending — linear interpolation
                                  z = (1-k)·A + k·B over matching keys.
                                  Blending two checkpoints of one training
                                  run plays the training trajectory itself.

Safetensors metadata (base_model etc.) is carried over from the (first)
source, plus a `bend` record describing what was done — bent adapters stay
reproducible and identifiable.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_NAME_RE = re.compile(r"^[A-Za-z0-9._ -]{1,80}$")


class LoraBendError(ValueError):
    pass


def _load(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    from safetensors import safe_open
    tensors: Dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        meta = dict(f.metadata() or {})
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors, meta


def _save(path: Path, tensors: Dict[str, torch.Tensor], meta: Dict[str, str]) -> None:
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={k: str(v) for k, v in meta.items()})


def _output_path(fine_tuned_dir: Path, name: str) -> Path:
    # The name becomes a directory: a leading dot would allow "..".
    if not _NAME_RE.match(name or "") or name.startswith("."):
        raise LoraBendError(
            "Output name must be 1-80 chars of letters, digits, dot, dash, "
            "underscore or space, and must not start with a dot.")
    run_dir = fine_tuned_dir / name / "checkpoints"
    if not run_dir.resolve().is_relative_to(Path(fine_tuned_dir).resolve()):
        raise LoraBendError("Output name resolves outside models/fine_tuned/.")
    out = run_dir / "bent.safetensors"
    counter = 2
    while out.exists():
        out = run_dir / f"bent_{counter}.safetensors"
        counter += 1
    return out


def _matches(key: str, keys_filter: Optional[List[str]]) -> bool:
    if not keys_filter:
        return True
    return any(sub in key for sub in keys_filter)


def bend_lora(src: Path, ops: List[Dict[str, Any]], *, fine_tuned_dir: Path,
              output_name: str) -> Path:
    """Apply an op chain to one adapter file. Ops:
        {"op": "scale", "factor": f, "keys": [substr, ...]?}
        {"op": "noise", "amount": a, "seed": s, "keys": ...}
        {"op": "mask", "keys": [...]}          # zero matching tensors
    """
    tensors, meta = _load(Path(src))
    for i, op in enumerate(ops or []):
        kind = str(op.get("op", ""))
        keys_filter = op.get("keys") or None
        if kind == "scale":
            factor = float(op.get("factor", 1.0))
            for k in tensors:
                if _matches(k, keys_filter) and tensors[k].dtype.is_floating_point:
                    tensors[k] = tensors[k] * factor
        elif kind == "noise":
            amount = float(op.get("amount", 0.0))
            g = torch.Generator().manual_seed(int(op.get("seed", 0)) & 0x7FFFFFFF)
            for k in tensors:
                if _matches(k, keys_filter) and tensors[k].dtype.is_floating_point:
                    t = tensors[k]
                    std = t.float().std()
                    if not torch.isfinite(std) or std == 0:
                        std = torch.tensor(1.0)
                    noise = torch.randn(t.shape, generator=g, dtype=torch.float32)
                    tensors[k] = (t.float() + noise * std * amount).to(t.dtype)
        elif kind == "mask":
            for k in tensors:
                if _matches(k, keys_filter):
                    tensors[k] = torch.zeros_like(tensors[k])
        else:
            raise LoraBendError(f"ops[{i}]: unknown op {kind!r}.")

    meta["bend"] = json.dumps({
        "kind": "bend", "source": Path(src).name, "ops": ops,
        "ts": time.time(),
    })
    out = _output_path(fine_tuned_dir, output_name)
    _save(out, tensors, meta)
    return out


def blend_loras(a: Path, b: Path, k: float, *, fine_tuned_dir: Path,
                output_name: str) -> Tuple[Path, List[str]]:
    """Model Blending: z = (1-k)·A + k·B over matching keys; keys missing
    from B or shape-mismatched keep A's tensor (reported as warnings)."""
    k = max(0.0, min(1.0, float(k)))
    ta, meta_a = _load(Path(a))
    tb, meta_b = _load(Path(b))

    base_a = meta_a.get("base_model") or meta_a.get("base_model_id")
    base_b = meta_b.get("base_model") or meta_b.get("base_model_id")
    strip = lambda m: m[:-5] if m and m.endswith("-base") else m
    if base_a and base_b and strip(base_a) != strip(base_b):
        raise LoraBendError(
            f"Adapters are trained against different backbones "
            f"({base_a} vs {base_b}) — blending needs matching architectures.")

    warnings: List[str] = []
    out_tensors: Dict[str, torch.Tensor] = {}
    for key, tens_a in ta.items():
        tens_b = tb.get(key)
        if tens_b is None or tens_b.shape != tens_a.shape \
                or not tens_a.dtype.is_floating_point:
            if tens_b is None:
                warnings.append(f"{key}: missing in B; kept A.")
            elif tens_b.shape != tens_a.shape:
                warnings.append(f"{key}: shape mismatch; kept A.")
            out_tensors[key] = tens_a
            continue
        out_tensors[key] = (
            tens_a.float() * (1.0 - k) + tens_b.float() * k
        ).to(tens_a.dtype)

    meta = dict(meta_a)
    meta["bend"] = json.dumps({
        "kind": "blend", "a": Path(a).name, "b": Path(b).name, "k": k,
        "ts": time.time(),
    })
    out = _output_path(fine_tuned_dir, output_name)
    _save(out, out_tensors, meta)
    return out, warnings
