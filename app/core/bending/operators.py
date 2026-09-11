"""Bend operators — pure tensor transformations.

Each operator is a pure function `(tensor, params, ctx) -> tensor` that must
preserve shape, dtype, and device. Operators never mutate their input.

`ctx` is an OpContext carrying everything an operator may need beyond its
own params: the axes that mean "features" and "time" for the tensor at hand
(they differ per bend stage — transformer activations are (B, seq, dim),
latents and decoder activations are (B, C, T), weights are (out, in) or
(n,)), plus a seeded torch.Generator for the stochastic operators so a
patch + seed reproduces bit-exactly.

Lineage per operator is noted inline — Broad et al. 2021/2022 (activation
taxonomy), Kotowski & Font NIME 2026 (parameter-level: scale / offset /
negate / roll / draw), torchbend (noise), Gillespie & Schachter DAFx 2022
(the protocol the whole registry serves).

Every application is followed by a NaN scrub in `apply_operator` — a bend
that explodes must degrade to silence/limits, never poison the run with
NaN/Inf (the WAV writer has a second scrub, but catching it here keeps
later stages meaningful).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch


@dataclass
class OpContext:
    feature_axis: int = -1     # axis whose entries are "features/channels"
    time_axis: Optional[int] = None   # sequence/time axis, None for weights
    generator: Optional[torch.Generator] = None  # CPU generator, pre-seeded

    def randn_like(self, t: torch.Tensor) -> torch.Tensor:
        # Draw on CPU with the seeded generator, then move — identical
        # results on cuda/mps/cpu for the same patch seed.
        return torch.randn(
            t.shape, generator=self.generator, dtype=torch.float32
        ).to(device=t.device, dtype=t.dtype)

    def rand_like(self, t: torch.Tensor) -> torch.Tensor:
        return torch.rand(
            t.shape, generator=self.generator, dtype=torch.float32
        ).to(device=t.device, dtype=t.dtype)


# --- Phase 1 ---------------------------------------------------------------

def _op_ablate(t, p, ctx):
    """Zero the targeted values. (Broad: ablation.)"""
    return torch.zeros_like(t)


def _op_scale(t, p, ctx):
    """Multiply. factor < 0 inverts (Broad's inversion; K&F's negate)."""
    return t * float(p.get("factor", 1.0))


def _op_bias(t, p, ctx):
    """Add a constant offset. (K&F: offset.)"""
    return t + float(p.get("offset", 0.0))


def _op_roll(t, p, ctx):
    """Circular shift along the feature axis (K&F's most productive op).

    `axis: "time"` rolls along the time axis instead where one exists.
    Shift is a fraction of the axis length so one slider works across
    layers of any width.
    """
    axis = ctx.time_axis if (p.get("axis") == "time" and ctx.time_axis is not None) \
        else ctx.feature_axis
    n = t.shape[axis]
    shift = int(round(float(p.get("amount", 0.0)) * n)) % max(n, 1)
    if shift == 0:
        return t
    return torch.roll(t, shifts=shift, dims=axis)


def _op_threshold(t, p, ctx):
    """Zero everything below |cutoff|; `hard` additionally binarises the
    survivors to ±1 × their sign scale. (Broad: binary threshold.)"""
    cutoff = float(p.get("cutoff", 0.0))
    mask = t.abs() >= cutoff
    if bool(p.get("hard", False)):
        return torch.where(mask, torch.sign(t) * cutoff, torch.zeros_like(t))
    return torch.where(mask, t, torch.zeros_like(t))


def _op_noise(t, p, ctx):
    """Add seeded gaussian noise scaled to the tensor's own std, so one
    amount slider behaves comparably across layers. (torchbend.)"""
    amount = float(p.get("amount", 0.0))
    if amount == 0.0:
        return t
    std = t.float().std()
    if not torch.isfinite(std) or std == 0:
        std = torch.tensor(1.0)
    return t + ctx.randn_like(t) * (std.to(t.dtype) * amount)


def _op_clamp(t, p, ctx):
    return t.clamp(float(p.get("lo", -1.0)), float(p.get("hi", 1.0)))


def _op_quantize(t, p, ctx):
    """Bit-crush: snap values to `levels` steps across the tensor's range."""
    levels = max(2, int(p.get("levels", 8)))
    lo, hi = t.min(), t.max()
    span = (hi - lo)
    if not torch.isfinite(span) or span == 0:
        return t
    q = torch.round((t - lo) / span * (levels - 1)) / (levels - 1)
    return q * span + lo


# --- Phase 2: temporal & structural ---------------------------------------

def _smooth_1d(t: torch.Tensor, axis: int, kernel: int, mode: str) -> torch.Tensor:
    """Shared engine for smear/erode/dilate: sliding window along `axis`
    via unfold. mode: 'mean' | 'min' | 'max'."""
    kernel = max(1, int(kernel))
    if kernel <= 1:
        return t
    moved = t.movedim(axis, -1)
    pad = kernel // 2
    # Replicate-pad the last dim so the output length matches the input.
    padded = torch.cat(
        [moved[..., :1].expand(*moved.shape[:-1], pad),
         moved,
         moved[..., -1:].expand(*moved.shape[:-1], kernel - 1 - pad)], dim=-1)
    windows = padded.unfold(-1, kernel, 1)          # (..., L, kernel)
    if mode == "mean":
        out = windows.mean(dim=-1)
    elif mode == "min":
        out = windows.min(dim=-1).values
    else:
        out = windows.max(dim=-1).values
    return out.movedim(-1, axis)


def _time_or_feature_axis(p, ctx):
    if p.get("axis") == "feature" or ctx.time_axis is None:
        return ctx.feature_axis
    return ctx.time_axis


def _op_smear(t, p, ctx):
    """1-D blur along the time axis — Broad's morphological filters
    translated to sequence data."""
    return _smooth_1d(t, _time_or_feature_axis(p, ctx), p.get("kernel", 5), "mean")


def _op_erode(t, p, ctx):
    return _smooth_1d(t, _time_or_feature_axis(p, ctx), p.get("kernel", 5), "min")


def _op_dilate(t, p, ctx):
    return _smooth_1d(t, _time_or_feature_axis(p, ctx), p.get("kernel", 5), "max")


def _op_reflect(t, p, ctx):
    """Reverse along the time axis (feature axis for weights)."""
    return torch.flip(t, dims=[_time_or_feature_axis(p, ctx)])


def _op_sort(t, p, ctx):
    """Sort values along the feature axis by magnitude — the databending
    idiom (pixel-sorting, transplanted)."""
    descending = str(p.get("direction", "desc")) == "desc"
    keys = t.abs() if bool(p.get("by_magnitude", True)) else t
    idx = keys.argsort(dim=ctx.feature_axis, descending=descending)
    return torch.gather(t, ctx.feature_axis, idx)


def _op_shuffle(t, p, ctx):
    """Permute entries along the feature axis with a seeded permutation
    (same permutation for every position — a rewiring, not white noise)."""
    n = t.shape[ctx.feature_axis]
    perm = torch.randperm(n, generator=ctx.generator).to(t.device)
    return t.index_select(ctx.feature_axis, perm)


def _op_draw(t, p, ctx):
    """K&F's freehand curve: `points` is a list of 2-128 floats forming an
    envelope over the feature axis, linearly resampled to the axis length,
    then applied as multiplier (`mode: "scale"`, curve value 1.0 = unity)
    or offset (`mode: "offset"`)."""
    pts = p.get("points") or []
    if len(pts) < 2:
        return t
    curve = torch.tensor([float(v) for v in pts], dtype=torch.float32)
    n = t.shape[ctx.feature_axis]
    curve = torch.nn.functional.interpolate(
        curve.view(1, 1, -1), size=n, mode="linear", align_corners=True
    ).view(-1).to(device=t.device, dtype=t.dtype)
    shape = [1] * t.ndim
    shape[ctx.feature_axis] = n
    curve = curve.view(shape)
    if str(p.get("mode", "scale")) == "offset":
        return t + curve
    return t * curve


# --- Registry ---------------------------------------------------------------

@dataclass
class OperatorSpec:
    name: str
    fn: Callable
    label: str
    phase: int
    # Param descriptors drive the frontend sliders: name -> {min,max,default,step,type}
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Which bend domains the operator makes sense in.
    domains: tuple = ("activation", "weight", "latent")
    description: str = ""

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name, "label": self.label, "phase": self.phase,
            "params": self.params, "domains": list(self.domains),
            "description": self.description,
        }


OPERATORS: Dict[str, OperatorSpec] = {}


def _register(spec: OperatorSpec) -> None:
    OPERATORS[spec.name] = spec


_register(OperatorSpec(
    "ablate", _op_ablate, "Ablate", 1, {},
    description="Silence the selected features entirely — the model computes around the hole."))
_register(OperatorSpec(
    "scale", _op_scale, "Scale", 1,
    {"factor": {"min": -4.0, "max": 4.0, "default": 1.5, "step": 0.01}},
    description="Amplify or invert. Negative values flip the feature's meaning."))
_register(OperatorSpec(
    "bias", _op_bias, "Offset", 1,
    {"offset": {"min": -3.0, "max": 3.0, "default": 0.5, "step": 0.01}},
    description="Push every value up or down by a constant."))
_register(OperatorSpec(
    "roll", _op_roll, "Roll", 1,
    {"amount": {"min": -1.0, "max": 1.0, "default": 0.25, "step": 0.005},
     "axis": {"type": "enum", "options": ["feature", "time"], "default": "feature"}},
    description="Circular shift — features land on their neighbours' wiring."))
_register(OperatorSpec(
    "threshold", _op_threshold, "Threshold", 1,
    {"cutoff": {"min": 0.0, "max": 3.0, "default": 0.5, "step": 0.01},
     "hard": {"type": "bool", "default": False}},
    description="Only values above the cutoff survive; hard mode binarises them."))
_register(OperatorSpec(
    "noise", _op_noise, "Noise", 1,
    {"amount": {"min": 0.0, "max": 3.0, "default": 0.5, "step": 0.01}},
    description="Seeded gaussian noise, scaled to the layer's own level."))
_register(OperatorSpec(
    "clamp", _op_clamp, "Clamp", 1,
    {"lo": {"min": -4.0, "max": 0.0, "default": -1.0, "step": 0.01},
     "hi": {"min": 0.0, "max": 4.0, "default": 1.0, "step": 0.01}},
    description="Hard-limit the range — squashes dynamics into distortion."))
_register(OperatorSpec(
    "quantize", _op_quantize, "Quantize", 1,
    {"levels": {"min": 2, "max": 64, "default": 8, "step": 1, "type": "int"}},
    description="Bit-crush the values into a handful of steps."))

_register(OperatorSpec(
    "smear", _op_smear, "Smear", 2,
    {"kernel": {"min": 2, "max": 64, "default": 9, "step": 1, "type": "int"},
     "axis": {"type": "enum", "options": ["time", "feature"], "default": "time"}},
    domains=("activation", "latent", "weight"),
    description="Blur along time — transients dissolve into wash."))
_register(OperatorSpec(
    "erode", _op_erode, "Erode", 2,
    {"kernel": {"min": 2, "max": 64, "default": 5, "step": 1, "type": "int"},
     "axis": {"type": "enum", "options": ["time", "feature"], "default": "time"}},
    domains=("activation", "latent", "weight"),
    description="Sliding minimum — hollows the signal out."))
_register(OperatorSpec(
    "dilate", _op_dilate, "Dilate", 2,
    {"kernel": {"min": 2, "max": 64, "default": 5, "step": 1, "type": "int"},
     "axis": {"type": "enum", "options": ["time", "feature"], "default": "time"}},
    domains=("activation", "latent", "weight"),
    description="Sliding maximum — peaks swallow their surroundings."))
_register(OperatorSpec(
    "reflect", _op_reflect, "Reflect", 2,
    {"axis": {"type": "enum", "options": ["time", "feature"], "default": "time"}},
    domains=("activation", "latent", "weight"),
    description="Mirror the axis — the computation runs backwards through itself."))
_register(OperatorSpec(
    "sort", _op_sort, "Sort", 2,
    {"direction": {"type": "enum", "options": ["desc", "asc"], "default": "desc"}},
    domains=("activation", "latent", "weight"),
    description="Sort features by magnitude — databending's pixel-sort, for sound."))
_register(OperatorSpec(
    "shuffle", _op_shuffle, "Shuffle", 2, {},
    domains=("activation", "latent", "weight"),
    description="Rewire the features with one seeded permutation."))
_register(OperatorSpec(
    "draw", _op_draw, "Draw", 2,
    {"points": {"type": "curve", "default": []},
     "mode": {"type": "enum", "options": ["scale", "offset"], "default": "scale"}},
    domains=("activation", "weight", "latent"),
    description="Draw a bend curve over the layer and apply it as gain or offset."))


def apply_operator(name: str, tensor: torch.Tensor, params: Dict[str, Any],
                   ctx: OpContext) -> torch.Tensor:
    """Apply one operator with the NaN/Inf scrub every bend must pass."""
    spec = OPERATORS.get(name)
    if spec is None:
        raise ValueError(f"Unknown bend operator: {name!r}")
    out = spec.fn(tensor, params or {}, ctx)
    if out.dtype.is_floating_point:
        out = torch.nan_to_num(out, nan=0.0, posinf=6e4, neginf=-6e4)
    return out


def describe_operators() -> List[Dict[str, Any]]:
    return [spec.describe() for spec in OPERATORS.values()]
