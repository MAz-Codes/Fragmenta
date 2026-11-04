"""Bend operators — pure tensor transformations.

Each operator is a pure function `(tensor, params, ctx) -> tensor` that must
preserve shape, dtype, and device. Operators never mutate their input.

`ctx` is an OpContext carrying everything an operator may need beyond its
own params: the axes that mean "features" and "time" for the tensor at hand
(they differ per bend stage — transformer activations are (B, seq, dim),
latents and decoder activations are (B, C, T), weights are (out, in) or
(n,)), a seeded torch.Generator for the stochastic operators so a
patch + seed reproduces bit-exactly, and `scale`, the tensor's own standard
deviation. Every additive or absolute parameter (an offset, a threshold, a
clamp limit, a drawn offset curve) is in units of that scale: measured on
sa3-small-music (4 Nov 2026), weight matrices have a std of 0.04-0.19, the
timestep embedding 0.15-0.5, latents 1-5 and the DiT's residual stream
24-73, so an absolute 0.5 was a blow-up on one and nothing on the other.
On weights an additive offset is a special case: shifting every entry of a
matrix the same way makes its dot products add up coherently, and fp16
overflows. Measured on sa3-small-music: offsets of 0.25 std keep the
music, 0.5 std and up turn it to noise, 1.5-2 std come back NaN. So in the
weight domain the offset unit is 0.1 std (the slider's ±3 reaches ±0.3).

Lineage per operator is noted inline — Broad et al. 2021/2022 (numerical:
ablation, inversion 1-x, multiplication, binary threshold; affine; morpho-
logical erosion/dilation; applied to activations), Kotowski & Font NIME 2026
(on weights and biases: offset, scaling, negation, circular shift, drawing
a parameter shape), Dzwonczyk et al. DAFx 2024 (the same point-wise and
morphological operators inside a diffusion model). Operators marked "ours"
have no source in those papers. The protocol the registry serves is
Ghazala's anti-theory: modify, listen, log.

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
    scale: float = 1.0         # the tensor's std; absolute params are × this
    domain: str = "activation"  # activation | weight | latent

    @property
    def offset_unit(self) -> float:
        """What an additive offset of 1.0 means for this tensor (see the
        module docstring for why weights get a tenth)."""
        return self.scale * (0.1 if self.domain == "weight" else 1.0)

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
    """Multiply (Broad: multiplication by a scalar; K&F: scaling). A
    negative factor negates (K&F's "negation (flipping)"). Not Broad's
    inversion, which is 1 - x and has no direct equivalent here."""
    return t * float(p.get("factor", 1.0))


def _op_bias(t, p, ctx):
    """Add a constant offset, in units of the tensor's std (a tenth of
    it on weights). (K&F: offset.)"""
    return t + float(p.get("offset", 0.0)) * ctx.offset_unit


def _op_roll(t, p, ctx):
    """Circular shift along the feature axis (K&F's circular shifting of
    the parameter array — "roll", the operation behind most rows of the
    Bending Log fragment they publish).

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
    """Zero everything below |cutoff|; `hard` additionally flattens the
    survivors to ±cutoff. After Broad's binary threshold (1 if x >= t else
    0), adapted so a bend keeps sign and scale: Broad's form, applied to
    a transformer's residual stream, replaces the signal with a constant."""
    cutoff = float(p.get("cutoff", 0.0)) * ctx.scale
    mask = t.abs() >= cutoff
    if bool(p.get("hard", False)):
        return torch.where(mask, torch.sign(t) * cutoff, torch.zeros_like(t))
    return torch.where(mask, t, torch.zeros_like(t))


def _op_noise(t, p, ctx):
    """Add seeded gaussian noise scaled to the tensor's own std, so one
    amount slider behaves comparably across layers. Ours: none of the
    source papers adds noise (Broad et al. 2022 get stochastic variation
    from random *feature selection*, not from added noise)."""
    amount = float(p.get("amount", 0.0))
    if amount == 0.0:
        return t
    std = t.float().std()
    if not torch.isfinite(std) or std == 0:
        std = torch.tensor(1.0)
    return t + ctx.randn_like(t) * (std.to(t.dtype) * amount)


def _op_clamp(t, p, ctx):
    """Hard limit at lo/hi × the tensor's std. Ours (a DSP idiom, not
    from the bending papers)."""
    return t.clamp(float(p.get("lo", -1.0)) * ctx.scale, float(p.get("hi", 1.0)) * ctx.scale)


def _op_quantize(t, p, ctx):
    """Bit-crush: snap values to `levels` steps across the tensor's range.
    Ours; a coarse relative of Broad's binary threshold (posterisation)."""
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
    """1-D mean blur along the time axis. Ours; the sliding-window
    engine it shares with erode/dilate is Broad's morphology made 1-D."""
    return _smooth_1d(t, _time_or_feature_axis(p, ctx), p.get("kernel", 5), "mean")


def _op_erode(t, p, ctx):
    """Sliding minimum — Broad's erosion (a circular structuring element
    on 2-D activation maps) as a 1-D window along time or features."""
    return _smooth_1d(t, _time_or_feature_axis(p, ctx), p.get("kernel", 5), "min")


def _op_dilate(t, p, ctx):
    """Sliding maximum — Broad's dilation, 1-D (see erode)."""
    return _smooth_1d(t, _time_or_feature_axis(p, ctx), p.get("kernel", 5), "max")


def _op_reflect(t, p, ctx):
    """Reverse along the time axis (feature axis for weights). After
    Broad's affine reflection of an activation map; Broad et al. 2022 report
    that their affine rotation and scaling did not transfer to audio, so
    only this one is kept."""
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
    (same permutation for every position — a rewiring, not white noise).
    Ours."""
    n = t.shape[ctx.feature_axis]
    perm = torch.randperm(n, generator=ctx.generator).to(t.device)
    return t.index_select(ctx.feature_axis, perm)


def _op_draw(t, p, ctx):
    """K&F's "draw a new parameter shape directly with the cursor":
    `points` is a list of 2-128 floats forming an
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
        return t + curve * ctx.offset_unit
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
    {"offset": {"min": -3.0, "max": 3.0, "default": 0.5, "step": 0.01, "unit": "level"}},
    description="Push every value up or down by a constant, in units of the layer's own level."))
_register(OperatorSpec(
    "roll", _op_roll, "Roll", 1,
    {"amount": {"min": -1.0, "max": 1.0, "default": 0.25, "step": 0.005},
     "axis": {"type": "enum", "options": ["feature", "time"], "default": "feature"}},
    description="Circular shift — features land on their neighbours' wiring."))
_register(OperatorSpec(
    "threshold", _op_threshold, "Threshold", 1,
    {"cutoff": {"min": 0.0, "max": 3.0, "default": 0.5, "step": 0.01, "unit": "level"},
     "hard": {"type": "bool", "default": False}},
    description="Only values above the cutoff (× the layer's own level) survive; hard mode binarises them."))
_register(OperatorSpec(
    "noise", _op_noise, "Noise", 1,
    {"amount": {"min": 0.0, "max": 3.0, "default": 0.5, "step": 0.01}},
    description="Seeded gaussian noise, scaled to the layer's own level."))
_register(OperatorSpec(
    "clamp", _op_clamp, "Clamp", 1,
    {"lo": {"min": -4.0, "max": 0.0, "default": -1.0, "step": 0.01, "unit": "level"},
     "hi": {"min": 0.0, "max": 4.0, "default": 1.0, "step": 0.01, "unit": "level"}},
    description="Hard-limit the range (× the layer's own level) — squashes dynamics into distortion."))
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
# Erode/dilate stay off weights: a running min/max turns a weight matrix
# into same-sign blocks whose dot products overflow fp16 — every such
# contact on sa3-small-music came back NaN (5 of 5, 4 Nov 2026).
_register(OperatorSpec(
    "erode", _op_erode, "Erode", 2,
    {"kernel": {"min": 2, "max": 64, "default": 5, "step": 1, "type": "int"},
     "axis": {"type": "enum", "options": ["time", "feature"], "default": "time"}},
    domains=("activation", "latent"),
    description="Sliding minimum — hollows the signal out."))
_register(OperatorSpec(
    "dilate", _op_dilate, "Dilate", 2,
    {"kernel": {"min": 2, "max": 64, "default": 5, "step": 1, "type": "int"},
     "axis": {"type": "enum", "options": ["time", "feature"], "default": "time"}},
    domains=("activation", "latent"),
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
    if tensor.dtype.is_floating_point and tensor.numel() > 1:
        std = tensor.detach().float().std()
        ctx.scale = float(std) if torch.isfinite(std) and float(std) > 0 else 1.0
    out = spec.fn(tensor, params or {}, ctx)
    if out.dtype.is_floating_point:
        out = torch.nan_to_num(out, nan=0.0, posinf=6e4, neginf=-6e4)
    return out


def describe_operators() -> List[Dict[str, Any]]:
    return [spec.describe() for spec in OPERATORS.values()]
