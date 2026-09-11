"""Bend patch schema — validation and normalization.

A bend patch is the unit of saving, sharing and reproducing a bend: a plain
JSON document (see BEND_PLAN.md §3). `validate_patch` normalizes one and
returns (patch, warnings). Validation is deliberately permissive where the
model is involved — a block index that doesn't exist on the loaded model is
a *warning at apply time*, not an error here, so patches survive across
model families and vendor bumps.

Top-level:
    {"version": 1, "name": "...", "model_id": "...", "seed": 7,
     "modules": [ <module>, ... ]}

Module:
    {"id": "b1", "enabled": true,
     "target": {"stage": "cond|timestep|dit|latent|decoder",
                "domain": "activation|weight|latent|structure",
                "blocks": [0, 1],           # blocks stages only
                "weight_slot": "ff",        # weight domain in dit, optional
                "param": "weight|bias|both",
                "features": {"mode": "all|random|indices",
                             "fraction": 0.3, "seed": 7, "indices": []}},
     "operator": "scale", "params": {...}, "mix": 1.0,
     "steps": {"from": 0.0, "to": 1.0},     # sampling-step gate
     "structure": {"type": "bypass|repeat|reorder|swap_nonlinearity", ...}}
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.core.bending.operators import OPERATORS

MAX_MODULES = 16
MAX_DRAW_POINTS = 256

_STAGES = {"cond", "timestep", "dit", "latent", "decoder"}
_DOMAINS = {"activation", "weight", "latent", "structure"}
_STAGE_DOMAINS = {
    "cond": {"activation", "weight"},
    "timestep": {"activation", "weight"},
    "dit": {"activation", "weight", "structure"},
    "latent": {"latent"},
    "decoder": {"activation", "weight"},
}
_STRUCTURE_TYPES = {"bypass", "repeat", "reorder", "swap_nonlinearity"}
SWAP_FNS = ("sin", "tanh", "relu", "abs", "square", "step")


class PatchError(ValueError):
    """Raised when a patch is structurally unusable (not merely degraded)."""


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _norm_features(raw: Any, warnings: List[str], mid: str) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"mode": "all"}
    mode = str(raw.get("mode", "all"))
    if mode not in ("all", "random", "indices", "cluster"):
        warnings.append(f"{mid}: unknown feature mode {mode!r}; using 'all'.")
        return {"mode": "all"}
    out: Dict[str, Any] = {"mode": mode}
    if mode == "random":
        out["fraction"] = _clamp(float(raw.get("fraction", 0.5)), 0.0, 1.0)
        out["seed"] = int(raw.get("seed", 0))
    elif mode == "cluster":
        out["k"] = int(_clamp(int(raw.get("k", 4)), 2, 16))
        out["index"] = int(_clamp(int(raw.get("index", 0)), 0, 15))
        out["seed"] = int(raw.get("seed", 0))
    elif mode == "indices":
        idx = raw.get("indices") or []
        out["indices"] = sorted({int(i) for i in idx if int(i) >= 0})[:4096]
    return out


def _norm_params(op_name: str, raw: Any, warnings: List[str], mid: str) -> Dict[str, Any]:
    spec = OPERATORS[op_name]
    raw = raw if isinstance(raw, dict) else {}
    out: Dict[str, Any] = {}
    for pname, pdesc in spec.params.items():
        ptype = pdesc.get("type", "float")
        if pname not in raw:
            if "default" in pdesc:
                out[pname] = pdesc["default"]
            continue
        v = raw[pname]
        try:
            if ptype == "bool":
                out[pname] = bool(v)
            elif ptype == "enum":
                out[pname] = v if v in pdesc.get("options", []) else pdesc.get("default")
            elif ptype == "curve":
                pts = [float(x) for x in (v or [])][:MAX_DRAW_POINTS]
                out[pname] = pts
            elif ptype == "int":
                out[pname] = int(_clamp(int(v), pdesc.get("min", -1e9), pdesc.get("max", 1e9)))
            else:
                out[pname] = float(_clamp(float(v), pdesc.get("min", -1e9), pdesc.get("max", 1e9)))
        except (TypeError, ValueError):
            warnings.append(f"{mid}: bad value for param {pname!r}; using default.")
            if "default" in pdesc:
                out[pname] = pdesc["default"]
    return out


def _norm_structure(raw: Any, warnings: List[str], mid: str) -> Dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    stype = str(raw.get("type", ""))
    if stype not in _STRUCTURE_TYPES:
        raise PatchError(f"{mid}: structure.type must be one of {sorted(_STRUCTURE_TYPES)}.")
    out: Dict[str, Any] = {"type": stype}
    if stype == "repeat":
        out["times"] = int(_clamp(int(raw.get("times", 2)), 2, 4))
    elif stype == "reorder":
        order = raw.get("order") or []
        try:
            out["order"] = [int(i) for i in order][:256]
        except (TypeError, ValueError):
            raise PatchError(f"{mid}: reorder.order must be a list of block indices.")
        if not out["order"]:
            raise PatchError(f"{mid}: reorder.order is empty.")
    elif stype == "swap_nonlinearity":
        fn = str(raw.get("fn", "sin"))
        if fn not in SWAP_FNS:
            warnings.append(f"{mid}: unknown swap fn {fn!r}; using 'sin'.")
            fn = "sin"
        out["fn"] = fn
    return out


def validate_patch(patch: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Normalize a raw patch dict. Raises PatchError only for structural
    problems; everything model-dependent is deferred to apply time."""
    if not isinstance(patch, dict):
        raise PatchError("bend_patch must be a JSON object.")
    warnings: List[str] = []

    modules_raw = patch.get("modules")
    if not isinstance(modules_raw, list) or not modules_raw:
        raise PatchError("bend_patch.modules must be a non-empty array.")
    if len(modules_raw) > MAX_MODULES:
        raise PatchError(f"bend_patch.modules exceeds the cap of {MAX_MODULES}.")

    norm_modules: List[Dict[str, Any]] = []
    for i, m in enumerate(modules_raw):
        mid = str(m.get("id") or f"module[{i}]") if isinstance(m, dict) else f"module[{i}]"
        if not isinstance(m, dict):
            raise PatchError(f"{mid}: module entries must be objects.")
        target = m.get("target")
        if not isinstance(target, dict):
            raise PatchError(f"{mid}: missing target.")
        stage = str(target.get("stage", ""))
        if stage not in _STAGES:
            raise PatchError(f"{mid}: unknown stage {stage!r}.")
        domain = str(target.get("domain") or
                     ("latent" if stage == "latent" else "activation"))
        if domain not in _DOMAINS or domain not in _STAGE_DOMAINS[stage]:
            raise PatchError(f"{mid}: domain {domain!r} not valid for stage {stage!r}.")

        nm: Dict[str, Any] = {
            "id": mid,
            "enabled": bool(m.get("enabled", True)),
            "target": {"stage": stage, "domain": domain},
            "mix": _clamp(float(m.get("mix", 1.0)), 0.0, 1.0),
        }

        if stage in ("dit", "decoder"):
            blocks = target.get("blocks")
            if blocks is None:
                nm["target"]["blocks"] = None      # None = all blocks
            else:
                try:
                    nm["target"]["blocks"] = sorted({int(b) for b in blocks if int(b) >= 0})[:256]
                except (TypeError, ValueError):
                    raise PatchError(f"{mid}: target.blocks must be a list of indices.")
                if not nm["target"]["blocks"]:
                    raise PatchError(f"{mid}: target.blocks is empty.")

        if domain == "structure":
            nm["structure"] = _norm_structure(m.get("structure"), warnings, mid)
        else:
            op = str(m.get("operator", ""))
            if op not in OPERATORS:
                raise PatchError(f"{mid}: unknown operator {op!r}.")
            spec_domains = OPERATORS[op].domains
            op_domain = "latent" if domain == "latent" else domain
            if op_domain not in spec_domains:
                raise PatchError(
                    f"{mid}: operator {op!r} does not support the "
                    f"{op_domain!r} domain.")
            nm["operator"] = op
            nm["params"] = _norm_params(op, m.get("params"), warnings, mid)
            nm["target"]["features"] = _norm_features(
                target.get("features"), warnings, mid)
            if domain == "weight":
                param = str(target.get("param", "weight"))
                if param not in ("weight", "bias", "both"):
                    warnings.append(f"{mid}: unknown param {param!r}; using 'weight'.")
                    param = "weight"
                nm["target"]["param"] = param
                slot = target.get("weight_slot")
                nm["target"]["weight_slot"] = str(slot) if slot else None

        # Step gating applies to anything inside the sampling loop.
        if stage in ("dit", "latent") and domain != "structure":
            steps = m.get("steps") if isinstance(m.get("steps"), dict) else {}
            s_from = _clamp(float(steps.get("from", 0.0)), 0.0, 1.0)
            s_to = _clamp(float(steps.get("to", 1.0)), 0.0, 1.0)
            if s_to < s_from:
                s_from, s_to = s_to, s_from
            nm["steps"] = {"from": s_from, "to": s_to}

        norm_modules.append(nm)

    normalized = {
        "version": int(patch.get("version", 1)),
        "name": str(patch.get("name") or "")[:120],
        "model_id": str(patch.get("model_id") or "")[:64],
        "seed": int(patch.get("seed", 0)),
        "modules": norm_modules,
    }
    return normalized, warnings


# --- Break mode: training-bend config --------------------------------------

def validate_train_bend(raw: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Normalize a Break-mode bend config (see train_bend.py's docstring).
    Unknown keys are dropped with a warning; values are clamped to sane
    creative ranges. An empty result means no interventions."""
    if not isinstance(raw, dict):
        raise PatchError("bend must be a JSON object.")
    warnings: List[str] = []
    out: Dict[str, Any] = {}

    if raw.get("grad_noise") is not None:
        out["grad_noise"] = _clamp(float(raw["grad_noise"]), 0.0, 3.0)
    if isinstance(raw.get("grad_flip"), dict):
        gf = raw["grad_flip"]
        out["grad_flip"] = {
            "fraction": _clamp(float(gf.get("fraction", 0.0)), 0.0, 1.0),
            "seed": int(gf.get("seed", 0)),
        }
    if isinstance(raw.get("amnesia"), dict):
        am = raw["amnesia"]
        out["amnesia"] = {
            "period": int(_clamp(int(am.get("period", 200)), 10, 100000)),
            "duty": _clamp(float(am.get("duty", 0.25)), 0.0, 0.9),
            "lr_scale": _clamp(float(am.get("lr_scale", -1.0)), -4.0, 4.0),
        }
    if raw.get("caption_shuffle") is not None:
        out["caption_shuffle"] = _clamp(float(raw["caption_shuffle"]), 0.0, 1.0)
    if raw.get("timestep_skew") is not None:
        skew = str(raw["timestep_skew"])
        if skew in ("none", "texture", "structure"):
            if skew != "none":
                out["timestep_skew"] = skew
        else:
            warnings.append(f"unknown timestep_skew {skew!r}; ignored.")
    if raw.get("loss_scale") is not None:
        out["loss_scale"] = _clamp(float(raw["loss_scale"]), -4.0, 4.0)
    if isinstance(raw.get("freeze_rotation"), dict):
        fr = raw["freeze_rotation"]
        out["freeze_rotation"] = {
            "period": int(_clamp(int(fr.get("period", 100)), 10, 100000)),
            "fraction": _clamp(float(fr.get("fraction", 0.5)), 0.0, 1.0),
            "seed": int(fr.get("seed", 0)),
        }

    known = {"grad_noise", "grad_flip", "amnesia", "caption_shuffle",
             "timestep_skew", "loss_scale", "freeze_rotation"}
    for key in raw:
        if key not in known:
            warnings.append(f"unknown bend key {key!r}; ignored.")
    return out, warnings
