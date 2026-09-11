"""BendSession — applies a validated bend patch to a loaded StableAudioModel
for exactly one generation, then restores the model bit-exactly.

Mechanisms (all zero-vendor-edit):

  * activation bends  -> torch forward hooks on resolved modules
  * weight bends      -> save-original / mutate-in-place / restore on the
                         underlying nn.Parameter data. Deliberately NOT a
                         parametrization: the LoRA loader's remove_lora()
                         strips every parametrization it finds, so foreign
                         parametrizations could corrupt the LoRA stack's
                         bookkeeping. With save/restore, a bend composes as
                         lora(bend(W)) — the bend hits the base weights and
                         the adapter rides on top. For parametrized params
                         we bend `module.parametrizations.<p>.original`.
  * latent bends      -> in-place transform of the sampler's evolving latent,
                         driven from the existing per-ODE-step callback
  * structural bends  -> swap the ContinuousTransformer's `layers`
                         ModuleList for a rebuilt one (bypass / repeat /
                         reorder) and/or swap activation-function submodules
                         (swap_nonlinearity); originals restored on remove()

The session is created and torn down inside AudioGenerator's generation
lock, so it can never race another generation. remove() is idempotent and
must be called in a finally: — after it, the model must behave as if the
session never existed (verified by the baseline-reproduction test).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from utils.logger import get_logger
from app.core.bending.operators import OpContext, apply_operator
from app.core.bending.targets import resolve_module_path

logger = get_logger("BendSession")

# Axes per stage: (feature_axis, time_axis) of the tensors seen there.
_STAGE_AXES = {
    "cond": (-1, 1),        # (B, seq, dim)
    "timestep": (-1, None),  # (B, dim)
    "dit": (-1, 1),         # (B, seq, dim)
    "latent": (1, -1),      # (B, C, T)
    "decoder": (1, -1),     # (B, C, T)
}

_SWAP_FN_FACTORY = {
    "sin": lambda: _Lambda(torch.sin, "sin"),
    "tanh": lambda: nn.Tanh(),
    "relu": lambda: nn.ReLU(),
    "abs": lambda: _Lambda(torch.abs, "abs"),
    "square": lambda: _Lambda(lambda x: x * x, "square"),
    "step": lambda: _Lambda(lambda x: (x > 0).to(x.dtype), "step"),
}


class _Lambda(nn.Module):
    def __init__(self, fn, name):
        super().__init__()
        self.fn = fn
        self._name = name

    def forward(self, x):
        return self.fn(x)

    def extra_repr(self):
        return self._name


class BendSession:
    def __init__(self, sam: Any, patch: Dict[str, Any]):
        """`sam` is the StableAudioModel wrapper; `patch` is the output of
        patch.validate_patch (already normalized)."""
        self.sam = sam
        self.inner = sam.model                      # ConditionedDiffusionModelWrapper
        self.patch = patch
        self.warnings: List[str] = []
        self._applied = False

        self._hooks: List[Any] = []                 # hook handles
        self._saved_params: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._latent_mods: List[Dict[str, Any]] = []
        self._layers_backup: List[Tuple[Any, Any]] = []   # (transformer, orig ModuleList)
        self._swapped_acts: List[Tuple[nn.Module, str, nn.Module]] = []
        self._mask_cache: Dict[Tuple[str, int], torch.Tensor] = {}

        # Sampling-step gate state, fed by on_sampler_step().
        self._step = 0
        self._total_steps = 1

        # One seeded CPU generator per session: patch seed drives every
        # stochastic op, so patch+seed reproduces bit-exactly.
        self._generator = torch.Generator().manual_seed(
            int(patch.get("seed", 0)) & 0x7FFFFFFF)

    # ------------------------------------------------------------------ gate
    def set_total_steps(self, n: int) -> None:
        self._total_steps = max(1, int(n))

    def on_sampler_step(self, info: Dict[str, Any]) -> None:
        """Called from the generator's per-ODE-step callback. Updates the
        step gate and applies latent bends in place."""
        i = info.get("i")
        if isinstance(i, int):
            self._step = i + 1
        x = info.get("x")
        if x is None or not self._latent_mods:
            return
        progress = self._progress()
        for mod in self._latent_mods:
            lo, hi = mod["steps"]["from"], mod["steps"]["to"]
            if not (lo <= progress <= hi):
                continue
            try:
                bent = self._bend_tensor(x, mod, "latent")
                x.copy_(bent)
            except Exception as exc:                     # never kill the run
                self._warn_once(f"latent bend {mod['id']} failed: {exc}")

    def _progress(self) -> float:
        return self._step / max(self._total_steps, 1)

    def _warn_once(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)
            logger.warning(msg)

    # ----------------------------------------------------------------- apply
    def apply(self) -> List[str]:
        """Attach every enabled module. Returns accumulated warnings.
        Missing targets are warnings, not errors (vendor-bump resilience)."""
        if self._applied:
            return self.warnings
        self._applied = True
        for mod in self.patch.get("modules", []):
            if not mod.get("enabled", True):
                continue
            stage = mod["target"]["stage"]
            domain = mod["target"]["domain"]
            try:
                if domain == "structure":
                    self._apply_structure(mod)
                elif domain == "latent":
                    self._latent_mods.append(mod)
                elif domain == "weight":
                    self._apply_weight(mod)
                else:
                    self._apply_activation(mod)
            except Exception as exc:
                self._warn_once(
                    f"bend module {mod.get('id')} ({stage}/{domain}) "
                    f"could not attach: {exc}")
        return self.warnings

    # ---------------------------------------------------------------- remove
    def remove(self) -> None:
        """Idempotent full teardown — the model must come back pristine."""
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []
        for param, backup in self._saved_params:
            try:
                with torch.no_grad():
                    param.data.copy_(backup)
            except Exception as exc:
                logger.error(f"weight restore failed: {exc}")
        self._saved_params = []
        for transformer, orig in self._layers_backup:
            try:
                transformer.layers = orig
            except Exception as exc:
                logger.error(f"layer-order restore failed: {exc}")
        self._layers_backup = []
        for parent, attr, orig in self._swapped_acts:
            try:
                setattr(parent, attr, orig)
            except Exception as exc:
                logger.error(f"activation restore failed: {exc}")
        self._swapped_acts = []
        self._latent_mods = []
        self._mask_cache = {}
        self._applied = False

    # ------------------------------------------------------------- resolvers
    def _stage_modules(self, mod: Dict[str, Any]) -> List[Tuple[str, nn.Module]]:
        """Resolve a module entry's target stage to live (path, module)s."""
        stage = mod["target"]["stage"]
        blocks = mod["target"].get("blocks")
        out: List[Tuple[str, nn.Module]] = []

        if stage == "cond":
            paths = ["model.model.to_cond_embed"]
        elif stage == "timestep":
            paths = ["model.model.to_timestep_embed"]
        elif stage == "dit":
            layers = resolve_module_path(self.inner, "model.model.transformer.layers")
            if layers is None:
                self._warn_once("DiT transformer.layers not found on this model.")
                return []
            n = len(layers)
            idxs = range(n) if blocks is None else [b for b in blocks if b < n]
            skipped = [] if blocks is None else [b for b in blocks if b >= n]
            if skipped:
                self._warn_once(
                    f"{mod['id']}: DiT blocks {skipped} beyond depth {n}; skipped.")
            paths = [f"model.model.transformer.layers.{i}" for i in idxs]
        elif stage == "decoder":
            layers = resolve_module_path(self.inner, "pretransform.model.decoder.layers")
            if layers is None:
                self._warn_once("VAE decoder.layers not found on this model.")
                return []
            # Indices 0..2 are Transpose/Linear plumbing; blocks are offset 3.
            offset, n = 3, len(layers)
            all_blocks = range(max(0, n - offset))
            idxs = all_blocks if blocks is None else [b for b in blocks if b < n - offset]
            paths = [f"pretransform.model.decoder.layers.{i + offset}" for i in idxs]
        else:
            return []

        for p in paths:
            m = resolve_module_path(self.inner, p)
            if m is None:
                self._warn_once(f"{mod['id']}: target module {p} not found; skipped.")
            else:
                out.append((p, m))
        return out

    # ------------------------------------------------------- tensor plumbing
    def _feature_mask(self, mod: Dict[str, Any], size: int,
                      device, dtype) -> Optional[torch.Tensor]:
        feats = mod["target"].get("features") or {"mode": "all"}
        if feats.get("mode") == "all":
            return None
        key = (mod["id"], size)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached.to(device=device, dtype=dtype)
        mask = torch.zeros(size)
        if feats["mode"] == "random":
            k = int(round(size * float(feats.get("fraction", 0.5))))
            if k > 0:
                g = torch.Generator().manual_seed(int(feats.get("seed", 0)) & 0x7FFFFFFF)
                idx = torch.randperm(size, generator=g)[:k]
                mask[idx] = 1.0
        else:  # indices
            idx = [i for i in feats.get("indices", []) if i < size]
            if idx:
                mask[torch.tensor(idx, dtype=torch.long)] = 1.0
        self._mask_cache[key] = mask
        return mask.to(device=device, dtype=dtype)

    def _bend_tensor(self, t: torch.Tensor, mod: Dict[str, Any],
                     stage: str, feature_axis: Optional[int] = None) -> torch.Tensor:
        f_ax, t_ax = _STAGE_AXES.get(stage, (-1, None))
        if feature_axis is not None:
            f_ax = feature_axis
        ctx = OpContext(feature_axis=f_ax, time_axis=t_ax, generator=self._generator)
        bent = apply_operator(mod["operator"], t, mod.get("params"), ctx)

        mix = float(mod.get("mix", 1.0))
        f_ax_n = f_ax % t.ndim
        mask = self._feature_mask(mod, t.shape[f_ax_n], t.device, t.dtype)
        if mask is not None:
            shape = [1] * t.ndim
            shape[f_ax_n] = t.shape[f_ax_n]
            amount = mask.view(shape) * mix
        elif mix >= 1.0:
            return bent
        else:
            amount = mix
        return t * (1 - amount) + bent * amount

    # ------------------------------------------------------------ activation
    def _apply_activation(self, mod: Dict[str, Any]) -> None:
        stage = mod["target"]["stage"]
        targets = self._stage_modules(mod)
        step_gate = mod.get("steps")

        def make_hook(entry):
            def hook(_module, _inputs, output):
                if step_gate is not None:
                    p = self._progress()
                    if not (step_gate["from"] <= p <= step_gate["to"]):
                        return output
                try:
                    if isinstance(output, tuple):
                        if not output or not torch.is_tensor(output[0]):
                            return output
                        return (self._bend_tensor(output[0], entry, stage),
                                *output[1:])
                    if not torch.is_tensor(output):
                        return output
                    return self._bend_tensor(output, entry, stage)
                except Exception as exc:
                    self._warn_once(
                        f"activation bend {entry['id']} failed mid-run: {exc}")
                    return output
            return hook

        for _path, m in targets:
            self._hooks.append(m.register_forward_hook(make_hook(mod)))

    # ---------------------------------------------------------------- weight
    def _iter_bendable_params(self, module: nn.Module, which: str):
        """Yield the actual nn.Parameter tensors to bend under `module`.
        For parametrized modules (LoRA), bend the .original so the adapter
        rides on top of the bent base weight."""
        names = ("weight", "bias") if which == "both" else (which,)
        for sub in module.modules():
            parametrizations = getattr(sub, "parametrizations", None)
            for name in names:
                if parametrizations is not None and name in parametrizations:
                    p = parametrizations[name].original
                elif isinstance(getattr(sub, name, None), torch.Tensor) \
                        and isinstance(sub.__dict__.get("_parameters", {}).get(name), torch.Tensor):
                    p = sub._parameters[name]
                else:
                    continue
                if p is not None and p.numel() > 0:
                    yield p

    def _apply_weight(self, mod: Dict[str, Any]) -> None:
        targets = self._stage_modules(mod)
        slot = mod["target"].get("weight_slot")
        which = mod["target"].get("param", "weight")
        bent_any = False
        for path, m in targets:
            scope = m
            if slot:
                scope = getattr(m, slot, None)
                if scope is None:
                    self._warn_once(
                        f"{mod['id']}: weight slot {slot!r} not found under {path}.")
                    continue
            for p in self._iter_bendable_params(scope, which):
                backup = p.detach().clone()
                with torch.no_grad():
                    # Weights are bent once, in fp32 for op fidelity, then
                    # cast back to the parameter's dtype.
                    bent = self._bend_tensor(
                        p.data.float(), mod, mod["target"]["stage"],
                        feature_axis=0)
                    p.data.copy_(bent.to(p.dtype))
                self._saved_params.append((p, backup))
                bent_any = True
        if not bent_any:
            self._warn_once(f"{mod['id']}: no bendable parameters resolved.")

    # ------------------------------------------------------------- structure
    def _transformer(self):
        return resolve_module_path(self.inner, "model.model.transformer")

    def _apply_structure(self, mod: Dict[str, Any]) -> None:
        transformer = self._transformer()
        if transformer is None or not hasattr(transformer, "layers"):
            self._warn_once(f"{mod['id']}: transformer not found for structural bend.")
            return
        struct = mod["structure"]
        stype = struct["type"]

        if stype == "swap_nonlinearity":
            self._swap_nonlinearity(mod, transformer)
            return

        orig = transformer.layers
        n = len(orig)
        blocks = mod["target"].get("blocks")
        selected = set(range(n)) if blocks is None else {b for b in blocks if b < n}

        if stype == "bypass":
            new_list = [orig[i] for i in range(n) if i not in selected]
            if not new_list:
                self._warn_once(f"{mod['id']}: bypass would remove every block; skipped.")
                return
        elif stype == "repeat":
            times = struct.get("times", 2)
            new_list = []
            for i in range(n):
                new_list.append(orig[i])
                if i in selected:
                    new_list.extend(orig[i] for _ in range(times - 1))
        else:  # reorder — explicit full or partial order of block indices
            order = [i for i in struct.get("order", []) if 0 <= i < n]
            if not order:
                self._warn_once(f"{mod['id']}: reorder order resolves to nothing; skipped.")
                return
            new_list = [orig[i] for i in order]

        self._layers_backup.append((transformer, orig))
        transformer.layers = nn.ModuleList(new_list)

    def _swap_nonlinearity(self, mod: Dict[str, Any], transformer) -> None:
        factory = _SWAP_FN_FACTORY.get(mod["structure"].get("fn", "sin"))
        if factory is None:
            return
        orig_layers = transformer.layers
        n = len(orig_layers)
        blocks = mod["target"].get("blocks")
        selected = range(n) if blocks is None else [b for b in blocks if b < n]
        swapped = 0
        for i in selected:
            block = orig_layers[i]
            for parent in block.modules():
                for name, child in list(parent.named_children()):
                    if isinstance(child, (nn.SiLU, nn.GELU)):
                        self._swapped_acts.append((parent, name, child))
                        setattr(parent, name, factory())
                        swapped += 1
        if swapped == 0:
            self._warn_once(f"{mod['id']}: no activation functions found to swap.")
