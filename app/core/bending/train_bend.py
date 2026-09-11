"""Bent-training entry point (Break mode) — Phase 3.

A sibling of vendor/stable-audio-3/scripts/train_lora.py that runs THE SAME
training pipeline with deliberate mis-training interventions injected. Zero
vendor edits: this script imports train_lora as a module, swaps its
DiffusionCondTrainingWrapper for a subclass, optionally wraps the caption
loader, then calls its main().

Interventions (bend.json, all optional — absent keys are inert):

    {"grad_noise": 0.5,                    # gaussian noise added to grads,
                                           #   scaled to each grad's own std
     "grad_flip": {"fraction": 0.25,       # fraction of trainable params
                   "seed": 0},             #   whose gradients are negated —
                                           #   layers that learn AWAY
     "amnesia": {"period": 200,            # every `period` steps, spend
                 "duty": 0.25,             #   duty·period steps at
                 "lr_scale": -1.0},        #   lr·lr_scale (negative = unlearn)
     "caption_shuffle": 0.3,               # probability a clip trains against
                                           #   a random OTHER clip's caption
     "timestep_skew": "texture",           # "texture" | "structure" — train
                                           #   only one denoising regime
     "loss_scale": 1.0,                    # global loss multiplier
                                           #   (negative = pure anti-learning)
     "freeze_rotation": {"period": 100,    # every `period` steps re-freeze a
                         "fraction": 0.5,  #   different random half of the
                         "seed": 0}}       #   adapter — split-brain training

Invocation (built by sa3_lora_runner.build_train_command):

    python train_bend.py --bend_config <run>/bend.json \
                         --train_script <vendor>/scripts/train_lora.py \
                         <every normal train_lora.py argument>

This file must stay importable with only the vendor on PYTHONPATH — it runs
in the training subprocess and must not import Fragmenta app modules.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_train_lora(script_path: Path):
    spec = importlib.util.spec_from_file_location("train_lora", str(script_path))
    module = importlib.util.module_from_spec(spec)
    # Registered under its own name so dill-serialized dataset fns that
    # reference the module can resolve inside DataLoader workers.
    sys.modules["train_lora"] = module
    spec.loader.exec_module(module)
    return module


def make_shuffled_caption_fn(probability: float):
    """Caption shuffle: with `probability`, a clip trains against a random
    OTHER clip's caption from the same directory — wrong word-sound
    associations, deterministically per path.

    Mirrors the vendor's caption_metadata_fn constraints: dill-serialized
    into spawned DataLoader workers, so all imports are local and no module
    globals are referenced.
    """
    def caption_metadata_fn(info, audio, _p=float(probability)):
        from pathlib import Path
        import random
        path = Path(info["path"])
        txt = path.with_suffix(".txt")
        if not txt.exists():
            return {"__reject__": True}
        rng = random.Random(hash(str(path)) & 0xFFFFFFFF)
        if rng.random() < _p:
            others = sorted(p for p in path.parent.glob("*.txt") if p != txt)
            if others:
                txt = rng.choice(others)
        return {"prompt": txt.read_text().strip()}
    return caption_metadata_fn


def make_bent_wrapper(base_cls, bend: dict):
    """Subclass the vendor's DiffusionCondTrainingWrapper with the
    intervention hooks. Only the configured interventions do anything."""
    import torch

    grad_noise = float(bend.get("grad_noise") or 0.0)
    grad_flip = bend.get("grad_flip") or {}
    flip_fraction = float(grad_flip.get("fraction") or 0.0)
    flip_seed = int(grad_flip.get("seed") or 0)
    amnesia = bend.get("amnesia") or {}
    amnesia_period = int(amnesia.get("period") or 0)
    amnesia_duty = min(0.9, max(0.0, float(amnesia.get("duty") or 0.0)))
    amnesia_scale = float(amnesia.get("lr_scale", -1.0))
    loss_scale = float(bend.get("loss_scale", 1.0))
    skew = str(bend.get("timestep_skew") or "none")
    freeze = bend.get("freeze_rotation") or {}
    freeze_period = int(freeze.get("period") or 0)
    freeze_fraction = min(1.0, max(0.0, float(freeze.get("fraction") or 0.0)))
    freeze_seed = int(freeze.get("seed") or 0)

    class BentTrainingWrapper(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Timestep skew: retarget the sampler at one denoising regime.
            # High log-SNR steps refine texture; low log-SNR steps lay down
            # structure. (Sampler attrs are read per-step in training_step.)
            if skew == "texture":
                self.timestep_sampler = "log_snr_uniform"
                self.min_logsnr, self.max_logsnr = 0.5, 5.0
            elif skew == "structure":
                self.timestep_sampler = "log_snr_uniform"
                self.min_logsnr, self.max_logsnr = -6.0, -1.0
            self._bend_flip_names = None
            self._bend_grad_gen = torch.Generator().manual_seed(flip_seed or 1)
            self._bend_base_lrs = None
            self._bend_frozen_names = set()

        # -- helpers -----------------------------------------------------
        def _trainable_names(self):
            return [n for n, p in self.named_parameters()
                    if p.requires_grad or n in self._bend_frozen_names]

        def _pick(self, names, fraction, seed):
            import random
            if fraction <= 0.0 or not names:
                return set()
            rng = random.Random(seed)
            k = max(1, int(round(len(names) * fraction)))
            return set(rng.sample(names, min(k, len(names))))

        # -- interventions -----------------------------------------------
        def on_train_batch_start(self, batch, batch_idx):
            out = super().on_train_batch_start(batch, batch_idx) \
                if hasattr(super(), "on_train_batch_start") else None
            step = int(self.global_step)

            if amnesia_period > 0 and amnesia_duty > 0.0:
                optimizers = self.trainer.optimizers or []
                if self._bend_base_lrs is None:
                    self._bend_base_lrs = [
                        [g["lr"] for g in opt.param_groups] for opt in optimizers]
                in_window = (step % amnesia_period) >= amnesia_period * (1 - amnesia_duty)
                for oi, opt in enumerate(optimizers):
                    for gi, group in enumerate(opt.param_groups):
                        base = self._bend_base_lrs[oi][gi]
                        group["lr"] = base * amnesia_scale if in_window else base

            if freeze_period > 0 and freeze_fraction > 0.0 \
                    and step % freeze_period == 0:
                # Thaw last rotation, freeze a fresh random subset.
                for n, p in self.named_parameters():
                    if n in self._bend_frozen_names:
                        p.requires_grad_(True)
                names = [n for n, p in self.named_parameters() if p.requires_grad]
                rotation = step // freeze_period
                self._bend_frozen_names = self._pick(
                    names, freeze_fraction, freeze_seed + rotation)
                for n, p in self.named_parameters():
                    if n in self._bend_frozen_names:
                        p.requires_grad_(False)
            return out

        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            if loss_scale != 1.0:
                if torch.is_tensor(loss):
                    loss = loss * loss_scale
                elif isinstance(loss, dict) and torch.is_tensor(loss.get("loss")):
                    loss = {**loss, "loss": loss["loss"] * loss_scale}
            return loss

        def on_after_backward(self):
            if hasattr(super(), "on_after_backward"):
                super().on_after_backward()
            if grad_noise <= 0.0 and flip_fraction <= 0.0:
                return
            if self._bend_flip_names is None:
                self._bend_flip_names = self._pick(
                    [n for n, p in self.named_parameters() if p.requires_grad],
                    flip_fraction, flip_seed)
            with torch.no_grad():
                for n, p in self.named_parameters():
                    if p.grad is None:
                        continue
                    if n in self._bend_flip_names:
                        p.grad.neg_()
                    if grad_noise > 0.0:
                        std = p.grad.float().std()
                        if torch.isfinite(std) and std > 0:
                            noise = torch.randn(
                                p.grad.shape, generator=self._bend_grad_gen,
                                dtype=torch.float32,
                            ).to(device=p.grad.device, dtype=p.grad.dtype)
                            p.grad.add_(noise * std.to(p.grad.dtype) * grad_noise)

    BentTrainingWrapper.__name__ = base_cls.__name__  # keep vendor log names
    return BentTrainingWrapper


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bend_config", required=True)
    parser.add_argument("--train_script", required=True)
    args, passthrough = parser.parse_known_args()

    with open(args.bend_config) as fh:
        bend = json.load(fh)

    train_lora = _load_train_lora(Path(args.train_script))

    train_lora.DiffusionCondTrainingWrapper = make_bent_wrapper(
        train_lora.DiffusionCondTrainingWrapper, bend)

    shuffle_p = float(bend.get("caption_shuffle") or 0.0)
    if shuffle_p > 0.0:
        train_lora.caption_metadata_fn = make_shuffled_caption_fn(shuffle_p)

    sys.argv = [str(Path(args.train_script))] + passthrough
    train_lora.main()


if __name__ == "__main__":
    main()
