"""Verify that the Probe board does what it says — on the real model.

For every contact on the board and a handful of wires, this script takes
the exact modules the frontend would send (tests/bend_board_patches.mjs runs
the board's own bendUtils.js), renders them through the loaded model and
checks, per patch:

  * the bend acted: its hooks fired (and on which sampling steps), and the
    audio differs from the clean render of the same seed;
  * `keep_level` holds: inside every hooked call the bent tensor leaves at
    the RMS of the tensor that came in;
  * the model is pristine afterwards: no hooks left, every parameter back
    bit-for-bit, and a clean render after all the bends is identical to the
    clean render before them.

It also exercises the Rack-only structural modules (bypass, repeat,
reorder, swap_nonlinearity) once each, and the step gate.

    python tests/verify_bend_board.py [model_id] [--out report.json]

Needs a downloaded PyTorch model (default sa3-small-music, the one the
board offers) and node on PATH.
Dev-only tooling: tracked on dev/bend, never on main.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import get_config                      # noqa: E402
from app.core.generation import audio_generator as G        # noqa: E402
from app.core.generation.engine import GenerationRequest    # noqa: E402
from app.core.bending import session as S                   # noqa: E402
from app.core.bending.operators import describe_operators   # noqa: E402
from app.core.bending.patch import SWAP_FNS, validate_patch  # noqa: E402
from app.core.bending.targets import get_targets, resolve_module_path  # noqa: E402

PROMPT = "warm synth melody over a simple drum groove"      # the board's default
SEED = 20261104
DURATION = 3.0


def db(x: float) -> float:
    return 20 * np.log10(max(float(x), 1e-12))


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a.astype(np.float64)))))


def param_fingerprint(module: torch.nn.Module) -> dict:
    """Cheap per-tensor fingerprint (exact sums), for a restore check after
    every patch; the full SHA-256 is taken once before and once after."""
    out = {}
    with torch.no_grad():
        for name, p in module.state_dict().items():
            if not torch.is_tensor(p) or not p.numel():
                continue
            f = p.detach().double()
            out[name] = (float(f.sum()), float(f.abs().sum()),
                         float(f.view(-1)[0]), float(f.view(-1)[-1]))
    return out


def param_sha(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in module.state_dict().items():
            if torch.is_tensor(p):
                h.update(name.encode())
                h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                         if p.dtype != torch.bool else p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def hook_count(module: torch.nn.Module) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in module.modules())


class Probe:
    """Instruments BendSession._bend_tensor: per module, which steps it ran
    on and the in/out RMS of every call."""

    def __init__(self):
        self.calls = {}
        self._orig = S.BendSession._bend_tensor
        probe = self

        def wrapped(self_, t, mod, stage, feature_axis=None, tag=None):
            out = probe._orig(self_, t, mod, stage, feature_axis, tag)
            r_in = float(t.detach().float().pow(2).mean().sqrt())
            r_out = float(out.detach().float().pow(2).mean().sqrt())
            rec = probe.calls.setdefault(mod["id"], {"steps": [], "ratio": [], "domain": mod["target"]["domain"]})
            rec["steps"].append(int(self_._step))
            rec["ratio"].append(r_out / r_in if r_in > 0 else None)
            return out
        S.BendSession._bend_tensor = wrapped

    def reset(self):
        self.calls = {}

    def restore(self):
        S.BendSession._bend_tensor = self._orig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", nargs="?", default="sa3-small-music")
    ap.add_argument("--out", default=str(ROOT / "tests" / "bend_board_report.json"))
    ap.add_argument("--wet", type=float, default=0.8)
    args = ap.parse_args()

    config = get_config()
    engine = G._make_engine("torch", config)
    t0 = time.time()
    engine.load(args.model_id)
    print(f"loaded {args.model_id} in {time.time() - t0:.1f}s")
    sam = engine.model
    inner = sam.model
    _, kind, _ = G._MODEL_INFO[args.model_id]
    steps = 50 if kind == "base" else 8
    cfg = 7.0 if kind == "base" else 1.0

    def request(seed=SEED):
        return GenerationRequest(prompt=PROMPT, model_id=args.model_id, duration=DURATION,
                                 steps=steps, cfg_scale=cfg, seed=seed, batch_size=1)

    def render(patch=None, nan_hunt=False):
        session = None
        info = {"fired": set(), "warnings": [], "params_finite": True, "first_nonfinite": None}
        if patch is not None:
            session = S.BendSession(sam, patch)
            session.set_total_steps(steps)
            session.apply()
            with torch.no_grad():
                info["params_finite"] = all(torch.isfinite(p).all().item()
                                            for p in inner.parameters())
        hunters = []
        if nan_hunt:
            # Name the first module whose output goes non-finite, in call order.
            def make(name):
                def h(_m, _i, out):
                    o = out[0] if isinstance(out, tuple) else out
                    if torch.is_tensor(o) and info["first_nonfinite"] is None \
                            and not torch.isfinite(o).all():
                        info["first_nonfinite"] = f"{name} (step {session._step if session else '?'})"
                return h
            for name, m in inner.named_modules():
                hunters.append(m.register_forward_hook(make(name)))

        def cb(i):
            if session is not None:
                session.on_sampler_step(i)
        t = time.time()
        try:
            audio = engine.generate(request(), cb)
        finally:
            for hk in hunters:
                hk.remove()
            if session is not None:
                info["fired"] = set(session._fired)
                info["warnings"] = list(session.warnings)
                session.remove()
        return audio, info, time.time() - t

    # --- registry → the board's own patches ---------------------------------
    reg = get_targets(args.model_id, config.get_path("models_pretrained"))
    reg["operators"] = describe_operators()
    reg["swap_fns"] = list(SWAP_FNS)
    reg_path = Path(args.out).with_suffix(".registry.json")
    reg_path.write_text(json.dumps(reg))
    patches = json.loads(subprocess.check_output(
        ["node", str(ROOT / "tests" / "bend_board_patches.mjs"), str(reg_path), str(args.wet)]))
    print(f"board: {len(patches)} contact/wire patches from bendUtils.js "
          f"(DiT depth {reg['dit_depth']}, decoder blocks {reg['decoder_blocks']})")

    # --- baselines ------------------------------------------------------------
    sha_before = param_sha(inner)
    fp_before = param_fingerprint(inner)
    hooks_before = hook_count(inner)
    clean_a, _, t_clean = render()
    clean_b, _, _ = render()
    deterministic = bool(np.array_equal(clean_a, clean_b))
    print(f"clean render {t_clean:.1f}s, RMS {db(rms(clean_a)):.1f} dBFS, "
          f"deterministic={deterministic}, hooks on model: {hooks_before}")

    probe = Probe()
    report = {
        "model_id": args.model_id, "steps": steps, "duration": DURATION, "seed": SEED,
        "prompt": PROMPT, "wet": args.wet, "dit_depth": reg["dit_depth"],
        "decoder_blocks": reg["decoder_blocks"], "clean_deterministic": deterministic,
        "clean_rms_dbfs": db(rms(clean_a)),
        "clean_flatness": G._spectral_flatness(clean_a),
        "clean_render_s": t_clean, "entries": [],
    }

    def run_patch(label, pads, modules, extra=None):
        probe.reset()
        norm, pwarn = validate_patch({"version": 1, "model_id": args.model_id,
                                      "seed": SEED, "modules": modules})
        bent, info, dt = render(norm)
        finite = bool(np.isfinite(bent).all())
        if not finite:
            probe.reset()
            _, info2, _ = render(norm, nan_hunt=True)
            info["first_nonfinite"] = info2["first_nonfinite"]
            bent = np.nan_to_num(bent, nan=0.0, posinf=0.0, neginf=0.0)
        diff = bent.astype(np.float64) - clean_a.astype(np.float64)
        fp_after = param_fingerprint(inner)
        restored = fp_after == fp_before
        entry = {
            "label": label, "pads": pads,
            "modules": [{k: v for k, v in m.items() if k != "id"} for m in norm["modules"]],
            "patch_warnings": pwarn, "run_warnings": info["warnings"],
            "fired": sorted(info["fired"]),
            "output_finite": finite, "params_finite_after_apply": info["params_finite"],
            "first_nonfinite_module": info["first_nonfinite"],
            "render_s": dt,
            "changed": bool(np.max(np.abs(diff)) > 1e-6),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "diff_rms_db": db(rms(diff)),
            "bent_rms_dbfs": db(rms(bent)),
            "bent_flatness": G._spectral_flatness(np.ascontiguousarray(bent)),
            "params_restored": restored,
            "hooks_after": hook_count(inner),
            "calls": {mid: {"domain": c["domain"], "n": len(c["steps"]),
                            "steps": sorted(set(c["steps"])),
                            "level_ratio_mean": (float(np.mean([r for r in c["ratio"] if r is not None]))
                                                 if any(r is not None for r in c["ratio"]) else None),
                            "level_ratio_max_dev": (float(np.max(np.abs(np.array(
                                [r for r in c["ratio"] if r is not None]) - 1.0)))
                                if any(r is not None for r in c["ratio"]) else None)}
                      for mid, c in probe.calls.items()},
        }
        if extra:
            entry.update(extra)
        # The safety stage the app applies to every bent output, for the record.
        rep = []
        safe = G._bend_output_safety(np.ascontiguousarray(bent), rep)
        entry["safety_notes"] = rep
        entry["safe_rms_dbfs"] = db(rms(safe))
        report["entries"].append(entry)
        flag = "" if entry["changed"] else "  <-- NO CHANGE"
        flag += "" if finite else f"  <-- NON-FINITE OUTPUT (first: {info['first_nonfinite']})"
        flag += "" if restored else "  <-- PARAMS NOT RESTORED"
        flag += "" if entry["hooks_after"] == hooks_before else "  <-- HOOKS LEFT"
        what = "; ".join(
            (f"{m.get('operator')} {json.dumps(m.get('params'))}"
             + (f" feat={m['target']['features']['mode']}" if m['target'].get('features', {}).get('mode', 'all') != 'all' else "")
             + (f" steps={m['steps']['from']:.2f}-{m['steps']['to']:.2f}" if m.get('steps') and (m['steps']['from'] > 0 or m['steps']['to'] < 1) else "")
             + f" mix={m.get('mix', 1):.2f}")
            if m.get("operator") else f"{m['structure']['type']} {json.dumps({k: v for k, v in m['structure'].items() if k != 'type'})} mix={m.get('mix', 1):.2f}"
            for m in norm["modules"])
        steps_fired = {mid: c["steps"] for mid, c in entry["calls"].items()}
        print(f"  {label:22s} Δ {entry['diff_rms_db'] - report['clean_rms_dbfs']:6.1f} dB  out {entry['bent_rms_dbfs']:6.1f} dBFS "
              f"flat {entry['bent_flatness']:.2f}  fired {len(entry['fired'])}/{len(norm['modules'])} "
              f"{flag}\n      {what}  calls={steps_fired}")
        return entry

    # --- every contact and wire on the board ---------------------------------
    print("board contacts and wires:")
    for p in patches:
        label = "+".join(p["pads"])
        if not p["modules"]:
            report["entries"].append({"label": label, "pads": p["pads"], "modules": [],
                                      "dropped_by_frontend": True})
            print(f"  {label:28s} (no modules — dropped by the board, as designed)")
            continue
        run_patch(label, p["pads"], p["modules"])

    # --- Rack-only structural modules, once each -----------------------------
    print("structural modules (Rack):")
    n = reg["dit_depth"]
    struct = [
        ("bypass 8-11", {"type": "bypass"}, list(range(8, min(n, 12)))),
        ("repeat 4-5 x2", {"type": "repeat", "times": 2}, [4, 5]),
        ("reorder reversed", {"type": "reorder", "order": list(range(n - 1, -1, -1))}, None),
        ("swap_nonlinearity sin 10-13", {"type": "swap_nonlinearity", "fn": "sin"},
         list(range(10, min(n, 14)))),
    ]
    for label, st, blocks in struct:
        mod = {"id": "s1", "enabled": True,
               "target": {"stage": "dit", "domain": "structure", "blocks": blocks},
               "structure": st}
        run_patch(label, [], [mod])
    layers_restored = len(resolve_module_path(inner, "model.model.transformer").layers) == n
    acts_restored = not any(isinstance(m, S._Lambda) for m in inner.modules())

    # --- step gate: one DiT activation bend confined to one window -----------
    print("step gate:")
    gate = {"id": "g1", "enabled": True,
            "target": {"stage": "dit", "domain": "activation", "blocks": [6]},
            "operator": "scale", "params": {"factor": 2.0}, "mix": 1.0,
            "steps": {"from": 0.4, "to": 0.6}}
    e = run_patch("gate 40-60% on dit.6", [], [gate])
    expected = [i for i in range(steps) if 0.4 <= i / max(steps - 1, 1) <= 0.6]
    e["gate_expected_steps"] = expected
    e["gate_ok"] = e["calls"].get("g1", {}).get("steps") == expected
    print(f"  fired on steps {e['calls'].get('g1', {}).get('steps')}, expected {expected}")

    # --- after everything: pristine? -----------------------------------------
    probe.restore()
    clean_after, _, _ = render()
    report["final"] = {
        "clean_after_identical": bool(np.array_equal(clean_a, clean_after)),
        "sha_before": sha_before, "sha_after": param_sha(inner),
        "hooks_after": hook_count(inner), "hooks_before": hooks_before,
        "layers_restored": layers_restored, "activations_restored": acts_restored,
    }
    report["final"]["params_identical"] = report["final"]["sha_before"] == report["final"]["sha_after"]
    print("final:", json.dumps({k: v for k, v in report["final"].items() if "sha" not in k}))

    # --- summary ----------------------------------------------------------------
    ran = [e for e in report["entries"] if e.get("modules")]
    for e in ran:
        # Effect relative to the clean signal: below -20 dB it is barely audible.
        e["effect_db"] = e["diff_rms_db"] - report["clean_rms_dbfs"]
    bad = [e["label"] for e in ran if not e["changed"] or not e["params_restored"]
           or e["hooks_after"] != hooks_before]
    # Only per-step modules (activation, latent, jumper) mark themselves as
    # fired; weight and structural modules act once at apply time.
    per_step = lambda m: m["target"]["domain"] in ("activation", "latent") or \
        m.get("structure", {}).get("type") == "jumper"
    not_fired = [e["label"] for e in ran
                 if len(e["fired"]) < sum(1 for m in e["modules"] if per_step(m))]
    weak = [e["label"] for e in ran if e["changed"] and e["effect_db"] < -20]
    nonfinite = [e["label"] for e in ran if not e.get("output_finite", True)]
    report["summary"] = {
        "patches_run": len(ran), "changed_and_restored": len(ran) - len(bad),
        "problems": bad, "per_step_modules_never_fired": not_fired,
        "weak_effect_below_-20dB": weak, "non_finite_output": nonfinite,
        "keep_level_max_dev": max([c["level_ratio_max_dev"] or 0.0 for e in ran
                                   for c in e["calls"].values()
                                   if any(m.get("keep_level") for m in e["modules"])] or [0.0]),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["summary"], indent=1))
    print(f"report: {args.out}")
    return 0 if not bad and report["final"]["params_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
