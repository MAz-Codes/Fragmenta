"""Partial (init-audio) sampling schedules start at init_noise_level and only
descend; full schedules are unchanged.

Run:  python tests/test_partial_schedule.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "stable-audio-3"))

import torch  # noqa: E402

from stable_audio_3.inference.distribution_shift import (  # noqa: E402
    DistributionShift, FluxDistributionShift, LogSNRShift)
from stable_audio_3.inference.sampling import build_schedule  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def upstream_full(steps, shift, seq_len):
    """The unpatched code path for sigma_max = 1."""
    t = torch.linspace(1.0, 0, steps + 1)
    t = shift.shift(t, seq_len)
    t[..., 0] = 1.0
    return t


SHIFTS = {
    "LogSNRShift (SA3 default)": LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0),
    "LogSNRShift rate=1": LogSNRShift(rate=1.0),
    "FluxDistributionShift": FluxDistributionShift(alpha_min=3.0, alpha_max=6.93),
    "DistributionShift": DistributionShift(),
}

for name, shift in SHIFTS.items():
    print(f"=== {name} ===")
    for steps in (8, 50):
        full = build_schedule(steps, 1.0, shift, 1033, include_endpoint=True)
        check(f"{steps} steps, full schedule unchanged", torch.equal(full, upstream_full(steps, shift, 1033)))
        for smax in (0.3, 0.5, 0.7, 0.9, 0.99):
            t = build_schedule(steps, smax, shift, 1033, include_endpoint=True)
            d = torch.diff(t)
            ok = len(t) == steps + 1 and abs(float(t[0]) - smax) < 1e-6 and float(t[-1]) == 0.0 and bool((d < 0).all())
            check(f"{steps} steps from {smax}: starts there, descends to 0", ok,
                  "" if ok else f"t[:3]={[round(float(v), 4) for v in t[:3]]}")
    t = build_schedule(8, 0.05, shift, 1033)
    check("tiny strength 0.05: starts there, never rises", abs(float(t[0]) - 0.05) < 1e-6 and float(t[-1]) == 0.0
          and bool((torch.diff(t) <= 0).all()), str([round(float(v), 3) for v in t]))
    per = build_schedule(50, 0.7, shift, torch.tensor([1033, 2066, 517]), include_endpoint=True)
    ok = per.shape == (3, 51) and torch.allclose(per[:, 0], torch.full((3,), 0.7)) and bool((torch.diff(per, dim=1) < 0).all())
    check("per-element lengths: one descending row each", ok, str(tuple(per.shape)))

print("=== no shift / zero strength: unchanged ===")
check("no dist_shift → plain linspace", torch.equal(build_schedule(10, 0.6, None), torch.linspace(0.6, 0, 11)))
check("sigma_max 0 → all zeros", float(build_schedule(10, 0.0, SHIFTS["LogSNRShift (SA3 default)"], 1033).abs().max()) == 0.0)

print()
if fails:
    print(f"FAILED: {fails}")
    sys.exit(1)
print("All partial-schedule checks passed.")
