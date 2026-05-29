#!/usr/bin/env python
"""Pre-encode wiring smoke test (W7 / Phase 6).

The offline assertions always run: they verify that
`build_train_command(..., encoded_dir=...)` threads `--encoded_dir` through to
the SA3 training subprocess, and omits it when no latents are present. This is
the contract that makes pre-encoded-latents training actually take effect.

Run:
    PYTHONPATH=vendor/stable-audio-3 python tests/smoke_test_pre_encode.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vendor" / "stable-audio-3"))

from app.core.training.sa3_lora_runner import build_train_command  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def main():
    print("Pre-encode wiring smoke test\n")
    common = dict(
        venv_python=sys.executable,
        sa3_vendor_dir=REPO / "vendor" / "stable-audio-3",
        sa3_model_name="small-music-base",
        data_dir=Path("/tmp/proj"),
        save_dir=Path("/tmp/proj/run/checkpoints"),
        adapter_type="dora-rows",
    )

    # With encoded_dir -> --encoded_dir present, pointing at the latents dir.
    cmd = build_train_command(**common, encoded_dir=Path("/tmp/proj/.latents"))
    check("--encoded_dir present when latents supplied", "--encoded_dir" in cmd)
    if "--encoded_dir" in cmd:
        val = cmd[cmd.index("--encoded_dir") + 1]
        check("--encoded_dir points at the .latents dir", val.endswith(".latents"), f"val={val}")

    # Without encoded_dir -> flag absent (SA3 falls back to on-the-fly encoding).
    cmd_none = build_train_command(**common, encoded_dir=None)
    check("--encoded_dir absent when no latents", "--encoded_dir" not in cmd_none)

    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("All pre-encode wiring checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
