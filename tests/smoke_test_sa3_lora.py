#!/usr/bin/env python
"""SA3 LoRA training smoke test (W7) — model-gated.

Builds a throwaway project from tests/fixtures/mini-project/ (4 clips +
captions), runs a short `dora-rows` LoRA training against
`sa3-small-music-base`, and asserts a `.safetensors` checkpoint is produced
and loadable. Cleans up the project and run afterward.

Self-skips (exit 0) when the base model isn't downloaded. Training a few steps
on CPU still takes a couple of minutes (model load dominates); tune via env:

    SMOKE_LORA_STEPS=20 SMOKE_LORA_TIMEOUT=600 \
        PYTHONPATH=vendor/stable-audio-3 python tests/smoke_test_sa3_lora.py
"""
import os
import sys
import time
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vendor" / "stable-audio-3"))

# Honour FRAGMENTA_FORCE_DEVICE if set (CI sets cpu); otherwise the trainer
# auto-detects. Forcing cpu on a box with flash-attn installed will fail.

from app.core.config import get_config  # noqa: E402
from app.core.model_manager import ModelManager  # noqa: E402

BASE = "sa3-small-music-base"
FIXTURE = REPO / "tests" / "fixtures" / "mini-project"
PROJECT = "smoke-lora-fixture"
STEPS = int(os.environ.get("SMOKE_LORA_STEPS", "20"))
CKPT_EVERY = max(5, STEPS // 2)
TIMEOUT = int(os.environ.get("SMOKE_LORA_TIMEOUT", "900"))


def main():
    cfg = get_config()
    if not ModelManager(cfg).is_model_downloaded(BASE):
        print(f"SKIP: {BASE} not downloaded — run the Checkpoint Manager to enable this test.")
        return 0

    from app.backend.data import projects as P
    from app.core.training import sa3_trainer as T
    from tests._fixture import ensure_mini_project

    ensure_mini_project(FIXTURE)  # generate the clips if not present

    # Fresh project from the fixture.
    try:
        P.delete_project(PROJECT)
    except Exception:
        pass
    print(f"Building project '{PROJECT}' from fixture…")
    P.create_project(PROJECT)
    P.ingest_folder(PROJECT, FIXTURE, mode="copy")
    for wav in sorted(FIXTURE.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if txt.exists():
            P.update_clip_prompt(PROJECT, wav.name, txt.read_text().strip())
    P.commit_project(PROJECT)

    ok = False
    detail = ""
    try:
        print(f"Starting {STEPS}-step dora-rows LoRA against {BASE} (CPU; this takes a few minutes)…")
        T.start_training({
            "modelName": PROJECT, "baseModel": BASE, "projectName": PROJECT,
            "steps": STEPS, "checkpointSteps": CKPT_EVERY, "batchSize": 1,
            "learningRate": 1e-4, "duration": 2.0, "loraRank": 8,
            "adapterType": "dora-rows", "precision": "bf16", "seed": 42,
            "overwrite": True,
        })
        run_dir = cfg.get_path("models_fine_tuned") / PROJECT

        def scan():
            return list((run_dir / "checkpoints").glob("*.safetensors")) if run_dir.exists() else []

        deadline = time.time() + TIMEOUT
        grace_deadline = None  # set once training reports finished
        while time.time() < deadline:
            st = T.get_training_status()
            # The trainer writes .ckpt during the run and converts them to
            # .safetensors *after* status flips to complete, so accept either
            # the files on disk or the trainer's own checkpoint list.
            ckpts = scan()
            if ckpts or st.get("checkpoints"):
                ok = True
                break
            if st.get("error"):
                detail = f"trainer error: {st['error']}"
                break
            finished = (not st.get("is_training")) and st.get("status") in ("complete", "idle", "failed", "stopped")
            if finished:
                # Grace window for the post-run .ckpt -> .safetensors conversion.
                if grace_deadline is None:
                    grace_deadline = time.time() + 30
                elif time.time() > grace_deadline:
                    ok = bool(scan())
                    detail = "" if ok else f"no checkpoint after grace (status={st.get('status')})"
                    break
            time.sleep(3)
        else:
            detail = f"timed out after {TIMEOUT}s"

        # Confirm a produced .safetensors loads and carries the base_model
        # metadata. Wait briefly for the post-run conversion to finish writing.
        if ok:
            ckpt = None
            for _ in range(10):
                found = scan()
                if found:
                    ckpt = found[0]
                    break
                time.sleep(3)
            if ckpt is None:
                detail = "trainer reported checkpoints but no .safetensors on disk"
                ok = False
            else:
                from safetensors import safe_open
                with safe_open(str(ckpt), framework="pt") as f:
                    meta = f.metadata() or {}
                ok = meta.get("base_model") == BASE
                detail = f"{ckpt.name} (base_model={meta.get('base_model')})"
    finally:
        try:
            T.stop_training()
        except Exception:
            pass
        time.sleep(2)
        shutil.rmtree(cfg.get_path("models_fine_tuned") / PROJECT, ignore_errors=True)
        try:
            P.delete_project(PROJECT)
        except Exception:
            pass

    print(f"  [{'PASS' if ok else 'FAIL'}] LoRA checkpoint produced + loadable — {detail}")
    print()
    if ok:
        print("SA3 LoRA training smoke passed.")
        return 0
    print("FAILED: " + detail)
    return 1


if __name__ == "__main__":
    sys.exit(main())
