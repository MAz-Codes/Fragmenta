#!/usr/bin/env python
"""Offline contract smoke test for the SA3 backend (W7).

Exercises the request/response contracts that do NOT require a model download
or audio generation, via the Flask test client. Safe to run on CI / a laptop
with no GPU and no checkpoints installed.

Run:
    PYTHONPATH=vendor/stable-audio-3 python scripts/smoke_test_sa3_contract.py

Covers:
  * /api/health, /api/environment capability flags
  * /api/checkpoints catalog shape
  * /api/loras list shape
  * /api/start-training validation (missing project_name, unknown project)
  * /api/generate LoRA path validation (bogus path → 400, no model load)

Exit code 0 = all pass, 1 = a failure.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "vendor" / "stable-audio-3"))

import app.backend.app as A  # noqa: E402

client = A.app.test_client()
failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def main():
    print("SA3 backend contract smoke test\n")

    r = client.get("/api/health")
    check("GET /api/health -> 200", r.status_code == 200, f"got {r.status_code}")

    r = client.get("/api/environment")
    env = r.get_json() or {}
    check("GET /api/environment has capability flags",
          r.status_code == 200 and all(k in env for k in
              ("platform", "cuda_available", "mps_available", "flash_attn_available")),
          f"got {env}")

    r = client.get("/api/checkpoints")
    cat = (r.get_json() or {}).get("checkpoints")
    check("GET /api/checkpoints -> catalog list",
          r.status_code == 200 and isinstance(cat, list) and len(cat) > 0,
          f"got {r.status_code}")
    if isinstance(cat, list) and cat:
        ids = {c.get("id") for c in cat}
        check("catalog includes sa3-small-music", "sa3-small-music" in ids, f"ids={ids}")

    r = client.get("/api/loras")
    check("GET /api/loras -> {loras: [...]}",
          r.status_code == 200 and isinstance((r.get_json() or {}).get("loras"), list),
          f"got {r.status_code}")

    # --- start-training validation -----------------------------------------
    r = client.post("/api/start-training", json={"modelName": "x", "baseModel": "sa3-medium-base"})
    check("POST /api/start-training without projectName -> 400",
          r.status_code == 400, f"got {r.status_code}")

    r = client.post("/api/start-training", json={
        "modelName": "x", "baseModel": "sa3-medium-base",
        "projectName": "definitely-not-a-real-project-xyz"})
    check("POST /api/start-training unknown project -> 400",
          r.status_code == 400, f"got {r.status_code}")

    # --- generate LoRA path validation (no model load) ----------------------
    r = client.post("/api/generate", json={
        "model_id": "sa3-small-music", "prompt": "techno kick", "duration": 4,
        "loras": [{"path": "models/fine_tuned/nope/checkpoints/step_1.safetensors",
                   "strength": 0.8}]})
    body = r.get_json() or {}
    check("POST /api/generate bogus LoRA path -> 400 (no model load)",
          r.status_code == 400, f"got {r.status_code}: {body}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("All contract checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
