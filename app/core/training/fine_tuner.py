"""SA3 LoRA training orchestrator — Phase 5.

Public surface (matches what app/backend/app.py imports):
    start_training(config)        -> dict
    get_training_status()         -> dict
    stop_training()               -> dict
    preview_training_plan(config) -> dict
    get_base_model_configs()      -> dict   # back-compat, returns {}
    class FineTuner

Training is dispatched as a subprocess running
`vendor/stable-audio-3/scripts/train_lora.py`. Progress comes back through
two channels:
  * stdout/stderr from the subprocess (parsed for tqdm "step X/Y" lines)
  * metrics.csv that train_lora.py writes under --save_dir

Config shape (from the frontend training form):
{
    "modelName":       "my-lora",            # used for run dir name
    "baseModel":       "sa3-medium-base",    # must end in -base
    "dataDir":         "data/",              # optional, default data/
    "steps":           5000,
    "checkpointSteps": 500,                  # checkpoint cadence
    "batchSize":       1,
    "learningRate":    1.0e-4,
    "duration":        30.0,                 # max clip seconds per sample
    "loraRank":        16,
    "loraAlpha":       16,                   # null → defaults to rank
    "loraDropout":     0.0,
    "adapterType":     "dora-rows",
    "precision":       "bf16",               # bf16|fp16
    "seed":            42,
    "include":         null,                 # list[str] or null
    "exclude":         null
}
"""
from __future__ import annotations

import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import get_config
from app.core.training.sa3_lora_runner import (
    SA3_BASE_MODELS,
    build_train_command,
    build_train_env,
    materialize_captions,
    prestage_base_model,
)


# --- Defaults --------------------------------------------------------------

DEFAULT_STEPS = 5000
DEFAULT_CHECKPOINT_STEPS = 500
DEFAULT_BATCH_SIZE = 1
DEFAULT_LR = 1e-4
DEFAULT_DURATION = 30.0
DEFAULT_RANK = 16
DEFAULT_ADAPTER = "dora-rows"
DEFAULT_PRECISION = "bf16"


def get_base_model_configs() -> Dict[str, Dict[str, str]]:
    """Back-compat shim — Phase 2 moved catalogues into ModelManager."""
    return get_config().model_configs


# --- FineTuner singleton ---------------------------------------------------

class FineTuner:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config or {}
        self.process: Optional[subprocess.Popen] = None
        self.run_dir: Optional[Path] = None
        self.metrics_csv: Optional[Path] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self.status: Dict[str, Any] = {
            "is_training": False,
            "status": "idle",
            "step": 0,
            "total_steps": 0,
            "loss": None,
            "message": "",
            "started_at": None,
            "ended_at": None,
            "log_tail": [],          # last ~50 stdout lines
            "checkpoints": [],       # safetensors written so far
            "error": None,
        }

    # --- Public API --------------------------------------------------------

    def start(self) -> Dict[str, Any]:
        try:
            self._resolve_paths()
            self._stage_dataset()
            self._stage_base_model()
            cmd, env = self._build_invocation()
            self._spawn(cmd, env)
            return {"success": True, "run_dir": str(self.run_dir)}
        except Exception as e:
            self.status["error"] = str(e)
            self.status["status"] = "failed"
            self.status["is_training"] = False
            return {"error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        return dict(self.status)

    def stop(self) -> Dict[str, Any]:
        if not self.process or self.process.poll() is not None:
            return {"error": "Nothing to stop — no active training run."}
        try:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.status["status"] = "stopped"
            self.status["is_training"] = False
            self.status["ended_at"] = time.time()
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}

    def preview_plan(self) -> Dict[str, Any]:
        try:
            self._resolve_paths(create_dirs=False)
        except FileNotFoundError as e:
            return {"error": str(e)}
        steps = int(self.config.get("steps") or DEFAULT_STEPS)
        ckpt_every = int(self.config.get("checkpointSteps") or DEFAULT_CHECKPOINT_STEPS)
        ckpts = max(1, steps // max(1, ckpt_every))
        return {
            "model_name": self.config.get("modelName", "fragmenta-lora"),
            "base_model": self.config.get("baseModel"),
            "data_dir": str(self.run_dir.parent.parent / "data") if self.run_dir else None,
            "save_dir": str(self.run_dir / "checkpoints") if self.run_dir else None,
            "steps": steps,
            "checkpoint_every": ckpt_every,
            "expected_checkpoints": ckpts,
            "rank": int(self.config.get("loraRank") or DEFAULT_RANK),
            "alpha": int(self.config.get("loraAlpha") or self.config.get("loraRank") or DEFAULT_RANK),
            "adapter_type": self.config.get("adapterType") or DEFAULT_ADAPTER,
            "batch_size": int(self.config.get("batchSize") or DEFAULT_BATCH_SIZE),
            "lr": float(self.config.get("learningRate") or DEFAULT_LR),
            "duration": float(self.config.get("duration") or DEFAULT_DURATION),
            "precision": self.config.get("precision") or DEFAULT_PRECISION,
        }

    # --- Internals ---------------------------------------------------------

    def _resolve_paths(self, create_dirs: bool = True) -> None:
        cfg = get_config()
        run_name = self._safe_name(self.config.get("modelName") or "lora-run")
        self.run_dir = cfg.get_path("models_fine_tuned") / run_name
        self.metrics_csv = self.run_dir / "metrics.csv"
        if create_dirs:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "checkpoints").mkdir(exist_ok=True)

    def _stage_dataset(self) -> None:
        cfg = get_config()
        data_dir = Path(self.config.get("dataDir") or cfg.get_path("data"))
        if not data_dir.is_absolute():
            data_dir = cfg.project_root / data_dir
        if not data_dir.exists():
            raise FileNotFoundError(f"data dir not found: {data_dir}")
        metadata_path = Path(cfg.get_metadata_json_path())
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"data/metadata.json not found — annotate clips first."
            )
        result = materialize_captions(metadata_path, data_dir)
        self.status["log_tail"].append(
            f"Caption sidecars: wrote {result['written']}, "
            f"skipped {result['skipped']} (already current); "
            f"missing audio: {len(result['missing_audio'])}"
        )
        if not (result["written"] + result["skipped"]):
            raise RuntimeError(
                "No usable caption rows found in metadata.json — "
                "make sure every entry has a non-empty `prompt`."
            )
        self._data_dir = data_dir

    def _stage_base_model(self) -> None:
        cfg = get_config()
        base_model = self.config.get("baseModel")
        if base_model not in SA3_BASE_MODELS:
            raise ValueError(
                f"baseModel must be one of {list(SA3_BASE_MODELS)}. "
                "Post-trained checkpoints (no -base suffix) can't be used "
                "as a LoRA training base — CFG distillation has collapsed "
                "the gradient signal LoRAs target."
            )
        hub_dir = cfg.get_path("models_pretrained") / "sa3" / "hub"
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None

        def _cb(pct: int, msg: str) -> None:
            self.status["message"] = msg
            self.status["log_tail"].append(f"[stage] {msg}")

        prestage_base_model(base_model, hub_dir, token=token, progress_callback=_cb)
        self._hub_dir = hub_dir

    def _build_invocation(self):
        cfg = get_config()
        sa3_vendor = cfg.get_path("stable_audio_3")
        sa3_name, _repo = SA3_BASE_MODELS[self.config["baseModel"]]

        # Use the Fragmenta venv's python so we share installed packages.
        venv_python = sys.executable

        precision_raw = (self.config.get("precision") or DEFAULT_PRECISION).lower()
        precision = "bf16" if precision_raw in ("bf16", "bfloat16", "auto", "") else "fp16"

        include = self.config.get("include")
        if include and isinstance(include, str):
            include = shlex.split(include)
        exclude = self.config.get("exclude")
        if exclude and isinstance(exclude, str):
            exclude = shlex.split(exclude)

        cmd = build_train_command(
            venv_python=venv_python,
            sa3_vendor_dir=sa3_vendor,
            sa3_model_name=sa3_name,
            data_dir=self._data_dir,
            save_dir=self.run_dir / "checkpoints",
            rank=int(self.config.get("loraRank") or DEFAULT_RANK),
            lora_alpha=self.config.get("loraAlpha"),
            adapter_type=self.config.get("adapterType") or DEFAULT_ADAPTER,
            dropout=float(self.config.get("loraDropout") or 0.0),
            lr=float(self.config.get("learningRate") or DEFAULT_LR),
            steps=int(self.config.get("steps") or DEFAULT_STEPS),
            batch_size=int(self.config.get("batchSize") or DEFAULT_BATCH_SIZE),
            duration=float(self.config.get("duration") or DEFAULT_DURATION),
            base_precision=precision,
            include=include,
            exclude=exclude,
            seed=int(self.config.get("seed") or 42),
            checkpoint_every=int(self.config.get("checkpointSteps") or DEFAULT_CHECKPOINT_STEPS),
            name=self.config.get("modelName") or "fragmenta-lora",
        )
        env = build_train_env(sa3_vendor, self._hub_dir)
        return cmd, env

    def _spawn(self, cmd: List[str], env: Dict[str, str]) -> None:
        log_path = self.run_dir / "training.log"
        # Stamp training_metadata.json so /api/loras can find the base_model
        # if the embedded safetensors metadata is missing it.
        (self.run_dir / "training_metadata.json").write_text(json.dumps({
            "mode": "lora",
            "engine": "sa3",
            "base_model": self.config.get("baseModel"),
            "model_name": self.config.get("modelName"),
            "started_at": time.time(),
            "lora_config": {
                "rank": int(self.config.get("loraRank") or DEFAULT_RANK),
                "alpha": self.config.get("loraAlpha"),
                "adapter_type": self.config.get("adapterType") or DEFAULT_ADAPTER,
                "dropout": float(self.config.get("loraDropout") or 0.0),
            },
            "steps": int(self.config.get("steps") or DEFAULT_STEPS),
            "lr": float(self.config.get("learningRate") or DEFAULT_LR),
            "batch_size": int(self.config.get("batchSize") or DEFAULT_BATCH_SIZE),
        }, indent=2))

        self.status.update({
            "is_training": True,
            "status": "running",
            "step": 0,
            "total_steps": int(self.config.get("steps") or DEFAULT_STEPS),
            "loss": None,
            "error": None,
            "started_at": time.time(),
            "ended_at": None,
            "checkpoints": [],
            "message": "Starting training subprocess...",
        })

        self.process = subprocess.Popen(
            cmd,
            cwd=str(get_config().project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._monitor_thread = threading.Thread(
            target=self._monitor,
            args=(log_path,),
            daemon=True,
            name=f"sa3-train-monitor:{self.run_dir.name}",
        )
        self._monitor_thread.start()

    def _monitor(self, log_path: Path) -> None:
        """Pull stdout, parse tqdm step lines, scan CSV, watch checkpoints."""
        step_pat = re.compile(r"(\d+)/(\d+)")
        last_log_flush = time.time()
        try:
            with open(log_path, "w") as logf:
                if self.process and self.process.stdout:
                    for line in self.process.stdout:
                        line = line.rstrip()
                        logf.write(line + "\n")
                        if time.time() - last_log_flush > 1:
                            logf.flush()
                            last_log_flush = time.time()
                        self.status["log_tail"].append(line)
                        if len(self.status["log_tail"]) > 80:
                            self.status["log_tail"] = self.status["log_tail"][-50:]
                        # tqdm progress: "12/5000 [00:21<...]" — pick the first
                        # such pair as current_step/total_steps.
                        m = step_pat.search(line)
                        if m:
                            cur, total = int(m.group(1)), int(m.group(2))
                            if total >= self.status["total_steps"]:
                                self.status["step"] = cur
                                self.status["total_steps"] = total
                rc = self.process.wait() if self.process else 1
        except Exception as e:
            self.status["error"] = str(e)
            rc = -1

        self.status["ended_at"] = time.time()
        self.status["is_training"] = False
        self.status["status"] = "complete" if rc == 0 else ("stopped" if rc == -2 else "failed")
        if rc != 0 and not self.status.get("error"):
            self.status["error"] = f"train_lora.py exited with code {rc}"

        # Final pass: enumerate written checkpoints + best-effort latest loss.
        ckpt_dir = self.run_dir / "checkpoints"
        if ckpt_dir.exists():
            self.status["checkpoints"] = sorted(
                str(p.relative_to(get_config().project_root))
                for p in ckpt_dir.glob("*.safetensors")
            )
        self._scrape_csv_loss()

    def _scrape_csv_loss(self) -> None:
        if not self.metrics_csv or not self.metrics_csv.exists():
            # train_lora.py writes its CSV under the lightning logger dir,
            # which is `<save_dir>/<name>/version_*/metrics.csv`. Walk to
            # find it.
            ckpt_dir = self.run_dir / "checkpoints"
            for p in ckpt_dir.rglob("metrics.csv"):
                self.metrics_csv = p
                break
        if not self.metrics_csv or not self.metrics_csv.exists():
            return
        try:
            with open(self.metrics_csv) as f:
                rows = list(csv.DictReader(f))
            for row in reversed(rows):
                for k in ("train/loss", "loss", "train_loss"):
                    v = row.get(k)
                    if v not in (None, ""):
                        try:
                            self.status["loss"] = float(v)
                            return
                        except ValueError:
                            pass
        except Exception:
            pass

    @staticmethod
    def _safe_name(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_") or "lora-run"


# --- Module-level singleton + public functions -----------------------------

_active: Optional[FineTuner] = None
_lock = threading.Lock()


def get_trainer() -> Optional[FineTuner]:
    return _active


def start_training(config: Dict[str, Any]) -> Dict[str, Any]:
    global _active
    with _lock:
        if _active and _active.status.get("is_training"):
            return {"error": "A training run is already in progress."}
        _active = FineTuner(config)
        return _active.start()


def get_training_status() -> Dict[str, Any]:
    if _active is None:
        return {
            "is_training": False,
            "status": "idle",
            "message": "No training run has been started yet.",
        }
    return _active.get_status()


def stop_training() -> Dict[str, Any]:
    if _active is None:
        return {"error": "No training run to stop."}
    return _active.stop()


def preview_training_plan(config: Dict[str, Any]) -> Dict[str, Any]:
    return FineTuner(config).preview_plan()
