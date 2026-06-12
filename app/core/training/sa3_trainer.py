"""SA3 LoRA training orchestrator — Phase 5.

Public surface (matches what app/backend/app.py imports):
    start_training(config)        -> dict
    get_training_status()         -> dict
    stop_training()               -> dict
    preview_training_plan(config) -> dict
    class SA3Trainer

Training is dispatched as a subprocess running
`vendor/stable-audio-3/scripts/train_lora.py`. Progress comes back through
two channels:
  * stdout/stderr from the subprocess (parsed for tqdm "step X/Y" lines)
  * metrics.csv that train_lora.py writes under --save_dir

Config shape (from the frontend training form):
{
    "modelName":       "my-lora",            # used for run dir name
    "baseModel":       "sa3-medium-base",    # must end in -base
    "projectName":     "my_first_track",     # Dataset Workbench project name
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
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.backend.data.projects import project_path
from app.core.config import get_config
from app.core.training.sa3_lora_runner import (
    SA3_BASE_MODELS,
    build_train_command,
    build_train_env,
    convert_run_checkpoints_to_safetensors,
    prestage_base_model,
)
from utils.logger import get_logger
from utils.process_control import graceful_stop, pid_alive

logger = get_logger("SA3Trainer")

# Written into each run dir on spawn so a restarted backend can detect (and
# offer to kill) a training subprocess it no longer holds a Popen handle for.
PID_FILENAME = "run.pid"


# --- Defaults --------------------------------------------------------------

DEFAULT_STEPS = 5000
DEFAULT_CHECKPOINT_STEPS = 500
DEFAULT_BATCH_SIZE = 1
DEFAULT_LR = 1e-4
DEFAULT_DURATION = 30.0
DEFAULT_RANK = 16
DEFAULT_ADAPTER = "dora-rows"
DEFAULT_PRECISION = "bf16"


# --- SA3Trainer singleton --------------------------------------------------

class SA3Trainer:
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
        # Fresh run on this trainer — clear any stop flag from a prior run.
        self._stop_requested = False
        # Mark training as in-flight BEFORE any blocking work. /api/start-training
        # can block for tens of seconds (T5Gemma sibling fetch, base-model
        # prestaging) — during that window the frontend polls
        # /api/training-status and would otherwise see is_training=False from
        # the __init__ default and interpret it as "training complete".
        self.status.update({
            "is_training": True,
            "status": "staging",
            "started_at": time.time(),
            "ended_at": None,
            "step": 0,
            "total_steps": int(self.config.get("steps") or DEFAULT_STEPS),
            "loss": None,
            "error": None,
            "checkpoints": [],
            # Surface the concrete seed (the backend rolls a random one when the
            # UI requests it) so the user can reproduce a run they liked.
            "seed": (int(self.config["seed"]) if self.config.get("seed") is not None else None),
            "message": "Preparing dataset and base model…",
        })
        try:
            self._refuse_if_run_dir_owned_by_live_pid()
            self._maybe_wipe_run_dir()
            self._resolve_paths()
            self._stage_dataset()
            self._stage_base_model()
            cmd, env = self._build_invocation()
            self._spawn(cmd, env)
            logger.info(
                "Training started · project=%s · base=%s · adapter=%s · "
                "rank=%s · steps=%s · batch=%s · lr=%s · duration=%ss",
                self.config.get("projectName"),
                self.config.get("baseModel"),
                self.config.get("adapterType") or DEFAULT_ADAPTER,
                self.config.get("loraRank") or DEFAULT_RANK,
                self.config.get("steps") or DEFAULT_STEPS,
                self.config.get("batchSize") or DEFAULT_BATCH_SIZE,
                self.config.get("learningRate") or DEFAULT_LR,
                self.config.get("duration") or DEFAULT_DURATION,
            )
            return {"success": True, "run_dir": str(self.run_dir)}
        except Exception as e:
            self.status["error"] = str(e)
            self.status["status"] = "failed"
            self.status["is_training"] = False
            self.status["ended_at"] = time.time()
            logger.error("Training failed to start: %s", e)
            return {"error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        # Snapshot + add a few derived fields the frontend already reads, so
        # the polling loop in App.js doesn't have to know about both names.
        # SA3 is step-based; we no longer expose `current_epoch`.
        # If the on-disk checkpoint count looks stale (run finished, glob
        # ran with the old filter, no live files surfaced), rescan once
        # lazily so the UI catches up without needing a backend restart.
        if not self.status.get("checkpoints") and self.run_dir is not None:
            ckpt_dir = self.run_dir / "checkpoints"
            if ckpt_dir.exists() and any(ckpt_dir.glob("*.ckpt")):
                self._scan_checkpoints()
        s = dict(self.status)
        total = int(s.get("total_steps") or 0)
        step = int(s.get("step") or 0)
        s["current_step"] = step
        s["progress"] = int(round(100 * step / total)) if total > 0 else 0
        s["checkpoints_saved"] = len(s.get("checkpoints") or [])
        return s

    def stop(self) -> Dict[str, Any]:
        if not self.process or self.process.poll() is not None:
            return {"error": "Nothing to stop — no active training run."}
        try:
            # Flag the stop so the monitor thread labels the exit "stopped"
            # rather than "failed" — SIGINT doesn't yield a stable rc==-2.
            self._stop_requested = True
            # Platform-split escalation lives in graceful_stop. The old code
            # sent SIGINT unconditionally, which raises ValueError on Windows
            # — and the terminate()/kill() fallbacks were nested inside the
            # TimeoutExpired branch, so they never ran: Stop was a dead
            # button on Windows while the child kept training.
            graceful_stop(self.process)
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
        proj_name = self.config.get("projectName") or self.config.get("project_name")
        data_dir = str(project_path(proj_name)) if proj_name else None
        return {
            "model_name": self.config.get("modelName", "fragmenta-lora"),
            "base_model": self.config.get("baseModel"),
            "project_name": proj_name,
            "data_dir": data_dir,
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
        # Lightning's CSVLogger writes metrics.csv under
        # `<save_dir>/lightning_logs/version_X/metrics.csv`. We don't know X
        # upfront, so leave this unset and let _scrape_loss_history /
        # _scrape_csv_loss rglob for it the first time they're called.
        self.metrics_csv = None
        if create_dirs:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "checkpoints").mkdir(exist_ok=True)

    @classmethod
    def existing_run_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """Look up an existing run dir for a given LoRA name. Returns a dict
        of countable artifacts if the dir exists with content, else None.

        Used by /api/start-training to refuse a same-name run unless the
        caller explicitly opts in to overwrite. Counts only *.ckpt and
        *.safetensors so a half-set-up dir with only a metadata file
        doesn't trip the prompt.
        """
        import shutil  # noqa: F401  # ensures shutil resolves if user calls _maybe_wipe later
        cfg = get_config()
        run_name = cls._safe_name(model_name or "lora-run")
        run_dir = cfg.get_path("models_fine_tuned") / run_name
        if not run_dir.exists():
            return None
        ckpt_dir = run_dir / "checkpoints"
        files = []
        if ckpt_dir.exists():
            for ext in ("*.safetensors", "*.ckpt"):
                files.extend(ckpt_dir.glob(ext))
        if not files and not (run_dir / "training.log").exists():
            return None
        return {
            "run_dir": str(run_dir),
            "run_name": run_name,
            "checkpoint_count": len(files),
            "has_log": (run_dir / "training.log").exists(),
        }

    def _run_dir_for_config(self) -> Path:
        cfg = get_config()
        run_name = self._safe_name(self.config.get("modelName") or "lora-run")
        return cfg.get_path("models_fine_tuned") / run_name

    def _refuse_if_run_dir_owned_by_live_pid(self) -> None:
        """Abort start-up when the target run dir belongs to a live process.

        An orphaned trainer (spawned by a backend that has since restarted)
        is still writing checkpoints into its run dir; wiping or re-staging
        that dir would corrupt the run. The user can kill it via the
        orphaned-runs surface in /api/training-status.
        """
        meta = _read_pid_file(self._run_dir_for_config())
        if meta and pid_alive(meta.get("pid"), meta.get("create_time")):
            raise RuntimeError(
                f"Run '{self._run_dir_for_config().name}' appears to still be "
                f"training under PID {meta['pid']} (probably orphaned by a "
                "backend restart). Kill it from the training status panel "
                "before starting or overwriting this run."
            )

    def _maybe_wipe_run_dir(self) -> None:
        """Honor the `overwrite` flag — wipe the run dir before staging."""
        if not self.config.get("overwrite"):
            return
        run_dir = self._run_dir_for_config()
        if run_dir.exists():
            import shutil
            shutil.rmtree(run_dir)
            logger.info("Cleared existing run dir before restart: %s", run_dir)

    def _stage_dataset(self) -> None:
        """Resolve --data_dir from a Dataset Workbench project.

        Training reads the committed `.txt` sidecars sitting next to each
        audio file inside `<projects_dir>/<projectName>/`. The Workbench's
        "Create Dataset" action materialised those sidecars; we don't
        rewrite anything here.
        """
        project_name = self.config.get("projectName") or self.config.get("project_name")
        if not project_name:
            raise FileNotFoundError(
                "projectName is required. Pick a project in the Training "
                "tab's Dataset picker before starting a run."
            )
        proj_dir = project_path(project_name)
        if not proj_dir.exists():
            raise FileNotFoundError(f"project not found: {project_name}")

        sidecars = list(proj_dir.glob("*.txt"))
        if not sidecars:
            raise RuntimeError(
                f"project “{project_name}” has no committed prompts yet — "
                "annotate the clips and click Create Dataset, then retry."
            )
        # SA3's caption_metadata_fn rejects clips whose sidecar is empty,
        # so they silently drop out of the training set. Count them upfront
        # so the user knows what they're actually training on (and refuse
        # to start if NONE have prompts — that would just waste GPU hours).
        non_empty = [p for p in sidecars if p.read_text(encoding="utf-8").strip()]
        if not non_empty:
            raise RuntimeError(
                f"project “{project_name}” has {len(sidecars)} clip(s) but every "
                "sidecar is empty — SA3 will reject all of them. Annotate at "
                "least one clip and re-commit before training."
            )
        blank = len(sidecars) - len(non_empty)
        if blank > 0:
            logger.warning(
                "%d of %d clip(s) in project '%s' have empty prompts — "
                "SA3 will silently drop them. Training on %d clip(s).",
                blank, len(sidecars), project_name, len(non_empty),
            )
            self.status["log_tail"].append(
                f"Warning: {blank}/{len(sidecars)} clips have empty prompts and "
                "will be dropped by SA3's data loader."
            )
        self.status["log_tail"].append(
            f"Dataset: project '{project_name}' · {len(non_empty)} usable clip(s) · {proj_dir}"
        )
        self._data_dir = proj_dir

        # Phase 6 — opt into pre-encoded latents if a compatible .latents/
        # cache exists. SA3's `train_lora.py --encoded_dir` then skips the
        # autoencoder pass per step. The cache is AE-bound (same-s vs
        # same-l) so we verify the manifest matches the picked base before
        # using it — otherwise we'd feed the DiT mis-shaped latents.
        self._encoded_dir: Optional[Path] = None
        try:
            from app.backend.data.pre_encoder import (
                latents_dir, latents_count, latents_match_base,
            )
            ldir = latents_dir(project_name)
            base_model = self.config.get("baseModel")
            if ldir.exists() and latents_count(project_name) > 0:
                if latents_match_base(project_name, base_model):
                    self._encoded_dir = ldir
                    self.status["log_tail"].append(
                        f"Using pre-encoded latents: {latents_count(project_name)} "
                        f"file(s) · {ldir}"
                    )
                    logger.info(
                        "Pre-encoded latents detected for project '%s' (%d files) — "
                        "skipping SAME autoencoder per step.",
                        project_name, latents_count(project_name),
                    )
                else:
                    logger.warning(
                        "Pre-encoded latents exist for project '%s' but were "
                        "produced by a different autoencoder than the chosen "
                        "base (%s) — falling back to live encoding.",
                        project_name, base_model,
                    )
                    self.status["log_tail"].append(
                        f"Note: project has cached latents but they're for a "
                        f"different autoencoder than {base_model}. Training "
                        "will re-encode audio per step."
                    )
        except Exception as exc:
            logger.warning("Pre-encoded latents probe failed: %s", exc)

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
            # Mirror to the project logger so the terminal shows what's
            # happening during long blocking operations (e.g. first-time
            # T5Gemma sibling fetch can take ~30s on medium-base).
            logger.info("[stage] %s", msg)

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

        adapter_type = self.config.get("adapterType") or DEFAULT_ADAPTER

        # -XS adapters can reuse a precomputed SVD-bases cache keyed by base
        # model, skipping the per-layer SVD at startup. SA3 only loads (never
        # writes) this file, so we pass it only when present; population is a
        # manual/precompute step. Ensure the dir exists so it's discoverable.
        svd_bases_path = None
        if adapter_type.endswith("-xs"):
            svd_cache_dir = get_config().get_path("models_fine_tuned") / ".svd_cache"
            svd_cache_dir.mkdir(parents=True, exist_ok=True)
            candidate = svd_cache_dir / f"{self.config['baseModel']}.pt"
            if candidate.exists():
                svd_bases_path = candidate

        cmd = build_train_command(
            venv_python=venv_python,
            sa3_vendor_dir=sa3_vendor,
            sa3_model_name=sa3_name,
            data_dir=self._data_dir,
            encoded_dir=getattr(self, "_encoded_dir", None),
            svd_bases_path=svd_bases_path,
            save_dir=self.run_dir / "checkpoints",
            rank=int(self.config.get("loraRank") or DEFAULT_RANK),
            lora_alpha=self.config.get("loraAlpha"),
            adapter_type=adapter_type,
            dropout=float(self.config.get("loraDropout") or 0.0),
            lr=float(self.config.get("learningRate") or DEFAULT_LR),
            steps=int(self.config.get("steps") or DEFAULT_STEPS),
            batch_size=int(self.config.get("batchSize") or DEFAULT_BATCH_SIZE),
            # Default to AND clamp at the base model's native training length
            # (medium ≈380s, small ≈120s) — SA3's DiT tops out at 4096 latent
            # tokens, so a longer window would exceed the model, not just cost
            # VRAM. A missing duration defaults to the model max.
            duration=min(
                float(self.config.get("duration") or (380.0 if "medium" in sa3_name else 120.0)),
                380.0 if "medium" in sa3_name else 120.0,
            ),
            base_precision=precision,
            include=include,
            exclude=exclude,
            seed=(int(self.config["seed"]) if self.config.get("seed") is not None else 42),
            checkpoint_every=int(self.config.get("checkpointSteps") or DEFAULT_CHECKPOINT_STEPS),
            name=self.config.get("modelName") or "fragmenta-lora",
        )
        env = build_train_env(sa3_vendor, self._hub_dir)
        return cmd, env

    def _spawn(self, cmd: List[str], env: Dict[str, str]) -> None:
        log_path = self.run_dir / "training.log"
        rank = int(self.config.get("loraRank") or DEFAULT_RANK)
        alpha_cfg = self.config.get("loraAlpha")
        alpha = int(alpha_cfg) if alpha_cfg not in (None, "") else rank
        # Stamp training_metadata.json so /api/loras can find the base_model
        # if the embedded safetensors metadata is missing it (legacy paths).
        (self.run_dir / "training_metadata.json").write_text(json.dumps({
            "mode": "lora",
            "engine": "sa3",
            "base_model": self.config.get("baseModel"),
            "model_name": self.config.get("modelName"),
            "started_at": time.time(),
            "lora_config": {
                "rank": rank,
                "alpha": alpha,
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
            # Without an explicit encoding, text mode decodes with the locale
            # default — cp1252 on Windows — and the first UTF-8 glyph the
            # child prints raises UnicodeDecodeError in the monitor thread:
            # the run gets marked failed while the child keeps training and
            # eventually blocks on a full stdout pipe.
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        _write_pid_file(self.run_dir, self.process.pid, self.run_dir.name)
        self._monitor_thread = threading.Thread(
            target=self._monitor,
            args=(log_path,),
            daemon=True,
            name=f"sa3-train-monitor:{self.run_dir.name}",
        )
        self._monitor_thread.start()

    def _monitor(self, log_path: Path) -> None:
        """Pull stdout, parse PyTorch Lightning progress, scrape loss, watch checkpoints.

        SA3 trains via PL whose default progress bar emits *per-epoch* step
        counts ("Epoch 6: 50%|...| 25/50 [00:07<00:07, 3.36it/s, train/loss=0.559]").
        We derive the global step as `epoch * batches_per_epoch + step_in_epoch`,
        capture `batches_per_epoch` from the first such line (it's stable across
        epochs since SampleDataset returns a fixed length), and clamp the
        result to the configured max_steps so the percentage doesn't go past
        100 if the final epoch overruns.
        """
        epoch_pat = re.compile(r"Epoch\s+(\d+):")
        in_epoch_pat = re.compile(r"\|\s*(\d+)/(\d+)\b")  # tqdm's "current/total"
        loss_pat = re.compile(r"train/loss=([\d.eE+\-]+)")
        speed_pat = re.compile(r"([\d.]+)it/s")
        last_log_flush = time.time()
        last_ckpt_scan = 0.0
        last_terminal_log = 0.0
        last_logged_step = -1
        prev_ckpt_count = 0
        current_epoch = 0
        batches_per_epoch = 0
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

                        # Only parse the step counter on lines that ARE the
                        # training progress bar (prefixed with "Epoch N:"),
                        # so unrelated tqdm bars during startup (e.g.
                        # "Loading checkpoint shards: 9/9") don't pollute
                        # batches_per_epoch.
                        m_epoch = epoch_pat.search(line)
                        if m_epoch:
                            current_epoch = int(m_epoch.group(1))
                            m_step = in_epoch_pat.search(line)
                            if m_step:
                                cur_in_epoch = int(m_step.group(1))
                                per_epoch = int(m_step.group(2))
                                if per_epoch > 0 and batches_per_epoch == 0:
                                    batches_per_epoch = per_epoch
                                if batches_per_epoch > 0:
                                    global_step = current_epoch * batches_per_epoch + cur_in_epoch
                                    max_steps = self.status.get("total_steps") or 0
                                    if max_steps > 0:
                                        global_step = min(global_step, max_steps)
                                    if global_step > self.status.get("step", 0):
                                        self.status["step"] = global_step

                        m_loss = loss_pat.search(line)
                        if m_loss:
                            try:
                                self.status["loss"] = float(m_loss.group(1))
                            except ValueError:
                                pass

                        # Live checkpoint enumeration + loss history scrape.
                        # Lightning writes *.ckpt every N steps; we want the
                        # count to climb in the UI as files appear, not only
                        # at end-of-run. Bucketed to ~2s so we don't pound
                        # the FS. The loss history scrape backfills step
                        # 0..49 from metrics.csv since PL's stdout postfix
                        # doesn't show train/loss until end-of-epoch-0.
                        now = time.time()
                        if now - last_ckpt_scan > 2.0:
                            last_ckpt_scan = now
                            self._scan_checkpoints()
                            self._scrape_loss_history()
                            cur_ckpt_count = len(self.status.get("checkpoints") or [])
                            if cur_ckpt_count > prev_ckpt_count:
                                logger.info(
                                    "Checkpoint saved · %d total · run=%s",
                                    cur_ckpt_count, self.run_dir.name,
                                )
                                prev_ckpt_count = cur_ckpt_count

                        # Throttled progress to the backend terminal log.
                        # Lightning emits step lines ~3× per second; we
                        # condense to one tidy summary every 5s. Omit the
                        # loss segment when we don't have a value yet (the
                        # CSV scrape runs every 2s but PL may not have
                        # logged anything during the very first second).
                        cur_step = self.status.get("step") or 0
                        if (cur_step > last_logged_step
                                and now - last_terminal_log >= 5.0):
                            total = self.status.get("total_steps") or 0
                            loss = self.status.get("loss")
                            pct = round(100 * cur_step / total) if total > 0 else 0
                            speed_m = speed_pat.search(line)
                            parts = [f"step {cur_step}/{total} ({pct}%)"]
                            if isinstance(loss, (int, float)):
                                parts.append(f"loss {loss:.4f}")
                            if speed_m:
                                parts.append(f"{speed_m.group(1)} it/s")
                            logger.info(" · ".join(parts))
                            last_terminal_log = now
                            last_logged_step = cur_step
                rc = self.process.wait() if self.process else 1
        except Exception as e:
            self.status["error"] = str(e)
            rc = -1

        self.status["ended_at"] = time.time()
        self.status["is_training"] = False
        # The subprocess is gone — drop its PID file so it can't be reported
        # as an orphan. PID files only outlive a run when the backend died
        # while the child was still training.
        if self.run_dir is not None:
            _remove_pid_file(self.run_dir)
        # A user-requested stop wins regardless of the exit code (SIGINT can
        # surface as various negative/non-zero codes across platforms).
        if getattr(self, "_stop_requested", False):
            self.status["status"] = "stopped"
        else:
            self.status["status"] = "complete" if rc == 0 else "failed"
        if self.status["status"] == "failed" and not self.status.get("error"):
            self.status["error"] = f"train_lora.py exited with code {rc}"

        # Convert PyTorch Lightning .ckpt files to SA3's native .safetensors
        # LoRA format — the inference loader (/api/loras) only sees
        # .safetensors, so unconverted .ckpt files would be functionally
        # orphaned. We also inject `base_model` into the safetensors header
        # so /api/loras' metadata filter passes without a JSON fallback.
        # Best-effort: failure here doesn't fail the run.
        if self.status["status"] in ("complete", "stopped") and self.run_dir:
            try:
                produced = convert_run_checkpoints_to_safetensors(
                    self.run_dir,
                    base_model=self.config.get("baseModel"),
                    model_name=self.config.get("modelName"),
                )
                if produced:
                    logger.info(
                        "Converted %d checkpoint(s) to .safetensors · run=%s",
                        len(produced), self.run_dir.name,
                    )
            except Exception as exc:
                logger.warning("Checkpoint conversion failed: %s", exc)

        # Final pass: enumerate written checkpoints + full loss history +
        # latest single-value loss.
        self._scan_checkpoints()
        self._scrape_loss_history()
        self._scrape_csv_loss()

        final_step = self.status.get("step") or 0
        final_total = self.status.get("total_steps") or 0
        final_loss = self.status.get("loss")
        final_ckpts = len(self.status.get("checkpoints") or [])
        loss_str = f"{final_loss:.4f}" if isinstance(final_loss, (int, float)) else "—"
        if self.status["status"] == "complete":
            logger.info(
                "Training complete · %d/%d steps · final loss %s · %d checkpoint(s) · run=%s",
                final_step, final_total, loss_str, final_ckpts, self.run_dir.name,
            )
        elif self.status["status"] == "stopped":
            logger.info(
                "Training stopped at step %d/%d · %d checkpoint(s) · run=%s",
                final_step, final_total, final_ckpts, self.run_dir.name,
            )
        else:
            logger.error(
                "Training failed (exit %s) · %d/%d steps · error: %s · run=%s",
                rc, final_step, final_total, self.status.get("error"), self.run_dir.name,
            )

    def _scrape_loss_history(self) -> None:
        """Refresh self.status['loss_history'] from Lightning's metrics.csv.

        PL's tqdm postfix only surfaces `train/loss=` *after* the first
        metrics flush (typically end-of-epoch-0), so step 0..49 of a fresh
        run never appear in stdout. metrics.csv, on the other hand, has
        per-step rows from step 0 — we just need to read it.

        Cheap: even at 10K steps a CSV scan is sub-10ms. Skipped silently
        if the file hasn't been created yet (early in the run, before PL's
        CSVLogger flushes anything).
        """
        if not self.metrics_csv or not self.metrics_csv.exists():
            # CSVLogger writes under <save_dir>/lightning_logs/version_*/
            if self.run_dir:
                for p in (self.run_dir / "checkpoints").rglob("metrics.csv"):
                    self.metrics_csv = p
                    break
        if not self.metrics_csv or not self.metrics_csv.exists():
            return
        try:
            with open(self.metrics_csv) as f:
                rows = list(csv.DictReader(f))
        except Exception:
            return
        points: List[Dict[str, Any]] = []
        loss_keys = ("train/loss", "loss", "train_loss")
        for row in rows:
            step_raw = row.get("step")
            if step_raw in (None, ""):
                continue
            try:
                step = int(step_raw)
            except ValueError:
                continue
            for k in loss_keys:
                v = row.get(k)
                if v not in (None, ""):
                    try:
                        points.append({"step": step, "loss": float(v)})
                    except ValueError:
                        pass
                    break
        # Dedupe: csv can have multiple rows per step (different metric flush
        # boundaries) — keep the last loss seen for each step.
        by_step: Dict[int, float] = {}
        for p in points:
            by_step[p["step"]] = p["loss"]
        ordered = sorted(by_step.items())
        self.status["loss_history"] = [{"step": s, "loss": l} for s, l in ordered]
        # Also surface the most recent loss as the scalar so the terminal
        # log and "Current Loss" field don't show "—" until end-of-epoch-0.
        # PL's tqdm postfix is async; the CSV row lands a beat ahead.
        if ordered:
            self.status["loss"] = ordered[-1][1]

    def _scan_checkpoints(self) -> None:
        """Update self.status['checkpoints'] from on-disk artifacts.

        SA3's train_lora.py uses PyTorch Lightning's ModelCheckpoint, which
        writes `.ckpt` files (Lightning pickle format). The diffusion wrapper's
        `on_save_checkpoint` hook strips the state_dict to LoRA-only weights
        plus the embedded `lora_config`, so each .ckpt IS a LoRA checkpoint.
        We also accept .safetensors for forward-compat with a future export
        path or manual conversion.
        """
        if not self.run_dir:
            return
        ckpt_dir = self.run_dir / "checkpoints"
        if not ckpt_dir.exists():
            return
        found = []
        for ext in ("*.safetensors", "*.ckpt"):
            found.extend(ckpt_dir.glob(ext))
        # Lightning writes nested lightning_logs/version_X/* — those aren't
        # the user-facing artifacts; skip recursion.
        project_root = get_config().project_root
        self.status["checkpoints"] = sorted(
            str(p.relative_to(project_root)) for p in found
        )

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


# --- PID files + orphan detection -------------------------------------------
# Run state used to live only in the in-memory `_active` trainer + its Popen
# handle. A backend restart (crash, packaged-app relaunch, dev reload) left
# the training subprocess running with nothing tracking it: invisible in the
# UI, unkillable, holding the GPU, and its run dir free to be overwritten.

def _write_pid_file(run_dir: Path, pid: int, run_name: str) -> None:
    try:
        create_time = None
        try:
            import psutil
            create_time = psutil.Process(pid).create_time()
        except Exception:
            pass
        (run_dir / PID_FILENAME).write_text(json.dumps({
            "pid": pid,
            # create_time pins the PID to THIS process so a recycled PID
            # belonging to something unrelated is never reported or killed.
            "create_time": create_time,
            "run_name": run_name,
            "started_at": time.time(),
        }, indent=2))
    except Exception as exc:
        logger.warning("Could not write PID file for %s: %s", run_name, exc)


def _read_pid_file(run_dir: Path) -> Optional[Dict[str, Any]]:
    pf = run_dir / PID_FILENAME
    if not pf.is_file():
        return None
    try:
        meta = json.loads(pf.read_text())
        meta["pid"] = int(meta.get("pid") or 0)
        return meta
    except Exception:
        return None


def _remove_pid_file(run_dir: Path) -> None:
    try:
        (run_dir / PID_FILENAME).unlink(missing_ok=True)
    except Exception:
        pass


_orphan_cache: Dict[str, Any] = {"at": 0.0, "runs": []}
_ORPHAN_SCAN_TTL = 10.0


def scan_orphaned_runs(force: bool = False) -> List[Dict[str, Any]]:
    """Run dirs whose PID file points at a live process we don't own.

    Cached for a few seconds — /api/training-status is polled continuously
    and the scan touches the filesystem + process table. Stale PID files
    (process gone, i.e. the child crashed alongside the backend) are
    cleaned up as a side effect.
    """
    now = time.time()
    if not force and now - _orphan_cache["at"] < _ORPHAN_SCAN_TTL:
        return list(_orphan_cache["runs"])

    orphans: List[Dict[str, Any]] = []
    try:
        root = get_config().get_path("models_fine_tuned")
        active_pid = None
        if _active and _active.process and _active.process.poll() is None:
            active_pid = _active.process.pid
        if root.exists():
            for run_dir in root.iterdir():
                if not run_dir.is_dir():
                    continue
                meta = _read_pid_file(run_dir)
                if not meta or not meta["pid"]:
                    continue
                if meta["pid"] == active_pid:
                    continue  # the run we're tracking, not an orphan
                if pid_alive(meta["pid"], meta.get("create_time")):
                    orphans.append({
                        "run_name": run_dir.name,
                        "pid": meta["pid"],
                        "started_at": meta.get("started_at"),
                    })
                else:
                    _remove_pid_file(run_dir)
    except Exception as exc:
        logger.warning("Orphan-run scan failed: %s", exc)

    _orphan_cache["at"] = now
    _orphan_cache["runs"] = orphans
    return list(orphans)


def kill_orphaned_run(run_name: str) -> Dict[str, Any]:
    """Terminate an orphaned training process identified by its run dir."""
    root = get_config().get_path("models_fine_tuned").resolve()
    run_dir = (root / run_name).resolve()
    try:
        run_dir.relative_to(root)  # no traversal via run_name
    except ValueError:
        return {"error": f"Invalid run name: {run_name!r}"}
    meta = _read_pid_file(run_dir)
    if not meta or not meta["pid"]:
        return {"error": f"No PID file for run '{run_name}'."}
    if _active and _active.process and _active.process.poll() is None \
            and _active.process.pid == meta["pid"]:
        return {"error": "That run is actively tracked — use Stop Training instead."}
    if not pid_alive(meta["pid"], meta.get("create_time")):
        _remove_pid_file(run_dir)
        return {"success": True, "message": "Process already gone; cleaned up its PID file."}
    try:
        import psutil
        proc = psutil.Process(meta["pid"])
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except psutil.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    except Exception as exc:
        return {"error": f"Could not kill PID {meta['pid']}: {exc}"}
    _remove_pid_file(run_dir)
    _orphan_cache["at"] = 0.0  # next status poll re-scans
    logger.info("Killed orphaned training run '%s' (PID %d)", run_name, meta["pid"])
    return {"success": True, "killed_pid": meta["pid"]}


# --- Module-level singleton + public functions -----------------------------

_active: Optional[SA3Trainer] = None
_lock = threading.Lock()


def get_trainer() -> Optional[SA3Trainer]:
    return _active


def start_training(config: Dict[str, Any]) -> Dict[str, Any]:
    global _active
    with _lock:
        if _active and _active.status.get("is_training"):
            return {"error": "A training run is already in progress."}
        trainer = SA3Trainer(config)
        # Claim the slot BEFORE releasing the lock. start() blocks on dataset
        # staging and base-model prestaging (network fetches that can take
        # minutes); holding the module lock through that made a second
        # start request hang on the lock instead of failing fast, and froze
        # every other locked operation behind the download.
        trainer.status["is_training"] = True
        trainer.status["status"] = "staging"
        _active = trainer
    return trainer.start()


def get_training_status() -> Dict[str, Any]:
    if _active is None:
        status = {
            "is_training": False,
            "status": "idle",
            "message": "No training run has been started yet.",
            "progress": 0,
            "current_step": 0,
            "total_steps": 0,
            "checkpoints_saved": 0,
            "loss": None,
        }
    else:
        status = _active.get_status()
    # Surface training subprocesses a previous backend left running so the
    # UI can warn and offer to kill them (POST /api/training/kill-orphan).
    status["orphaned_runs"] = scan_orphaned_runs()
    return status


def stop_training() -> Dict[str, Any]:
    if _active is None:
        return {"error": "No training run to stop."}
    return _active.stop()


def preview_training_plan(config: Dict[str, Any]) -> Dict[str, Any]:
    return SA3Trainer(config).preview_plan()
