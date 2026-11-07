"""Engine tab backend: start and stop the perpetual engine as a subprocess.

The tab talks to the engine's own API (`/v1` on its port) directly; these
three routes only manage the process, so the headless command line and the
Start button run the same thing. Off on the Space and in Docker.

    POST /api/engine/start   {scene, program, layout, port, model_id, loras, renderer, sink, osc_port}
    POST /api/engine/stop
    GET  /api/engine/status
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

from app.core.config import get_config
from utils.logger import get_logger

logger = get_logger("EngineAPI")
engine_bp = Blueprint("engine", __name__)

_proc: Optional[subprocess.Popen] = None
_info: Dict[str, Any] = {}
_lock = threading.Lock()
DEFAULT_PORT = 5002


def available() -> bool:
    return not (os.environ.get("FRAGMENTA_DOCKER", "0") == "1"
                or os.environ.get("FRAGMENTA_HF_SPACE", "0") == "1"
                or bool(os.environ.get("SPACE_ID")))


def _log_path() -> Path:
    p = get_config().get_path("logs")
    p.mkdir(parents=True, exist_ok=True)
    return p / "engine.log"


def _running() -> bool:
    return _proc is not None and _proc.poll() is None


def _status() -> Dict[str, Any]:
    running = _running()
    return {
        "available": available(), "running": running, "pid": _proc.pid if _proc else None,
        "port": _info.get("port", DEFAULT_PORT), "url": _info.get("url", f"http://127.0.0.1:{DEFAULT_PORT}/v1"),
        "started_at": _info.get("started_at"), "exit_code": (None if running or _proc is None else _proc.returncode),
        "log": str(_log_path()), "args": _info.get("args", []),
    }


def _ping(url: str, timeout: float = 0.8) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/status", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


@engine_bp.get("/api/engine/status")
def engine_status():
    s = _status()
    if s["running"]:
        s["api_up"] = _ping(s["url"])
    return jsonify(s)


@engine_bp.post("/api/engine/start")
def engine_start():
    global _proc, _info
    if not available():
        return jsonify({"error": "The engine runs on a desktop install, not on the Space or in Docker."}), 403
    body = request.get_json(force=True, silent=True) or {}
    with _lock:
        if _running():
            return jsonify({**_status(), "already": True})
        port = int(body.get("port") or DEFAULT_PORT)
        cfg = get_config()
        args = [sys.executable, "-m", "app.engine", "--port", str(port),
                "--scene", str(body.get("scene") or "places"), "--layout", str(body.get("layout") or "stereo"),
                "--renderer", "fake" if body.get("renderer") == "fake" else "torch",
                "--sink", str(body.get("sink") or "auto"), "--osc-port", str(int(body.get("osc_port", 9000)))]
        if body.get("program"):
            args += ["--program", str(body["program"])]
        if body.get("model_id"):
            args += ["--model", str(body["model_id"])]
        for l in body.get("loras") or []:
            if isinstance(l, dict) and l.get("path"):
                args += ["--lora", f"{l['path']}:{float(l.get('strength', 1.0))}"]
            elif isinstance(l, str):
                args += ["--lora", l]
        if body.get("device") not in (None, ""):
            args += ["--device", str(body["device"])]
        if body.get("channels"):
            args += ["--channels", str(int(body["channels"]))]
        for u in body.get("radio") or []:
            if isinstance(u, str) and u.strip():
                args += ["--radio", u.strip()]
        if body.get("clock"):
            args.append("--clock")
        if body.get("play"):
            args.append("--play")
        if body.get("no_resume"):
            args.append("--no-resume")
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        # The tab calls the engine's API from this backend's origin; let it in
        # whatever port this backend runs on (the engine allows 5001 and 3000 itself).
        flask_port = os.environ.get("FLASK_PORT", "5001")
        origins = [o for o in env.get("FRAGMENTA_ALLOWED_ORIGINS", "").split(",") if o.strip()]
        origins += [f"http://127.0.0.1:{flask_port}", f"http://localhost:{flask_port}"]
        env["FRAGMENTA_ALLOWED_ORIGINS"] = ",".join(dict.fromkeys(origins))
        log = open(_log_path(), "a", encoding="utf-8")
        log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} start: {' '.join(args)}\n")
        log.flush()
        try:
            _proc = subprocess.Popen(args, cwd=str(cfg.project_root), env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL)
        except OSError as exc:
            return jsonify({"error": f"could not start the engine: {exc}"}), 500
        _info = {"port": port, "url": f"http://127.0.0.1:{port}/v1", "started_at": time.time(), "args": args[2:]}
        logger.info("engine started pid %s on port %s", _proc.pid, port)
    # give it a moment so the tab's first poll finds the API
    deadline = time.time() + 3.0
    while time.time() < deadline and _running() and not _ping(_info["url"], 0.4):
        time.sleep(0.2)
    return jsonify({**_status(), "api_up": _ping(_info["url"])}), 201


@engine_bp.post("/api/engine/stop")
def engine_stop():
    global _proc
    with _lock:
        if not _running():
            return jsonify({**_status(), "stopped": False})
        url = _info.get("url")
        try:
            req = urllib.request.Request(f"{url}/shutdown", data=b"{}", method="POST",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            pass
        try:
            _proc.wait(timeout=6)
        except subprocess.TimeoutExpired:
            _proc.terminate()
            try:
                _proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                _proc.kill()
        logger.info("engine stopped (%s)", _proc.returncode)
    return jsonify({**_status(), "stopped": True})


def stop_on_exit() -> None:
    """Called at backend exit so an engine never outlives the app."""
    if _running():
        try:
            _proc.terminate()
            _proc.wait(timeout=3)
        except Exception:
            try:
                _proc.kill()
            except Exception:
                pass
