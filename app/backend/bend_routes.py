"""Bend tab API — a separate blueprint so the 3.4k-line app.py stays out of
it. Registered from app.py with one line. Everything here is additive; no
existing endpoint changes behaviour when the Bend tab is unused.

Routes:
    GET    /api/bend/targets?model_id=…   signal-path registry + operator specs
    POST   /api/bend/validate             dry-run a patch, return warnings
    GET    /api/bend/presets              list saved patches (bends/presets/)
    POST   /api/bend/presets              save {name, patch}
    DELETE /api/bend/presets/<name>       delete one
    GET    /api/bend/log?model_id=&limit= Bending Log, newest first
    PATCH  /api/bend/log/<id>             set the sonic-result note
    DELETE /api/bend/log/<id>             remove one entry
    POST   /api/bend/lora                 bend / blend adapter files (Phase 2)
    GET    /api/bend/boards/<model_id>    the Probe board's names (user's finds)
    PUT    /api/bend/boards/<model_id>    save them
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from flask import Blueprint, jsonify, request

from app.core.bending.bendlog import BendLog
from app.core.bending.lora_bend import LoraBendError, bend_lora, blend_loras
from app.core.bending.operators import describe_operators
from app.core.bending.patch import SWAP_FNS, PatchError, validate_patch
from app.core.bending.targets import get_targets
from app.core.config import get_config
from utils.logger import get_logger

logger = get_logger("BendAPI")

bend_bp = Blueprint("bend", __name__)

_PRESET_NAME_RE = re.compile(r"^[A-Za-z0-9._ -]{1,80}$")


def _bends_dir() -> Path:
    d = get_config().project_root / "bends"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _presets_dir() -> Path:
    d = _bends_dir() / "presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _boards_dir() -> Path:
    d = _bends_dir() / "boards"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _log() -> BendLog:
    return BendLog(_bends_dir())


# --- registry ---------------------------------------------------------------

@bend_bp.route("/api/bend/targets", methods=["GET"])
def bend_targets():
    model_id = (request.args.get("model_id") or "sa3-small-music-base").strip()
    config = get_config()
    targets = get_targets(model_id, config.get_path("models_pretrained"))
    targets["operators"] = describe_operators()
    targets["swap_fns"] = list(SWAP_FNS)
    return jsonify(targets)


@bend_bp.route("/api/bend/validate", methods=["POST"])
def bend_validate():
    data = request.json or {}
    try:
        normalized, warnings = validate_patch(data.get("patch"))
    except PatchError as e:
        return jsonify({"valid": False, "error": str(e)}), 400
    return jsonify({"valid": True, "patch": normalized, "warnings": warnings})


# --- presets ----------------------------------------------------------------

@bend_bp.route("/api/bend/presets", methods=["GET"])
def bend_presets_list():
    presets = []
    for f in sorted(_presets_dir().glob("*.json")):
        try:
            with open(f) as fh:
                patch = json.load(fh)
            presets.append({
                "name": f.stem,
                "model_id": patch.get("model_id") or "",
                "modules": len(patch.get("modules") or []),
                "patch": patch,
            })
        except Exception:
            continue
    return jsonify({"presets": presets})


@bend_bp.route("/api/bend/presets", methods=["POST"])
def bend_presets_save():
    data = request.json or {}
    name = str(data.get("name") or "").strip()
    if not _PRESET_NAME_RE.match(name):
        return jsonify({"error": "Preset name must be 1-80 chars of letters, "
                                 "digits, dot, dash, underscore or space."}), 400
    try:
        normalized, warnings = validate_patch(data.get("patch"))
    except PatchError as e:
        return jsonify({"error": f"Patch invalid: {e}"}), 400
    normalized["name"] = name
    with open(_presets_dir() / f"{name}.json", "w") as fh:
        json.dump(normalized, fh, indent=2)
    return jsonify({"saved": name, "warnings": warnings})


@bend_bp.route("/api/bend/presets/<name>", methods=["DELETE"])
def bend_presets_delete(name):
    if not _PRESET_NAME_RE.match(name or ""):
        return jsonify({"error": "Bad preset name."}), 400
    f = _presets_dir() / f"{name}.json"
    if not f.exists():
        return jsonify({"error": "Preset not found."}), 404
    f.unlink()
    return jsonify({"deleted": name})


# --- Bending Log ------------------------------------------------------------

def _fill_loras(entries):
    """Every entry's LoRAs, project-relative as the LoRA picker lists them.
    Entries logged before the log recorded LoRAs get them from the
    fragment's own sidecar, which has always recorded them."""
    root = get_config().project_root.resolve()
    out_dir = get_config().get_path("output")

    def rel(path):
        p = Path(str(path))
        try:
            return str(p.resolve().relative_to(root))
        except ValueError:
            return str(p)

    for e in entries:
        raw = e.get("loras")
        if raw is None:
            raw = []
            if e.get("fragment"):
                try:
                    with open(out_dir / (e["fragment"] + ".json")) as fh:
                        raw = json.load(fh).get("loras") or []
                except (OSError, ValueError):
                    pass
        e["loras"] = [{"path": rel(l["path"]), "strength": float(l.get("strength", 1.0))}
                      for l in raw if isinstance(l, dict) and l.get("path")]
    return entries


@bend_bp.route("/api/bend/log", methods=["GET"])
def bend_log_list():
    model_id = (request.args.get("model_id") or "").strip() or None
    try:
        limit = int(request.args.get("limit", 200))
    except ValueError:
        limit = 200
    return jsonify({"entries": _fill_loras(_log().list(model_id=model_id, limit=limit))})


@bend_bp.route("/api/bend/log/<entry_id>", methods=["PATCH"])
def bend_log_note(entry_id):
    data = request.json or {}
    if _log().set_note(entry_id, str(data.get("note", ""))):
        return jsonify({"updated": entry_id})
    return jsonify({"error": "Log entry not found."}), 404


@bend_bp.route("/api/bend/log/<entry_id>", methods=["DELETE"])
def bend_log_delete(entry_id):
    if _log().delete(entry_id):
        return jsonify({"deleted": entry_id})
    return jsonify({"error": "Log entry not found."}), 404


# --- adapter bending / Model Blending (Phase 2) ----------------------------

def _resolve_lora_path(raw: str) -> Path:
    config = get_config()
    p = Path(str(raw))
    if not p.is_absolute():
        p = config.project_root / p
    p = p.resolve()
    fine_tuned = config.get_path("models_fine_tuned").resolve()
    # Only adapters inside models/fine_tuned/ may be read — this endpoint
    # takes client-supplied paths and must not become a file oracle.
    if not p.is_relative_to(fine_tuned):
        raise LoraBendError("LoRA paths must live under models/fine_tuned/.")
    if not p.exists():
        raise LoraBendError(f"LoRA not found: {raw}")
    return p


@bend_bp.route("/api/bend/lora", methods=["POST"])
def bend_lora_route():
    data = request.json or {}
    mode = str(data.get("mode", "bend"))
    output_name = str(data.get("output_name") or "").strip()
    config = get_config()
    fine_tuned = config.get_path("models_fine_tuned")
    try:
        if mode == "blend":
            a = _resolve_lora_path(data.get("source"))
            b = _resolve_lora_path(data.get("source_b"))
            k = float(data.get("k", 0.5))
            out, warnings = blend_loras(
                a, b, k, fine_tuned_dir=fine_tuned, output_name=output_name)
            return jsonify({"output": str(out), "warnings": warnings})
        elif mode == "bend":
            src = _resolve_lora_path(data.get("source"))
            ops = data.get("ops") or []
            if not isinstance(ops, list) or not ops:
                return jsonify({"error": "ops must be a non-empty array."}), 400
            out = bend_lora(
                src, ops, fine_tuned_dir=fine_tuned, output_name=output_name)
            return jsonify({"output": str(out), "warnings": []})
        return jsonify({"error": f"Unknown mode {mode!r}."}), 400
    except LoraBendError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("LoRA bend failed")
        return jsonify({"error": f"LoRA bend failed: {e}"}), 500


# --- Probe boards -----------------------------------------------------------
# What each contact on the board does is fixed (bendUtils.padModule on the
# client), so all that is stored is what the user found: the names they gave
# contacts and wires after listening, one board per model. Plain JSON in
# bends/boards/, readable and portable outside the app.

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_PAD_ID_RE = re.compile(r"^[a-z0-9._-]{1,32}$")
_PAD_NAME_MAX = 60
_MAX_PADS = 512


def _board_file(model_id: str) -> Path:
    return _boards_dir() / f"{model_id}.json"


@bend_bp.route("/api/bend/boards/<model_id>", methods=["GET"])
def bend_board_get(model_id):
    if not _MODEL_ID_RE.match(model_id):
        return jsonify({"error": "Bad model_id."}), 400
    try:
        with open(_board_file(model_id)) as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("pads"), dict):
            return jsonify(data)
    except (FileNotFoundError, ValueError):
        pass
    return jsonify({"model_id": model_id, "pads": {}})


@bend_bp.route("/api/bend/boards/<model_id>", methods=["PUT"])
def bend_board_put(model_id):
    if not _MODEL_ID_RE.match(model_id):
        return jsonify({"error": "Bad model_id."}), 400
    raw = (request.json or {}).get("pads")
    if not isinstance(raw, dict):
        return jsonify({"error": "pads must be an object."}), 400
    pads = {}
    for pad_id, entry in list(raw.items())[:_MAX_PADS]:
        name = str((entry or {}).get("name", "")).strip()[:_PAD_NAME_MAX] \
            if isinstance(entry, dict) else ""
        if _PAD_ID_RE.match(str(pad_id)) and name:
            pads[str(pad_id)] = {"name": name}
    board = {"model_id": model_id, "pads": pads, "updated": time.time()}
    path = _board_file(model_id)
    if not pads:
        path.unlink(missing_ok=True)
        return jsonify(board)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(board, fh, indent=2)
    tmp.replace(path)
    return jsonify(board)
