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
    POST   /api/bend/live/start           live on: {loop, chunk_sec, …params}
    POST   /api/bend/live/set             change params / patch / loop while live
    POST   /api/bend/live/advance         Next: move on to the following chunk
    POST   /api/bend/live/stop            live mode off
    GET    /api/bend/live/status
    GET    /api/bend/live/next?after=N    long-poll the next rendered chunk (WAV)
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

# app.py hands us its (lazily built) AudioGenerator so live mode renders on the
# same engine and behind the same lock as /api/generate.
generator_provider = None
_live = None


def _live_session():
    global _live
    if _live is None:
        from app.core.bending.live import LiveBend
        if generator_provider is None:
            raise RuntimeError("live bending has no generator")
        _live = LiveBend(generator_provider())
    return _live

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
# client), so what is stored is what the user found: the names they gave
# contacts, wires and chords (several contacts heard together, keyed
# `c.<hash>` with their `pads`) after listening, one board per model, plus
# `does`, the client's fingerprint of what the circuit did when it was
# named, so a changed draw can be noticed. Plain JSON in bends/boards/,
# readable and portable outside the app.

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_PAD_ID_RE = re.compile(r"^[a-z0-9._-]{1,32}$")
_PAD_NAME_MAX = 60
_MAX_PADS = 512
_FIND_DOES_MAX = 2048
_CHORD_MAX_PADS = 64


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
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()[:_PAD_NAME_MAX]
        if not (_PAD_ID_RE.match(str(pad_id)) and name):
            continue
        find = {"name": name}
        chord = entry.get("pads")
        if isinstance(chord, list):
            chord = [str(p) for p in chord if isinstance(p, str) and _PAD_ID_RE.match(p)]
            if chord:
                find["pads"] = chord[:_CHORD_MAX_PADS]
        does = entry.get("does")
        if isinstance(does, str) and does.strip():
            find["does"] = does.strip()[:_FIND_DOES_MAX]
        pads[str(pad_id)] = find
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


# --- live mode ---------------------------------------------------------------

_LIVE_FIELDS = ("loop", "model_id", "prompt", "steps", "seed",
                "chunk_sec", "context_sec", "past_sec")


def _live_params(data):
    """Validate the live request body; the bend patch goes through the same
    normalisation as /api/generate's. Returns (params, error)."""
    params = {}
    for k in _LIVE_FIELDS:
        if k in data and data[k] is not None:
            params[k] = data[k]
    if "loop" in params:
        params["loop"] = bool(params["loop"])
    for k in ("chunk_sec", "context_sec", "past_sec"):
        if k in params:
            try:
                params[k] = max(0.5, min(60.0, float(params[k])))
            except (TypeError, ValueError):
                return None, f"{k} must be a number"
    if "steps" in params:
        try:
            params["steps"] = int(params["steps"])
        except (TypeError, ValueError):
            return None, "steps must be an integer"
    if "seed" in params:
        try:
            params["seed"] = int(params["seed"])
        except (TypeError, ValueError):
            return None, "seed must be an integer"
    if "bend_patch" in data:
        raw = data["bend_patch"]
        if raw:
            try:
                patch, _warnings = validate_patch(raw)
            except PatchError as e:
                return None, f"bend_patch invalid: {e}"
            params["bend_patch"] = patch
        else:
            params["bend_patch"] = None
    if "loras" in data:
        loras = []
        for item in data["loras"] or []:
            if not isinstance(item, dict) or not item.get("path"):
                continue
            path = Path(str(item["path"]))
            if not path.is_absolute():
                path = get_config().project_root / path
            if path.exists():
                loras.append({"path": str(path),
                              "strength": max(-2.0, min(2.0, float(item.get("strength", 1.0))))})
        params["loras"] = loras
    return params, None


@bend_bp.route("/api/bend/live/start", methods=["POST"])
def bend_live_start():
    data = request.get_json(silent=True) or {}
    params, err = _live_params(data)
    if err:
        return jsonify({"error": err}), 400
    if not params.get("model_id"):
        return jsonify({"error": "model_id is required"}), 400
    try:
        return jsonify(_live_session().start(params))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@bend_bp.route("/api/bend/live/set", methods=["POST"])
def bend_live_set():
    data = request.get_json(silent=True) or {}
    params, err = _live_params(data)
    if err:
        return jsonify({"error": err}), 400
    return jsonify(_live_session().update(params))


@bend_bp.route("/api/bend/live/advance", methods=["POST"])
def bend_live_advance():
    if _live is None:
        return jsonify({"error": "live mode is off"}), 409
    return jsonify(_live.advance())


@bend_bp.route("/api/bend/live/stop", methods=["POST"])
def bend_live_stop():
    if _live is None:
        return jsonify({"running": False})
    return jsonify(_live.stop())


@bend_bp.route("/api/bend/live/status", methods=["GET"])
def bend_live_status():
    if _live is None:
        return jsonify({"running": False, "seq": 0})
    return jsonify(_live.status())


@bend_bp.route("/api/bend/live/next", methods=["GET"])
def bend_live_next():
    """Long-poll: the first chunk newer than `after`, as WAV. 204 on timeout,
    409 when live mode is off, 500 with the message when a render failed."""
    if _live is None:
        return jsonify({"error": "live mode is off"}), 409
    try:
        after = int(request.args.get("after", 0))
        timeout = max(1.0, min(60.0, float(request.args.get("timeout", 25))))
    except ValueError:
        return jsonify({"error": "bad query"}), 400
    result = _live.next_chunk(after, timeout)
    st = _live.status()
    if result is None:
        if st.get("error"):
            return jsonify({"error": st["error"]}), 500
        if not st.get("running"):
            return jsonify({"error": "live mode is off"}), 409
        return ("", 204)
    seq, wav, meta = result
    from flask import Response
    resp = Response(wav, mimetype="audio/wav")
    resp.headers["X-Live-Seq"] = str(seq)
    resp.headers["X-Live-Chunk"] = str(meta.get("chunk"))
    resp.headers["X-Live-Loop"] = "1" if meta.get("loop") else "0"
    resp.headers["X-Live-Version"] = str(meta.get("version"))
    resp.headers["X-Live-Render-Ms"] = str(meta.get("render_ms"))
    resp.headers["X-Live-Seconds"] = str(meta.get("seconds"))
    if meta.get("warnings"):
        resp.headers["X-Live-Warnings"] = json.dumps(meta["warnings"][:8], ensure_ascii=True)
    resp.headers["Cache-Control"] = "no-store"
    return resp
