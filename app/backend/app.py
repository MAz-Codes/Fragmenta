from utils.validators import Validator
from utils.exceptions import ModelNotFoundError, ValidationError, GenerationError
from utils.api_responses import APIResponse, handle_api_error
from utils.logger import setup_logging, get_logger
from app.core.generation.audio_generator import AudioGenerator
from app.core.training.sa3_trainer import start_training as start_training_func, get_training_status, stop_training, preview_training_plan
from app.core.config import get_config
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import re
from pathlib import Path
import sys
import threading
import time
import json
import logging
from typing import Any, Dict, Optional
from werkzeug.serving import WSGIRequestHandler

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))


LOG_LEVEL = os.environ.get('FRAGMENTA_LOG_LEVEL', 'INFO')
if not any(handler.name == 'fragmenta' for handler in logging.root.handlers):
    setup_logging(log_level=LOG_LEVEL, log_file=True)

logger = get_logger("BackendAPI")

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class QuietWSGIRequestHandler(WSGIRequestHandler):
    def log_request(self, code='-', size='-'):
        if str(code).startswith('4') or str(code).startswith('5'):
            super().log_request(code, size)


app = Flask(__name__, static_folder='../frontend/build', static_url_path='')

app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 

CORS(app,
     resources={r"/api/*": {"origins": "*"}},
     supports_credentials=True,
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'File(s) too large. Maximum upload size is 1 GB total.',
        'max_size_mb': 1024,
        'details': 'Try uploading fewer files at once or compress your audio files.'
    }), 413

DEBUG_MODE = os.environ.get('FRAGMENTA_DEBUG', 'false').lower() == 'true'

config = None
generator = None
model_manager = None
_components_initialised = False
_init_error = None


_LEGACY_FINETUNED_CONFIG_PATH = "models/config/model_config_small.json"


def _resolve_finetuned_metadata(model_dir: Path, config) -> Dict[str, Optional[str]]:
    """Pick the architecture config + base-model identity for a fine-tuned model.

    New runs drop a per-run model_config.json and a training_metadata.json
    breadcrumb (with `base_model`) into the model folder. Older runs from
    before that change have neither, so fall back to the small base config
    and infer the base model from the config filename.
    """
    config_path: Optional[str] = None
    per_run_config = model_dir / "model_config.json"
    if per_run_config.exists():
        try:
            config_path = str(per_run_config.relative_to(config.project_root))
        except ValueError:
            config_path = str(per_run_config)

    base_model: Optional[str] = None
    metadata_path = model_dir / "training_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            base_model = metadata.get("base_model")
            if config_path is None:
                base_config = metadata.get("base_config_path")
                if base_config and (config.project_root / base_config).exists():
                    config_path = base_config
        except (OSError, json.JSONDecodeError):
            pass

    if config_path is None:
        config_path = _LEGACY_FINETUNED_CONFIG_PATH

    if base_model is None:
        # Best-effort fallback for legacy runs: infer from the config filename
        # (the small base config has "_small" in its name).
        base_model = (
            "stable-audio-open-small"
            if "small" in config_path.lower()
            else "stable-audio-open-1.0"
        )

    return {"config_path": config_path, "base_model": base_model}


def _resolve_finetuned_config_path(model_dir: Path, config) -> str:
    """Back-compat shim — call sites that only need the config path."""
    return _resolve_finetuned_metadata(model_dir, config)["config_path"]


def _ensure_components():
    global config, generator, model_manager
    global _components_initialised, _init_error

    if _components_initialised:
        return
    if _init_error:
        raise RuntimeError(f"Backend failed to initialise earlier: {_init_error}")

    try:
        logger.info("Initializing Backend API components (lazy)…")
        config = get_config()
        generator = AudioGenerator(config)

        from app.core.model_manager import ModelManager
        model_manager = ModelManager(config)

        _components_initialised = True
        logger.info("Backend components initialized successfully")

    except Exception as e:
        _init_error = str(e)
        logger.error(f"Failed to initialize backend components: {e}")
        raise


@app.before_request
def lazy_init():
    if request.path == '/api/health':
        return
    try:
        _ensure_components()
    except Exception as e:
        if request.path.startswith('/api/'):
            return jsonify({'error': f'Backend not ready: {e}'}), 503
        return None


@app.route('/api/health')
def health_check():
    import torch
    status = {
        'status': 'ok' if _components_initialised else 'degraded',
        'components_ready': _components_initialised,
        'init_error': _init_error,
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    # Return 200 even in degraded mode so Docker HEALTHCHECK doesn't kill
    # the container before components finish loading.
    return jsonify(status), 200


@app.route('/')
def serve_react_app():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static_files(path):
    response = send_from_directory(app.static_folder, path)

    if path.endswith('.js') or path.endswith('.css'):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'

    return response




_SA3_LORA_BASES = ['sa3-small-music-base', 'sa3-small-sfx-base', 'sa3-medium-base']


@app.route('/api/start-training', methods=['POST'])
def start_training():
    try:
        training_config = request.json
        if training_config is None:
            raise ValueError("No training configuration provided")

        # Required fields.
        required_fields = ['modelName', 'baseModel', 'projectName']
        missing = [f for f in required_fields if not training_config.get(f)]
        if missing:
            return jsonify({'error': f"Missing required fields: {missing}"}), 400

        # Project must exist on disk (Dataset Workbench created it).
        from app.backend.data.projects import project_path
        proj_dir = project_path(training_config['projectName'])
        if not proj_dir.exists():
            return jsonify({
                'error': f"Project not found: {training_config['projectName']}. "
                         "Create or load it in the Dataset tab first.",
            }), 400

        # SA3 base validation. LoRA training requires a CFG-aware *-base
        # checkpoint; the post-trained / distilled checkpoints have had
        # the gradient signal LoRAs target collapsed away.
        base_model = training_config.get('baseModel')
        if base_model not in _SA3_LORA_BASES:
            return jsonify({
                'error': (
                    f"baseModel '{base_model}' is not a valid LoRA target. "
                    f"Pick one of: {_SA3_LORA_BASES}. SA2 models are gone in 0.2.0; "
                    f"post-trained SA3 checkpoints (no -base suffix) can't be used "
                    f"as a training base."
                )
            }), 400

        # Same-name collision check. If a previous run for this modelName
        # already wrote checkpoints, refuse unless the caller passes
        # overwrite=true. Stops a re-train from quietly co-mingling with
        # stale artifacts from the previous run.
        if not training_config.get('overwrite'):
            from app.core.training.sa3_trainer import SA3Trainer
            existing = SA3Trainer.existing_run_info(training_config['modelName'])
            if existing:
                return jsonify({
                    'error': 'run_exists',
                    'code': 'run_exists',
                    'message': (
                        f"A run named “{existing['run_name']}” already exists "
                        f"with {existing['checkpoint_count']} checkpoint(s). "
                        "Confirm overwrite to replace it."
                    ),
                    **existing,
                }), 409

        # SA3-aligned defaults. Phase 5 ships LoRA-only — no `mode` switch.
        training_config['mode'] = 'lora'
        training_config.setdefault('steps', 5000)
        training_config.setdefault('checkpointSteps', 500)
        training_config.setdefault('batchSize', 1)
        training_config.setdefault('learningRate', 1e-4)
        training_config.setdefault('duration', 30.0)
        training_config.setdefault('precision', 'bf16')
        training_config.setdefault('loraRank', 16)
        training_config.setdefault('loraAlpha', training_config['loraRank'])
        training_config.setdefault('loraDropout', 0.0)
        training_config.setdefault('adapterType', 'dora-rows')

        logger.info(
            f"Training request: base={base_model}, name={training_config['modelName']}, "
            f"rank={training_config['loraRank']}, adapter={training_config['adapterType']}, "
            f"steps={training_config['steps']}, batch={training_config['batchSize']}, "
            f"lr={training_config['learningRate']}"
        )

        result = start_training_func(training_config)

        if "error" in result:
            print(f"TRAINING START FAILED: {result['error']}")
            return jsonify(result), 400

        print(f"TRAINING STARTED SUCCESSFULLY")
        print("="*80)
        return jsonify(result)

    except Exception as e:
        error_msg = f"API Error: {str(e)}"
        print(f"API EXCEPTION: {error_msg}")
        print("="*80)
        return jsonify({'error': error_msg}), 500


_SA3_MODEL_IDS = {
    "sa3-small-music", "sa3-small-sfx", "sa3-medium",
    "sa3-small-music-base", "sa3-small-sfx-base", "sa3-medium-base",
}


@app.route('/api/generate', methods=['POST'])
def generate_audio():
    """SA3 inference. Phase 3 + Phase 7 of SA3_INTEGRATION_PLAN.md.

    Schema:
        {
            "model_id":           "sa3-small-music",   # required, SA3 IDs only
            "prompt":             "techno kick",
            "duration":           5.0,
            "steps":              8,                    # optional; default by model
            "cfg_scale":          1.0,                  # optional; default by model
            "seed":               -1,
            "negative_prompt":    null,
            "batch_size":         1,
            "align_bars":         null,                 # bars-mode passthrough
            "align_bpm":          null,
            "chunked_decode":     null,
            "loras":              [{"path":"…","strength":1.0}, …],
            "init_audio_path":    null,                 # audio-to-audio source
            "init_noise_level":   1.0,
            "inpaint_audio_path": null,                 # inpainting source
            "inpaint_starts":     [4.0],                # list or single float
            "inpaint_ends":       [8.0],
            "loop_stitch":        null                  # "inpaint" | "crossfade" | null
        }

    Phase 7 seamless looping: when `loop_stitch` is set, `align_bars` and
    `align_bpm` are required (the seam-smoothing pass needs bar duration
    for mask sizing). `align_to_grid` is skipped when a stitch is applied
    so head-trim / end-snap don't break the seamless boundary.
    """
    if not request.json:
        return jsonify(APIResponse.error("No JSON data provided", status_code=400)), 400

    data = request.json
    try:
        prompt = Validator.string(
            data.get('prompt', ''), 'prompt', min_length=1, max_length=500)

        # Legacy callers send model_name; new callers send model_id. Honour either.
        model_id = (data.get('model_id') or data.get('model_name') or '').strip()
        if not model_id:
            return jsonify(APIResponse.error(
                "model_id is required (e.g. 'sa3-small-music').",
                status_code=400)), 400
        if model_id not in _SA3_MODEL_IDS:
            return jsonify(APIResponse.error(
                f"'{model_id}' is not a SA3 model. The SA2/SAO engine was "
                f"removed in 0.2.0 (see v0.1.x-legacy tag for legacy use). "
                f"Pick one of: {sorted(_SA3_MODEL_IDS)}.",
                status_code=400)), 400

        duration = Validator.number(
            data.get('duration', 10.0), 'duration', min_value=1, max_value=380)
        seed = Validator.number(
            data.get('seed', -1), 'seed',
            min_value=-1, max_value=2**32 - 1, integer_only=True)
        # `/api/generate` returns exactly one WAV (single-file response), and
        # the engine's _finalize writes one clip. Batching is done client-side
        # — the UI loops this endpoint with distinct seeds (see App.js
        # batchCount / PerformancePanel generateForChannel). So a server-side
        # batch_size>1 would silently drop all but the first member; reject it
        # with a clear message instead of failing quietly.
        batch_size = Validator.number(
            data.get('batch_size', 1), 'batch_size',
            min_value=1, max_value=1, integer_only=True)

        steps_raw = data.get('steps')
        # 250 matches the frontend slider max. Past ~80–100 the marginal
        # quality gain from more steps is negligible on SA3 base models —
        # the cap is a sanity boundary, not a recommendation.
        steps = Validator.number(
            steps_raw, 'steps', min_value=1, max_value=250, integer_only=True
        ) if steps_raw is not None else None

        cfg_raw = data.get('cfg_scale')
        cfg_scale = Validator.number(
            cfg_raw, 'cfg_scale', min_value=0.1, max_value=20.0
        ) if cfg_raw is not None else None

        negative_prompt_raw = data.get('negative_prompt')
        negative_prompt = Validator.string(
            negative_prompt_raw, 'negative_prompt', min_length=0, max_length=500
        ) if negative_prompt_raw else None

        align_bars_raw = data.get('align_bars')
        align_bpm_raw = data.get('align_bpm')
        align_bars = Validator.number(
            align_bars_raw, 'align_bars', min_value=1, max_value=64,
            integer_only=True) if align_bars_raw is not None else None
        align_bpm = Validator.number(
            align_bpm_raw, 'align_bpm', min_value=20, max_value=300
        ) if align_bpm_raw is not None else None
        do_align = align_bars is not None and align_bpm is not None

        chunked_decode = data.get('chunked_decode')  # tri-state: True / False / None

        # Phase 7: audio-to-audio + inpainting --------------------------
        def _resolve_src(p):
            if not p:
                return None
            ap = Path(str(p))
            if not ap.is_absolute():
                ap = config.project_root / ap
            if not ap.exists():
                raise FileNotFoundError(f"Source audio not found: {p}")
            return str(ap)

        try:
            init_audio_path = _resolve_src(data.get('init_audio_path'))
            inpaint_audio_path = _resolve_src(data.get('inpaint_audio_path'))
        except FileNotFoundError as e:
            return jsonify(APIResponse.error(str(e), status_code=400)), 400

        init_noise_level = Validator.number(
            data.get('init_noise_level', 1.0), 'init_noise_level',
            min_value=0.0, max_value=1.0,
        )

        def _normalize_seconds(raw):
            if raw is None:
                return None
            if isinstance(raw, (int, float)):
                return [float(raw)]
            if isinstance(raw, list):
                return [float(x) for x in raw]
            raise ValueError("must be a number or list of numbers")
        try:
            inpaint_starts = _normalize_seconds(data.get('inpaint_starts'))
            inpaint_ends = _normalize_seconds(data.get('inpaint_ends'))
        except (TypeError, ValueError) as e:
            return jsonify(APIResponse.error(
                f"inpaint_starts/inpaint_ends invalid: {e}", status_code=400)), 400
        if (inpaint_starts is None) != (inpaint_ends is None):
            return jsonify(APIResponse.error(
                "inpaint_starts and inpaint_ends must both be set or both omitted.",
                status_code=400)), 400
        if inpaint_starts and inpaint_ends and len(inpaint_starts) != len(inpaint_ends):
            return jsonify(APIResponse.error(
                "inpaint_starts and inpaint_ends must be the same length.",
                status_code=400)), 400

        # Phase 7: seamless looping. Bars/BPM are required so the seam-
        # smoothing pass knows how much audio to regenerate at the join.
        loop_stitch = data.get('loop_stitch')
        if loop_stitch is not None:
            if loop_stitch not in ("inpaint", "crossfade"):
                return jsonify(APIResponse.error(
                    f"loop_stitch must be 'inpaint', 'crossfade', or null; got {loop_stitch!r}.",
                    status_code=400)), 400
            if not do_align:
                return jsonify(APIResponse.error(
                    "loop_stitch requires align_bars and align_bpm "
                    "(seamless loops are tempo-aware).",
                    status_code=400)), 400
            if loop_stitch == "inpaint" and inpaint_audio_path:
                return jsonify(APIResponse.error(
                    "loop_stitch='inpaint' is incompatible with inpaint_audio_path "
                    "(the loop algorithm itself uses inpainting as a second pass).",
                    status_code=400)), 400

        # LoRA stack: list of { path, strength }. Validate each entry.
        loras_raw = data.get('loras') or []
        if not isinstance(loras_raw, list):
            return jsonify(APIResponse.error(
                "loras must be an array of {path, strength} entries.",
                status_code=400)), 400
        loras = []
        for i, item in enumerate(loras_raw):
            if not isinstance(item, dict) or 'path' not in item:
                return jsonify(APIResponse.error(
                    f"loras[{i}] missing 'path'.", status_code=400)), 400
            lora_path = str(item['path']).strip()
            if not lora_path:
                continue
            strength = float(item.get('strength', 1.0))
            strength = max(-2.0, min(2.0, strength))
            # Resolve relative paths against the project root.
            lora_abs = Path(lora_path)
            if not lora_abs.is_absolute():
                lora_abs = config.project_root / lora_abs
            if not lora_abs.exists():
                return jsonify(APIResponse.error(
                    f"LoRA not found: {lora_path}", status_code=400)), 400
            # Compatibility gate (Phase 4 contract #5): a LoRA's embedded
            # base_model must share a backbone with the active model. A
            # `*-base` LoRA also runs on its distilled sibling (same
            # architecture, differ only in CFG state), so compare with a
            # trailing `-base` stripped from both sides. Unknown/missing
            # base metadata is allowed through (legacy LoRAs) rather than
            # blocking generation on a metadata gap.
            try:
                from safetensors import safe_open
                with safe_open(str(lora_abs), framework="pt") as _f:
                    _lora_meta = _f.metadata() or {}
            except Exception:
                _lora_meta = {}
            _lora_base = _lora_meta.get('base_model') or _lora_meta.get('base_model_id')
            if _lora_base and str(_lora_base).startswith('sa3-'):
                _strip = lambda m: m[:-5] if m.endswith('-base') else m
                if _strip(str(_lora_base)) != _strip(model_id):
                    return jsonify(APIResponse.error(
                        f"LoRA base mismatch: '{Path(lora_path).name}' was trained "
                        f"against {_lora_base}, which is incompatible with {model_id}.",
                        status_code=400,
                        details={'error_code': 'lora_base_mismatch',
                                 'lora_id': lora_path,
                                 'expected': model_id,
                                 'actual': _lora_base})), 400
            loras.append({'path': str(lora_abs), 'strength': strength})

    except ValidationError as e:
        field = e.details.get('field', 'unknown') if e.details else 'unknown'
        logger.warning(f"/api/generate validation failed on '{field}': {e}")
        return jsonify(APIResponse.validation_error({field: [str(e)]})), 400

    logger.info(
        f"Audio generation request: model={model_id} duration={duration}s "
        f"prompt='{prompt[:50]}{'…' if len(prompt) > 50 else ''}'"
    )

    # Variable-length lets us ask SA3 for exactly the bars-mode target duration
    # up front — no time-stretch needed in the common path. Headroom gives the
    # post-processor room for head-trim + drift correction without running short.
    # Proportional (8% of target, clamped to [0.5s, 2.0s]) so fast tempos don't
    # waste 30% of generation time on a fixed 1.5s buffer and slow ones get
    # enough margin. Skipped when loop_stitch is set: the loop pass needs the
    # exact requested duration so the seam lands at the right sample.
    effective_duration = duration
    if do_align and not loop_stitch:
        headroom = max(0.5, min(2.0, duration * 0.08))
        effective_duration = duration + headroom
        logger.debug(
            f"Bars-mode alignment: bars={align_bars}, bpm={align_bpm}; "
            f"target={duration:.2f}s +{headroom:.2f}s headroom = "
            f"{effective_duration:.2f}s requested"
        )

    try:
        output_path = generator.generate_audio(
            prompt,
            model_id=model_id,
            duration=float(effective_duration),
            steps=int(steps) if steps is not None else None,
            cfg_scale=float(cfg_scale) if cfg_scale is not None else None,
            seed=int(seed),
            negative_prompt=negative_prompt,
            batch_size=int(batch_size),
            chunked_decode=chunked_decode,
            loop_mode=do_align,
            loras=loras,
            init_audio_path=init_audio_path,
            init_noise_level=float(init_noise_level),
            inpaint_audio_path=inpaint_audio_path,
            inpaint_starts=inpaint_starts,
            inpaint_ends=inpaint_ends,
            loop_stitch=loop_stitch,
            loop_bars=int(align_bars) if (loop_stitch and align_bars) else None,
            loop_bpm=float(align_bpm) if (loop_stitch and align_bpm) else None,
        )

        if not output_path.exists():
            raise GenerationError(prompt, model_id, "Generated audio file not found")

        # Skip align_to_grid when a stitch ran — head-trim / end-snap would
        # shift the seam off its sample-exact location and reintroduce a
        # click. The stitch pass already gave us a duration-exact loop.
        if do_align and not loop_stitch:
            try:
                from app.core.generation.audio_post_process import align_to_grid
                align_to_grid(
                    output_path,
                    target_bpm=float(align_bpm),
                    target_bars=int(align_bars),
                )
                logger.info(f"Aligned to grid: bars={align_bars}, bpm={align_bpm}")
            except Exception as exc:
                # Never fail the request because alignment failed — the user
                # would rather have the raw clip than an error toast.
                logger.warning(f"Grid alignment skipped after error: {exc}")
        elif loop_stitch:
            logger.info(
                f"loop_stitch={loop_stitch} applied (bars={align_bars}, "
                f"bpm={align_bpm}); align_to_grid skipped to preserve seam"
            )

        logger.info(
            f"Audio generation completed: {output_path.name} "
            f"({output_path.stat().st_size} bytes)"
        )

        # Sidecar metadata — lets the frontend restore the "Generated
        # Fragments" panel across page reloads. Failure to write is non-
        # fatal (the WAV is the only mandatory artifact).
        sidecar_path = output_path.with_suffix(output_path.suffix + ".json")
        try:
            edit_mode = None
            if init_audio_path:
                edit_mode = 'style'
            elif inpaint_audio_path:
                edit_mode = 'inpaint/extend'
            sidecar = {
                "filename": output_path.name,
                "created_at": time.time(),
                "prompt": prompt,
                "model_id": model_id,
                "duration": float(duration),
                "seed": int(seed),
                "negative_prompt": negative_prompt,
                "cfg_scale": float(cfg_scale) if cfg_scale is not None else None,
                "steps": int(steps) if steps is not None else None,
                "batch_size": int(batch_size),
                "align_bars": int(align_bars) if align_bars else None,
                "align_bpm": float(align_bpm) if align_bpm else None,
                "loop_stitch": loop_stitch,
                "loras": loras or [],
                "init_audio_path": init_audio_path,
                "init_noise_level": float(init_noise_level) if init_audio_path else None,
                "inpaint_audio_path": inpaint_audio_path,
                "inpaint_starts": list(inpaint_starts) if inpaint_starts else None,
                "inpaint_ends": list(inpaint_ends) if inpaint_ends else None,
                "edit_mode": edit_mode,
            }
            with open(sidecar_path, "w") as f:
                json.dump(sidecar, f, indent=2)
        except Exception as exc:
            logger.warning(f"Failed to write fragment sidecar at {sidecar_path}: {exc}")

        return send_file(
            str(output_path),
            mimetype='audio/wav',
            as_attachment=True,
            download_name='generated_audio.wav',
        )

    except (ModelNotFoundError, GenerationError, ValidationError) as e:
        logger.error(f"Generation error: {e}")
        return jsonify(APIResponse.error(str(e), status_code=400)), 400
    except Exception as e:
        from app.core.generation.audio_generator import GenerationStopped
        if isinstance(e, GenerationStopped):
            logger.info("Generation stopped by user request")
            return jsonify({'stopped': True, 'message': 'Generation stopped'}), 499
        logger.exception("Unexpected error during audio generation")
        return jsonify(APIResponse.error(f"Unexpected error: {e}", status_code=500)), 500


@app.route('/api/loras', methods=['GET'])
def list_loras():
    """Enumerate SA3 LoRAs under models/fine_tuned/<run>/checkpoints/.

    SA3 LoRAs are .safetensors files with config (rank, adapter_type,
    base_model, etc.) embedded in the safetensors metadata header.
    `train_lora.py` from vendor/stable-audio-3/scripts/ writes one per
    checkpoint step. We surface every step so the user can A/B-test
    checkpoints inside a single run.

    Lightning .ckpt files from prior runs get lazily converted to
    .safetensors on demand so the picker can see them.

    Query:
        base_model (optional) — filter to LoRAs compatible with this base
            (e.g. ?base_model=sa3-small-music or ?base_model=sa3-small-music-base).
            A small-music LoRA is compatible with both small-music and
            small-music-base (same backbone, different CFG distillation
            state); the matcher strips a trailing `-base` from both sides
            before comparing.
    """
    requested_base = (request.args.get('base_model') or '').strip()

    def _base_root(model_id: str) -> str:
        """Strip the `-base` suffix so LoRAs trained against `*-base` filter
        as compatible with their post-trained sibling (same architecture)."""
        if not model_id:
            return ''
        return model_id[:-5] if model_id.endswith('-base') else model_id

    requested_root = _base_root(requested_base)

    try:
        config = get_config()
        fine_tuned_dir = config.get_path("models_fine_tuned")
        # Grouping: one entry per LoRA *run* (modelName), with `all_checkpoints`
        # listing every snapshot oldest→latest. `path` defaults to the latest
        # checkpoint so picking the LoRA without changing the sub-picker uses
        # the final state of training.
        loras_by_name: Dict[str, Dict[str, Any]] = {}
        if fine_tuned_dir.exists():
            try:
                from safetensors import safe_open
            except ImportError:
                return jsonify({"loras": []})

            # Lazy migration: any run dir that still has Lightning .ckpt
            # files (from a training run that completed before the
            # auto-convert step landed) gets a one-time conversion here so
            # the picker can see it. Cheap: no-op if .safetensors already
            # exists for each .ckpt.
            from app.core.training.sa3_lora_runner import convert_run_checkpoints_to_safetensors

            for run_dir in sorted(fine_tuned_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                ckpt_dir = run_dir / "checkpoints"
                if ckpt_dir.is_dir() and any(ckpt_dir.glob("*.ckpt")):
                    # Read the run's base_model from its training_metadata.json
                    # so the converted .safetensors carries the right tag.
                    meta_path = run_dir / "training_metadata.json"
                    base_model = None
                    model_name = run_dir.name
                    if meta_path.exists():
                        try:
                            rm = json.loads(meta_path.read_text())
                            base_model = rm.get("base_model")
                            model_name = rm.get("model_name") or model_name
                        except Exception:
                            pass
                    if base_model:
                        try:
                            convert_run_checkpoints_to_safetensors(
                                run_dir, base_model=base_model, model_name=model_name,
                            )
                        except Exception as conv_err:
                            logger.warning("Could not auto-convert %s: %s", run_dir.name, conv_err)

                # SA3 checkpoints live under <run>/checkpoints/. Fall back
                # to the run dir itself for runs that don't follow that
                # convention.
                search_dirs = [run_dir / "checkpoints", run_dir]
                ckpt_files = []
                for d in search_dirs:
                    if d.is_dir():
                        ckpt_files = sorted(
                            d.glob("*.safetensors"),
                            key=lambda p: p.stat().st_mtime,
                        )
                        if ckpt_files:
                            break
                if not ckpt_files:
                    continue

                for ckpt in ckpt_files:
                    try:
                        with safe_open(str(ckpt), framework="pt") as f:
                            meta = f.metadata() or {}
                    except Exception:
                        meta = {}
                    # SA3 canonically nests rank/alpha/adapter_type inside a
                    # `lora_config` JSON metadata key (see save_lora_safetensors
                    # in vendor/.../models/lora/utils.py). Parse it so the
                    # picker can surface those values; fall back to top-level
                    # for forward-compat with any future shape.
                    lora_config = {}
                    if meta.get("lora_config"):
                        try:
                            lora_config = json.loads(meta["lora_config"])
                        except Exception:
                            lora_config = {}

                    # `train_lora.py` doesn't embed base_model itself — we add
                    # it during the .ckpt→.safetensors conversion step.
                    # Fall back to training_metadata.json for legacy runs.
                    base_model = meta.get("base_model") or meta.get("base_model_id")
                    if not base_model:
                        run_meta_path = run_dir / "training_metadata.json"
                        if run_meta_path.exists():
                            try:
                                rm = json.loads(run_meta_path.read_text())
                                base_model = rm.get("base_model")
                            except Exception:
                                pass
                    if not base_model or not str(base_model).startswith("sa3-"):
                        continue  # not a SA3 LoRA

                    # Filter by requested base if the caller specified one.
                    # Treat `sa3-small-music` and `sa3-small-music-base` as
                    # compatible: same backbone, only differ in CFG state.
                    if requested_root and _base_root(base_model) != requested_root:
                        continue

                    rel_path = str(ckpt.relative_to(config.project_root))
                    rank = _safe_int(lora_config.get("rank") or meta.get("rank"))
                    alpha = _safe_int(
                        lora_config.get("alpha")
                        or meta.get("lora_alpha")
                        or meta.get("alpha")
                    )
                    adapter_type = (
                        lora_config.get("adapter_type")
                        or meta.get("adapter_type")
                        or "lora"
                    )

                    entry = loras_by_name.get(run_dir.name)
                    if entry is None:
                        entry = {
                            "id": run_dir.name,
                            "name": run_dir.name,
                            "base_model": base_model,
                            "rank": rank,
                            "alpha": alpha,
                            "adapter_type": adapter_type,
                            "all_checkpoints": [],
                        }
                        loras_by_name[run_dir.name] = entry

                    entry["all_checkpoints"].append({
                        "path": rel_path,
                        "checkpoint": ckpt.stem,
                        "size_bytes": ckpt.stat().st_size,
                        "mtime": ckpt.stat().st_mtime,
                    })

        # Finalize each LoRA entry — sort checkpoints by training step
        # extracted from the filename (Lightning writes "epoch=X-step=Y.ckpt"
        # which converts to "epoch=X-step=Y.safetensors"). mtime is unreliable
        # because the lazy .ckpt→.safetensors converter can rewrite files in
        # alphabetical (not training-step) order.
        import re as _re
        _step_pat = _re.compile(r"step=(\d+)")

        def _step_of(checkpoint_stem: str) -> int:
            m = _step_pat.search(checkpoint_stem)
            return int(m.group(1)) if m else -1

        loras = []
        for entry in loras_by_name.values():
            ckpts = sorted(entry["all_checkpoints"], key=lambda c: _step_of(c["checkpoint"]))
            entry["all_checkpoints"] = [c["path"] for c in ckpts]
            latest = ckpts[-1]
            entry["path"] = latest["path"]
            entry["checkpoint"] = latest["checkpoint"]
            entry["size_bytes"] = latest["size_bytes"]
            entry["mtime"] = latest["mtime"]
            loras.append(entry)
        loras.sort(key=lambda e: e["mtime"], reverse=True)

        return jsonify({"loras": loras})
    except Exception as e:
        logger.exception("Failed to enumerate LoRAs")
        return jsonify(APIResponse.error(f"Failed to list LoRAs: {e}", status_code=500)), 500


def _safe_int(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


@app.route('/api/audio/upload', methods=['POST'])
def upload_source_audio():
    """Accept a user-uploaded audio file to use as init_audio / inpaint_audio.

    Stores under <output>/uploads/<timestamp>_<safe-name> so the file is
    inside the project tree. Returns {path, name} where `path` is relative
    to project_root so /api/generate can resolve it.
    """
    if 'file' not in request.files:
        return jsonify(APIResponse.error("No file provided.", status_code=400)), 400
    fileobj = request.files['file']
    if not fileobj.filename:
        return jsonify(APIResponse.error("Empty filename.", status_code=400)), 400

    name = Path(fileobj.filename).name
    # Strip path components + restrict to known extensions.
    ext = Path(name).suffix.lower()
    if ext not in {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}:
        return jsonify(APIResponse.error(
            f"Unsupported audio format '{ext}'. Use wav/mp3/flac/m4a/ogg/opus.",
            status_code=400)), 400
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(name).stem)[:60] or "upload"

    cfg = get_config()
    uploads_dir = cfg.get_path("output") / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = uploads_dir / f"{ts}_{safe}{ext}"
    fileobj.save(str(dest))

    rel = dest.relative_to(cfg.project_root)
    return jsonify({"path": str(rel), "name": dest.name, "size_bytes": dest.stat().st_size})


@app.route('/api/fragments', methods=['GET'])
def list_fragments():
    """List previously-generated audio fragments (latest first).

    Returns the union of:
      • Generations with a sidecar JSON (full metadata: prompt, seed, etc.)
      • Orphan WAVs in output/ (no sidecar — happens for clips made by
        older versions of Fragmenta; metadata is recovered from the
        filename + mtime).

    Query: ?limit=<int>  (default 100, capped at 500)
    """
    cfg = get_config()
    output_dir = cfg.get_path("output")
    try:
        limit = max(1, min(500, int(request.args.get('limit', 100))))
    except (TypeError, ValueError):
        limit = 100

    if not output_dir.exists():
        return jsonify({"fragments": []})

    fragments = []
    seen_wavs = set()

    # Sidecared generations
    for sidecar_path in output_dir.glob("*.wav.json"):
        try:
            with open(sidecar_path) as f:
                meta = json.load(f)
            wav_name = meta.get("filename") or sidecar_path.name[:-len(".json")]
            wav_path = output_dir / wav_name
            if not wav_path.exists():
                continue  # sidecar without its WAV — skip silently
            seen_wavs.add(wav_name)
            meta["filename"] = wav_name
            meta["size_bytes"] = wav_path.stat().st_size
            fragments.append(meta)
        except Exception as exc:
            logger.warning(f"Failed to read fragment sidecar {sidecar_path}: {exc}")

    # Orphan WAVs (no sidecar) — recover what we can.
    # Filename format from _finalize: <ts>_<model_id>_<slugified_prompt>.wav
    for wav_path in output_dir.glob("*.wav"):
        if wav_path.name in seen_wavs:
            continue
        try:
            stem = wav_path.stem
            parts = stem.split("_")
            # Try to parse "YYYYMMDD_HHMMSS" as the first two tokens.
            created_at = wav_path.stat().st_mtime
            model_id = ""
            prompt = stem
            if len(parts) >= 2:
                try:
                    created_at = time.mktime(
                        time.strptime(f"{parts[0]}_{parts[1]}", "%Y%m%d_%H%M%S")
                    )
                    rest = "_".join(parts[2:])
                    # rest = "<model_id>_<slugified_prompt>". model_id is
                    # contiguous hyphenated; we use the longest known prefix
                    # by matching against SA3 model ids in _MODEL_INFO.
                    rest_low = rest.lower()
                    for mid in sorted(_SA3_MODEL_IDS, key=len, reverse=True):
                        if rest_low.startswith(mid):
                            model_id = mid
                            prompt = rest[len(mid):].lstrip("_").replace("_", " ")
                            break
                    else:
                        prompt = rest.replace("_", " ")
                except ValueError:
                    pass
            fragments.append({
                "filename": wav_path.name,
                "created_at": created_at,
                "prompt": prompt or "(unknown)",
                "model_id": model_id,
                "duration": None,
                "seed": None,
                "cfg_scale": None,
                "steps": None,
                "size_bytes": wav_path.stat().st_size,
                "_orphan": True,
            })
        except Exception as exc:
            logger.warning(f"Failed to list orphan fragment {wav_path}: {exc}")

    fragments.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    return jsonify({"fragments": fragments[:limit]})


@app.route('/api/fragments/<path:filename>', methods=['GET'])
def serve_fragment(filename):
    """Serve a WAV from output/ by name. Path traversal is rejected."""
    if "/" in filename or "\\" in filename or ".." in filename:
        return jsonify(APIResponse.error("Invalid filename.", status_code=400)), 400
    if not filename.endswith(".wav"):
        return jsonify(APIResponse.error("Only .wav files are served.", status_code=400)), 400
    cfg = get_config()
    full = cfg.get_path("output") / filename
    if not full.exists() or not full.is_file():
        return jsonify(APIResponse.error("File not found.", status_code=404)), 404
    return send_file(str(full), mimetype="audio/wav")


@app.route('/api/fragments/<path:filename>', methods=['DELETE'])
def delete_fragment(filename):
    """Delete a single fragment (WAV + its sidecar JSON) from output/."""
    if "/" in filename or "\\" in filename or ".." in filename:
        return jsonify(APIResponse.error("Invalid filename.", status_code=400)), 400
    if not filename.endswith(".wav"):
        return jsonify(APIResponse.error("Only .wav files are deletable.", status_code=400)), 400
    cfg = get_config()
    output_dir = cfg.get_path("output")
    wav_path = output_dir / filename
    sidecar_path = output_dir / f"{filename}.json"
    if not wav_path.exists():
        return jsonify(APIResponse.error("File not found.", status_code=404)), 404
    removed = []
    try:
        wav_path.unlink()
        removed.append(wav_path.name)
        if sidecar_path.exists():
            sidecar_path.unlink()
            removed.append(sidecar_path.name)
    except Exception as exc:
        logger.error(f"Failed to delete fragment {filename}: {exc}")
        return jsonify(APIResponse.error(f"Delete failed: {exc}", status_code=500)), 500
    logger.info(f"Deleted fragment: {', '.join(removed)}")
    return jsonify({"deleted": removed})


@app.route('/api/fragments', methods=['DELETE'])
def clear_fragments():
    """Delete EVERY .wav + .wav.json directly under output/.

    Does NOT recurse — uploaded source clips live under output/uploads/
    and are intentionally left alone (they may still be referenced by
    in-flight Edit-mode work, and the user uploaded them deliberately).
    """
    cfg = get_config()
    output_dir = cfg.get_path("output")
    if not output_dir.exists():
        return jsonify({"deleted": 0})
    removed = 0
    errors = []
    for pattern in ("*.wav", "*.wav.json"):
        for p in output_dir.glob(pattern):
            if not p.is_file():
                continue
            try:
                p.unlink()
                removed += 1
            except Exception as exc:
                errors.append(f"{p.name}: {exc}")
    if errors:
        logger.warning(f"clear_fragments: removed {removed}, errors: {errors}")
    else:
        logger.info(f"clear_fragments: removed {removed} file(s)")
    return jsonify({"deleted": removed, "errors": errors})


@app.route('/api/lora-strength', methods=['POST'])
def update_lora_strength():
    """Live-update a loaded LoRA's strength without regenerating.

    Performance Mode uses this when the user drags a strength slider —
    the next generate() picks up the new value, but the model itself
    doesn't need to be reloaded. Returns 409 if no LoRAs are loaded yet
    or the index is out of range.
    """
    data = request.json or {}
    try:
        index = int(data.get('index', -1))
        strength = float(data.get('strength', 1.0))
    except (TypeError, ValueError):
        return jsonify(APIResponse.error("index and strength are required.", status_code=400)), 400

    try:
        if not getattr(generator, 'model', None):
            return jsonify(APIResponse.error("No model loaded.", status_code=409)), 409
        ok = generator.set_lora_strength(index, strength)
        if not ok:
            return jsonify(APIResponse.error(
                f"LoRA index {index} not loaded.", status_code=409)), 409
        return jsonify({'success': True, 'index': index, 'strength': strength})
    except Exception as e:
        logger.exception("set_lora_strength failed")
        return jsonify(APIResponse.error(str(e), status_code=500)), 500


@app.route('/api/training/suggest-hyperparams', methods=['GET'])
def training_suggest_hyperparams():
    """Heuristic hyperparameter suggester for the Training tab's Suggest button.

    Query:
        project_name (required) — Dataset Workbench project to analyse
        base_model  (optional) — picked SA3 base, e.g. sa3-medium-base. Used to
                                  pick a -XS adapter when VRAM is tight and to
                                  emit base-model-aware warnings.
    Returns: {ok, stats, config, rationale, warnings} — see hyperparam_suggester.suggest.
    """
    try:
        from app.backend.data.projects import project_path
        from app.core.training.hyperparam_suggester import suggest
        project_name = request.args.get('project_name', '').strip()
        base_model = request.args.get('base_model', '').strip() or None
        if not project_name:
            return jsonify({'ok': False, 'error': "project_name is required."}), 400
        proj_dir = project_path(project_name)
        if not proj_dir.exists():
            return jsonify({
                'ok': False,
                'error': f"Project not found: {project_name}",
            }), 404
        result = suggest(proj_dir, base_model=base_model)
        return jsonify(result)
    except Exception as exc:
        logger.exception("hyperparam suggestion failed")
        return jsonify({'ok': False, 'error': str(exc)}), 500


@app.route('/api/training/checkpoint-preview', methods=['POST'])
def training_checkpoint_preview():
    """Resolve checkpoint cadence + step counts for the current training config.

    Body: trainingConfig dict (epochs, batchSize, checkpointSteps, ...). Send
    checkpointSteps as null/0 to request the auto value.
    """
    payload = request.json or {}
    try:
        plan = preview_training_plan(payload)
    except Exception as exc:
        logger.warning("checkpoint preview failed: %s", exc)
        return jsonify({'valid': False, 'error': str(exc)}), 200
    return jsonify(plan)


@app.route('/api/training-status', methods=['GET'])
def get_training_status_route():
    _log_api_call('training_status')
    try:
        return jsonify(get_training_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stop-training', methods=['POST'])
def stop_training_route():
    try:
        result = stop_training()
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        config = get_config()
        models_dir = config.get_path("models_fine_tuned")
        if not models_dir.exists():
            return jsonify({'models': []})

        models = []
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                # Skip LoRA runs — they're surfaced separately via /api/loras.
                # The training_metadata.json mode field is the source of truth;
                # full FTs have mode="full" (or no metadata at all on legacy runs).
                meta_path = model_dir / "training_metadata.json"
                if meta_path.exists():
                    try:
                        if json.loads(meta_path.read_text()).get("mode") == "lora":
                            continue
                    except Exception:
                        pass

                checkpoint_files = list(model_dir.glob("*.ckpt"))
                config_files = list(model_dir.glob("*.json")) + \
                    list(model_dir.glob("*.yaml"))
                has_checkpoint = len(checkpoint_files) > 0
                has_config = len(config_files) > 0

                checkpoints = []
                for ckpt_file in checkpoint_files:
                    import re
                    name = ckpt_file.stem
                    epoch_match = re.search(r'epoch=(\d+)', name)
                    step_match = re.search(r'step=(\d+)', name)

                    checkpoint_info = {
                        'name': name,
                        'path': str(ckpt_file.relative_to(config.project_root)),
                        'size_mb': round(ckpt_file.stat().st_size / (1024 * 1024), 1),
                        'created': ckpt_file.stat().st_mtime
                    }

                    if epoch_match:
                        checkpoint_info['epoch'] = int(epoch_match.group(1))
                    if step_match:
                        checkpoint_info['step'] = int(step_match.group(1))

                    checkpoints.append(checkpoint_info)

                checkpoints.sort(key=lambda x: x['created'], reverse=True)

                latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat(
                ).st_mtime) if checkpoint_files else None
                latest_config = max(
                    config_files, key=lambda x: x.stat().st_mtime) if config_files else None

                # Resolve the architecture config + base-model identity for
                # this fine-tuned model. Order: per-run copy in the model
                # folder, then training_metadata breadcrumb, then legacy
                # fallback to the small base config.
                resolved = _resolve_finetuned_metadata(model_dir, config)

                models.append({
                    'name': model_dir.name,
                    'path': str(model_dir.relative_to(config.project_root)),
                    'has_checkpoint': has_checkpoint,
                    'has_config': has_config,
                    'ckpt_path': str(latest_checkpoint.relative_to(config.project_root)) if latest_checkpoint else None,
                    'config_path': resolved['config_path'],
                    'base_model': resolved['base_model'],
                    'checkpoints': checkpoints,
                    'created': model_dir.stat().st_mtime if model_dir.exists() else None
                })

        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Checkpoint Manager — SA3 catalog endpoints (Phase 2a of SA3_INTEGRATION_PLAN)
# ============================================================================

@app.route('/api/checkpoints', methods=['GET'])
def list_checkpoints():
    try:
        # ?include=all → also returns base + standalone AE entries (training
        # subprocess uses this; the manager UI relies on the default).
        include_hidden = request.args.get('include') == 'all'
        return jsonify({
            'checkpoints': model_manager.get_catalog(include_hidden=include_hidden),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/checkpoints/storage', methods=['GET'])
def checkpoints_storage():
    try:
        return jsonify(model_manager.get_storage_info())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/checkpoints/<model_id>', methods=['GET'])
def get_checkpoint(model_id):
    try:
        info = model_manager.get_model_info(model_id)
        if not info:
            return jsonify({'error': 'Unknown checkpoint'}), 404
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/checkpoints/<model_id>/download', methods=['POST'])
def start_checkpoint_download(model_id):
    try:
        result = model_manager.start_download(model_id)
        # _DownloadJob.to_dict() always includes the "error" key (None when
        # ok); use a truthy check so a successful job doesn't 400.
        if result.get('error'):
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/checkpoints/<model_id>/cancel-download', methods=['POST'])
def cancel_checkpoint_download(model_id):
    try:
        # Cancel every in-flight job for this checkpoint (usually one).
        jobs = [j for j in model_manager.list_jobs()
                if j['model_id'] == model_id and j['status'] in ('queued', 'running')]
        cancelled = [j['job_id'] for j in jobs if model_manager.cancel_job(j['job_id'])]
        return jsonify({'cancelled': cancelled})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/checkpoints/<model_id>', methods=['DELETE'])
def delete_checkpoint_download(model_id):
    try:
        if model_manager.delete_model(model_id):
            return jsonify({'success': True})
        return jsonify({'error': 'Nothing to delete'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/checkpoints/jobs/<job_id>', methods=['GET'])
def get_checkpoint_job(job_id):
    try:
        job = model_manager.get_job(job_id)
        if not job:
            return jsonify({'error': 'Unknown job'}), 404
        return jsonify(job)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- HuggingFace auth -------------------------------------------------------

@app.route('/api/hf-auth/status', methods=['GET'])
def hf_auth_status():
    try:
        return jsonify(model_manager.hf_auth_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/hf-auth', methods=['POST'])
def hf_auth_login():
    try:
        data = request.json or {}
        token = (data.get('token') or '').strip()
        if not token:
            return jsonify({'error': 'Token is required'}), 400
        import huggingface_hub
        try:
            huggingface_hub.login(token=token, add_to_git_credential=False)
            info = huggingface_hub.whoami(token=token)
            return jsonify({
                'success': True,
                'username': info.get('name') or info.get('fullname'),
            })
        except Exception as e:
            return jsonify({'error': f'Invalid token: {e}'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/hf-auth', methods=['DELETE'])
def hf_auth_logout():
    try:
        import huggingface_hub
        huggingface_hub.logout()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/link/state', methods=['GET'])
def link_state():
    from app.core.audio.link_sync import get_link_bridge
    return jsonify(get_link_bridge().get_state())


@app.route('/api/link/enable', methods=['POST'])
def link_enable():
    from app.core.audio.link_sync import get_link_bridge
    bridge = get_link_bridge()
    ok = bridge.enable()
    state = bridge.get_state()
    if not ok:
        # 503 so the frontend can distinguish "not installed" from "normal reply"
        return jsonify({**state, 'error': 'Ableton Link binding not installed. Run: pip install LinkPython-extern'}), 503
    return jsonify(state)


@app.route('/api/link/disable', methods=['POST'])
def link_disable():
    from app.core.audio.link_sync import get_link_bridge
    bridge = get_link_bridge()
    bridge.disable()
    return jsonify(bridge.get_state())


@app.route('/api/link/bpm', methods=['POST'])
def link_set_bpm():
    from app.core.audio.link_sync import get_link_bridge
    data = request.get_json(silent=True) or {}
    try:
        bpm = float(data.get('bpm', 120))
    except (TypeError, ValueError):
        return jsonify({'error': 'bpm must be numeric'}), 400
    if bpm < 20 or bpm > 300:
        return jsonify({'error': 'bpm out of range [20, 300]'}), 400
    bridge = get_link_bridge()
    bridge.set_bpm(bpm)
    return jsonify(bridge.get_state())


@app.route('/api/link/install', methods=['POST'])
def link_install():
    """Install LinkPython-extern via pip, on-demand and with user consent.

    Restricted to localhost — running pip as a side effect of an HTTP call is
    only reasonable when the caller is the local user. In frozen PyInstaller
    builds there's no pip, so we report that clearly instead of failing weirdly.
    """
    # Localhost-only: prevents a remote attacker on the same Flask host (e.g. a
    # Docker image bound to 0.0.0.0) from installing arbitrary packages.
    remote = request.remote_addr or ''
    if remote not in ('127.0.0.1', '::1', 'localhost'):
        return jsonify({'error': 'install endpoint is only accessible from localhost'}), 403

    if getattr(sys, 'frozen', False):
        return jsonify({
            'success': False,
            'error': 'Automatic install is not available in the packaged desktop build. '
                     'Run from source (pip install LinkPython-extern) or use a Docker image '
                     'that includes it.',
        }), 400

    try:
        import subprocess
        import importlib
        logger.info("Installing LinkPython-extern via pip...")
        proc = subprocess.run(
            [sys.executable, '-m', 'pip', 'install',
             '--disable-pip-version-check', '--quiet',
             'LinkPython-extern'],
            capture_output=True, text=True, timeout=240,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or '').strip().splitlines()[-10:]
            return jsonify({
                'success': False,
                'error': 'pip install failed',
                'detail': '\n'.join(tail),
            }), 500

        # Force get_link_bridge() to re-probe imports next call.
        importlib.invalidate_caches()
        from app.core.audio import link_sync
        link_sync._bridge = None

        bridge = link_sync.get_link_bridge()
        if not bridge.available:
            return jsonify({
                'success': False,
                'error': 'Package installed but module did not import. Restart the backend.',
            }), 500

        return jsonify({'success': True, 'available': True})
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'pip install timed out (>4 min)'}), 500
    except Exception as e:
        logger.error(f"Link install failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/models/fine-tuned/<model_name>', methods=['DELETE'])
def delete_fine_tuned_model(model_name):
    try:
        import shutil
        config = get_config()
        models_dir = config.get_path("models_fine_tuned").resolve()
        target = (models_dir / model_name).resolve()
        # Path-traversal guard: target must be a direct child of models_fine_tuned.
        if target.parent != models_dir:
            return jsonify({'error': 'Invalid model name'}), 400
        if not target.exists() or not target.is_dir():
            return jsonify({'error': f'Fine-tuned model not found: {model_name}'}), 404
        shutil.rmtree(target)
        logger.info(f"Deleted fine-tuned model: {model_name}")
        return jsonify({'success': True, 'message': f'Deleted {model_name}'})
    except Exception as e:
        logger.error(f"Failed to delete fine-tuned model {model_name}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generation-progress', methods=['GET'])
def get_generation_progress_route():
    """Live progress for the in-flight `/api/generate` call.

    Returns the same dict as audio_generator.get_generation_progress():
        is_generating, phase ("idle"|"loading"|"sampling"|"decoding"|
        "complete"|"failed"), step, total_steps, progress (0-100),
        batch_index, batch_total, started_at, ended_at, error.

    Cheap (just a dict copy under a lock); safe to poll at ~200ms.
    """
    from app.core.generation.audio_generator import get_generation_progress
    return jsonify(get_generation_progress())


@app.route('/api/stop-generation', methods=['POST'])
def stop_generation_route():
    if generator is None:
        return jsonify({'stopped': False, 'message': 'Generator not initialised'}), 200
    newly_set = generator.request_stop()
    return jsonify({
        'stopped': True,
        'newly_set': newly_set,
        'message': 'Stop signal sent to generator'
    })


@app.route('/api/free-gpu-memory', methods=['POST'])
def free_gpu_memory():
    try:
        import subprocess
        import torch
        import os
        import time
        import gc

        print(" FREEING GPU MEMORY...")

        # The dominant VRAM consumer in this process is the loaded generator
        # model. cuda.empty_cache() only releases *unused* cached blocks, so
        # we must drop the live references first.
        global generator
        if generator is not None and getattr(generator, 'model', None) is not None:
            print("    Unloading in-process generator model")
            try:
                generator.model = None
            except Exception as exc:
                print(f"     Could not drop generator model: {exc}")

        try:
            from app.backend.data.auto_annotator import unload_clap
            unload_clap()
            print("    Unloaded CLAP tagger (if loaded)")
        except Exception as exc:
            print(f"     Could not unload CLAP: {exc}")

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            print("    Cleared PyTorch CUDA cache")

        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            print("    Cleared MPS cache")

        current_pid = os.getpid()
        print(f"     Current process PID: {current_pid}")

        try:
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory,process_name', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=10)

            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 3:
                            pid, mem_mb, process_name = parts[0], parts[1], parts[2]
                            try:
                                pid_int = int(pid)
                                mem_gb = float(mem_mb) / 1024

                                if pid_int == current_pid:
                                    print(
                                        f"     Skipping current process PID: {pid_int}")
                                    continue

                                if 'python' in process_name.lower() and mem_gb > 1.0:
                                    print(
                                        f"    Found Python process PID: {pid_int} using {mem_gb:.1f}GB")
                                    print(f"      Process: {process_name}")

                                    try:
                                        subprocess.run(
                                            ['kill', '-TERM', str(pid_int)], check=False, timeout=5)
                                        print(
                                            f"    Sent SIGTERM to PID: {pid_int}")

                                        time.sleep(2)

                                        try:
                                            os.kill(pid_int, 0)
                                            print(
                                                f"     Process {pid_int} still running, sending SIGKILL")
                                            subprocess.run(
                                                ['kill', '-KILL', str(pid_int)], check=False, timeout=5)
                                        except OSError:
                                            print(
                                                f"    Process {pid_int} stopped gracefully")

                                    except subprocess.TimeoutExpired:
                                        print(
                                            f"     Timeout stopping process {pid_int}")
                                    except Exception as e:
                                        print(
                                            f"     Could not stop process {pid_int}: {e}")

                            except (ValueError, IndexError) as e:
                                print(
                                    f"     Could not parse process info: {e}")
            else:
                print("     No CUDA processes found")

        except subprocess.TimeoutExpired:
            print("     Timeout getting CUDA processes")
        except Exception as e:
            print(f"     Could not check CUDA processes: {e}")

        time.sleep(3)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("    Cleared PyTorch CUDA cache again")

        memory_info = {}
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(
                0).total_memory / (1024**3)
            torch.cuda.synchronize()
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
            cached_memory = torch.cuda.memory_reserved(0) / (1024**3)
            free_memory = total_memory - allocated_memory

            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                                        capture_output=True, text=True, timeout=5)
                if result.stdout.strip():
                    used_mb, total_mb = result.stdout.strip().split(', ')
                    nvidia_used_gb = float(used_mb) / 1024
                    nvidia_total_gb = float(total_mb) / 1024
                    nvidia_free_gb = nvidia_total_gb - nvidia_used_gb
                else:
                    nvidia_used_gb = 0
                    nvidia_total_gb = total_memory
                    nvidia_free_gb = total_memory
            except Exception as e:
                print(f"Could not get nvidia-smi info: {e}")
                nvidia_used_gb = 0
                nvidia_total_gb = total_memory
                nvidia_free_gb = total_memory

            # PyTorch sometimes reports 0 for externally-allocated memory; fall back to nvidia-smi.
            if allocated_memory > 0:
                final_allocated = allocated_memory
                final_free = free_memory
            else:
                final_allocated = nvidia_used_gb
                final_free = nvidia_free_gb

            memory_info['cuda'] = {
                'total': nvidia_total_gb,
                'allocated': final_allocated,
                'cached': cached_memory,
                'free': final_free,
                'pytorch_allocated': allocated_memory,
                'pytorch_cached': cached_memory,
                'nvidia_used': nvidia_used_gb
            }

        print(f"    GPU Memory after clearing:")
        if 'cuda' in memory_info:
            print(f"      - Total: {memory_info['cuda']['total']:.2f} GB")
            print(
                f"      - Allocated: {memory_info['cuda']['allocated']:.2f} GB")
            print(f"      - Cached: {memory_info['cuda']['cached']:.2f} GB")
            print(f"      - Free: {memory_info['cuda']['free']:.2f} GB")

        return jsonify({
            'success': True,
            'message': 'GPU memory freed and training processes stopped',
            'memory_info': memory_info
        })

    except Exception as e:
        print(f" ERROR FREEING GPU MEMORY: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/toggle-debug', methods=['POST'])
def toggle_debug():
    global DEBUG_MODE
    try:
        data = request.json
        if data and 'debug' in data:
            DEBUG_MODE = bool(data['debug'])
            print(f" Debug mode {'enabled' if DEBUG_MODE else 'disabled'}")
            return jsonify({
                'success': True,
                'debug_mode': DEBUG_MODE,
                'message': f"Debug mode {'enabled' if DEBUG_MODE else 'disabled'}"
            })
        else:
            return jsonify({'error': 'Missing debug parameter'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/debug-status', methods=['GET'])
def get_debug_status():
    return jsonify({
        'debug_mode': DEBUG_MODE,
        'message': f"Debug mode is {'enabled' if DEBUG_MODE else 'disabled'}"
    })


_api_call_stats = {
    'gpu_memory_status': 0,
    'status': 0,
    'training_status': 0,
    'last_reset': time.time()
}


def _log_api_call(endpoint):
    global _api_call_stats
    _api_call_stats[endpoint] = _api_call_stats.get(endpoint, 0) + 1

    if time.time() - _api_call_stats['last_reset'] > 3600:
        _api_call_stats = {endpoint: 1, 'last_reset': time.time()}


@app.route('/api/debug-stats', methods=['GET'])
def get_debug_stats():
    return jsonify({
        'api_call_stats': _api_call_stats,
        'uptime_hours': (time.time() - _api_call_stats['last_reset']) / 3600,
        'calls_per_minute': {
            'gpu_memory_status': _api_call_stats.get('gpu_memory_status', 0) / max(1, (time.time() - _api_call_stats['last_reset']) / 60),
            'status': _api_call_stats.get('status', 0) / max(1, (time.time() - _api_call_stats['last_reset']) / 60),
            'training_status': _api_call_stats.get('training_status', 0) / max(1, (time.time() - _api_call_stats['last_reset']) / 60)
        }
    })


_gpu_memory_cache = {}
_gpu_memory_cache_time = 0
_gpu_memory_cache_duration = 2.0

_last_memory_warning_time = 0
_memory_warning_interval = 30


@app.route('/api/open-output-folder', methods=['POST'])
def open_output_folder():
    try:
        import subprocess
        import platform
        
        output_path = Path("output")
        output_path.mkdir(exist_ok=True)
        
        system = platform.system()
        if system == "Windows":
            subprocess.run(["explorer", str(output_path.absolute())])
        elif system == "Darwin":  # macOS
            subprocess.run(["open", str(output_path.absolute())])
        else:  # Linux
            subprocess.run(["xdg-open", str(output_path.absolute())])
            
        return jsonify({"success": True, "message": "Output folder opened"})
    except Exception as e:
        logger.error(f"Error opening output folder: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/open-documentation', methods=['POST'])
def open_documentation():
    try:
        import webbrowser

        payload = request.get_json(silent=True) or {}
        doc_key = payload.get('doc_key', 'about')

        docs_map = {
            'about': 'https://www.misaghazimi.com/fragmenta',
            'documentation': 'https://github.com/MAz-Codes/Fragmenta',
        }

        target_url = docs_map.get(doc_key)
        if not target_url:
            return jsonify({
                "success": False,
                "error": f"Unsupported documentation target: {doc_key}"
            }), 400

        webbrowser.open(target_url)

        return jsonify({
            "success": True,
            "message": f"Opened {doc_key}",
            "doc_key": doc_key,
            "url": target_url,
        })
    except Exception as e:
        logger.error(f"Error opening documentation: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

_welcome_page_closed = False

@app.route('/api/welcome-page-closed', methods=['POST'])
def welcome_page_closed():
    global _welcome_page_closed
    try:
        _welcome_page_closed = True
        logger.info("Welcome page closed signal received")
        return jsonify({"success": True, "message": "Welcome page closure recorded"})
    except Exception as e:
        logger.error(f"Error recording welcome page closure: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/welcome-page-status', methods=['GET'])
def get_welcome_page_status():
    global _welcome_page_closed
    return jsonify({"closed": _welcome_page_closed})

@app.route('/api/license-info', methods=['GET'])
def get_license_info():
    try:
        project_root = Path(__file__).parent.parent.parent

        license_file = project_root / "LICENSE"
        license_text = ""
        if license_file.exists():
            with open(license_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:50]
                license_text = ''.join(lines)

        notice_file = project_root / "NOTICE.md"
        notice_text = ""
        if notice_file.exists():
            with open(notice_file, 'r', encoding='utf-8') as f:
                notice_text = f.read()
        
        return jsonify({
            "license": "GNU Affero General Public License v3.0",
            "copyright": "Copyright 2025-2026 Misagh Azimi",
            "license_text": license_text,
            "notice_text": notice_text,
            "license_url": "https://www.gnu.org/licenses/agpl-3.0"
        })
    except Exception as e:
        logger.error(f"Error reading license info: {e}")
        return jsonify({
            "license": "GNU Affero General Public License v3.0",
            "copyright": "Copyright 2025-2026 Misagh Azimi",
            "error": str(e)
        }), 500

@app.route('/api/gpu-memory-status', methods=['GET'])
def get_gpu_memory_status():
    _log_api_call('gpu_memory_status')
    global _gpu_memory_cache, _gpu_memory_cache_time

    current_time = time.time()
    if current_time - _gpu_memory_cache_time < _gpu_memory_cache_duration:
        return jsonify({'memory_info': _gpu_memory_cache})

    try:
        import torch
        import subprocess
        import psutil

        memory_info = {}
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(
                0).total_memory / (1024**3)

            torch.cuda.synchronize()
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
            cached_memory = torch.cuda.memory_reserved(0) / (1024**3)
            free_memory = total_memory - allocated_memory

            nvidia_used_gb = 0
            nvidia_total_gb = total_memory
            nvidia_free_gb = total_memory

            # PyTorch reports 0 when memory is held by other processes; ask nvidia-smi instead.
            if allocated_memory == 0:
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                                            capture_output=True, text=True, timeout=5)
                    if result.stdout.strip():
                        used_mb, total_mb = result.stdout.strip().split(', ')
                        nvidia_used_gb = float(used_mb) / 1024
                        nvidia_total_gb = float(total_mb) / 1024
                        nvidia_free_gb = nvidia_total_gb - nvidia_used_gb
                except Exception as e:
                    if "Could not get nvidia-smi info" not in str(e):
                        print(f"GPU Memory Error: {e}")

            cuda_capability = torch.cuda.get_device_capability(0)
            device_name = torch.cuda.get_device_name(0)

            if allocated_memory > 0:
                final_allocated = allocated_memory
                final_cached = cached_memory
                final_free = free_memory
                memory_source = "PyTorch"
            else:
                final_allocated = nvidia_used_gb
                final_cached = cached_memory
                final_free = nvidia_free_gb
                memory_source = "nvidia-smi"

            memory_info['cuda'] = {
                'total': nvidia_total_gb,
                'allocated': final_allocated,
                'cached': final_cached,
                'free': final_free,
                'device': device_name,
                'cuda_capability': cuda_capability,
                'memory_source': memory_source,
                'pytorch_allocated': allocated_memory,
                'pytorch_cached': cached_memory,
                'nvidia_used': nvidia_used_gb
            }

            global _last_memory_warning_time
            if (current_time - _last_memory_warning_time) > _memory_warning_interval:
                if final_allocated > 10.0:
                    print(
                        f"  High GPU Memory Usage: {final_allocated:.2f}GB allocated, {final_free:.2f}GB free")
                    _last_memory_warning_time = current_time
                elif final_free < 1.0:
                    print(
                        f"  Low GPU Memory: {final_free:.2f}GB free, {final_allocated:.2f}GB allocated")
                    _last_memory_warning_time = current_time
        else:
            memory_info['cpu'] = {
                'total': psutil.virtual_memory().total / (1024**3),
                'available': psutil.virtual_memory().available / (1024**3),
                'used': psutil.virtual_memory().used / (1024**3),
                'device': 'CPU',
                'type': 'cpu'
            }

        _gpu_memory_cache = memory_info
        _gpu_memory_cache_time = current_time

        return jsonify({'memory_info': memory_info})
    except Exception as e:
        print(f"Error getting GPU memory status: {e}")
        return jsonify({'error': str(e)}), 500



def _annotator_labels_default_path():
    return Path(get_config().project_root) / 'config' / 'annotator_labels.json'


def _annotator_labels_user_path():
    return Path(get_config().project_root) / 'config' / 'annotator_labels.user.json'


def _annotator_labels_path():
    user_path = _annotator_labels_user_path()
    return user_path if user_path.exists() else _annotator_labels_default_path()


_LABEL_CATEGORIES = ('genre', 'mood', 'instruments')


def _read_labels_from(path):
    if not path.exists():
        return {cat: [] for cat in _LABEL_CATEGORIES}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {cat: list(data.get(cat) or []) for cat in _LABEL_CATEGORIES}


@app.route('/api/annotator-labels', methods=['GET'])
def get_annotator_labels():
    user_path = _annotator_labels_user_path()
    overridden = user_path.exists()
    effective = _read_labels_from(user_path if overridden else _annotator_labels_default_path())
    defaults = _read_labels_from(_annotator_labels_default_path())
    return jsonify({'labels': effective, 'defaults': defaults, 'overridden': overridden})


@app.route('/api/annotator-labels', methods=['PUT'])
def put_annotator_labels():
    payload = request.json or {}
    cleaned = {}
    for cat in _LABEL_CATEGORIES:
        raw = payload.get(cat)
        if raw is None:
            cleaned[cat] = []
            continue
        if not isinstance(raw, list):
            return jsonify({'error': f'{cat} must be a list of strings'}), 400
        seen = set()
        out = []
        for item in raw:
            if not isinstance(item, str):
                return jsonify({'error': f'{cat} entries must be strings'}), 400
            label = item.strip()
            key = label.lower()
            if not label or key in seen:
                continue
            seen.add(key)
            out.append(label)
        cleaned[cat] = out
    user_path = _annotator_labels_user_path()
    user_path.parent.mkdir(parents=True, exist_ok=True)
    with open(user_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, indent=4)
    return jsonify({'labels': cleaned, 'overridden': True})


@app.route('/api/annotator-labels', methods=['DELETE'])
def delete_annotator_labels():
    user_path = _annotator_labels_user_path()
    if user_path.exists():
        user_path.unlink()
    defaults = _read_labels_from(_annotator_labels_default_path())
    return jsonify({'labels': defaults, 'overridden': False})


def _clap_ckpt_path():
    from app.backend.data.auto_annotator import clap_checkpoint_path
    return clap_checkpoint_path(get_config().get_path('models_pretrained'))


@app.route('/api/environment', methods=['GET'])
def environment():
    return jsonify({
        'docker': os.environ.get('FRAGMENTA_DOCKER', '0') == '1',
    })


@app.route('/api/upload-folder', methods=['POST'])
def upload_folder():
    # Browser-native folder upload path for containerised deployments
    # (e.g. HF Space) where no display server is available for a native dialog.
    audio_exts = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}

    files = request.files.getlist('files')
    rel_paths = request.form.getlist('rel_paths')

    if not files:
        return jsonify({'error': 'No files uploaded.'}), 400
    if len(rel_paths) != len(files):
        return jsonify({'error': 'rel_paths count does not match files count.'}), 400

    first_rel = (rel_paths[0] or '').replace('\\', '/').lstrip('/')
    folder_name = first_rel.split('/', 1)[0] if '/' in first_rel else 'folder'
    safe_folder = ''.join(c for c in folder_name if c.isalnum() or c in '-_') or 'folder'

    staging_root = get_config().get_path('data') / 'uploads'
    staging_root.mkdir(parents=True, exist_ok=True)
    target_dir = staging_root / f"{int(time.time())}-{safe_folder}"
    target_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for file_obj, rel in zip(files, rel_paths):
        rel_norm = (rel or file_obj.filename or '').replace('\\', '/').lstrip('/')
        if not rel_norm or '..' in rel_norm.split('/'):
            continue
        if Path(rel_norm).suffix.lower() not in audio_exts:
            continue

        dest = (target_dir / rel_norm).resolve()
        try:
            dest.relative_to(target_dir.resolve())
        except ValueError:
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        file_obj.save(dest)
        saved += 1

    if saved == 0:
        import shutil
        shutil.rmtree(target_dir, ignore_errors=True)
        return jsonify({'error': 'No audio files found in the selected folder.'}), 400

    return jsonify({'path': str(target_dir), 'file_count': saved})


@app.route('/api/pick-folder', methods=['POST'])
def pick_folder():
    import subprocess
    import shutil as _shutil

    payload = request.json or {}
    start_dir = payload.get('start_dir') or str(Path.home())

    def _try(cmd):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if out.returncode == 0:
                path = out.stdout.strip()
                if path:
                    return path
        except Exception as exc:
            logger.debug("folder dialog attempt failed (%s): %s", cmd[0], exc)
        return None

    chosen = None

    if sys.platform.startswith('linux'):
        if _shutil.which('zenity'):
            chosen = _try(['zenity', '--file-selection', '--directory',
                           f'--filename={start_dir}/', '--title=Choose audio folder'])
        if not chosen and _shutil.which('kdialog'):
            chosen = _try(['kdialog', '--getexistingdirectory', start_dir])
        if not chosen:
            # fall back to system python3's tkinter (the venv may not have tk)
            script = (
                "import tkinter as tk; from tkinter import filedialog; "
                "r = tk.Tk(); r.withdraw(); "
                f"p = filedialog.askdirectory(initialdir={start_dir!r}, title='Choose audio folder'); "
                "print(p or '')"
            )
            chosen = _try(['python3', '-c', script])
    elif sys.platform == 'darwin':
        chosen = _try(['osascript', '-e',
                       f'POSIX path of (choose folder with prompt "Choose audio folder" default location POSIX file "{start_dir}")'])
    elif sys.platform == 'win32':
        safe_start = (start_dir or '').replace("'", "''")
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
            f"$d.SelectedPath = '{safe_start}'; "
            "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.SelectedPath }"
        )
        chosen = _try(['powershell', '-NoProfile', '-Command', ps])

    if not chosen:
        return jsonify({'path': None, 'cancelled': True})
    return jsonify({'path': chosen})



# --- SA3 sidecar-native dataset prep -----------------------------------------
# Projects are folders under <user_data_dir>/projects/<name>/. Editing happens
# against an in-memory session per loaded project; persistence is explicit via
# Save (writes .draft.json) and Commit (writes .txt sidecars + marks audio
# committed). See DATASET_PREP_REDESIGN.md.

_project_annotate_jobs: Dict[str, dict] = {}
_project_annotate_jobs_lock = threading.Lock()


def _get_project_annotate_job(project_name: str) -> dict:
    with _project_annotate_jobs_lock:
        job = _project_annotate_jobs.get(project_name)
        if job is None:
            job = {
                'state': 'idle',
                'current': 0,
                'total': 0,
                'current_file': '',
                'tier': None,
                'annotated': 0,
                'skipped_existing': 0,
                'errors': 0,
                'error': None,
                'started_at': None,
                'finished_at': None,
                'cancelled': False,
            }
            _project_annotate_jobs[project_name] = job
        return job


@app.route('/api/projects', methods=['GET'])
def list_projects_route():
    from app.backend.data.projects import list_projects
    try:
        return jsonify({'projects': list_projects()})
    except Exception as exc:
        logger.exception("Failed to list projects")
        return jsonify({'error': str(exc)}), 500


@app.route('/api/projects', methods=['POST'])
def create_project_route():
    from app.backend.data.projects import create_project, sanitize_project_name
    payload = request.json or {}
    raw_name = payload.get('name', '')
    try:
        name = sanitize_project_name(raw_name)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    try:
        project = create_project(name)
    except FileExistsError as exc:
        return jsonify({'error': str(exc)}), 409
    except Exception as exc:
        logger.exception("Failed to create project %s", name)
        return jsonify({'error': str(exc)}), 500
    return jsonify(project), 201


@app.route('/api/projects/<name>', methods=['GET'])
def get_project_route(name):
    from app.backend.data.projects import get_project
    try:
        return jsonify(get_project(name))
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to get project %s", name)
        return jsonify({'error': str(exc)}), 500


@app.route('/api/projects/<name>/template', methods=['PATCH'])
def patch_project_template_route(name):
    """Update the project's annotation-template preset.

    Body: { "preset": "music" | "instrument" | "sfx" }
    """
    from app.backend.data.projects import update_project_template_preset
    payload = request.json or {}
    if 'preset' not in payload:
        return jsonify({'error': 'preset is required'}), 400
    try:
        return jsonify(update_project_template_preset(name, payload['preset']))
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to update template preset for %s", name)
        return jsonify({'error': str(exc)}), 500


@app.route('/api/projects/<name>/health', methods=['GET'])
def project_health_route(name):
    """Per-clip health checks. Returns counts + file lists the UI can route
    into the existing selection model."""
    from app.backend.data.projects import compute_health
    try:
        short_th = float(request.args.get('short_threshold_sec', 1.0))
        loud_th = float(request.args.get('loudness_outlier_db', 6.0))
    except (TypeError, ValueError):
        short_th, loud_th = 1.0, 6.0
    try:
        return jsonify(compute_health(name, short_th, loud_th))
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        logger.exception("Health check failed for %s", name)
        return jsonify({'error': str(exc)}), 500


@app.route('/api/projects/<name>', methods=['DELETE'])
def delete_project_route(name):
    """Nuke the project folder + drop the in-memory session. Irreversible."""
    from app.backend.data.projects import delete_project
    try:
        delete_project(name)
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to delete project %s", name)
        return jsonify({'error': str(exc)}), 500
    return jsonify({'name': name, 'deleted': True})


@app.route('/api/projects/<name>/ingest', methods=['POST'])
def ingest_into_project_route(name):
    """Body: { folder_path: string, mode: "copy" | "symlink" }"""
    from app.backend.data.projects import ingest_folder, INGEST_MODES, get_project
    payload = request.json or {}
    folder = (payload.get('folder_path') or '').strip()
    mode = payload.get('mode', 'copy')
    if mode not in INGEST_MODES:
        return jsonify({'error': f'Invalid ingest mode: {mode}'}), 400
    if not folder:
        return jsonify({'error': 'folder_path is required'}), 400
    folder_path = Path(folder).expanduser()
    try:
        result = ingest_folder(name, folder_path, mode)
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        logger.exception("Ingest failed for project %s", name)
        return jsonify({'error': str(exc)}), 500
    logger.info(
        "Ingest into project=%s mode=%s added=%d (copied=%d symlinked=%d skipped=%d)",
        name, mode, result['added'], result['copied'], result['symlinked'], result['skipped'],
    )
    return jsonify({**result, 'project': get_project(name)})


@app.route('/api/projects/<name>/clip/<path:file_name>', methods=['PATCH'])
def patch_clip_route(name, file_name):
    """In-memory prompt edit. Persists only on Save or Commit."""
    from app.backend.data.projects import update_clip_prompt
    payload = request.json or {}
    if 'prompt' not in payload:
        return jsonify({'error': 'prompt is required'}), 400
    try:
        clip = update_clip_prompt(name, file_name, payload['prompt'])
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to update clip %s in project %s", file_name, name)
        return jsonify({'error': str(exc)}), 500
    return jsonify(clip)


@app.route('/api/projects/<name>/clip/<path:file_name>', methods=['DELETE'])
def delete_clip_route(name, file_name):
    """Immediate delete — cannot be discarded back."""
    from app.backend.data.projects import delete_clip, get_project
    try:
        delete_clip(name, file_name)
    except Exception as exc:
        logger.exception("Failed to delete clip %s in project %s", file_name, name)
        return jsonify({'error': str(exc)}), 500
    return jsonify({'name': name, 'file_name': file_name, 'deleted': True, 'project': get_project(name)})


@app.route('/api/projects/<name>/clip/<path:file_name>/audio', methods=['GET'])
def clip_audio_route(name, file_name):
    """Stream raw audio bytes for a clip. Range requests work via send_file."""
    from app.backend.data.projects import get_session_handle
    try:
        session = get_session_handle(name)
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    with session.lock:
        clip = session.clips.get(file_name)
        if clip is None:
            return jsonify({'error': f"Clip not found: {file_name}"}), 404
        audio_path = Path(clip.path)
    if not audio_path.exists():
        return jsonify({'error': 'Audio file missing on disk'}), 404
    return send_file(str(audio_path), conditional=True)


@app.route('/api/projects/<name>/clip/<path:file_name>/slice', methods=['POST'])
def clip_slice_route(name, file_name):
    """Split one clip into N children. Body: { target_duration, overlap_sec, strategy }."""
    from app.backend.data.projects import slice_clip
    from app.backend.data.slicing import VALID_STRATEGIES
    payload = request.json or {}
    try:
        target = float(payload.get('target_duration', 0))
        overlap = float(payload.get('overlap_sec', 0))
        strategy = str(payload.get('strategy', 'hard')).lower()
    except (TypeError, ValueError):
        return jsonify({'error': 'target_duration / overlap_sec must be numeric'}), 400
    if target <= 0 or target > 600:
        return jsonify({'error': 'target_duration must be in (0, 600]'}), 400
    if overlap < 0 or overlap >= target:
        return jsonify({'error': 'overlap_sec must be >= 0 and < target_duration'}), 400
    if strategy not in VALID_STRATEGIES:
        return jsonify({'error': f"strategy must be one of {VALID_STRATEGIES}"}), 400
    try:
        result = slice_clip(name, file_name, target, overlap, strategy)
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        logger.exception("Slice failed for %s/%s", name, file_name)
        return jsonify({'error': str(exc)}), 500
    return jsonify(result)


@app.route('/api/projects/<name>/clip/<path:file_name>/peaks', methods=['GET'])
def clip_peaks_route(name, file_name):
    """Return waveform peaks + duration JSON for a clip. Cached per session."""
    from app.backend.data.projects import get_session_handle, get_or_compute_peaks
    try:
        n = int(request.args.get('n', 200))
    except (TypeError, ValueError):
        n = 200
    n = max(20, min(n, 500))
    try:
        session = get_session_handle(name)
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    with session.lock:
        clip = session.clips.get(file_name)
        if clip is None:
            return jsonify({'error': f"Clip not found: {file_name}"}), 404
        audio_path = Path(clip.path)
    if not audio_path.exists():
        return jsonify({'error': 'Audio file missing on disk'}), 404
    try:
        peaks, duration = get_or_compute_peaks(session, file_name, audio_path, n)
    except Exception as exc:
        logger.exception("Peak computation failed for %s/%s", name, file_name)
        return jsonify({'error': f'Peak computation failed: {exc}'}), 500
    return jsonify({'peaks': peaks, 'duration': duration})


@app.route('/api/projects/<name>/save', methods=['POST'])
def save_project_route(name):
    """Persist in-memory diffs as a hidden draft (not the SA3 sidecars)."""
    from app.backend.data.projects import save_project
    try:
        return jsonify(save_project(name))
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to save project %s", name)
        return jsonify({'error': str(exc)}), 500


@app.route('/api/projects/<name>/commit', methods=['POST'])
def commit_project_route(name):
    """Flush in-memory state to .txt sidecars; overwrites the previous commit."""
    from app.backend.data.projects import commit_project
    try:
        return jsonify(commit_project(name))
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to commit project %s", name)
        return jsonify({'error': str(exc)}), 500


@app.route('/api/projects/<name>/discard', methods=['POST'])
def discard_project_route(name):
    """Drop uncommitted state and delete audio files added since the last commit."""
    from app.backend.data.projects import discard_project
    try:
        return jsonify(discard_project(name))
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to discard project %s", name)
        return jsonify({'error': str(exc)}), 500


@app.route('/api/projects/<name>/annotate', methods=['POST'])
def annotate_project_route(name):
    """Kick off auto-annotation. Updates the in-memory session; sidecars on
    disk are not touched until the user commits.

    Body: {
      tier: "basic" | "rich",
      scope?: "all" | ["file_name1", ...],          # default: "all"
      skip_existing?: bool                          # default: true
    }
    """
    from app.backend.data.projects import get_session_handle, reset_cancel, project_path
    from app.backend.data.auto_annotator import (
        annotate_file, load_label_sets, get_clap_tagger, clap_checkpoint_available,
    )

    payload = request.json or {}
    tier = payload.get('tier', 'basic')
    scope = payload.get('scope', 'all')
    skip_existing = bool(payload.get('skip_existing', True))
    if tier not in ('basic', 'rich'):
        return jsonify({'error': f'Invalid tier: {tier}'}), 400

    try:
        session = get_session_handle(name)
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404

    if tier == 'rich' and not clap_checkpoint_available(get_config().get_path('models_pretrained')):
        return jsonify({
            'error': 'CLAP checkpoint not downloaded yet. Open Model Management to download it.',
            'code': 'clap_not_available',
        }), 409

    with session.lock:
        all_clips = sorted(session.clips.values(), key=lambda c: c.file_name)
        if scope == 'all':
            target = list(all_clips)
        elif isinstance(scope, list):
            wanted = set(scope)
            target = [c for c in all_clips if c.file_name in wanted]
            missing = wanted - {c.file_name for c in target}
            if missing:
                return jsonify({'error': f'Clips not in project: {sorted(missing)}'}), 404
        else:
            return jsonify({'error': 'scope must be "all" or a list of file names'}), 400

        if skip_existing:
            run_targets = [c for c in target if not (c.prompt or '').strip()]
            skipped_existing = len(target) - len(run_targets)
        else:
            run_targets = list(target)
            skipped_existing = 0
        target_names = [c.file_name for c in run_targets]

    job = _get_project_annotate_job(name)
    with _project_annotate_jobs_lock:
        if job['state'] == 'running':
            return jsonify({'error': f'Annotation already running for project {name}.'}), 409
        job.update({
            'state': 'running',
            'current': 0,
            'total': len(target_names),
            'current_file': '',
            'tier': tier,
            'annotated': 0,
            'skipped_existing': skipped_existing,
            'errors': 0,
            'error': None,
            'started_at': time.time(),
            'finished_at': None,
            'cancelled': False,
        })

    reset_cancel(session)
    labels = load_label_sets(_annotator_labels_path())
    clap_tagger = None
    if tier == 'rich':
        clap_tagger = get_clap_tagger(_clap_ckpt_path())
        try:
            clap_tagger.ensure_loaded()
        except FileNotFoundError as exc:
            # File-existence check passed earlier but the actual load failed.
            with _project_annotate_jobs_lock:
                job['state'] = 'idle'
            return jsonify({
                'error': str(exc),
                'code': 'clap_not_available',
            }), 409
        except ImportError as exc:
            # The .pt weights are on disk but one of CLAP's Python deps isn't
            # installed in the venv. Could be laion_clap itself or anything it
            # imports transitively (e.g. torchvision). Model Manager can't fix
            # this — the user has to pip install in their environment.
            with _project_annotate_jobs_lock:
                job['state'] = 'idle'
            missing = getattr(exc, 'name', None) or 'laion_clap'
            # Module name → PyPI name when they differ; default identical.
            _PYPI_NAME = {'laion_clap': 'laion-clap'}
            pip_name = _PYPI_NAME.get(missing, missing)
            install_command = f'pip install {pip_name}'
            # torch-family packages need the CUDA index URL to match the
            # pinned torch build; otherwise pip installs the CPU-only wheel.
            if missing in {'torchvision', 'torchaudio'}:
                install_command += ' --extra-index-url https://download.pytorch.org/whl/cu128'
            return jsonify({
                'error': (
                    f"The '{missing}' Python package is required for Rich-tier annotation "
                    "but isn't installed. Install it in Fragmenta's venv, then restart the app:"
                ),
                'code': 'clap_package_missing',
                'install_command': install_command,
            }), 409
        except Exception as exc:
            with _project_annotate_jobs_lock:
                job['state'] = 'idle'
            logger.exception("CLAP load failed for project %s", name)
            return jsonify({
                'error': f'CLAP failed to load: {exc}',
                'code': 'clap_load_failed',
            }), 500

    proj_path = project_path(name)
    # Resolve the active preset to a template string once per job.
    from app.backend.data.projects import resolve_prompt_template
    active_template = resolve_prompt_template(session)

    def runner():
        try:
            logger.info(
                "Project annotate started: name=%s tier=%s targets=%d skip_existing=%s",
                name, tier, len(target_names), skip_existing,
            )
            for i, file_name in enumerate(target_names, start=1):
                if session.cancel_event.is_set():
                    logger.info("Project annotate cancelled mid-run: name=%s", name)
                    with _project_annotate_jobs_lock:
                        job['cancelled'] = True
                    break
                with _project_annotate_jobs_lock:
                    job['current_file'] = file_name
                logger.info("  annotating %d/%d: %s", i, len(target_names), file_name)
                audio_path = proj_path / file_name
                try:
                    result = annotate_file(
                        audio_path, tier, clap_tagger, labels,
                        prompt_template=active_template,
                    )
                except Exception as exc:
                    logger.warning("annotate_file failed for %s: %s", file_name, exc)
                    with _project_annotate_jobs_lock:
                        job['errors'] += 1
                        job['current'] += 1
                    continue
                if result.get('error'):
                    with _project_annotate_jobs_lock:
                        job['errors'] += 1
                        job['current'] += 1
                    continue
                prompt = result.get('prompt', '') or ''
                with session.lock:
                    clip = session.clips.get(file_name)
                    if clip is not None:
                        clip.prompt = prompt
                with _project_annotate_jobs_lock:
                    job['annotated'] += 1
                    job['current'] += 1
            with _project_annotate_jobs_lock:
                job['state'] = 'done'
                job['finished_at'] = time.time()
                job['current_file'] = ''
            logger.info(
                "Project annotate done: name=%s annotated=%d errors=%d skipped_existing=%d cancelled=%s",
                name, job['annotated'], job['errors'], job['skipped_existing'], job['cancelled'],
            )
        except Exception as exc:
            logger.exception("Project annotate failed: name=%s", name)
            with _project_annotate_jobs_lock:
                job['state'] = 'error'
                job['error'] = str(exc)
                job['finished_at'] = time.time()

    threading.Thread(target=runner, daemon=True).start()
    with _project_annotate_jobs_lock:
        snapshot = dict(job)
    return jsonify({'name': name, 'job': snapshot}), 202


@app.route('/api/projects/<name>/annotate/cancel', methods=['POST'])
def annotate_project_cancel_route(name):
    """Stop a running annotate job after the in-flight clip completes."""
    from app.backend.data.projects import get_session_handle
    try:
        session = get_session_handle(name)
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    session.cancel_event.set()
    return jsonify({'name': name, 'cancel_signal_set': True})


@app.route('/api/projects/<name>/annotate/status', methods=['GET'])
def annotate_project_status_route(name):
    from app.backend.data.projects import project_path
    if not project_path(name).exists():
        return jsonify({'error': f'Project not found: {name}'}), 404
    with _project_annotate_jobs_lock:
        job = _project_annotate_jobs.get(name)
        snapshot = dict(job) if job else {'state': 'idle'}
    return jsonify({'name': name, 'job': snapshot})


# --- Phase 6: pre-encoded latents -----------------------------------------

@app.route('/api/projects/<name>/pre-encode', methods=['POST'])
def pre_encode_project_route(name):
    """Kick off SA3 pre-encoding for a project. Returns the job state (202)."""
    from app.backend.data.projects import project_path
    from app.backend.data.pre_encoder import start_pre_encode
    if not project_path(name).exists():
        return jsonify({'error': f'Project not found: {name}'}), 404
    # silent=True so an empty/no-Content-Type body is treated as {} instead of
    # Flask returning 415 — callers usually fire-and-forget without a payload.
    body = request.get_json(silent=True) or {}
    autoencoder = (body.get('autoencoder') or '').strip() or None
    try:
        job = start_pre_encode(name, autoencoder=autoencoder)
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to start pre-encode")
        return jsonify({'error': str(exc)}), 500
    return jsonify({'name': name, 'job': job}), 202


@app.route('/api/projects/<name>/pre-encode/status', methods=['GET'])
def pre_encode_status_route(name):
    """Poll the current job state. Cheap (dict copy under lock)."""
    from app.backend.data.projects import project_path
    from app.backend.data.pre_encoder import get_pre_encode_job
    if not project_path(name).exists():
        return jsonify({'error': f'Project not found: {name}'}), 404
    return jsonify({'name': name, 'job': get_pre_encode_job(name)})


@app.route('/api/projects/<name>/pre-encode/cancel', methods=['POST'])
def pre_encode_cancel_route(name):
    """Cooperative cancel. Signals SIGINT → SIGTERM → SIGKILL on the subprocess."""
    from app.backend.data.projects import project_path
    from app.backend.data.pre_encoder import cancel_pre_encode
    if not project_path(name).exists():
        return jsonify({'error': f'Project not found: {name}'}), 404
    cancelled = cancel_pre_encode(name)
    return jsonify({'name': name, 'cancelled': cancelled})


@app.route('/api/projects/<name>/pre-encode/prompt', methods=['PATCH'])
def pre_encode_prompt_route(name):
    """Persist the 'Don't ask again' choice from the post-commit dialog.

    Body: { "suppress": bool }
    """
    from app.backend.data.projects import project_path, update_pre_encode_suppression
    if not project_path(name).exists():
        return jsonify({'error': f'Project not found: {name}'}), 404
    body = request.get_json(silent=True) or {}
    if 'suppress' not in body:
        return jsonify({'error': "Body must contain 'suppress': bool."}), 400
    updated = update_pre_encode_suppression(name, bool(body['suppress']))
    return jsonify(updated)


@app.route('/api/clap/unload', methods=['POST'])
def clap_unload_route():
    """Free CLAP weights from VRAM (e.g. before starting training or generation)."""
    from app.backend.data.auto_annotator import unload_clap
    unload_clap()
    return jsonify({'message': 'CLAP unloaded from memory.'})


@app.route('/shutdown', methods=['POST'])
def shutdown():
    try:
        print(" Shutting down Flask server...")
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()
        return jsonify({'message': 'Server shutting down'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', '5001'))
    app.run(debug=True, host=host, port=port)
