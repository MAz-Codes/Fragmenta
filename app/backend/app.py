from utils.validators import Validator
from utils.exceptions import ModelNotFoundError, ValidationError, GenerationError
from utils.api_responses import APIResponse, handle_api_error
from utils.logger import setup_logging, get_logger
from app.core.generation.audio_generator import AudioGenerator
from app.core.training.fine_tuner import start_training as start_training_func, get_training_status, stop_training
from app.backend.data.simple_audio_processor import SimpleAudioProcessor
from app.core.config import get_config
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path
import sys
import threading
import time
import json
import logging
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
audio_processor = None
generator = None
model_manager = None
_components_initialised = False
_init_error = None


def _ensure_components():
    global config, audio_processor, generator, model_manager
    global _components_initialised, _init_error

    if _components_initialised:
        return
    if _init_error:
        raise RuntimeError(f"Backend failed to initialise earlier: {_init_error}")

    try:
        logger.info("Initializing Backend API components (lazy)…")
        config = get_config()
        audio_processor = SimpleAudioProcessor(
            model_config_path=config.get_path("models_config") / "model_config.json"
        )
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


@app.route('/api/process-files', methods=['POST'])
def process_files():
    try:
        content_length = request.content_length
        if content_length:
            print(f"Upload request size: {content_length / (1024*1024):.2f} MB")
        else:
            print("Upload request size: Unknown")
        
        max_size = app.config.get('MAX_CONTENT_LENGTH', 1000 * 1024 * 1024)
        if content_length and content_length > max_size:
            return jsonify({
                'error': f'Upload too large: {content_length / (1024*1024):.2f} MB exceeds {max_size / (1024*1024):.0f} MB limit',
                'max_size_mb': max_size // (1024*1024)
            }), 413

        config = get_config()
        data_dir = config.get_path("data")
        data_dir.mkdir(exist_ok=True, parents=True)

        files_and_prompts = []
        for key in request.files:
            if key.startswith('file_'):
                index = key.split('_')[1]
                prompt_key = f'prompt_{index}'

                if prompt_key in request.form:
                    file_obj = request.files[key]
                    prompt = request.form[prompt_key]

                    if file_obj and prompt and prompt.strip():
                        files_and_prompts.append((file_obj, prompt.strip()))

        if not files_and_prompts:
            return jsonify({'error': 'No files uploaded or no prompts provided'}), 400

        target_length = int(request.form.get('target_length', 30))
        sample_rate = int(request.form.get('sample_rate', 44100))
        channels = int(request.form.get('channels', 2))

        saved_files = []
        prompts_data = []

        for file_obj, prompt in files_and_prompts:
            try:
                file_path = data_dir / file_obj.filename
                file_obj.save(file_path)
                saved_files.append(file_obj.filename)

                prompts_data.append((file_obj.filename, prompt))

            except Exception as e:
                print(f"Error saving {file_obj.filename}: {e}")
                continue

        chunks_preview_data = []
        for filename, prompt in prompts_data:
            chunks_preview_data.append([filename, filename, prompt, "original"])

        # Merge into existing metadata instead of overwriting, so repeated
        # uploads accumulate into one dataset.
        json_path = Path(config.get_metadata_json_path())
        existing_metadata = []

        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    import json
                    existing_metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load existing metadata: {e}")
                existing_metadata = []
        
        existing_files = {item['file_name']: item for item in existing_metadata}
        
        for filename, prompt in prompts_data:
            existing_files[filename] = {
                "file_name": filename,
                "prompt": prompt,
                "path": f"app/backend/data/{filename}"
            }
        
        # Convert back to list and save
        final_metadata = list(existing_files.values())
        
        with open(json_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(final_metadata, f, indent=2)

        try:
            config.update_dataset_config()
        except Exception as exc:
            print(f"Warning: failed to refresh dataset-config.json: {exc}")

        return jsonify({
            'message': f'Files saved successfully! {len(saved_files)} original files saved to data folder',
            'saved_files': saved_files,
            'processed_count': len(saved_files),
            'chunks_preview': chunks_preview_data,
            'data_folder': str(data_dir),
            'metadata_json': str(json_path),
            'approach': 'original_files_only'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/start-training', methods=['POST'])
def start_training():
    try:
        training_config = request.json
        if training_config is None:
            raise ValueError("No training configuration provided")

        print("\n" + "="*80)
        print("API TRAINING REQUEST RECEIVED")
        print("="*80)
        print(f"RECEIVED CONFIG FROM FRONTEND:")
        print(f"   - Model Name: {training_config.get('modelName', 'untitled')}")
        print(f"   - Base Model: {training_config.get('baseModel', 'NOT SET')}")
        print(f"   - Epochs: {training_config.get('epochs', 'NOT SET')}")
        print(f"   - Checkpoint Steps: {training_config.get('checkpointSteps', 'NOT SET')}")
        print(f"   - Batch Size: {training_config.get('batchSize', 'NOT SET')}")
        print(f"   - Learning Rate: {training_config.get('learningRate', 'NOT SET')}")
        print(f"   - Save Wrapped Checkpoint: {training_config.get('saveWrappedCheckpoint', False)}")

        required_fields = ['modelName', 'baseModel']
        missing_fields = [field for field in required_fields if field not in training_config]
        if missing_fields:
            error_msg = f"Missing required fields: {missing_fields}"
            print(f"API ERROR: {error_msg}")
            return jsonify({'error': error_msg}), 400

        valid_models = ['stable-audio-open-small', 'stable-audio-open-1.0']
        base_model = training_config.get('baseModel')
        if base_model not in valid_models:
            error_msg = f"Invalid base model '{base_model}'. Must be one of: {valid_models}"
            print(f"API ERROR: {error_msg}")
            return jsonify({'error': error_msg}), 400

        if 'epochs' not in training_config:
            training_config['epochs'] = 30
            print(f"   Setting default epochs: 30")
        if 'checkpointSteps' not in training_config:
            training_config['checkpointSteps'] = 50
            print(f"   Setting default checkpointSteps: 50")
        if 'batchSize' not in training_config:
            training_config['batchSize'] = 1
            print(f"   Setting default batch size: 1")
        if 'learningRate' not in training_config:
            training_config['learningRate'] = 1e-4
            print(f"   Setting default learning rate: 1e-4")
        if 'saveWrappedCheckpoint' not in training_config:
            training_config['saveWrappedCheckpoint'] = False
            print(f"   Setting default saveWrappedCheckpoint: False")
        if 'precision' not in training_config or not training_config['precision']:
            training_config['precision'] = 'auto'
            print(f"   Setting default precision: auto")

        print(f"\nVALIDATED CONFIG:")
        print(f"   - Model Name: {training_config['modelName']}")
        print(f"   - Base Model: {training_config['baseModel']}")
        print(f"   - Epochs: {training_config['epochs']}")
        print(f"   - Checkpoint Steps: {training_config['checkpointSteps']}")
        print(f"   - Batch Size: {training_config['batchSize']}")
        print(f"   - Learning Rate: {training_config['learningRate']}")
        print(f"   - Save Wrapped Checkpoint: {training_config['saveWrappedCheckpoint']}")
        print(f"   - Precision: {training_config['precision']}")

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


@app.route('/api/generate', methods=['POST'])
def generate_audio():
    if not request.json:
        return jsonify(APIResponse.error("No JSON data provided", status_code=400)), 400

    data = request.json
    try:
        prompt = Validator.string(
            data.get('prompt', ''), 'prompt', min_length=1, max_length=500)
        duration = Validator.number(
            data.get('duration', 10.0), 'duration', min_value=1, max_value=60)
        cfg_scale = Validator.number(
            data.get('cfg_scale', 7.0), 'cfg_scale', min_value=0.1, max_value=20.0)
        seed = Validator.number(
            data.get('seed', -1), 'seed', min_value=-1, max_value=2**32 - 1, integer_only=True)
        batch_index = Validator.number(
            data.get('batch_index', 1), 'batch_index', min_value=1, max_value=10, integer_only=True)
        batch_total = Validator.number(
            data.get('batch_total', 1), 'batch_total', min_value=1, max_value=10, integer_only=True)
        model_name = data.get('model_name', data.get('model', 'default'))
        model_path = data.get('model_path')
        unwrapped_model_path = data.get('unwrapped_model_path')

    except ValidationError as e:
        return jsonify(APIResponse.validation_error({e.details['field']: [str(e)]})), 400

    logger.info(f"Audio generation request received")
    logger.debug(f"Request details: prompt='{prompt[:50]}...', duration={duration}s, model={model_name}")
    if DEBUG_MODE:
        logger.debug(f"Model paths: model_path={model_path}, unwrapped_model_path={unwrapped_model_path}")

    def determine_model_config(model_name, model_path, unwrapped_model_path):
        config_file = None
        model_file_path = None

        # Priority: unwrapped_model_path > model_path > base model.
        if unwrapped_model_path:
            model_file_path = Path(unwrapped_model_path)
            if not model_file_path.exists():
                raise ModelNotFoundError(
                    f"unwrapped_model:{model_name}", str(model_file_path))
            logger.debug(f"Using unwrapped model: {model_file_path}")

        elif model_path:
            model_file_path = Path(model_path)
            if not model_file_path.exists():
                raise ModelNotFoundError(
                    f"model_path:{model_name}", str(model_file_path))
            logger.debug(f"Using model path: {model_file_path}")

        # Small and full models use different configs; pick by file size when the name is ambiguous.
        if model_file_path:
            file_size_gb = model_file_path.stat().st_size / (1024**3)
            config_file = "model_config_small.json" if file_size_gb < 2.0 else "model_config.json"
            logger.debug(
                f"Model file size: {file_size_gb:.2f} GB, using {'small' if file_size_gb < 2.0 else 'large'} config")

        elif model_name in ['stable-audio-open-small', 'stable-audio-open-1.0']:
            config_file = "model_config_small.json" if 'small' in model_name else "model_config.json"
            logger.debug(f"Using base model config for {model_name}")
        else:
            logger.warning(f"No config determined for model: {model_name}")
            config_file = "model_config_small.json"

        return config_file, model_file_path

    config_file, determined_model_path = determine_model_config(
        model_name, model_path, unwrapped_model_path)
    logger.info(f"Starting generation with config: {config_file}")
    try:
        if determined_model_path and determined_model_path.exists():
            output_path = generator.generate_audio(
                prompt,
                unwrapped_model_path=unwrapped_model_path if unwrapped_model_path else None,
                model_path=determined_model_path if not unwrapped_model_path else None,
                config_file=config_file,
                duration=duration,
                cfg_scale=cfg_scale,
                seed=seed,
                batch_index=batch_index,
                batch_total=batch_total
            )
        elif model_name in ['stable-audio-open-small', 'stable-audio-open-1.0']:
            model_file_mapping = {
                'stable-audio-open-small': 'stable-audio-open-small-model.safetensors',
                'stable-audio-open-1.0': 'stable-audio-open-model.safetensors'
            }
            model_file_name = model_file_mapping.get(
                model_name, f"{model_name}-model.safetensors")
            model_file_path = config.project_root / \
                "models" / "pretrained" / model_file_name

            if not model_file_path.exists():
                raise ModelNotFoundError(model_name, str(model_file_path))

            output_path = generator.generate_audio(
                prompt,
                model_path=model_file_path,
                config_file=config_file,
                duration=duration,
                cfg_scale=cfg_scale,
                seed=seed,
                batch_index=batch_index,
                batch_total=batch_total
            )
        elif model_name and model_name != 'default':
            fine_tuned_path = config.get_path("models_fine_tuned") / model_name
            if not fine_tuned_path.exists():
                raise ModelNotFoundError(model_name, str(fine_tuned_path))

            output_path = generator.generate_audio(
                prompt, fine_tuned_path, duration=duration,
                cfg_scale=cfg_scale, seed=seed,
                batch_index=batch_index, batch_total=batch_total)
        else:
            logger.debug("Using default model")
            output_path = generator.generate_audio(
                prompt, duration=duration, cfg_scale=cfg_scale, seed=seed,
                batch_index=batch_index, batch_total=batch_total)

        if not output_path.exists():
            raise GenerationError(prompt, model_name, "Generated audio file not found")

        logger.info(f"Audio generation completed: {output_path.name} ({output_path.stat().st_size} bytes)")
        return send_file(
            str(output_path),
            mimetype='audio/wav',
            as_attachment=True,
            download_name='generated_audio.wav'
        )

    except (ModelNotFoundError, GenerationError, ValidationError) as e:
        logger.error(f"Generation error: {str(e)}")
        return jsonify(APIResponse.error(str(e), status_code=400)), 400
    except Exception as e:
        from app.core.generation.audio_generator import GenerationStopped
        if isinstance(e, GenerationStopped):
            logger.info("Generation stopped by user request")
            return jsonify({'stopped': True, 'message': 'Generation stopped'}), 499
        logger.exception("Unexpected error during audio generation")
        return jsonify(APIResponse.error(f"Unexpected error: {str(e)}", status_code=500)), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    _log_api_call('status')
    try:
        config = get_config()
        data_dir = config.get_path("data")

        audio_files = list(data_dir.glob("*.wav")) + \
            list(data_dir.glob("*.mp3")) + list(data_dir.glob("*.flac"))
        config = get_config()
        metadata_json = Path(config.get_metadata_json_path())
        custom_metadata = Path(config.get_custom_metadata_path())
        total_duration = 0.0
        try:
            import torchaudio
        except ImportError:
            torchaudio = None
        try:
            import soundfile as sf
        except ImportError:
            sf = None

        for audio_file in audio_files:
            try:
                if torchaudio is not None:
                    info = torchaudio.info(str(audio_file))
                    total_duration += info.num_frames / info.sample_rate
                elif sf is not None:
                    f = sf.SoundFile(str(audio_file))
                    total_duration += len(f) / f.samplerate
            except Exception as e:
                print(f"Error reading {audio_file}: {e}")
                continue

        status_response = {
            'status': 'running',
            'raw_files': len(audio_files),
            'processed_segments': len(audio_files),
            'raw_file_names': [f.name for f in audio_files[:10]],
            'total_duration': total_duration,
            'has_metadata_json': metadata_json.exists(),
            'has_custom_metadata': custom_metadata.exists(),
            'trained_models': len(list(config.get_path("models_fine_tuned").glob("*"))) if config.get_path("models_fine_tuned").exists() else 0,
            'training': get_training_status()
        }

        return jsonify(status_response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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

                unwrapped_dir = model_dir / "unwrapped"
                unwrapped_models = []
                if unwrapped_dir.exists():
                    for unwrapped_file in unwrapped_dir.glob("*.safetensors"):
                        unwrapped_models.append({
                            'name': unwrapped_file.stem,
                            'path': str(unwrapped_file.relative_to(config.project_root)),
                            'size_mb': round(unwrapped_file.stat().st_size / (1024 * 1024), 1),
                            'created': unwrapped_file.stat().st_mtime
                        })

                    unwrapped_models.sort(
                        key=lambda x: x['created'], reverse=True)

                # Fine-tuned models reuse the base model's config for unwrapping.
                base_config_path = "models/config/model_config_small.json"

                models.append({
                    'name': model_dir.name,
                    'path': str(model_dir.relative_to(config.project_root)),
                    'has_checkpoint': has_checkpoint,
                    'has_config': has_config,
                    'ckpt_path': str(latest_checkpoint.relative_to(config.project_root)) if latest_checkpoint else None,
                    'config_path': base_config_path,
                    'checkpoints': checkpoints,
                    'unwrapped_models': unwrapped_models,
                    'created': model_dir.stat().st_mtime if model_dir.exists() else None
                })

        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/available', methods=['GET'])
def get_available_models():
    try:
        models = model_manager.get_available_models()
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/<model_id>/info', methods=['GET'])
def get_model_info(model_id):
    try:
        model_info = model_manager.get_model_info(model_id)
        if not model_info:
            return jsonify({'error': 'Model not found'}), 404
        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/<model_id>/accept-terms', methods=['POST'])
def accept_model_terms(model_id):
    try:
        success = model_manager.accept_terms(model_id)
        if success:
            return jsonify({'success': True, 'message': f'Terms accepted for {model_id}'})
        else:
            return jsonify({'error': 'Failed to accept terms'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/<model_id>/download', methods=['POST'])
def download_model(model_id):
    try:
        if not model_manager.is_terms_accepted(model_id):
            return jsonify({'error': 'Terms not accepted for this model'}), 400

        success = model_manager.download_model(model_id)
        if success:
            return jsonify({
                'success': True,
                'message': f'Model {model_id} downloaded successfully'
            })
        else:
            return jsonify({'error': f'Failed to download {model_id}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/hf-login', methods=['POST'])
def hf_login():
    try:
        data = request.json
        token = data.get('token')
        if not token:
            return jsonify({'error': 'Token is required'}), 400
            
        import huggingface_hub
        try:
            huggingface_hub.login(token=token, add_to_git_credential=False)
            user_info = huggingface_hub.whoami(token=token)
            return jsonify({'success': True, 'user': user_info.get('name', 'User')})
        except Exception as e:
            return jsonify({'error': f'Invalid token or connection error: {str(e)}'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/base-models/status', methods=['GET'])
def get_base_models_status():
    try:
        import os
        from pathlib import Path

        base_models = {
            'stable-audio-open-1.0': {
                'name': 'Stable Audio Open 1.0',
                'path': 'models/pretrained',
                'file': 'stable-audio-open-model.safetensors',
                'downloaded': False
            },
            'stable-audio-open-small': {
                'name': 'Stable Audio Open Small',
                'path': 'models/pretrained',
                'file': 'stable-audio-open-small-model.safetensors',
                'downloaded': False
            }
        }

        for model_id, info in base_models.items():
            model_dir = Path(info['path'])
            model_file = model_dir / info['file']

            if model_file.exists() and model_file.is_file():
                info['downloaded'] = True
            else:
                # Legacy layout: model stored in a subdirectory.
                old_path = model_dir / model_id
                if old_path.exists() and old_path.is_dir():
                    has_files = any([
                        (old_path / 'model.safetensors').exists(),
                        (old_path / 'pytorch_model.bin').exists(),
                        (old_path / 'model.ckpt').exists(),
                        len(list(old_path.glob('*.safetensors'))) > 0,
                        len(list(old_path.glob('*.bin'))) > 0
                    ])
                    info['downloaded'] = has_files
        
        return jsonify({'base_models': base_models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/<model_id>/delete', methods=['DELETE'])
def delete_model(model_id):
    try:
        success = model_manager.delete_model(model_id)
        if success:
            return jsonify({'success': True, 'message': f'Model {model_id} deleted'})
        else:
            return jsonify({'error': f'Failed to delete {model_id}'}), 400
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


@app.route('/api/models/storage', methods=['GET'])
def get_model_storage():
    try:
        storage_info = model_manager.get_storage_info()
        return jsonify(storage_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/start-fresh', methods=['POST'])
def start_fresh():
    try:
        config = get_config()
        data_dir = config.get_path("data")
        config_dir = config.get_path("models_config")

        data_files_deleted = 0
        if data_dir.exists():
            for file_path in data_dir.glob("*"):
                if file_path.is_file() and not file_path.name.endswith('.py'):
                    file_path.unlink()
                    data_files_deleted += 1

        config_files_deleted = 0
        if config_dir.exists():
            for file_path in config_dir.glob("custom_metadata.py"):
                if file_path.is_file():
                    file_path.unlink()
                    config_files_deleted += 1

        labels_reset = False
        user_labels_path = _annotator_labels_user_path()
        if user_labels_path.exists():
            user_labels_path.unlink()
            labels_reset = True

        data_dir.mkdir(exist_ok=True, parents=True)

        return jsonify({
            'message': f'Fresh start completed! Deleted {data_files_deleted} data files and {config_files_deleted} config metadata files.',
            'data_files_deleted': data_files_deleted,
            'config_files_deleted': config_files_deleted,
            'annotator_labels_reset': labels_reset
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/unwrap-model', methods=['POST'])
def unwrap_model():
    try:
        data = request.json
        model_config = data.get('model_config')
        ckpt_path = data.get('ckpt_path')
        name = data.get('name', 'model_unwrap')

        if not model_config or not ckpt_path:
            return jsonify({'error': 'model_config and ckpt_path are required'}), 400

        import subprocess
        from pathlib import Path

        config = get_config()
        repo_root = config.project_root

        model_config_path = repo_root / \
            model_config if not Path(
                model_config).is_absolute() else Path(model_config)
        ckpt_path_resolved = repo_root / \
            ckpt_path if not Path(ckpt_path).is_absolute() else Path(ckpt_path)

        if not model_config_path.exists():
            return jsonify({'error': f'Model config not found: {model_config_path}'}), 400
        if not ckpt_path_resolved.exists():
            return jsonify({'error': f'Checkpoint not found: {ckpt_path_resolved}'}), 400

        model_dir = ckpt_path_resolved.parent
        unwrapped_dir = model_dir / "unwrapped"
        unwrapped_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable, 'unwrap_model.py',
            '--model-config', str(model_config_path),
            '--ckpt-path', str(ckpt_path_resolved),
            '--name', name,
            '--use-safetensors'
        ]

        # unwrap_model.py writes next to its CWD, so run from stable-audio-tools/.
        stable_audio_dir = repo_root / "stable-audio-tools"

        proc = subprocess.run(cmd, cwd=stable_audio_dir,
                              capture_output=True, text=True)

        if proc.returncode == 0:

            import glob
            pattern = str(stable_audio_dir / f"{name}*.safetensors")
            created_files = glob.glob(pattern)

            moved_files = []
            for created_file in created_files:
                created_path = Path(created_file)
                target_path = unwrapped_dir / created_path.name

                try:
                    created_path.rename(target_path)
                    moved_files.append(str(target_path))
                    print(f"Moved {created_path.name} to {target_path}")
                except Exception as e:
                    print(f"Error moving {created_path}: {e}")

            unwrapped_files = list(unwrapped_dir.glob("*.safetensors"))

            return jsonify({
                'status': 'success',
                'output': proc.stdout,
                'unwrapped_path': moved_files[0] if moved_files else None,
                'unwrapped_files': [str(f) for f in unwrapped_files],
                'moved_files': moved_files
            })
        else:
            return jsonify({'status': 'error', 'error': proc.stderr, 'output': proc.stdout}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete-checkpoint', methods=['POST'])
def delete_checkpoint():
    try:
        data = request.json
        checkpoint_path = data.get('checkpoint_path')

        if not checkpoint_path:
            return jsonify({'error': 'checkpoint_path is required'}), 400

        config = get_config()
        repo_root = config.project_root

        ckpt_path_resolved = repo_root / \
            checkpoint_path if not Path(
                checkpoint_path).is_absolute() else Path(checkpoint_path)

        if not ckpt_path_resolved.exists():
            return jsonify({'error': f'Checkpoint file not found: {ckpt_path_resolved}'}), 404

        # Restrict deletion to .ckpt to avoid accidental loss of unwrapped models.
        if not ckpt_path_resolved.suffix == '.ckpt':
            return jsonify({'error': f'Only .ckpt files can be deleted: {ckpt_path_resolved}'}), 400

        try:
            ckpt_path_resolved.unlink()
            return jsonify({
                'status': 'success',
                'message': f'Checkpoint deleted successfully',
                'deleted_file': str(ckpt_path_resolved.name)
            })
        except Exception as e:
            return jsonify({'error': f'Failed to delete checkpoint: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete-wrapped-checkpoint', methods=['POST'])
def delete_wrapped_checkpoint():
    try:
        data = request.json
        model_name = data.get('model_name')

        if not model_name:
            return jsonify({'error': 'model_name is required'}), 400

        config = get_config()
        models_dir = config.get_path("models_fine_tuned")
        model_dir = models_dir / model_name

        if not model_dir.exists():
            return jsonify({'error': f'Model directory not found: {model_dir}'}), 404

        deleted_files = []
        for ckpt_file in model_dir.glob("*.ckpt"):
            try:
                ckpt_file.unlink()
                deleted_files.append(str(ckpt_file.name))
            except Exception as e:
                return jsonify({'error': f'Failed to delete {ckpt_file.name}: {str(e)}'}), 500

        if not deleted_files:
            return jsonify({'message': 'No wrapped checkpoint files found to delete'})

        return jsonify({
            'status': 'success',
            'message': f'Deleted {len(deleted_files)} wrapped checkpoint file(s)',
            'deleted_files': deleted_files
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
            "license": "Apache License 2.0",
            "copyright": "Copyright 2025-2026 Misagh Azimi",
            "license_text": license_text,
            "notice_text": notice_text,
            "license_url": "http://www.apache.org/licenses/LICENSE-2.0"
        })
    except Exception as e:
        logger.error(f"Error reading license info: {e}")
        return jsonify({
            "license": "Apache License 2.0",
            "copyright": "Copyright 2025-2026 Misagh Azimi",
            "error": str(e)
        }), 500

@app.route('/api/models-status', methods=['GET'])
def get_models_status():
    try:
        required_models = ['stable-audio-open-small', 'stable-audio-open-1.0']
        downloaded_models = [
            model_id for model_id in required_models if model_manager.is_model_downloaded(model_id)
        ]
        models_exist = len(downloaded_models) > 0
        models_message = (
            "Required base models are available."
            if models_exist
            else "No required base model is downloaded yet."
        )

        hf_authenticated = False
        try:
            from huggingface_hub import HfApi
            HfApi().whoami()
            hf_authenticated = True
        except Exception:
            hf_authenticated = False

        should_show = (not models_exist) and (not hf_authenticated)
        auth_reason = (
            "Hugging Face authentication is required to download gated models."
            if should_show
            else "Authentication already available or models already downloaded."
        )
        
        return jsonify({
            "models_exist": models_exist,
            "models_message": models_message,
            "should_show_auth_dialog": should_show,
            "auth_reason": auth_reason
        })
    except Exception as e:
        logger.error(f"Error checking models status: {e}")
        return jsonify({
            "error": str(e),
            "models_exist": False,
            "should_show_auth_dialog": True,
            "auth_reason": f"Error checking models: {str(e)}"
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
                                            capture_output=True, text=True, timeout=1)
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


_annotate_job_lock = threading.Lock()
_annotate_job = {
    'state': 'idle',   # idle | running | done | error
    'current': 0,
    'total': 0,
    'current_file': '',
    'tier': None,
    'folder': None,
    'results': [],
    'error': None,
}
_clap_download_job = {
    'state': 'idle',   # idle | running | done | error
    'message': '',
    'error': None,
}


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


@app.route('/api/bulk-annotate/status', methods=['GET'])
def bulk_annotate_status():
    from app.backend.data.auto_annotator import clap_checkpoint_available
    with _annotate_job_lock:
        snapshot = {k: v for k, v in _annotate_job.items() if k != 'results'}
        snapshot['result_count'] = len(_annotate_job['results'])
    snapshot['clap_available'] = clap_checkpoint_available(get_config().get_path('models_pretrained'))
    snapshot['clap_download'] = dict(_clap_download_job)
    return jsonify(snapshot)


@app.route('/api/bulk-annotate/results', methods=['GET'])
def bulk_annotate_results():
    with _annotate_job_lock:
        return jsonify({'results': list(_annotate_job['results']), 'state': _annotate_job['state']})


@app.route('/api/bulk-annotate', methods=['POST'])
def bulk_annotate():
    payload = request.json or {}
    folder = payload.get('folder_path', '').strip()
    tier = payload.get('tier', 'basic')
    if tier not in ('basic', 'rich'):
        return jsonify({'error': f"Invalid tier: {tier}"}), 400
    if not folder:
        return jsonify({'error': 'folder_path is required'}), 400

    folder_path = Path(folder).expanduser()
    if not folder_path.exists() or not folder_path.is_dir():
        return jsonify({'error': f'Folder not found: {folder_path}'}), 400

    from app.backend.data.auto_annotator import (
        annotate_folder, load_label_sets, clap_checkpoint_available,
    )

    if tier == 'rich' and not clap_checkpoint_available(get_config().get_path('models_pretrained')):
        return jsonify({'error': 'CLAP checkpoint not downloaded yet.'}), 409

    with _annotate_job_lock:
        if _annotate_job['state'] == 'running':
            return jsonify({'error': 'An annotation job is already running.'}), 409
        _annotate_job.update({
            'state': 'running', 'current': 0, 'total': 0, 'current_file': '',
            'tier': tier, 'folder': str(folder_path), 'results': [], 'error': None,
        })

    labels = load_label_sets(_annotator_labels_path())

    def progress_cb(i, total, name):
        with _annotate_job_lock:
            _annotate_job['current'] = i
            _annotate_job['total'] = total
            _annotate_job['current_file'] = name

    def runner():
        try:
            results = annotate_folder(
                folder_path, tier=tier, label_sets=labels,
                clap_ckpt_path=_clap_ckpt_path() if tier == 'rich' else None,
                progress_cb=progress_cb,
            )
            with _annotate_job_lock:
                _annotate_job['results'] = results
                _annotate_job['state'] = 'done'
        except Exception as exc:
            logger.exception("Bulk annotation failed")
            with _annotate_job_lock:
                _annotate_job['state'] = 'error'
                _annotate_job['error'] = str(exc)

    threading.Thread(target=runner, daemon=True).start()
    return jsonify({'message': 'Annotation started', 'tier': tier, 'folder': str(folder_path)})


@app.route('/api/bulk-annotate/commit', methods=['POST'])
def bulk_annotate_commit():
    """Merge user-reviewed annotation results into metadata.json.

    Body: { entries: [{ file_name, prompt, path }, ...], copy_files: bool }
    """
    payload = request.json or {}
    entries = payload.get('entries') or []
    copy_files = bool(payload.get('copy_files', True))
    if not entries:
        return jsonify({'error': 'No entries to commit.'}), 400

    config = get_config()
    data_dir = config.get_path('data')
    data_dir.mkdir(exist_ok=True, parents=True)

    json_path = Path(config.get_metadata_json_path())
    existing_metadata = []
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
        except Exception as exc:
            logger.warning("Could not load existing metadata: %s", exc)
            existing_metadata = []
    existing_files = {item['file_name']: item for item in existing_metadata}

    import shutil
    committed = 0
    for entry in entries:
        file_name = entry.get('file_name')
        prompt = (entry.get('prompt') or '').strip()
        src_path = entry.get('path')
        if not file_name or not prompt or not src_path:
            continue

        src = Path(src_path)
        if copy_files and src.exists() and src.parent.resolve() != data_dir.resolve():
            dst = data_dir / file_name
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                try:
                    shutil.copy2(src, dst)
                except Exception as exc:
                    logger.warning("Copy failed for %s: %s", src, exc)
                    continue
            stored_path = f"app/backend/data/{file_name}"
        else:
            stored_path = str(src)

        existing_files[file_name] = {
            'file_name': file_name,
            'prompt': prompt,
            'path': stored_path,
        }
        committed += 1

    final_metadata = list(existing_files.values())
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_metadata, f, indent=2)

    try:
        config.update_dataset_config()
    except Exception as exc:
        logger.warning("Failed to refresh dataset-config.json: %s", exc)

    return jsonify({
        'message': f'Committed {committed} annotations.',
        'committed': committed,
        'metadata_json': str(json_path),
    })


@app.route('/api/bulk-annotate/download-clap', methods=['POST'])
def bulk_annotate_download_clap():
    from app.backend.data.auto_annotator import download_clap_checkpoint

    with _annotate_job_lock:
        if _clap_download_job['state'] == 'running':
            return jsonify({'error': 'CLAP download already in progress.'}), 409
        _clap_download_job.update({'state': 'running', 'message': 'Starting download…', 'error': None})

    def runner():
        try:
            target = download_clap_checkpoint(
                get_config().get_path('models_pretrained'),
                progress_cb=lambda m: _clap_download_job.update({'message': m}),
            )
            _clap_download_job.update({'state': 'done', 'message': f'Downloaded to {target}'})
        except Exception as exc:
            logger.exception("CLAP download failed")
            _clap_download_job.update({'state': 'error', 'error': str(exc)})

    threading.Thread(target=runner, daemon=True).start()
    return jsonify({'message': 'CLAP download started'})


@app.route('/api/bulk-annotate/unload-clap', methods=['POST'])
def bulk_annotate_unload_clap():
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
