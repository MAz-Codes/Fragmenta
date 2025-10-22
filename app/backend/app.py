from utils.validators import Validator
from utils.exceptions import ModelNotFoundError, ValidationError, GenerationError
from utils.api_responses import APIResponse, handle_api_error
from utils.logger import setup_logging, get_logger
from app.core.generation.audio_generator import AudioGenerator
from app.core.training.lora_trainer import start_training as start_training_func, get_training_status, stop_training
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

logger.info("Initializing Backend API")

try:
    config = get_config()
    audio_processor = SimpleAudioProcessor(
        model_config_path=config.get_path("models_config") / "model_config.json"
    )
    generator = AudioGenerator(config)

    from app.core.model_manager import ModelManager
    model_manager = ModelManager(config)

    logger.info("Backend components initialized successfully")

except Exception as e:
    logger.error(f"Failed to initialize backend components: {e}")
    raise

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
            chunks_preview_data.append([
                filename,  # Original filename (not chunked)
                filename,  # Source file
                prompt,    # User's prompt
                "original" # Not chunked
            ])

        # Do not overwrite the metadata! keeps dataset creation more sustainable
        json_path = Path(config.get_metadata_json_path())
        existing_metadata = []
        
        # Load existing metadata if file exists
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

        return jsonify({
            'message': f'Files saved successfully! {len(saved_files)} original files saved to data folder',
            'saved_files': saved_files,
            'processed_count': len(saved_files),
            'chunks_preview': chunks_preview_data,  # Show all files (no chunking)
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
            training_config['epochs'] = 3
            print(f"   Setting default epochs: 3")
        if 'batchSize' not in training_config:
            training_config['batchSize'] = 1
            print(f"   Setting default batch size: 1")
        if 'learningRate' not in training_config:
            training_config['learningRate'] = 1e-4
            print(f"   Setting default learning rate: 1e-4")
        if 'saveWrappedCheckpoint' not in training_config:
            training_config['saveWrappedCheckpoint'] = False
            print(f"   Setting default saveWrappedCheckpoint: False")

        print(f"\nVALIDATED CONFIG:")
        print(f"   - Model Name: {training_config['modelName']}")
        print(f"   - Base Model: {training_config['baseModel']}")
        print(f"   - Epochs: {training_config['epochs']}")
        print(f"   - Batch Size: {training_config['batchSize']}")
        print(f"   - Learning Rate: {training_config['learningRate']}")
        print(f"   - Save Wrapped Checkpoint: {training_config['saveWrappedCheckpoint']}")

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

        # Priority: unwrapped_model_path > model_path > base model
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

        # Determine config based on file size or model name
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
            # Use the determined model path
            output_path = generator.generate_audio(
                prompt,
                unwrapped_model_path=unwrapped_model_path if unwrapped_model_path else None,
                model_path=determined_model_path if not unwrapped_model_path else None,
                config_file=config_file,
                duration=duration
            )
        elif model_name in ['stable-audio-open-small', 'stable-audio-open-1.0']:
            # Handle base models
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
                duration=duration
            )
        elif model_name and model_name != 'default':
            fine_tuned_path = config.get_path("models_fine_tuned") / model_name
            if not fine_tuned_path.exists():
                raise ModelNotFoundError(model_name, str(fine_tuned_path))

            output_path = generator.generate_audio(
                prompt, fine_tuned_path, duration=duration)
        else:
            logger.debug("Using default model")
            output_path = generator.generate_audio(prompt, duration=duration)

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

                # Create detailed checkpoint information
                checkpoints = []
                for ckpt_file in checkpoint_files:
                    # Extract epoch and step from filename if possible
                    import re
                    name = ckpt_file.stem
                    epoch_match = re.search(r'epoch=(\d+)', name)
                    step_match = re.search(r'step=(\d+)', name)

                    checkpoint_info = {
                        'name': name,
                        # Use relative path
                        'path': str(ckpt_file.relative_to(config.project_root)),
                        'size_mb': round(ckpt_file.stat().st_size / (1024 * 1024), 1),
                        'created': ckpt_file.stat().st_mtime
                    }

                    if epoch_match:
                        checkpoint_info['epoch'] = int(epoch_match.group(1))
                    if step_match:
                        checkpoint_info['step'] = int(step_match.group(1))

                    checkpoints.append(checkpoint_info)

                # Sort checkpoints by creation time (newest first)
                checkpoints.sort(key=lambda x: x['created'], reverse=True)

                # Get the latest checkpoint and config files
                latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat(
                ).st_mtime) if checkpoint_files else None
                latest_config = max(
                    config_files, key=lambda x: x.stat().st_mtime) if config_files else None

                # Check for unwrapped models
                unwrapped_dir = model_dir / "unwrapped"
                unwrapped_models = []
                if unwrapped_dir.exists():
                    for unwrapped_file in unwrapped_dir.glob("*.safetensors"):
                        unwrapped_models.append({
                            'name': unwrapped_file.stem,
                            # Use relative path
                            'path': str(unwrapped_file.relative_to(config.project_root)),
                            'size_mb': round(unwrapped_file.stat().st_size / (1024 * 1024), 1),
                            'created': unwrapped_file.stat().st_mtime
                        })

                    # Sort unwrapped models by creation time (newest first)
                    unwrapped_models.sort(
                        key=lambda x: x['created'], reverse=True)

                # For fine-tuned models, use the base model's config
                base_config_path = "models/config/model_config_small.json"  # Use relative path

                models.append({
                    'name': model_dir.name,
                    # Use relative path
                    'path': str(model_dir.relative_to(config.project_root)),
                    'has_checkpoint': has_checkpoint,
                    'has_config': has_config,
                    # Use relative path
                    'ckpt_path': str(latest_checkpoint.relative_to(config.project_root)) if latest_checkpoint else None,
                    'config_path': base_config_path,  # Use base model config for unwrapping
                    'checkpoints': checkpoints,  # Detailed checkpoint list
                    'unwrapped_models': unwrapped_models,
                    'created': model_dir.stat().st_mtime if model_dir.exists() else None
                })

        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/available', methods=['GET'])
def get_available_models():
    """Get list of available models from Hugging Face"""
    try:
        models = model_manager.get_available_models()
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/<model_id>/info', methods=['GET'])
def get_model_info(model_id):
    """Get information about a specific model"""
    try:
        model_info = model_manager.get_model_info(model_id)
        if not model_info:
            return jsonify({'error': 'Model not found'}), 404
        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/<model_id>/accept-terms', methods=['POST'])
def accept_model_terms(model_id):
    """Accept terms for a specific model"""
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
    """Download a model from Hugging Face"""
    try:
        # Check if terms are accepted
        if not model_manager.is_terms_accepted(model_id):
            return jsonify({'error': 'Terms not accepted for this model'}), 400

        # Start download
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


@app.route('/api/base-models/status', methods=['GET'])
def get_base_models_status():
    """Get the download status of base models"""
    try:
        import os
        from pathlib import Path
        
        base_models = {
            'stable-audio-open-1.0': {
                'name': 'Stable Audio Open 1.0',
                'path': 'models/pretrained',  # Updated to correct path
                'file': 'stable-audio-open-model.safetensors',  # Specific file to check
                'downloaded': False
            },
            'stable-audio-open-small': {
                'name': 'Stable Audio Open Small', 
                'path': 'models/pretrained',  # Updated to correct path
                'file': 'stable-audio-open-small-model.safetensors',  # Specific file to check
                'downloaded': False
            }
        }
        
        # Check if models are actually downloaded by looking for specific files
        for model_id, info in base_models.items():
            model_dir = Path(info['path'])
            model_file = model_dir / info['file']
            
            # Check if the specific model file exists
            if model_file.exists() and model_file.is_file():
                info['downloaded'] = True
            else:
                # Fallback: check subdirectory structure (old format)
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
    """Delete a downloaded model"""
    try:
        success = model_manager.delete_model(model_id)
        if success:
            return jsonify({'success': True, 'message': f'Model {model_id} deleted'})
        else:
            return jsonify({'error': f'Failed to delete {model_id}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/storage', methods=['GET'])
def get_model_storage():
    """Get storage information for models"""
    try:
        storage_info = model_manager.get_storage_info()
        return jsonify(storage_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/start-fresh', methods=['POST'])
def start_fresh():
    """Delete all data and start fresh"""
    try:
        config = get_config()
        data_dir = config.get_path("data")
        config_dir = config.get_path("models_config")

        # Delete all data files
        data_files_deleted = 0
        if data_dir.exists():
            for file_path in data_dir.glob("*"):
                if file_path.is_file() and not file_path.name.endswith('.py'):  # Don't delete Python files
                    file_path.unlink()
                    data_files_deleted += 1

        # Delete config metadata files (but keep the model configs)
        config_files_deleted = 0
        if config_dir.exists():
            for file_path in config_dir.glob("custom_metadata.py"):
                if file_path.is_file():
                    file_path.unlink()
                    config_files_deleted += 1

        # Recreate empty data directory
        data_dir.mkdir(exist_ok=True, parents=True)

        return jsonify({
            'message': f'Fresh start completed! Deleted {data_files_deleted} data files and {config_files_deleted} config metadata files.',
            'data_files_deleted': data_files_deleted,
            'config_files_deleted': config_files_deleted
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/unwrap-model', methods=['POST'])
def unwrap_model():
    """Unwrap a specific model checkpoint"""
    try:
        data = request.json
        model_config = data.get('model_config')
        ckpt_path = data.get('ckpt_path')
        name = data.get('name', 'model_unwrap')

        if not model_config or not ckpt_path:
            return jsonify({'error': 'model_config and ckpt_path are required'}), 400

        # Use the stable-audio-tools unwrap_model.py script directly for individual checkpoints
        import subprocess
        from pathlib import Path

        # Get config to resolve relative paths
        config = get_config()
        repo_root = config.project_root

        # Resolve paths relative to project root
        model_config_path = repo_root / \
            model_config if not Path(
                model_config).is_absolute() else Path(model_config)
        ckpt_path_resolved = repo_root / \
            ckpt_path if not Path(ckpt_path).is_absolute() else Path(ckpt_path)

        # Validate paths exist
        if not model_config_path.exists():
            return jsonify({'error': f'Model config not found: {model_config_path}'}), 400
        if not ckpt_path_resolved.exists():
            return jsonify({'error': f'Checkpoint not found: {ckpt_path_resolved}'}), 400

        # Get the model directory and create unwrapped subdirectory
        model_dir = ckpt_path_resolved.parent
        unwrapped_dir = model_dir / "unwrapped"
        unwrapped_dir.mkdir(exist_ok=True)

        cmd = [
            # Just the script name since we're running from stable-audio-tools dir
            sys.executable, 'unwrap_model.py',
            '--model-config', str(model_config_path),
            '--ckpt-path', str(ckpt_path_resolved),
            '--name', name,
            '--use-safetensors'
        ]

        # Run from repo root and set working directory to stable-audio-tools
        stable_audio_dir = repo_root / "stable-audio-tools"

        proc = subprocess.run(cmd, cwd=stable_audio_dir,
                              capture_output=True, text=True)

        if proc.returncode == 0:
            # The unwrap_model.py script creates files in the stable-audio-tools directory
            # We need to move them to the correct unwrapped directory

            # Find the created file in stable-audio-tools directory
            import glob
            pattern = str(stable_audio_dir / f"{name}*.safetensors")
            created_files = glob.glob(pattern)

            moved_files = []
            for created_file in created_files:
                created_path = Path(created_file)
                target_path = unwrapped_dir / created_path.name

                try:
                    # Move the file to the unwrapped directory
                    created_path.rename(target_path)
                    moved_files.append(str(target_path))
                    print(f"Moved {created_path.name} to {target_path}")
                except Exception as e:
                    print(f"Error moving {created_path}: {e}")

            # Find all unwrapped files in the unwrapped directory
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
    """Delete a specific checkpoint file"""
    try:
        data = request.json
        checkpoint_path = data.get('checkpoint_path')

        if not checkpoint_path:
            return jsonify({'error': 'checkpoint_path is required'}), 400

        # Get config to resolve relative paths
        config = get_config()
        repo_root = config.project_root

        # Resolve path relative to project root
        ckpt_path_resolved = repo_root / \
            checkpoint_path if not Path(
                checkpoint_path).is_absolute() else Path(checkpoint_path)

        if not ckpt_path_resolved.exists():
            return jsonify({'error': f'Checkpoint file not found: {ckpt_path_resolved}'}), 404

        # Ensure it's a .ckpt file for safety
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
    """Delete wrapped checkpoint files for a specific model"""
    try:
        data = request.json
        model_name = data.get('model_name')

        if not model_name:
            return jsonify({'error': 'model_name is required'}), 400

        # Find the model directory
        config = get_config()
        models_dir = config.get_path("models_fine_tuned")
        model_dir = models_dir / model_name

        if not model_dir.exists():
            return jsonify({'error': f'Model directory not found: {model_dir}'}), 404

        # Find and delete wrapped checkpoint files (.ckpt)
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


@app.route('/api/free-gpu-memory', methods=['POST'])
def free_gpu_memory():
    """Free GPU memory by clearing cache and stopping training processes"""
    try:
        import subprocess
        import torch
        import os
        import time

        print(" FREEING GPU MEMORY...")

        # Clear PyTorch CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("    Cleared PyTorch CUDA cache")

        # Clear MPS cache if available
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            print("    Cleared MPS cache")

        # Get current process ID to avoid killing ourselves
        current_pid = os.getpid()
        print(f"     Current process PID: {current_pid}")

        # Check for training processes and stop them safely
        try:
            # Get all CUDA processes
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

                                # Skip our own process
                                if pid_int == current_pid:
                                    print(
                                        f"     Skipping current process PID: {pid_int}")
                                    continue

                                # Check if it's a Python process using significant memory
                                if 'python' in process_name.lower() and mem_gb > 1.0:
                                    print(
                                        f"    Found Python process PID: {pid_int} using {mem_gb:.1f}GB")
                                    print(f"      Process: {process_name}")

                                    # Try to gracefully stop the process
                                    try:
                                        # Send SIGTERM first (graceful)
                                        subprocess.run(
                                            ['kill', '-TERM', str(pid_int)], check=False, timeout=5)
                                        print(
                                            f"    Sent SIGTERM to PID: {pid_int}")

                                        # Wait a moment
                                        time.sleep(2)

                                        # Check if process is still running
                                        try:
                                            # Check if process exists
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

        # Wait a moment for processes to stop
        time.sleep(3)

        # Clear cache again after stopping processes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("    Cleared PyTorch CUDA cache again")

        # Get memory info after clearing
        memory_info = {}
        if torch.cuda.is_available():
            # Use the same improved memory detection as the status endpoint
            total_memory = torch.cuda.get_device_properties(
                0).total_memory / (1024**3)
            torch.cuda.synchronize()
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
            cached_memory = torch.cuda.memory_reserved(0) / (1024**3)
            free_memory = total_memory - allocated_memory

            # Get nvidia-smi info
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

            # Use the most accurate reading
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
    """Toggle debug mode for GPU memory logging"""
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
    """Get current debug mode status"""
    return jsonify({
        'debug_mode': DEBUG_MODE,
        'message': f"Debug mode is {'enabled' if DEBUG_MODE else 'disabled'}"
    })


# Add API call statistics for debugging
_api_call_stats = {
    'gpu_memory_status': 0,
    'status': 0,
    'training_status': 0,
    'last_reset': time.time()
}


def _log_api_call(endpoint):
    """Log API call for debugging"""
    global _api_call_stats
    _api_call_stats[endpoint] = _api_call_stats.get(endpoint, 0) + 1

    # Reset stats every hour
    if time.time() - _api_call_stats['last_reset'] > 3600:
        _api_call_stats = {endpoint: 1, 'last_reset': time.time()}


@app.route('/api/debug-stats', methods=['GET'])
def get_debug_stats():
    """Get API call statistics for debugging"""
    return jsonify({
        'api_call_stats': _api_call_stats,
        'uptime_hours': (time.time() - _api_call_stats['last_reset']) / 3600,
        'calls_per_minute': {
            'gpu_memory_status': _api_call_stats.get('gpu_memory_status', 0) / max(1, (time.time() - _api_call_stats['last_reset']) / 60),
            'status': _api_call_stats.get('status', 0) / max(1, (time.time() - _api_call_stats['last_reset']) / 60),
            'training_status': _api_call_stats.get('training_status', 0) / max(1, (time.time() - _api_call_stats['last_reset']) / 60)
        }
    })


# Add caching for GPU memory status to reduce overhead
_gpu_memory_cache = {}
_gpu_memory_cache_time = 0
_gpu_memory_cache_duration = 2.0  # Cache for 2 seconds

# Throttle memory warnings (only show every 30 seconds)
_last_memory_warning_time = 0
_memory_warning_interval = 30  # seconds


@app.route('/api/open-output-folder', methods=['POST'])
def open_output_folder():
    """Open the output folder in the system file explorer"""
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
    """Open the documentation URL in the default browser"""
    try:
        import webbrowser
        
        # TODO: Replace with actual documentation URL
        documentation_url = "https://github.com/your-repo/fragmenta-docs"
        webbrowser.open(documentation_url)
        
        return jsonify({"success": True, "message": "Documentation opened"})
    except Exception as e:
        logger.error(f"Error opening documentation: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# Global flag for welcome page state
_welcome_page_closed = False

@app.route('/api/welcome-page-closed', methods=['POST'])
def welcome_page_closed():
    """Signal that the welcome page has been closed"""
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
    """Check if welcome page has been closed"""
    global _welcome_page_closed
    return jsonify({"closed": _welcome_page_closed})

@app.route('/api/license-info', methods=['GET'])
def get_license_info():
    """Get license and attribution information"""
    try:
        project_root = Path(__file__).parent.parent.parent
        
        # Read LICENSE file
        license_file = project_root / "LICENSE"
        license_text = ""
        if license_file.exists():
            with open(license_file, 'r', encoding='utf-8') as f:
                # Read first 50 lines for summary
                lines = f.readlines()[:50]
                license_text = ''.join(lines)
        
        # Read NOTICE.md for attribution info
        notice_file = project_root / "NOTICE.md"
        notice_text = ""
        if notice_file.exists():
            with open(notice_file, 'r', encoding='utf-8') as f:
                notice_text = f.read()
        
        return jsonify({
            "license": "Apache License 2.0",
            "copyright": "Copyright 2025 Misagh Azimi",
            "license_text": license_text,
            "notice_text": notice_text,
            "license_url": "http://www.apache.org/licenses/LICENSE-2.0"
        })
    except Exception as e:
        logger.error(f"Error reading license info: {e}")
        return jsonify({
            "license": "Apache License 2.0",
            "copyright": "Copyright 2025 Misagh Azimi",
            "error": str(e)
        }), 500

@app.route('/api/models-status', methods=['GET'])
def get_models_status():
    """Check if required models exist and if auth dialog should be shown"""
    try:
        from app.core.hf_auth_dialog import check_required_models_exist, should_show_auth_dialog
        
        models_exist, models_message = check_required_models_exist()
        should_show, auth_reason = should_show_auth_dialog()
        
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
    """Get current GPU memory status with caching to reduce overhead"""
    _log_api_call('gpu_memory_status')
    global _gpu_memory_cache, _gpu_memory_cache_time

    # Check cache first
    current_time = time.time()
    if current_time - _gpu_memory_cache_time < _gpu_memory_cache_duration:
        return jsonify({'memory_info': _gpu_memory_cache})

    try:
        import torch
        import subprocess
        import psutil

        memory_info = {}
        if torch.cuda.is_available():
            # Get PyTorch memory info with better tracking
            total_memory = torch.cuda.get_device_properties(
                0).total_memory / (1024**3)

            # Force PyTorch to synchronize before reading memory
            torch.cuda.synchronize()
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
            cached_memory = torch.cuda.memory_reserved(0) / (1024**3)
            free_memory = total_memory - allocated_memory

            # Get nvidia-smi info for comparison (only if PyTorch shows 0 usage)
            nvidia_used_gb = 0
            nvidia_total_gb = total_memory
            nvidia_free_gb = total_memory

            if allocated_memory == 0:
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                                            capture_output=True, text=True, timeout=1)  # Add timeout
                    if result.stdout.strip():
                        used_mb, total_mb = result.stdout.strip().split(', ')
                        nvidia_used_gb = float(used_mb) / 1024
                        nvidia_total_gb = float(total_mb) / 1024
                        nvidia_free_gb = nvidia_total_gb - nvidia_used_gb
                except Exception as e:
                    # Only log if there's an actual error, not just missing nvidia-smi
                    if "Could not get nvidia-smi info" not in str(e):
                        print(f"GPU Memory Error: {e}")

            # Get CUDA capability and device info
            cuda_capability = torch.cuda.get_device_capability(0)
            device_name = torch.cuda.get_device_name(0)

            # Use the most accurate memory reading
            # If PyTorch shows 0 but nvidia-smi shows usage, use nvidia-smi
            # If PyTorch shows usage, use PyTorch
            if allocated_memory > 0:
                final_allocated = allocated_memory
                final_cached = cached_memory
                final_free = free_memory
                memory_source = "PyTorch"
            else:
                final_allocated = nvidia_used_gb
                final_cached = cached_memory  # Keep PyTorch cached
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

            # Only log if there are significant issues AND enough time has passed
            global _last_memory_warning_time
            if (current_time - _last_memory_warning_time) > _memory_warning_interval:
                if final_allocated > 10.0:  # More than 10GB used
                    print(
                        f"  High GPU Memory Usage: {final_allocated:.2f}GB allocated, {final_free:.2f}GB free")
                    _last_memory_warning_time = current_time
                elif final_free < 1.0:  # Less than 1GB free
                    print(
                        f"  Low GPU Memory: {final_free:.2f}GB free, {final_allocated:.2f}GB allocated")
                    _last_memory_warning_time = current_time
        else:
            # CPU fallback
            memory_info['cpu'] = {
                'total': psutil.virtual_memory().total / (1024**3),
                'available': psutil.virtual_memory().available / (1024**3),
                'used': psutil.virtual_memory().used / (1024**3),
                'device': 'CPU',
                'type': 'cpu'
            }

        # Update cache
        _gpu_memory_cache = memory_info
        _gpu_memory_cache_time = current_time

        return jsonify({'memory_info': memory_info})
    except Exception as e:
        print(f"Error getting GPU memory status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown the Flask server gracefully"""
    try:
        print(" Shutting down Flask server...")
        # Use a function to shutdown the server
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()
        return jsonify({'message': 'Server shutting down'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
