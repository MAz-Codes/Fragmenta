import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime
import requests
from huggingface_hub import snapshot_download, hf_hub_download
import hashlib


class ModelManager:

    def __init__(self, config):
        self.config = config
        self.models_dir = config.get_path("models_pretrained")
        self.models_dir.mkdir(exist_ok=True, parents=True)

        self.available_models = {
            'stable-audio-open-small': {
                'name': 'Stable Audio Open Small',
                'repo': 'stabilityai/stable-audio-open-small',
                'files': ['model.safetensors', 'config.json'],
                'size': '2.1 GB',
                'description': 'Fast generation, good quality, lower memory usage',
                'best_for': 'Beginners, quick experiments, limited GPU',
                'license': 'Stability AI License',
                'checksum': 'sha256:abc123...'
            },
            'stable-audio-open-1.0': {
                'name': 'Stable Audio Open 1.0',
                'repo': 'stabilityai/stable-audio-open-1.0',
                'files': ['model.safetensors', 'config.json'],
                'size': '8.2 GB',
                'description': 'Highest quality, more detailed audio',
                'best_for': 'Professional use, high-end GPUs',
                'license': 'Stability AI License',
                'checksum': 'sha256:def456...'
            }
        }

        self.terms_file = Path("config/terms_accepted.json")
        self.terms_file.parent.mkdir(exist_ok=True)

    def get_available_models(self) -> List[Dict]:

        models = []

        for model_id, info in self.available_models.items():
            is_downloaded = self.is_model_downloaded(model_id)
            
            downloaded_size = None
            if is_downloaded:
                if model_id == 'stable-audio-open-small':
                    model_file = self.models_dir / 'stable-audio-open-small-model.safetensors'
                    downloaded_size = self._get_file_size(model_file) if model_file.exists() else None
                elif model_id == 'stable-audio-open-1.0':
                    model_file = self.models_dir / 'stable-audio-open-model.safetensors'
                    downloaded_size = self._get_file_size(model_file) if model_file.exists() else None
                else:
                    model_path = self.models_dir / model_id
                    downloaded_size = self._get_downloaded_size(model_path) if model_path.exists() else None

            models.append({
                'id': model_id,
                'name': info['name'],
                'size': info['size'],
                'description': info['description'],
                'best_for': info['best_for'],
                'license': info['license'],
                'downloaded': is_downloaded,
                'downloaded_size': downloaded_size,
                'terms_accepted': self.is_terms_accepted(model_id)
            })

        return models

    def _get_file_size(self, file_path: Path) -> str:

        if not file_path.exists() or not file_path.is_file():
            return "0 B"
        
        size = file_path.stat().st_size
        return self._bytes_to_human(size)

    def _get_downloaded_size(self, model_path: Path) -> str:

        if not model_path.exists():
            return "0 B"

        total_size = 0
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024.0:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024.0
        return f"{total_size:.1f} TB"

    def get_model_info(self, model_id: str) -> Optional[Dict]:

        if model_id not in self.available_models:
            return None

        info = self.available_models[model_id].copy()
        info['id'] = model_id
        info['downloaded'] = self.is_model_downloaded(model_id)
        info['terms_accepted'] = self.is_terms_accepted(model_id)

        return info

    def is_model_downloaded(self, model_id: str) -> bool:

        if model_id == 'stable-audio-open-small':
            model_file = self.models_dir / 'stable-audio-open-small-model.safetensors'
            return model_file.exists() and model_file.is_file()
        elif model_id == 'stable-audio-open-1.0':
            model_file = self.models_dir / 'stable-audio-open-model.safetensors'
            return model_file.exists() and model_file.is_file()
        else:
            model_path = self.models_dir / model_id
            if model_path.exists() and model_path.is_dir():
                return any(model_path.iterdir())
            pattern = f"*{model_id}*.safetensors"
            matching_files = list(self.models_dir.glob(pattern))
            return len(matching_files) > 0

    def is_terms_accepted(self, model_id: str) -> bool:

        if not self.terms_file.exists():
            return False

        try:
            with open(self.terms_file, 'r') as f:
                terms_data = json.load(f)
            return terms_data.get(model_id, {}).get('accepted', False)
        except:
            return False

    def accept_terms(self, model_id: str) -> bool:

        if model_id not in self.available_models:
            return False

        terms_data = {}
        if self.terms_file.exists():
            try:
                with open(self.terms_file, 'r') as f:
                    terms_data = json.load(f)
            except:
                terms_data = {}

        terms_data[model_id] = {
            'accepted': True,
            'accepted_at': datetime.now().isoformat(),
            'model_name': self.available_models[model_id]['name'],
            'license': self.available_models[model_id]['license']
        }

        try:
            with open(self.terms_file, 'w') as f:
                json.dump(terms_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving terms acceptance: {e}")
            return False

    def download_model(self, model_id: str, progress_callback: Optional[Callable] = None) -> bool:

        if model_id not in self.available_models:
            return False

        if not self.is_terms_accepted(model_id):
            print(f"Terms not accepted for {model_id}")
            self.accept_terms(model_id)
            print(f"Automatically accepted terms for {model_id}")

        model_info = self.available_models[model_id]
        target_dir = self.models_dir
        target_dir.mkdir(exist_ok=True, parents=True)

        try:
            print(f"Downloading {model_info['name']} to {target_dir}")

            if progress_callback:
                progress_callback(
                    0, f"Starting download of {model_info['name']}...")

            from huggingface_hub import HfApi
            api = HfApi()

            try:
                user = api.whoami()
                print(f"Authenticated as: {user}")
                if progress_callback:
                    progress_callback(10, "Authentication verified...")
            except Exception as auth_error:
                print(f"Not authenticated with Hugging Face: {auth_error}")
                if progress_callback:
                    progress_callback(0, "Authentication required...")

                try:
                    from app.core.hf_auth_dialog import show_hf_auth_dialog
                    success = show_hf_auth_dialog()
                    
                    if not success:
                        print("Authentication dialog was cancelled")
                        if progress_callback:
                            progress_callback(0, "Authentication cancelled")
                        return False

                    try:
                        user = api.whoami()
                        print(f"Now authenticated as: {user}")
                        if progress_callback:
                            progress_callback(
                                10, "Authentication successful...")
                    except Exception as retry_error:
                        print(f"Still not authenticated: {retry_error}")
                        if progress_callback:
                            progress_callback(0, "Authentication failed")
                        return False

                except ImportError:
                    print("To download models, you need to:")
                    print(
                        "1. Visit https://huggingface.co/stabilityai/stable-audio-open-small")
                    print("2. Accept the terms and conditions")
                    print("3. Log in to your Hugging Face account")
                    print(
                        "4. Get your access token from https://huggingface.co/settings/tokens")
                    print("5. Run: huggingface-cli login")
                    if progress_callback:
                        progress_callback(0, "Manual authentication required")
                    return False

            if progress_callback:
                progress_callback(20, "Starting file download...")

            class ProgressWrapper:
                def __init__(self, callback):
                    self.callback = callback
                    self.last_percent = 20

                def __call__(self, bytes_downloaded, total_bytes):
                    if total_bytes > 0:
                        percent = 20 + \
                            int((bytes_downloaded / total_bytes) * 70)
                        if percent > self.last_percent:
                            self.last_percent = percent
                            if self.callback:
                                self.callback(
                                    percent, f"Downloaded {bytes_downloaded}/{total_bytes} bytes")

            progress_wrapper = ProgressWrapper(
                progress_callback) if progress_callback else None

            if progress_callback:
                progress_callback(0, "Starting model download...")

            try:
                from huggingface_hub import snapshot_download
                import shutil
                from huggingface_hub import hf_hub_download

                downloaded_files = []
                total_files = len(model_info['files'])

                for i, file_pattern in enumerate(model_info['files']):
                    if progress_callback:
                        progress_callback(
                            0, f"Starting download of {file_pattern}...")

                    try:
                        if file_pattern == 'model.safetensors':
                            if model_id == 'stable-audio-open-small':
                                final_filename = 'stable-audio-open-small-model.safetensors'
                            elif model_id == 'stable-audio-open-1.0':
                                final_filename = 'stable-audio-open-model.safetensors'
                            else:
                                final_filename = f"{model_id}-model.safetensors"
                        else:
                            final_filename = f"{model_id}-{file_pattern}"

                        downloaded_file = hf_hub_download(
                            repo_id=model_info['repo'],
                            filename=file_pattern,
                            resume_download=True
                        )
                        
                        downloaded_path = Path(downloaded_file)
                        final_path = target_dir / final_filename
                        
                        final_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        shutil.copy2(str(downloaded_path), str(final_path))
                        print(f"Saved as {final_filename}")
                            
                        downloaded_files.append(str(final_path))

                        if progress_callback:
                            progress_callback(100, f"Completed {file_pattern}")

                    except Exception as file_error:
                        print(
                            f"Failed to download {file_pattern}: {file_error}")
                        if progress_callback:
                            progress_callback(
                                0, f"Failed to download {file_pattern}")
                        continue

                print(f"Downloaded {len(downloaded_files)} files")

                if progress_callback:
                    progress_callback(
                        95, "Download completed, verifying files...")

            except Exception as download_error:
                print(f"Error during download: {download_error}")
                if progress_callback:
                    progress_callback(
                        0, f"Download failed: {str(download_error)}")
                return False

            if progress_callback:
                progress_callback(95, "Verifying download...")

            expected_files = []
            if model_id == 'stable-audio-open-small':
                expected_files.append(
                    'stable-audio-open-small-model.safetensors')
            elif model_id == 'stable-audio-open-1.0':
                expected_files.append('stable-audio-open-model.safetensors')
            else:
                expected_files.append(f"{model_id}-model.safetensors")

            files_exist = any((target_dir / expected_file).exists()
                              for expected_file in expected_files)

            if files_exist:
                if progress_callback:
                    progress_callback(100, "Download complete!")
                print(f"Successfully downloaded {model_info['name']}")
                return True
            else:
                if progress_callback:
                    progress_callback(0, "Download verification failed")
                print(f"Expected files not found: {expected_files}")
                return False

        except Exception as e:
            print(f"Error downloading {model_info['name']}: {e}")
            if progress_callback:
                progress_callback(0, f"Error: {str(e)}")

            if "403" in str(e) and "gated repositories" in str(e).lower():
                print("Token permission issue detected!")
                print(
                    "Your Hugging Face token needs 'Read access to public gated repositories'")
                print("Please:")
                print("1. Go to https://huggingface.co/settings/tokens")
                print("2. Edit your token or create a new one")
                print("3. Enable 'Read access to public gated repositories'")
                print("4. Try the download again")
            elif "401" in str(e) or "restricted" in str(e).lower():
                print("This model requires Hugging Face authentication.")
                print("Please visit the model page and accept terms first:")
                print(f"https://huggingface.co/{model_info['repo']}")
            return False

    def delete_model(self, model_id: str) -> bool:

        deleted_something = False
        
        if model_id == 'stable-audio-open-small':
            model_file = self.models_dir / 'stable-audio-open-small-model.safetensors'
            config_file = self.models_dir / 'stable-audio-open-small-config.json'
        elif model_id == 'stable-audio-open-1.0':
            model_file = self.models_dir / 'stable-audio-open-model.safetensors'
            config_file = self.models_dir / 'stable-audio-open-1.0-config.json'
        else:
            model_file = self.models_dir / f"{model_id}-model.safetensors"
            config_file = self.models_dir / f"{model_id}-config.json"
        
        for file_path in [model_file, config_file]:
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"Deleted {file_path.name}")
                    deleted_something = True
                except Exception as e:
                    print(f"Error deleting {file_path.name}: {e}")
        
        model_path = self.models_dir / model_id
        if model_path.exists() and model_path.is_dir():
            try:
                shutil.rmtree(model_path)
                print(f"Deleted {model_id} directory")
                deleted_something = True
            except Exception as e:
                print(f"Error deleting {model_id} directory: {e}")

        if deleted_something:
            print(f"Deleted {model_id}")
            return True
        else:
            print(f"No files found for {model_id}")
            return False

    def get_download_progress(self, model_id: str) -> Dict:

        return {
            'model_id': model_id,
            'downloaded': self.is_model_downloaded(model_id),
            'size': self.available_models.get(model_id, {}).get('size', 'Unknown')
        }

    def get_storage_info(self) -> Dict:

        total_size = 0
        model_count = 0

        if self.models_dir.exists():
            for model_id in self.available_models.keys():
                if self.is_model_downloaded(model_id):
                    model_count += 1
            
            for file_path in self.models_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size

        return {
            'total_size_bytes': total_size,
            'total_size_human': self._bytes_to_human(total_size),
            'model_count': model_count,
            'models_dir': str(self.models_dir)
        }

    def _bytes_to_human(self, bytes_value: int) -> str:

        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"
