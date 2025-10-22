from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import os

from .logger import get_logger
from .exceptions import ConfigurationError

logger = get_logger(__name__)

class ConfigValidator:
    
    def __init__(self, config):
        self.config = config
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_all(self) -> Dict[str, Any]:
        logger.info("Starting configuration validation...")
        
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        self._validate_paths()
        self._validate_models()
        self._validate_environment()
        self._validate_dependencies()
        self._validate_permissions()
        
        results = {
            "valid": len(self.validation_errors) == 0,
            "errors": self.validation_errors,
            "warnings": self.validation_warnings,
            "total_errors": len(self.validation_errors),
            "total_warnings": len(self.validation_warnings)
        }
        
        if results["valid"]:
            logger.info(f"Configuration validation passed ({len(self.validation_warnings)} warnings)")
        else:
            logger.error(f"Configuration validation failed ({len(self.validation_errors)} errors, {len(self.validation_warnings)} warnings)")
        
        return results
    
    def _validate_paths(self):
        logger.debug("Validating paths...")
        
        critical_paths = [
            ("project_root", "Project root directory"),
            ("models", "Models directory"),
            ("models_config", "Model configuration directory"),
            ("backend", "Backend directory"),
            ("frontend", "Frontend directory")
        ]
        
        for path_name, description in critical_paths:
            try:
                path = self.config.get_path(path_name)
                if not path.exists():
                    self._add_error(f"{description} does not exist: {path}")
                elif not path.is_dir():
                    self._add_error(f"{description} is not a directory: {path}")
                else:
                    logger.debug(f"{description}: {path}")
            except Exception as e:
                self._add_error(f"Failed to validate {description}: {e}")
        
        optional_paths = [
            ("models_pretrained", "Pretrained models directory"),
            ("models_fine_tuned", "Fine-tuned models directory"),
            ("data_raw", "Raw data directory"),
            ("data_processed", "Processed data directory")
        ]
        
        for path_name, description in optional_paths:
            try:
                path = self.config.get_path(path_name)
                if not path.exists():
                    self._add_warning(f"{description} will be created: {path}")
                else:
                    logger.debug(f"{description}: {path}")
            except Exception as e:
                self._add_warning(f"Could not check {description}: {e}")
    
    def _validate_models(self):
        logger.debug("Validating model configurations...")
        
        try:
            model_configs = self.config.model_configs
            
            for model_name, model_config in model_configs.items():
                config_file = Path(model_config.get("config", ""))
                if not config_file.exists():
                    self._add_warning(f"Model config file not found for {model_name}: {config_file}")
                else:
                    try:
                        with open(config_file, 'r') as f:
                            json.load(f)
                        logger.debug(f"Model config valid: {model_name}")
                    except json.JSONDecodeError as e:
                        self._add_error(f"Invalid JSON in model config {model_name}: {e}")
                
                ckpt_file = Path(model_config.get("ckpt", ""))
                if not ckpt_file.exists():
                    self._add_warning(f"Model checkpoint not found for {model_name}: {ckpt_file}")
                else:
                    logger.debug(f"Model checkpoint exists: {model_name}")
        
        except Exception as e:
            self._add_error(f"Failed to validate model configurations: {e}")
    
    def _validate_environment(self):
        logger.debug("Validating environment...")
        
        import sys
        python_version = sys.version_info
        if python_version < (3, 8):
            self._add_error(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        else:
            logger.debug(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                logger.debug(f"CUDA available: {device_count} device(s), {device_name}")
            else:
                self._add_warning("CUDA not available, will use CPU (slower)")
        except ImportError:
            self._add_error("PyTorch not installed or not accessible")
        
        env_vars = [
            ("HOME", "User home directory"),
            ("PATH", "System PATH")
        ]
        
        for var_name, description in env_vars:
            if not os.environ.get(var_name):
                self._add_warning(f"Environment variable not set: {var_name} ({description})")
    
    def _validate_dependencies(self):
        logger.debug("Validating dependencies...")
        
        required_packages = [
            ("torch", "PyTorch"),
            ("torchaudio", "TorchAudio"),
            ("flask", "Flask"),
            ("transformers", "Transformers"),
            ("diffusers", "Diffusers"),
            ("librosa", "Librosa"),
            ("soundfile", "SoundFile"),
            ("numpy", "NumPy"),
            ("scipy", "SciPy")
        ]
        
        for package_name, description in required_packages:
            try:
                __import__(package_name)
                logger.debug(f"{description} available")
            except ImportError:
                self._add_error(f"Required package not installed: {package_name} ({description})")
        
        optional_packages = [
            ("wandb", "Weights & Biases"),
            ("gradio", "Gradio"),
            ("matplotlib", "Matplotlib")
        ]
        
        for package_name, description in optional_packages:
            try:
                __import__(package_name)
                logger.debug(f"{description} available")
            except ImportError:
                self._add_warning(f"Optional package not installed: {package_name} ({description})")
    
    def _validate_permissions(self):
        logger.debug("Validating permissions...")
        
        write_dirs = [
            ("models", "Models directory"),
            ("data_raw", "Raw data directory"),
            ("data_processed", "Processed data directory")
        ]
        
        for path_name, description in write_dirs:
            try:
                path = self.config.get_path(path_name)
                path.mkdir(exist_ok=True, parents=True)
                
                test_file = path / ".permission_test"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                    logger.debug(f"Write permission: {description}")
                except PermissionError:
                    self._add_error(f"No write permission for {description}: {path}")
            except Exception as e:
                self._add_error(f"Failed to check permissions for {description}: {e}")
    
    def _add_error(self, message: str):
        self.validation_errors.append(message)
        logger.error(f"Validation Error: {message}")
    
    def _add_warning(self, message: str):
        self.validation_warnings.append(message)
        logger.warning(f"Validation Warning: {message}")

def validate_config(config) -> Dict[str, Any]:
    validator = ConfigValidator(config)
    return validator.validate_all()

def ensure_config_valid(config) -> bool:
    results = validate_config(config)
    
    if not results["valid"]:
        error_messages = "\n".join(results["errors"])
        raise ConfigurationError(
            "configuration_validation",
            "valid configuration",
            f"{results['total_errors']} validation errors"
        )
    
    return True