from typing import Any, Dict, List, Optional, Union, Callable
import re
import os
from pathlib import Path

from .logger import get_logger
from .exceptions import ValidationError

logger = get_logger(__name__)

class Validator:
    
    @staticmethod
    def required(value: Any, field_name: str = "field") -> Any:
        if value is None:
            raise ValidationError(field_name, None, "value is required")
        
        if isinstance(value, str) and not value.strip():
            raise ValidationError(field_name, value, "value cannot be empty")
        
        if isinstance(value, (list, dict)) and len(value) == 0:
            raise ValidationError(field_name, str(value), "value cannot be empty")
        
        return value
    
    @staticmethod
    def string(
        value: Any,
        field_name: str = "field",
        min_length: int = None,
        max_length: int = None,
        pattern: str = None
    ) -> str:
        if not isinstance(value, str):
            raise ValidationError(field_name, str(value), "must be a string")
        
        if min_length is not None and len(value) < min_length:
            raise ValidationError(
                field_name, 
                value, 
                f"must be at least {min_length} characters long"
            )
        
        if max_length is not None and len(value) > max_length:
            raise ValidationError(
                field_name, 
                value, 
                f"must be no more than {max_length} characters long"
            )
        
        if pattern is not None and not re.match(pattern, value):
            raise ValidationError(
                field_name, 
                value, 
                f"must match pattern: {pattern}"
            )
        
        return value
    
    @staticmethod
    def number(
        value: Any,
        field_name: str = "field",
        min_value: Union[int, float] = None,
        max_value: Union[int, float] = None,
        integer_only: bool = False
    ) -> Union[int, float]:
        try:
            if integer_only:
                num_value = int(value)
            else:
                num_value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(
                field_name, 
                str(value), 
                f"must be a {'integer' if integer_only else 'number'}"
            )
        
        if min_value is not None and num_value < min_value:
            raise ValidationError(
                field_name, 
                str(value), 
                f"must be at least {min_value}"
            )
        
        if max_value is not None and num_value > max_value:
            raise ValidationError(
                field_name, 
                str(value), 
                f"must be no more than {max_value}"
            )
        
        return num_value
    
    @staticmethod
    def file_path(
        value: Any,
        field_name: str = "field",
        must_exist: bool = True,
        allowed_extensions: List[str] = None
    ) -> Path:
        if not isinstance(value, (str, Path)):
            raise ValidationError(field_name, str(value), "must be a valid file path")
        
        path = Path(value)
        
        if must_exist and not path.exists():
            raise ValidationError(field_name, str(value), "file does not exist")
        
        if allowed_extensions:
            extension = path.suffix.lower()
            if extension not in [ext.lower() for ext in allowed_extensions]:
                raise ValidationError(
                    field_name, 
                    str(value), 
                    f"must have one of these extensions: {', '.join(allowed_extensions)}"
                )
        
        return path
    
    @staticmethod
    def choice(
        value: Any,
        field_name: str = "field",
        choices: List[Any] = None
    ) -> Any:
        if choices is not None and value not in choices:
            raise ValidationError(
                field_name, 
                str(value), 
                f"must be one of: {', '.join(str(c) for c in choices)}"
            )
        
        return value
    
    @staticmethod
    def email(value: Any, field_name: str = "field") -> str:
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not isinstance(value, str):
            raise ValidationError(field_name, str(value), "must be a string")
        
        if not re.match(email_pattern, value):
            raise ValidationError(field_name, value, "must be a valid email address")
        
        return value.lower()
    
    @staticmethod
    def url(value: Any, field_name: str = "field") -> str:
        url_pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:\w)*)?)?$'
        
        if not isinstance(value, str):
            raise ValidationError(field_name, str(value), "must be a string")
        
        if not re.match(url_pattern, value):
            raise ValidationError(field_name, value, "must be a valid URL")
        
        return value

def validate_request_data(schema: Dict[str, Dict[str, Any]]):
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    errors = {}
    
    required_fields = ['model_type', 'sample_rate']
    for field in required_fields:
        try:
            Validator.required(config.get(field), field)
        except ValidationError as e:
            errors[field] = [str(e)]
    
    if 'sample_rate' in config:
        try:
            Validator.number(
                config['sample_rate'], 
                'sample_rate',
                min_value=8000,
                max_value=48000,
                integer_only=True
            )
        except ValidationError as e:
            errors['sample_rate'] = [str(e)]
    
    if 'model_type' in config:
        try:
            Validator.choice(
                config['model_type'],
                'model_type',
                choices=['autoencoder', 'diffusion', 'lm']
            )
        except ValidationError as e:
            errors['model_type'] = [str(e)]
    
    if errors:
        logger.error(f"Model configuration validation failed: {errors}")
        raise ValidationError("model_config", str(config), f"validation failed: {errors}")
    
    return config

def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    errors = {}
    
    if 'modelName' in config:
        try:
            Validator.string(
                config['modelName'],
                'modelName',
                min_length=1,
                max_length=100
            )
        except ValidationError as e:
            errors['modelName'] = [str(e)]
    
    if 'epochs' in config:
        try:
            Validator.number(
                config['epochs'],
                'epochs',
                min_value=1,
                max_value=1000,
                integer_only=True
            )
        except ValidationError as e:
            errors['epochs'] = [str(e)]
    
    if 'batchSize' in config:
        try:
            Validator.number(
                config['batchSize'],
                'batchSize',
                min_value=1,
                max_value=64,
                integer_only=True
            )
        except ValidationError as e:
            errors['batchSize'] = [str(e)]
    
    if 'learningRate' in config:
        try:
            Validator.number(
                config['learningRate'],
                'learningRate',
                min_value=1e-6,
                max_value=1e-1
            )
        except ValidationError as e:
            errors['learningRate'] = [str(e)]
    
    if errors:
        logger.error(f"Training configuration validation failed: {errors}")
        raise ValidationError("training_config", str(config), f"validation failed: {errors}")
    
    return config