
class FragmentaError(Exception):
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            details_str = ", ".join([f"{k}={v}" for k, v in self.details.items()])
            return f"{self.message} ({details_str})"
        return self.message

class ModelNotFoundError(FragmentaError):
    
    def __init__(self, model_name: str, model_path: str = None):
        details = {"model_name": model_name}
        if model_path:
            details["model_path"] = model_path
        
        message = f"Model '{model_name}' not found"
        if model_path:
            message += f" at path '{model_path}'"
        
        super().__init__(message, details)

class ConfigurationError(FragmentaError):
    
    def __init__(self, config_item: str, expected: str = None, actual: str = None):
        details = {"config_item": config_item}
        if expected:
            details["expected"] = expected
        if actual:
            details["actual"] = actual
        
        message = f"Configuration error for '{config_item}'"
        if expected and actual:
            message += f": expected '{expected}', got '{actual}'"
        
        super().__init__(message, details)

class AuthenticationError(FragmentaError):
    
    def __init__(self, service: str, reason: str = None):
        details = {"service": service}
        if reason:
            details["reason"] = reason
        
        message = f"Authentication failed for service '{service}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(message, details)

class ValidationError(FragmentaError):
    
    def __init__(self, field: str, value: str = None, constraint: str = None):
        details = {"field": field}
        if value:
            details["value"] = value
        if constraint:
            details["constraint"] = constraint
        
        message = f"Validation failed for field '{field}'"
        if constraint:
            message += f": {constraint}"
        
        super().__init__(message, details)

class ModelDownloadError(FragmentaError):
    
    def __init__(self, model_id: str, reason: str = None):
        details = {"model_id": model_id}
        if reason:
            details["reason"] = reason
        
        message = f"Failed to download model '{model_id}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(message, details)

class GenerationError(FragmentaError):
    
    def __init__(self, prompt: str = None, model: str = None, reason: str = None):
        details = {}
        if prompt:
            details["prompt"] = prompt
        if model:
            details["model"] = model
        if reason:
            details["reason"] = reason
        
        message = "Audio generation failed"
        if reason:
            message += f": {reason}"
        
        super().__init__(message, details)

class TrainingError(FragmentaError):
    
    def __init__(self, stage: str = None, reason: str = None):
        details = {}
        if stage:
            details["stage"] = stage
        if reason:
            details["reason"] = reason
        
        message = "Training failed"
        if stage:
            message += f" during {stage}"
        if reason:
            message += f": {reason}"
        
        super().__init__(message, details)

# Exception mapping for common errors
def map_common_exception(exception: Exception, context: str = None) -> FragmentaError:
    
    if isinstance(exception, FileNotFoundError):
        if "model" in str(exception).lower():
            return ModelNotFoundError("Unknown", str(exception))
        else:
            return ConfigurationError("file_path", "existing file", "missing file")
    
    elif isinstance(exception, PermissionError):
        return ConfigurationError("permissions", "read/write access", "permission denied")
    
    elif isinstance(exception, ImportError):
        return ConfigurationError("dependencies", "installed package", "missing dependency")
    
    elif isinstance(exception, ValueError):
        return ValidationError("input_value", str(exception))
    
    else:
        details = {"original_type": type(exception).__name__}
        if context:
            details["context"] = context
        return FragmentaError(f"Unexpected error: {str(exception)}", details)