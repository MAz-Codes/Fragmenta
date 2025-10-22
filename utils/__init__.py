"""
Fragmenta Desktop Utilities Package
Centralized utilities for improved code organization and reusability
"""

from .logger import get_logger, setup_logging
from .config_validator import validate_config
from .exceptions import (
    FragmentaError,
    ModelNotFoundError,
    ConfigurationError,
    AuthenticationError,
    ValidationError
)

__all__ = [
    'get_logger',
    'setup_logging',
    'validate_config',
    'FragmentaError',
    'ModelNotFoundError',
    'ConfigurationError',
    'AuthenticationError',
    'ValidationError'
]