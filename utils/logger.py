"""
Centralized Logging System for Fragmenta Desktop
Replaces scattered print statements with structured logging
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import os

# Color codes for console output


class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels"""

    COLORS = {
        'DEBUG': Colors.CYAN,
        'INFO': Colors.GREEN,
        'WARNING': Colors.YELLOW,
        'ERROR': Colors.RED,
        'CRITICAL': Colors.RED + Colors.BOLD
    }

    def format(self, record):
        # Add color to level name
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, Colors.RESET)
            record.levelname = f"{color}{record.levelname}{Colors.RESET}"

        return super().format(record)


class FragmentaLogger:

    _instance = None
    _loggers = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.setup_logging()
            self._initialized = True

    def setup_logging(self, log_level: str = None, log_file: bool = True):

        if log_level is None:
            log_level = os.environ.get('FRAGMENTA_LOG_LEVEL', 'INFO').upper()

        numeric_level = getattr(logging, log_level, logging.INFO)

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)

        root_logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        console_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        console_formatter = ColoredFormatter(
            console_format, datefmt='%H:%M:%S')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        if log_file:
            log_filename = f"fragmenta_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_dir / log_filename)
            file_handler.setLevel(logging.DEBUG)

            file_format = "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
            file_formatter = logging.Formatter(
                file_format, datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        logger = self.get_logger('FragmentaLogger')
        logger.info(f"Logging system initialized (Level: {log_level})")
        if log_file:
            logger.info(f"Log file: {log_dir / log_filename}")

    def get_logger(self, name: str) -> logging.Logger:
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        return self._loggers[name]


_fragmenta_logger = FragmentaLogger()


def setup_logging(log_level: str = None, log_file: bool = True):
    _fragmenta_logger.setup_logging(log_level, log_file)


def get_logger(name: str) -> logging.Logger:
    return _fragmenta_logger.get_logger(name)


def log_function_call(func):
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(
            f"🔧 Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            raise
    return wrapper


def log_performance(func):
    def wrapper(*args, **kwargs):
        import time
        logger = get_logger(func.__module__)
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise
    return wrapper

def print_info(message: str, component: str = "Legacy"):
    logger = get_logger(component)
    logger.info(message)


def print_error(message: str, component: str = "Legacy"):
    logger = get_logger(component)
    logger.error(message)


def print_warning(message: str, component: str = "Legacy"):
    logger = get_logger(component)
    logger.warning(message)


def print_debug(message: str, component: str = "Legacy"):
    logger = get_logger(component)
    logger.debug(message)
