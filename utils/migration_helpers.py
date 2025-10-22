import functools
from typing import Any, Callable
from pathlib import Path

from .logger import get_logger
from .exceptions import FragmentaError, map_common_exception
from .api_responses import APIResponse

logger = get_logger(__name__)

def enhanced_print(message: str, component: str = "Legacy", level: str = "info"):
    comp_logger = get_logger(component)
    
    print(message)
    
    log_method = getattr(comp_logger, level.lower(), comp_logger.info)
    log_method(f"[MIGRATED] {message}")

def migrate_exception_handling(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FragmentaError:
            raise
        except Exception as e:
            fragmenta_error = map_common_exception(e, context=func.__name__)
            logger.error(f"Mapped exception in {func.__name__}: {e} -> {fragmenta_error}")
            raise fragmenta_error
    
    return wrapper

def enhance_api_endpoint(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            if isinstance(result, tuple) and len(result) == 2:
                response_data, status_code = result
                if status_code >= 400:
                    return APIResponse.error(response_data, status_code), status_code
                else:
                    return APIResponse.success(response_data), status_code
            
            if isinstance(result, dict) and 'error' in result:
                return APIResponse.error(result['error'], 400), 400
            
            return APIResponse.success(result), 200
            
        except FragmentaError as e:
            error_response = APIResponse.error(e)
            return error_response, error_response['error']['code']
        
        except Exception as e:
            logger.exception(f"Unexpected error in {func.__name__}")
            error_response = APIResponse.error(
                "Internal server error",
                status_code=500,
                details={"function": func.__name__}
            )
            return error_response, 500
    
    return wrapper

def log_performance_metrics(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time
        
        func_logger = get_logger(func.__module__ or "Unknown")
        start_time = time.time()
        
        func_logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            func_logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        
        except Exception as e:
            elapsed = time.time() - start_time
            func_logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise
    
    return wrapper

def safe_file_operation(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_logger = get_logger(func.__module__ or "FileOps")
        
        try:
            return func(*args, **kwargs)
        
        except FileNotFoundError as e:
            func_logger.error(f"File not found in {func.__name__}: {e}")
            raise FragmentaError(f"File not found: {e}")
        
        except PermissionError as e:
            func_logger.error(f"Permission denied in {func.__name__}: {e}")
            raise FragmentaError(f"Permission denied: {e}")
        
        except OSError as e:
            func_logger.error(f"OS error in {func.__name__}: {e}")
            raise FragmentaError(f"File system error: {e}")
        
        except Exception as e:
            func_logger.error(f"Unexpected error in file operation {func.__name__}: {e}")
            raise
    
    return wrapper

def migrate_print_statements_example():
    enhanced_print("Starting Fragmenta...", "DesktopApp", "info")
    enhanced_print("Backend server starting on http://127.0.0.1:5001", "DesktopApp", "info")
    
    app_logger = get_logger("DesktopApp")
    app_logger.info("Starting Fragmenta...")
    app_logger.info(" Backend server starting on http://127.0.0.1:5001")

def migrate_exception_handling_example():
    @migrate_exception_handling
    def some_operation():
        raise FileNotFoundError("Model file not found")

def find_print_statements(directory: Path):
    print_statements = []
    
    for py_file in directory.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                if 'print(' in line and not line.strip().startswith('#'):
                    print_statements.append({
                        'file': py_file,
                        'line': line_num,
                        'content': line.strip()
                    })
        except Exception as e:
            logger.warning(f"Could not scan {py_file}: {e}")
    
    return print_statements

def create_migration_report(project_root: Path):
    logger.info("Creating migration report...")
    
    print_statements = find_print_statements(project_root)
    
    report = {
        'print_statements': len(print_statements),
        'files_with_prints': len(set(ps['file'] for ps in print_statements)),
        'sample_prints': print_statements[:10],
    }
    
    logger.info(f"Migration Report:")
    logger.info(f"   - Print statements found: {report['print_statements']}")
    logger.info(f"   - Files with prints: {report['files_with_prints']}")
    
    return report