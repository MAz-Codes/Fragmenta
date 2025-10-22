from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import traceback

from .logger import get_logger
from .exceptions import FragmentaError

logger = get_logger(__name__)

class APIResponse:
    
    @staticmethod
    def success(data: Any = None, message: str = None, meta: Dict[str, Any] = None) -> Dict[str, Any]:
        response = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        if message:
            response["message"] = message
        
        if meta:
            response["meta"] = meta
        
        logger.debug(f"API Success Response: {message or 'Operation completed'}")
        return response
    
    @staticmethod
    def error(
        error: Union[str, Exception, FragmentaError],
        status_code: int = 500,
        details: Dict[str, Any] = None,
        include_traceback: bool = False
    ) -> Dict[str, Any]:
        if isinstance(error, FragmentaError):
            message = error.message
            error_details = error.details.copy() if error.details else {}
            if details:
                error_details.update(details)
        elif isinstance(error, Exception):
            message = str(error)
            error_details = {
                "type": type(error).__name__,
                **(details or {})
            }
        else:
            message = str(error)
            error_details = details or {}
        
        response = {
            "success": False,
            "timestamp": datetime.utcnow().isoformat(),
            "error": {
                "message": message,
                "code": status_code,
                "details": error_details
            }
        }
        
        if include_traceback and isinstance(error, Exception):
            response["error"]["traceback"] = traceback.format_exc()
        
        logger.error(f"API Error Response ({status_code}): {message}")
        return response
    
    @staticmethod
    def validation_error(field_errors: Dict[str, List[str]]) -> Dict[str, Any]:
        return APIResponse.error(
            "Validation failed",
            status_code=400,
            details={
                "validation_errors": field_errors,
                "total_errors": sum(len(errors) for errors in field_errors.values())
            }
        )
    
    @staticmethod
    def not_found(resource: str, identifier: str = None) -> Dict[str, Any]:
        message = f"{resource.title()} not found"
        if identifier:
            message += f": {identifier}"
        
        return APIResponse.error(
            message,
            status_code=404,
            details={
                "resource_type": resource,
                "identifier": identifier
            }
        )
    
    @staticmethod
    def unauthorized(message: str = "Authentication required") -> Dict[str, Any]:
        return APIResponse.error(
            message,
            status_code=401,
            details={"auth_required": True}
        )
    
    @staticmethod
    def forbidden(message: str = "Access denied") -> Dict[str, Any]:
        return APIResponse.error(
            message,
            status_code=403,
            details={"access_denied": True}
        )
    
    @staticmethod
    def progress(
        current: int,
        total: int,
        message: str = None,
        data: Any = None
    ) -> Dict[str, Any]:
        percentage = (current / total * 100) if total > 0 else 0
        
        response = {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "progress": {
                "current": current,
                "total": total,
                "percentage": round(percentage, 2),
                "completed": current >= total
            }
        }
        
        if message:
            response["progress"]["message"] = message
        
        if data:
            response["data"] = data
        
        return response

def handle_api_error(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, dict) and "success" in result:
                return result
            return APIResponse.success(result)
        
        except FragmentaError as e:
            return APIResponse.error(e)
        
        except Exception as e:
            logger.exception(f"Unexpected error in {func.__name__}")
            return APIResponse.error(
                "Internal server error",
                status_code=500,
                details={"function": func.__name__}
            )
    
    return wrapper

def paginate_response(
    data: List[Any],
    page: int = 1,
    per_page: int = 10,
    total: int = None
) -> Dict[str, Any]:
    if total is None:
        total = len(data)
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_data = data[start_idx:end_idx]
    
    total_pages = (total + per_page - 1) // per_page
    
    meta = {
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_items": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "next_page": page + 1 if page < total_pages else None,
            "prev_page": page - 1 if page > 1 else None
        }
    }
    
    return APIResponse.success(
        data=page_data,
        meta=meta
    )