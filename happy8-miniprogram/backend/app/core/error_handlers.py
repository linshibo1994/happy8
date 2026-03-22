"""全局异常处理器"""

import traceback
import uuid
from typing import Union
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import SQLAlchemyError
from redis.exceptions import RedisError

from app.core.exceptions import APIException, ErrorCode, create_error_response
from app.core.logging import app_logger, api_logger


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """API异常处理器"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    api_logger.error(
        f"API异常 - ID: {request_id}, 错误码: {exc.error_code}, "
        f"消息: {exc.message}, 路径: {request.url.path}"
    )
    
    return create_error_response(
        error_code=exc.error_code,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """HTTP异常处理器"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    api_logger.error(
        f"HTTP异常 - ID: {request_id}, 状态码: {exc.status_code}, "
        f"消息: {exc.detail}, 路径: {request.url.path}"
    )
    
    # 映射HTTP状态码到错误码
    error_code_map = {
        400: ErrorCode.INVALID_PARAMETER,
        401: ErrorCode.AUTH_FAILED,
        403: ErrorCode.PERMISSION_DENIED,
        404: ErrorCode.DATA_NOT_FOUND,
        422: ErrorCode.VALIDATION_ERROR,
        429: ErrorCode.PREDICTION_LIMIT_EXCEEDED,
        500: ErrorCode.UNKNOWN_ERROR,
    }
    
    error_code = error_code_map.get(exc.status_code, ErrorCode.UNKNOWN_ERROR)
    
    return create_error_response(
        error_code=error_code,
        message=str(exc.detail),
        status_code=exc.status_code
    )


async def validation_exception_handler(
    request: Request, 
    exc: RequestValidationError
) -> JSONResponse:
    """参数验证异常处理器"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # 提取验证错误详情
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        errors.append({"field": field, "message": message})
    
    api_logger.error(
        f"参数验证失败 - ID: {request_id}, 路径: {request.url.path}, "
        f"错误数量: {len(errors)}, 详情: {errors}"
    )
    
    return create_error_response(
        error_code=ErrorCode.VALIDATION_ERROR,
        message="参数验证失败",
        status_code=422,
        details={"errors": errors}
    )


async def sqlalchemy_exception_handler(
    request: Request, 
    exc: SQLAlchemyError
) -> JSONResponse:
    """数据库异常处理器"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    app_logger.error(
        f"数据库异常 - ID: {request_id}, 路径: {request.url.path}, "
        f"异常类型: {type(exc).__name__}, 详情: {str(exc)}"
    )
    
    # 记录详细错误信息（仅用于调试）
    if hasattr(exc, "orig"):
        app_logger.error(f"原始数据库错误: {exc.orig}")
    
    return create_error_response(
        error_code=ErrorCode.DATABASE_ERROR,
        message="数据库操作失败",
        status_code=500
    )


async def redis_exception_handler(
    request: Request, 
    exc: RedisError
) -> JSONResponse:
    """Redis异常处理器"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    app_logger.error(
        f"Redis异常 - ID: {request_id}, 路径: {request.url.path}, "
        f"异常类型: {type(exc).__name__}, 详情: {str(exc)}"
    )
    
    return create_error_response(
        error_code=ErrorCode.CACHE_ERROR,
        message="缓存服务异常",
        status_code=500
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """通用异常处理器"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # 记录完整的异常信息
    app_logger.error(
        f"未处理异常 - ID: {request_id}, 路径: {request.url.path}, "
        f"异常类型: {type(exc).__name__}, 详情: {str(exc)}"
    )
    
    # 记录堆栈跟踪（仅在调试模式下）
    from app.core.config import settings
    if settings.DEBUG:
        app_logger.error(f"堆栈跟踪: {traceback.format_exc()}")
    
    return create_error_response(
        error_code=ErrorCode.UNKNOWN_ERROR,
        message="服务器内部错误",
        status_code=500
    )


def register_exception_handlers(app):
    """注册所有异常处理器"""
    
    # API自定义异常
    app.add_exception_handler(APIException, api_exception_handler)
    
    # HTTP异常
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # 参数验证异常
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # 数据库异常
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
    
    # Redis异常
    app.add_exception_handler(RedisError, redis_exception_handler)
    
    # 通用异常（必须放在最后）
    app.add_exception_handler(Exception, generic_exception_handler)