"""请求中间件"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import log_api_request, log_api_response, api_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 记录请求开始时间
        start_time = time.time()
        
        # 获取用户信息（如果已认证）
        user_id = getattr(request.state, "user_id", None)
        
        # 记录请求日志
        log_api_request(
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            user_id=user_id
        )
        
        # 记录请求详情
        api_logger.debug(
            f"请求详情 - ID: {request_id}, Headers: {dict(request.headers)}, "
            f"Query: {dict(request.query_params)}"
        )
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算执行时间
            execution_time = time.time() - start_time
            
            # 记录响应日志
            log_api_response(
                request_id=request_id,
                status_code=response.status_code,
                execution_time=execution_time
            )
            
            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Execution-Time"] = f"{execution_time:.3f}"
            
            return response
            
        except Exception as e:
            # 计算执行时间
            execution_time = time.time() - start_time
            
            # 记录异常日志
            api_logger.error(
                f"请求异常 - ID: {request_id}, 执行时间: {execution_time:.3f}s, "
                f"异常: {str(e)}"
            )
            
            # 重新抛出异常，让异常处理器处理
            raise


class CORSMiddleware(BaseHTTPMiddleware):
    """CORS中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 处理预检请求
        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            response.headers["Access-Control-Max-Age"] = "86400"
            return response
        
        # 处理正常请求
        response = await call_next(request)
        
        # 添加CORS头
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response


def register_middleware(app):
    """注册所有中间件"""
    
    # 请求日志中间件
    app.add_middleware(RequestLoggingMiddleware)
    
    # CORS中间件
    app.add_middleware(CORSMiddleware)