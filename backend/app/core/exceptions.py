"""统一异常处理模块"""

from typing import Any, Dict, Optional
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse


class APIException(HTTPException):
    """API标准异常类"""
    
    def __init__(
        self,
        error_code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[Dict[str, Any]] = None
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        super().__init__(status_code=status_code, detail=message)


class ErrorCode:
    """错误码定义"""
    
    # 通用错误 1000-1999
    UNKNOWN_ERROR = "1000"
    INVALID_PARAMETER = "1001"
    MISSING_PARAMETER = "1002"
    VALIDATION_ERROR = "1003"
    
    # 认证相关错误 2000-2099
    AUTH_FAILED = "2000"
    TOKEN_INVALID = "2001"
    TOKEN_EXPIRED = "2002"
    PERMISSION_DENIED = "2003"
    
    # 用户相关错误 2100-2199
    USER_NOT_FOUND = "2100"
    USER_EXISTS = "2101"
    USER_INACTIVE = "2102"
    PHONE_EXISTS = "2103"
    EMAIL_EXISTS = "2104"
    
    # 会员相关错误 2200-2299
    MEMBERSHIP_NOT_FOUND = "2200"
    MEMBERSHIP_EXPIRED = "2201"
    INSUFFICIENT_PERMISSION = "2202"
    PLAN_NOT_FOUND = "2203"
    PLAN_INACTIVE = "2204"
    
    # 订单相关错误 2300-2399
    ORDER_NOT_FOUND = "2300"
    ORDER_STATUS_ERROR = "2301"
    PAYMENT_FAILED = "2302"
    ORDER_EXPIRED = "2303"
    REFUND_FAILED = "2304"
    
    # 预测相关错误 2400-2499
    PREDICTION_LIMIT_EXCEEDED = "2400"
    ALGORITHM_NOT_FOUND = "2401"
    ALGORITHM_DISABLED = "2402"
    INSUFFICIENT_DATA = "2403"
    PREDICTION_FAILED = "2404"
    
    # 数据相关错误 2500-2599
    DATA_NOT_FOUND = "2500"
    DATA_INVALID = "2501"
    DATABASE_ERROR = "2502"
    CACHE_ERROR = "2503"
    
    # 外部服务错误 2600-2699
    WECHAT_API_ERROR = "2600"
    PAYMENT_API_ERROR = "2601"
    SMS_SERVICE_ERROR = "2602"
    EMAIL_SERVICE_ERROR = "2603"


class BusinessException(APIException):
    """业务逻辑异常"""
    
    @classmethod
    def auth_failed(cls, message: str = "认证失败") -> "BusinessException":
        return cls(ErrorCode.AUTH_FAILED, message, status.HTTP_401_UNAUTHORIZED)
    
    @classmethod
    def token_invalid(cls, message: str = "令牌无效") -> "BusinessException":
        return cls(ErrorCode.TOKEN_INVALID, message, status.HTTP_401_UNAUTHORIZED)
    
    @classmethod
    def token_expired(cls, message: str = "令牌已过期") -> "BusinessException":
        return cls(ErrorCode.TOKEN_EXPIRED, message, status.HTTP_401_UNAUTHORIZED)
    
    @classmethod
    def permission_denied(cls, message: str = "权限不足") -> "BusinessException":
        return cls(ErrorCode.PERMISSION_DENIED, message, status.HTTP_403_FORBIDDEN)
    
    @classmethod
    def user_not_found(cls, message: str = "用户不存在") -> "BusinessException":
        return cls(ErrorCode.USER_NOT_FOUND, message, status.HTTP_404_NOT_FOUND)
    
    @classmethod
    def user_exists(cls, message: str = "用户已存在") -> "BusinessException":
        return cls(ErrorCode.USER_EXISTS, message, status.HTTP_409_CONFLICT)
    
    @classmethod
    def user_inactive(cls, message: str = "用户未激活") -> "BusinessException":
        return cls(ErrorCode.USER_INACTIVE, message, status.HTTP_403_FORBIDDEN)
    
    @classmethod
    def membership_expired(cls, message: str = "会员已过期") -> "BusinessException":
        return cls(ErrorCode.MEMBERSHIP_EXPIRED, message, status.HTTP_403_FORBIDDEN)
    
    @classmethod
    def insufficient_permission(cls, message: str = "会员权限不足") -> "BusinessException":
        return cls(ErrorCode.INSUFFICIENT_PERMISSION, message, status.HTTP_403_FORBIDDEN)
    
    @classmethod
    def order_not_found(cls, message: str = "订单不存在") -> "BusinessException":
        return cls(ErrorCode.ORDER_NOT_FOUND, message, status.HTTP_404_NOT_FOUND)
    
    @classmethod
    def prediction_limit_exceeded(cls, message: str = "预测次数已达上限") -> "BusinessException":
        return cls(ErrorCode.PREDICTION_LIMIT_EXCEEDED, message, status.HTTP_429_TOO_MANY_REQUESTS)
    
    @classmethod
    def algorithm_not_found(cls, message: str = "算法不存在") -> "BusinessException":
        return cls(ErrorCode.ALGORITHM_NOT_FOUND, message, status.HTTP_404_NOT_FOUND)
    
    @classmethod
    def data_not_found(cls, message: str = "数据不存在") -> "BusinessException":
        return cls(ErrorCode.DATA_NOT_FOUND, message, status.HTTP_404_NOT_FOUND)
    
    @classmethod
    def validation_error(cls, message: str = "参数验证失败", details: Dict[str, Any] = None) -> "BusinessException":
        return cls(ErrorCode.VALIDATION_ERROR, message, status.HTTP_422_UNPROCESSABLE_ENTITY, details)


def create_error_response(
    error_code: str,
    message: str,
    status_code: int = status.HTTP_400_BAD_REQUEST,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """创建标准错误响应"""
    content = {
        "success": False,
        "error_code": error_code,
        "message": message,
        "code": status_code,
        "data": None
    }
    
    if details:
        content["details"] = details
    
    return JSONResponse(
        status_code=status_code,
        content=content
    )


def create_success_response(
    data: Any = None,
    message: str = "操作成功",
    status_code: int = status.HTTP_200_OK
) -> Dict[str, Any]:
    """创建标准成功响应"""
    return {
        "success": True,
        "error_code": None,
        "message": message,
        "data": data,
        "code": status_code
    }
