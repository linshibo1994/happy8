"""日志配置模块"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any

from app.core.config import settings


def setup_logging() -> None:
    """配置日志系统"""
    
    # 确保日志目录存在
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": (
                    "%(asctime)s %(name)s %(levelname)s %(filename)s "
                    "%(lineno)d %(funcName)s %(message)s"
                )
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "default",
                "stream": sys.stdout,
            },
            "file_info": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": "logs/info.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8",
            },
            "file_error": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": "logs/error.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8",
            },
            "file_debug": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": "logs/debug.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 3,
                "encoding": "utf8",
            },
        },
        "loggers": {
            # 应用主日志器
            "app": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console", "file_info", "file_error"],
                "propagate": False,
            },
            # API请求日志器
            "app.api": {
                "level": "INFO",
                "handlers": ["console", "file_info"],
                "propagate": False,
            },
            # 业务逻辑日志器
            "app.service": {
                "level": "INFO",
                "handlers": ["console", "file_info", "file_error"],
                "propagate": False,
            },
            # 数据库日志器
            "app.database": {
                "level": "WARNING",
                "handlers": ["file_info", "file_error"],
                "propagate": False,
            },
            # 预测算法日志器
            "app.prediction": {
                "level": "INFO",
                "handlers": ["console", "file_info", "file_error"],
                "propagate": False,
            },
            # 支付日志器
            "app.payment": {
                "level": "INFO",
                "handlers": ["file_info", "file_error"],
                "propagate": False,
            },
            # 微信API日志器
            "app.wechat": {
                "level": "INFO",
                "handlers": ["file_info", "file_error"],
                "propagate": False,
            },
            # 第三方库日志器
            "sqlalchemy.engine": {
                "level": "WARNING",
                "handlers": ["file_debug"] if settings.DEBUG else [],
                "propagate": False,
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["console", "file_error"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["file_info"],
                "propagate": False,
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["console"],
        },
    }
    
    # 开发环境增加调试日志
    if settings.DEBUG:
        logging_config["loggers"]["app"]["level"] = "DEBUG"
        logging_config["loggers"]["app"]["handlers"].append("file_debug")
    
    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志器"""
    return logging.getLogger(name)


# 预定义日志器
app_logger = get_logger("app")
api_logger = get_logger("app.api")
service_logger = get_logger("app.service")
database_logger = get_logger("app.database")
prediction_logger = get_logger("app.prediction")
payment_logger = get_logger("app.payment")
wechat_logger = get_logger("app.wechat")


class LoggerMixin:
    """日志器混入类"""
    
    @property
    def logger(self) -> logging.Logger:
        """获取当前类的日志器"""
        return get_logger(f"app.{self.__class__.__module__.split('.')[-1]}")


def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None):
    """记录函数调用日志"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger("app.service")
            logger.info(f"调用函数: {func_name}, 参数: args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"函数执行成功: {func_name}")
                return result
            except Exception as e:
                logger.error(f"函数执行失败: {func_name}, 错误: {str(e)}")
                raise
        return wrapper
    return decorator


def log_api_request(request_id: str, method: str, path: str, user_id: int = None):
    """记录API请求日志"""
    api_logger.info(
        f"API请求 - ID: {request_id}, 方法: {method}, 路径: {path}, 用户: {user_id}"
    )


def log_api_response(request_id: str, status_code: int, execution_time: float):
    """记录API响应日志"""
    api_logger.info(
        f"API响应 - ID: {request_id}, 状态码: {status_code}, 执行时间: {execution_time:.3f}s"
    )


def log_business_event(event_type: str, user_id: int, details: dict = None):
    """记录业务事件日志"""
    service_logger.info(
        f"业务事件 - 类型: {event_type}, 用户: {user_id}, 详情: {details}"
    )


def log_prediction_event(
    user_id: int,
    algorithm: str,
    target_issue: str,
    success: bool,
    execution_time: float = None,
    error: str = None
):
    """记录预测事件日志"""
    if success:
        prediction_logger.info(
            f"预测成功 - 用户: {user_id}, 算法: {algorithm}, "
            f"期号: {target_issue}, 执行时间: {execution_time:.3f}s"
        )
    else:
        prediction_logger.error(
            f"预测失败 - 用户: {user_id}, 算法: {algorithm}, "
            f"期号: {target_issue}, 错误: {error}"
        )


def log_payment_event(
    user_id: int,
    order_no: str,
    amount: int,
    event_type: str,
    details: dict = None
):
    """记录支付事件日志"""
    payment_logger.info(
        f"支付事件 - 用户: {user_id}, 订单: {order_no}, "
        f"金额: {amount}, 事件: {event_type}, 详情: {details}"
    )