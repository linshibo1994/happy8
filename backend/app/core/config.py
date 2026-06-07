"""系统配置"""

import os
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置类"""
    
    # 数据库配置
    DATABASE_URL: str = "mysql://happy8_user:happy8_pass_2025@localhost:3306/happy8_miniprogram"
    
    # Redis配置
    REDIS_URL: str = "redis://:happy8_redis_2025@localhost:6379/0"
    
    # JWT配置
    SECRET_KEY: str = "happy8_jwt_secret_key_2025_very_long_and_secure"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 7天
    
    # 应用配置
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # 微信小程序配置
    WECHAT_APP_ID: str = ""
    WECHAT_APP_SECRET: str = ""
    
    # 微信支付配置
    WECHAT_PAY_MCHID: str = ""
    WECHAT_PAY_PRIVATE_KEY_PATH: str = ""
    WECHAT_PAY_CERT_SERIAL: str = ""
    WECHAT_PAY_APIV3_KEY: str = ""
    WECHAT_PAY_NOTIFY_URL: str = ""
    
    # 数据库连接池配置
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    
    # Redis连接池配置
    REDIS_POOL_SIZE: int = 10
    REDIS_TIMEOUT: int = 5
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 预测算法配置
    PREDICTION_CACHE_TTL: int = 3600  # 1小时
    PREDICTION_TIMEOUT: int = 30  # 30秒
    MAX_PREDICTIONS_PER_DAY_FREE: int = 5
    MAX_PREDICTIONS_PER_DAY_VIP: int = 50
    
    # 文件上传配置
    UPLOAD_PATH: str = "/app/uploads"
    MAX_FILE_SIZE: int = 10485760  # 10MB
    
    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """验证数据库URL"""
        if not v:
            raise ValueError("DATABASE_URL is required")
        return v
    
    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """验证JWT密钥"""
        if not v or len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """获取配置实例（单例模式）"""
    return Settings()


# 全局配置实例
settings = get_settings()
