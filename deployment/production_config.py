#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8预测系统生产环境配置
Happy8 Prediction System Production Configuration
"""

import os
import logging
from pathlib import Path

class ProductionConfig:
    """生产环境配置"""
    
    # 应用配置
    APP_NAME = "快乐8智能预测系统"
    APP_VERSION = "1.0.0"
    DEBUG = False
    
    # 服务器配置
    HOST = "0.0.0.0"
    PORT = 8501
    MAX_WORKERS = 4
    
    # 数据配置
    DATA_DIR = Path("data")
    DATA_FILE = DATA_DIR / "happy8_results.csv"
    BACKUP_DIR = DATA_DIR / "backups"
    MAX_DATA_PERIODS = 2000
    AUTO_UPDATE_INTERVAL = 300  # 5分钟
    
    # 缓存配置
    CACHE_ENABLED = True
    CACHE_TTL = 300  # 5分钟
    CACHE_MAX_SIZE = 1000
    
    # 性能配置
    PARALLEL_PROCESSING = True
    GPU_ENABLED = False
    MEMORY_LIMIT = "2GB"
    
    # 安全配置
    RATE_LIMIT = 100  # 每分钟请求数
    AUTH_REQUIRED = False
    ALLOWED_HOSTS = ["*"]
    
    # 日志配置
    LOG_LEVEL = logging.INFO
    LOG_FILE = "logs/happy8_system.log"
    LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # 预测配置
    DEFAULT_PREDICTION_METHOD = "ensemble"
    DEFAULT_ANALYSIS_PERIODS = 300
    DEFAULT_PREDICTION_COUNT = 30
    
    # 数据源配置
    DATA_SOURCES = [
        "lottery_gov",  # 官方数据源
        "zhcw",        # 中彩网
        "500wan"       # 500彩票网
    ]
    
    # 监控配置
    MONITORING_ENABLED = True
    HEALTH_CHECK_INTERVAL = 60  # 1分钟
    PERFORMANCE_TRACKING = True
    
    @classmethod
    def init_directories(cls):
        """初始化目录结构"""
        directories = [
            cls.DATA_DIR,
            cls.BACKUP_DIR,
            Path("logs"),
            Path("cache"),
            Path("models")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def setup_logging(cls):
        """设置日志"""
        cls.init_directories()
        
        # 创建日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 文件处理器
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            cls.LOG_FILE,
            maxBytes=cls.LOG_MAX_SIZE,
            backupCount=cls.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(cls.LOG_LEVEL)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(cls.LOG_LEVEL)
        
        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(cls.LOG_LEVEL)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        return root_logger
    
    @classmethod
    def get_database_url(cls):
        """获取数据库URL（如果使用数据库）"""
        return os.getenv("DATABASE_URL", f"sqlite:///{cls.DATA_DIR}/happy8.db")
    
    @classmethod
    def get_redis_url(cls):
        """获取Redis URL（如果使用Redis缓存）"""
        return os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    @classmethod
    def validate_config(cls):
        """验证配置"""
        errors = []
        
        # 检查端口
        if not (1 <= cls.PORT <= 65535):
            errors.append(f"无效端口号: {cls.PORT}")
        
        # 检查工作进程数
        if cls.MAX_WORKERS < 1:
            errors.append(f"工作进程数必须大于0: {cls.MAX_WORKERS}")
        
        # 检查数据目录
        if not cls.DATA_DIR.exists():
            try:
                cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"无法创建数据目录: {e}")
        
        # 检查日志目录
        log_dir = Path(cls.LOG_FILE).parent
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"无法创建日志目录: {e}")
        
        if errors:
            raise ValueError("配置验证失败:\n" + "\n".join(errors))
        
        return True

class DevelopmentConfig(ProductionConfig):
    """开发环境配置"""
    DEBUG = True
    LOG_LEVEL = logging.DEBUG
    CACHE_TTL = 60  # 1分钟
    AUTO_UPDATE_INTERVAL = 60  # 1分钟

class TestingConfig(ProductionConfig):
    """测试环境配置"""
    DEBUG = True
    DATA_DIR = Path("test_data")
    LOG_LEVEL = logging.DEBUG
    CACHE_ENABLED = False
    AUTO_UPDATE_INTERVAL = 30  # 30秒

# 根据环境变量选择配置
def get_config():
    """获取当前环境配置"""
    env = os.getenv("HAPPY8_ENV", "production").lower()
    
    if env == "development":
        return DevelopmentConfig
    elif env == "testing":
        return TestingConfig
    else:
        return ProductionConfig

# 当前配置
Config = get_config()
