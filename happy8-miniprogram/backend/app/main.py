"""应用主程序"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.database import init_database, close_database
from app.core.cache import init_redis, close_redis
from app.core.error_handlers import register_exception_handlers
from app.core.middleware import register_middleware
from app.core.auth import init_token_blacklist


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    setup_logging()
    init_database()
    init_redis()
    init_token_blacklist()

    yield

    # 关闭时清理
    close_database()
    await close_redis()


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    
    app = FastAPI(
        title="Happy8小程序API",
        description="快乐8智能预测微信小程序后端服务",
        version="1.0.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan
    )
    
    # 注册CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应该限制具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册自定义中间件
    register_middleware(app)
    
    # 注册异常处理器
    register_exception_handlers(app)
    
    # 注册路由
    register_routers(app)
    
    return app


def register_routers(app: FastAPI):
    """注册所有路由"""
    
    # 健康检查路由
    @app.get("/health")
    async def health_check():
        """健康检查接口"""
        from app.core.database import DatabaseHealthCheck
        from app.core.cache import HealthCheck
        
        db_status = await DatabaseHealthCheck.check_async_connection()
        redis_status = await HealthCheck.check_connection()
        
        return {
            "status": "healthy" if db_status and redis_status else "unhealthy",
            "database": "connected" if db_status else "disconnected",
            "redis": "connected" if redis_status else "disconnected",
            "version": "1.0.0"
        }
    
    # 根路径
    @app.get("/")
    async def root():
        """根路径"""
        return {
            "message": "Happy8小程序API服务",
            "version": "1.0.0",
            "docs": "/docs" if settings.DEBUG else "文档已禁用"
        }
    
    # 注册业务路由
    from app.api.v1.router import api_router
    app.include_router(api_router, prefix="/api/v1")


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD and settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )