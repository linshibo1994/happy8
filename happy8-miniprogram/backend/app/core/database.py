"""数据库连接管理"""

from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.engine import Engine

from app.core.config import settings
from app.core.logging import database_logger


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self._engine: Optional[Engine] = None
        self._async_engine = None
        self._session_factory: Optional[sessionmaker] = None
        self._async_session_factory = None
    
    def get_engine(self) -> Engine:
        """获取同步数据库引擎"""
        if self._engine is None:
            self._engine = create_engine(
                str(settings.DATABASE_URL),
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW,
                pool_timeout=settings.DB_POOL_TIMEOUT,
                pool_pre_ping=True,  # 连接前检查
                pool_recycle=3600,   # 1小时回收连接
                echo=settings.DEBUG,  # 开发环境显示SQL
            )
            
            # 配置数据库引擎事件
            self._setup_engine_events(self._engine)
            
        return self._engine
    
    def get_async_engine(self):
        """获取异步数据库引擎"""
        if self._async_engine is None:
            # 将同步URL转换为异步URL
            db_url = str(settings.DATABASE_URL)

            # 根据数据库类型替换驱动
            if "mysql+pymysql://" in db_url:
                async_url = db_url.replace("mysql+pymysql://", "mysql+aiomysql://")
            elif db_url.startswith("mysql://"):
                async_url = db_url.replace("mysql://", "mysql+aiomysql://", 1)
            elif db_url.startswith("postgresql://"):
                async_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            else:
                async_url = db_url
            
            self._async_engine = create_async_engine(
                async_url,
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW,
                pool_timeout=settings.DB_POOL_TIMEOUT,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=settings.DEBUG,
            )
            
        return self._async_engine
    
    def get_session_factory(self) -> sessionmaker:
        """获取同步会话工厂"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.get_engine(),
                autoflush=False,
                autocommit=False,
                expire_on_commit=False
            )
        return self._session_factory
    
    def get_async_session_factory(self) -> async_sessionmaker:
        """获取异步会话工厂"""
        if self._async_session_factory is None:
            self._async_session_factory = async_sessionmaker(
                bind=self.get_async_engine(),
                class_=AsyncSession,
                autoflush=False,
                autocommit=False,
                expire_on_commit=False
            )
        return self._async_session_factory
    
    def _setup_engine_events(self, engine: Engine):
        """设置数据库引擎事件监听"""
        
        @event.listens_for(engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            """连接建立时触发"""
            database_logger.debug("数据库连接建立")
            
            # MySQL特定配置
            if "mysql" in str(settings.DATABASE_URL):
                with dbapi_connection.cursor() as cursor:
                    # 设置字符集
                    cursor.execute("SET NAMES utf8mb4")
                    # 设置时区
                    cursor.execute("SET time_zone = '+08:00'")
                    # 设置SQL模式
                    cursor.execute("SET sql_mode = 'STRICT_TRANS_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO'")
        
        @event.listens_for(engine, "close")
        def receive_close(dbapi_connection, connection_record):
            """连接关闭时触发"""
            database_logger.debug("数据库连接关闭")

        # Note: 'invalid' event is removed in SQLAlchemy 2.0, using 'close' and pool events instead
        # @event.listens_for(engine, "invalid")
        # def receive_invalid(dbapi_connection, connection_record, exception):
        #     """连接失效时触发"""
        #     database_logger.warning(f"数据库连接失效: {exception}")
    
    def close(self):
        """关闭数据库连接"""
        if self._engine:
            self._engine.dispose()
            database_logger.info("同步数据库引擎已关闭")
        
        if self._async_engine:
            self._async_engine.dispose()
            database_logger.info("异步数据库引擎已关闭")


# 全局数据库管理器实例
db_manager = DatabaseManager()


def get_db() -> Session:
    """获取数据库会话（同步版本 - 用于依赖注入）"""
    session_factory = db_manager.get_session_factory()
    session = session_factory()
    try:
        yield session
    except Exception as e:
        database_logger.error(f"数据库会话异常: {e}")
        session.rollback()
        raise
    finally:
        session.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话（异步版本 - 用于依赖注入）"""
    async_session_factory = db_manager.get_async_session_factory()
    async with async_session_factory() as session:
        try:
            yield session
        except Exception as e:
            database_logger.error(f"异步数据库会话异常: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话（上下文管理器）"""
    async_session_factory = db_manager.get_async_session_factory()
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            database_logger.error(f"数据库事务异常: {e}")
            await session.rollback()
            raise


class TransactionManager:
    """事务管理器"""
    
    @staticmethod
    @asynccontextmanager
    async def transaction(session: AsyncSession):
        """异步事务上下文管理器"""
        try:
            yield session
            await session.commit()
            database_logger.debug("事务提交成功")
        except Exception as e:
            await session.rollback()
            database_logger.error(f"事务回滚: {e}")
            raise
    
    @staticmethod
    def sync_transaction(session: Session):
        """同步事务上下文管理器"""
        try:
            yield session
            session.commit()
            database_logger.debug("同步事务提交成功")
        except Exception as e:
            session.rollback()
            database_logger.error(f"同步事务回滚: {e}")
            raise


class DatabaseHealthCheck:
    """数据库健康检查"""
    
    @staticmethod
    async def check_async_connection() -> bool:
        """检查异步数据库连接"""
        try:
            async with get_db_session() as session:
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            database_logger.error(f"异步数据库连接检查失败: {e}")
            return False
    
    @staticmethod
    def check_sync_connection() -> bool:
        """检查同步数据库连接"""
        try:
            engine = db_manager.get_engine()
            with engine.connect() as conn:
                result = conn.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            database_logger.error(f"同步数据库连接检查失败: {e}")
            return False
    
    @staticmethod
    async def get_connection_info() -> dict:
        """获取数据库连接信息"""
        try:
            async with get_db_session() as session:
                # MySQL特定查询
                if "mysql" in str(settings.DATABASE_URL):
                    result = await session.execute("""
                        SELECT 
                            VERSION() as version,
                            @@character_set_database as charset,
                            @@time_zone as timezone,
                            CONNECTION_ID() as connection_id
                    """)
                    row = result.first()
                    return {
                        "database_type": "MySQL",
                        "version": row[0],
                        "charset": row[1],
                        "timezone": row[2],
                        "connection_id": row[3]
                    }
                # PostgreSQL特定查询
                elif "postgresql" in str(settings.DATABASE_URL):
                    result = await session.execute("""
                        SELECT 
                            version() as version,
                            current_setting('TimeZone') as timezone,
                            pg_backend_pid() as pid
                    """)
                    row = result.first()
                    return {
                        "database_type": "PostgreSQL",
                        "version": row[0],
                        "timezone": row[1],
                        "backend_pid": row[2]
                    }
                else:
                    return {"database_type": "Unknown"}
        except Exception as e:
            database_logger.error(f"获取数据库连接信息失败: {e}")
            return {"error": str(e)}


# 数据库连接池监控
class ConnectionPoolMonitor:
    """连接池监控"""
    
    @staticmethod
    def get_pool_status() -> dict:
        """获取连接池状态"""
        engine = db_manager.get_engine()
        pool = engine.pool
        
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
        }
    
    @staticmethod
    def log_pool_status():
        """记录连接池状态"""
        status = ConnectionPoolMonitor.get_pool_status()
        database_logger.info(f"连接池状态: {status}")


# 启动时初始化数据库连接
def init_database():
    """初始化数据库连接"""
    try:
        # 初始化同步引擎
        engine = db_manager.get_engine()
        database_logger.info("同步数据库引擎初始化成功")
        
        # 初始化异步引擎
        async_engine = db_manager.get_async_engine()
        database_logger.info("异步数据库引擎初始化成功")
        
        # 检查连接
        if DatabaseHealthCheck.check_sync_connection():
            database_logger.info("数据库连接检查通过")
        else:
            database_logger.error("数据库连接检查失败")
            
    except Exception as e:
        database_logger.error(f"数据库初始化失败: {e}")
        raise


# 关闭时清理数据库连接
def close_database():
    """关闭数据库连接"""
    db_manager.close()