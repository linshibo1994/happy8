"""Alembic环境配置文件"""

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.ext.asyncio import AsyncEngine

# 添加app目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import settings  # noqa
from app.models.base import Base  # noqa

# Alembic配置对象
config = context.config

# 设置数据库URL
config.set_main_option("sqlalchemy.url", str(settings.DATABASE_URL))

# 解释配置文件的日志记录
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 添加模型的元数据对象用于自动生成迁移
target_metadata = Base.metadata

# 其他配置值可以从配置中获取:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """在'离线'模式下运行迁移。
    
    这个配置仅使用URL配置数据库，
    而不需要Engine，虽然在这里需要一个Engine，
    但它没有关联的DBAPI连接。只是
    将SQL发出到文件中。
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """运行迁移的核心逻辑"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """在'在线'模式下运行迁移。
    
    在这种情况下，我们需要创建一个Engine
    并将连接与上下文关联。
    """
    
    # 创建异步引擎配置
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = str(settings.DATABASE_URL)
    
    # 如果是异步数据库URL，使用异步引擎
    if str(settings.DATABASE_URL).startswith(("postgresql+asyncpg://", "mysql+aiomysql://")):
        connectable = AsyncEngine(
            engine_from_config(
                configuration,
                prefix="sqlalchemy.",
                poolclass=pool.NullPool,
                future=True,
            )
        )

        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)

        await connectable.dispose()
    else:
        # 同步数据库连接
        connectable = engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

        with connectable.connect() as connection:
            do_run_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())