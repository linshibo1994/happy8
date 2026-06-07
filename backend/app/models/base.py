"""数据库基础配置"""

from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

# 创建基础模型类
Base = declarative_base()

# 设置命名约定，确保索引和约束有一致的命名
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

# 将命名约定应用到Base
Base.metadata = metadata
