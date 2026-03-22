"""数据库版本管理工具"""

import os
import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

# 添加app目录到Python路径
sys.path.append(str(Path(__file__).parent))

from app.core.config import settings
from app.core.logging import setup_logging, get_logger

# 设置日志
setup_logging()
logger = get_logger("app.database")


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.alembic_cfg = Config("alembic.ini")
        self.alembic_cfg.set_main_option("sqlalchemy.url", str(settings.DATABASE_URL))
        
    def check_database_connection(self) -> bool:
        """检查数据库连接"""
        try:
            engine = create_engine(str(settings.DATABASE_URL))
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("数据库连接成功")
            return True
        except OperationalError as e:
            logger.error(f"数据库连接失败: {e}")
            return False
        except Exception as e:
            logger.error(f"数据库连接检查出错: {e}")
            return False
    
    def create_database_if_not_exists(self) -> bool:
        """创建数据库（如果不存在）"""
        try:
            # 解析数据库URL获取数据库名
            from urllib.parse import urlparse
            parsed_url = urlparse(str(settings.DATABASE_URL))
            
            if parsed_url.scheme.startswith('mysql'):
                # MySQL数据库创建
                db_name = parsed_url.path[1:]  # 去掉开头的'/'
                base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
                
                engine = create_engine(base_url)
                with engine.connect() as conn:
                    # 检查数据库是否存在
                    result = conn.execute(
                        text(f"SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '{db_name}'")
                    )
                    if not result.fetchone():
                        conn.execute(text(f"CREATE DATABASE {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
                        logger.info(f"数据库 {db_name} 创建成功")
                    else:
                        logger.info(f"数据库 {db_name} 已存在")
                        
            elif parsed_url.scheme.startswith('postgresql'):
                # PostgreSQL数据库创建
                db_name = parsed_url.path[1:]
                base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/postgres"
                
                engine = create_engine(base_url)
                with engine.connect() as conn:
                    conn.execute(text("COMMIT"))  # 退出事务
                    result = conn.execute(
                        text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
                    )
                    if not result.fetchone():
                        conn.execute(text(f"CREATE DATABASE {db_name}"))
                        logger.info(f"数据库 {db_name} 创建成功")
                    else:
                        logger.info(f"数据库 {db_name} 已存在")
            
            return True
            
        except Exception as e:
            logger.error(f"创建数据库失败: {e}")
            return False
    
    def init_migrations(self) -> bool:
        """初始化迁移环境"""
        try:
            if not Path("alembic").exists():
                command.init(self.alembic_cfg, "alembic")
                logger.info("Alembic迁移环境初始化成功")
            else:
                logger.info("Alembic迁移环境已存在")
            return True
        except Exception as e:
            logger.error(f"初始化迁移环境失败: {e}")
            return False
    
    def create_migration(self, message: str) -> bool:
        """创建新的迁移文件"""
        try:
            command.revision(self.alembic_cfg, autogenerate=True, message=message)
            logger.info(f"创建迁移成功: {message}")
            return True
        except Exception as e:
            logger.error(f"创建迁移失败: {e}")
            return False
    
    def upgrade_database(self, revision: str = "head") -> bool:
        """升级数据库"""
        try:
            command.upgrade(self.alembic_cfg, revision)
            logger.info(f"数据库升级成功到版本: {revision}")
            return True
        except Exception as e:
            logger.error(f"数据库升级失败: {e}")
            return False
    
    def downgrade_database(self, revision: str) -> bool:
        """降级数据库"""
        try:
            command.downgrade(self.alembic_cfg, revision)
            logger.info(f"数据库降级成功到版本: {revision}")
            return True
        except Exception as e:
            logger.error(f"数据库降级失败: {e}")
            return False
    
    def get_current_revision(self) -> Optional[str]:
        """获取当前数据库版本"""
        try:
            from alembic.runtime.migration import MigrationContext
            from alembic.operations import Operations
            
            engine = create_engine(str(settings.DATABASE_URL))
            with engine.connect() as conn:
                context = MigrationContext.configure(conn)
                current_rev = context.get_current_revision()
                return current_rev
        except Exception as e:
            logger.error(f"获取当前版本失败: {e}")
            return None
    
    def get_migration_history(self) -> List[dict]:
        """获取迁移历史"""
        try:
            from alembic.script import ScriptDirectory
            
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            revisions = []
            
            for revision in script_dir.walk_revisions():
                revisions.append({
                    "revision": revision.revision,
                    "down_revision": revision.down_revision,
                    "branch_labels": revision.branch_labels,
                    "message": revision.doc,
                    "create_date": getattr(revision.module, 'create_date', None)
                })
            
            return revisions
        except Exception as e:
            logger.error(f"获取迁移历史失败: {e}")
            return []
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """备份数据库"""
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"backup_{timestamp}.sql"
            
            from urllib.parse import urlparse
            parsed_url = urlparse(str(settings.DATABASE_URL))
            
            if parsed_url.scheme.startswith('mysql'):
                # MySQL备份
                cmd = (
                    f"mysqldump -h {parsed_url.hostname} -P {parsed_url.port or 3306} "
                    f"-u {parsed_url.username} -p{parsed_url.password} "
                    f"{parsed_url.path[1:]} > {backup_path}"
                )
            elif parsed_url.scheme.startswith('postgresql'):
                # PostgreSQL备份
                cmd = (
                    f"pg_dump -h {parsed_url.hostname} -p {parsed_url.port or 5432} "
                    f"-U {parsed_url.username} {parsed_url.path[1:]} > {backup_path}"
                )
            else:
                logger.error("不支持的数据库类型用于备份")
                return False
            
            os.system(cmd)
            logger.info(f"数据库备份成功: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"数据库备份失败: {e}")
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """恢复数据库"""
        try:
            if not Path(backup_path).exists():
                logger.error(f"备份文件不存在: {backup_path}")
                return False
            
            from urllib.parse import urlparse
            parsed_url = urlparse(str(settings.DATABASE_URL))
            
            if parsed_url.scheme.startswith('mysql'):
                # MySQL恢复
                cmd = (
                    f"mysql -h {parsed_url.hostname} -P {parsed_url.port or 3306} "
                    f"-u {parsed_url.username} -p{parsed_url.password} "
                    f"{parsed_url.path[1:]} < {backup_path}"
                )
            elif parsed_url.scheme.startswith('postgresql'):
                # PostgreSQL恢复
                cmd = (
                    f"psql -h {parsed_url.hostname} -p {parsed_url.port or 5432} "
                    f"-U {parsed_url.username} {parsed_url.path[1:]} < {backup_path}"
                )
            else:
                logger.error("不支持的数据库类型用于恢复")
                return False
            
            os.system(cmd)
            logger.info(f"数据库恢复成功: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"数据库恢复失败: {e}")
            return False
    
    def validate_migrations(self) -> bool:
        """验证迁移文件"""
        try:
            from alembic.script import ScriptDirectory
            
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            
            # 检查迁移文件的完整性
            try:
                for revision in script_dir.walk_revisions():
                    # 确保迁移文件可以正确加载
                    revision.module
                logger.info("迁移文件验证成功")
                return True
            except Exception as e:
                logger.error(f"迁移文件验证失败: {e}")
                return False
                
        except Exception as e:
            logger.error(f"验证迁移失败: {e}")
            return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据库版本管理工具")
    parser.add_argument("command", choices=[
        "init", "create", "upgrade", "downgrade", "current", 
        "history", "backup", "restore", "validate", "check"
    ])
    parser.add_argument("-m", "--message", help="迁移消息")
    parser.add_argument("-r", "--revision", help="目标版本")
    parser.add_argument("-f", "--file", help="备份/恢复文件路径")
    
    args = parser.parse_args()
    
    db_manager = DatabaseManager()
    
    if args.command == "check":
        success = db_manager.check_database_connection()
        sys.exit(0 if success else 1)
    
    elif args.command == "init":
        success = (
            db_manager.create_database_if_not_exists() and
            db_manager.init_migrations()
        )
        sys.exit(0 if success else 1)
    
    elif args.command == "create":
        message = args.message or input("请输入迁移描述: ")
        success = db_manager.create_migration(message)
        sys.exit(0 if success else 1)
    
    elif args.command == "upgrade":
        revision = args.revision or "head"
        success = db_manager.upgrade_database(revision)
        sys.exit(0 if success else 1)
    
    elif args.command == "downgrade":
        revision = args.revision or "-1"
        success = db_manager.downgrade_database(revision)
        sys.exit(0 if success else 1)
    
    elif args.command == "current":
        current_rev = db_manager.get_current_revision()
        if current_rev:
            print(f"当前版本: {current_rev}")
        else:
            print("无法获取当前版本")
        sys.exit(0)
    
    elif args.command == "history":
        history = db_manager.get_migration_history()
        for rev in history:
            print(f"版本: {rev['revision']}")
            print(f"  消息: {rev['message']}")
            print(f"  上一版本: {rev['down_revision']}")
            print(f"  创建时间: {rev['create_date']}")
            print()
    
    elif args.command == "backup":
        backup_path = args.file
        success = db_manager.backup_database(backup_path)
        sys.exit(0 if success else 1)
    
    elif args.command == "restore":
        if not args.file:
            print("错误: 需要指定备份文件路径")
            sys.exit(1)
        success = db_manager.restore_database(args.file)
        sys.exit(0 if success else 1)
    
    elif args.command == "validate":
        success = db_manager.validate_migrations()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()