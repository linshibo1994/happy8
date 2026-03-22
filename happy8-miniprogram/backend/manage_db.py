#!/usr/bin/env python3
"""
数据库迁移管理脚本

使用方法:
    python manage_db.py init     # 初始化迁移环境
    python manage_db.py migrate  # 创建新迁移
    python manage_db.py upgrade  # 升级数据库
    python manage_db.py downgrade # 降级数据库
    python manage_db.py reset    # 重置数据库
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加app目录到Python路径
sys.path.append(str(Path(__file__).parent))

from app.core.config import settings


def run_command(command: list, description: str):
    """运行命令并处理错误"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        print(f"✅ {description}成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description}失败:")
        print(f"错误代码: {e.returncode}")
        if e.stdout:
            print(f"输出: {e.stdout}")
        if e.stderr:
            print(f"错误: {e.stderr}")
        return False


def init_migrations():
    """初始化迁移环境"""
    if Path("alembic").exists():
        print("⚠️ Alembic已经初始化，跳过初始化步骤")
        return True
    
    return run_command(
        ["alembic", "init", "alembic"],
        "初始化Alembic迁移环境"
    )


def create_migration(message: str = None):
    """创建新的迁移文件"""
    if not message:
        message = input("请输入迁移描述: ")
    
    if not message:
        message = "Auto migration"
    
    return run_command(
        ["alembic", "revision", "--autogenerate", "-m", message],
        f"创建迁移: {message}"
    )


def upgrade_database(revision: str = "head"):
    """升级数据库到指定版本"""
    return run_command(
        ["alembic", "upgrade", revision],
        f"升级数据库到 {revision}"
    )


def downgrade_database(revision: str = "-1"):
    """降级数据库到指定版本"""
    return run_command(
        ["alembic", "downgrade", revision],
        f"降级数据库到 {revision}"
    )


def show_current_revision():
    """显示当前数据库版本"""
    return run_command(
        ["alembic", "current"],
        "查看当前数据库版本"
    )


def show_history():
    """显示迁移历史"""
    return run_command(
        ["alembic", "history"],
        "查看迁移历史"
    )


def reset_database():
    """重置数据库（危险操作）"""
    confirm = input("⚠️ 这将删除所有数据，确定要继续吗？(yes/no): ")
    if confirm.lower() != "yes":
        print("操作已取消")
        return False
    
    # 降级到base
    if not run_command(["alembic", "downgrade", "base"], "降级到base"):
        return False
    
    # 升级到最新版本
    return run_command(["alembic", "upgrade", "head"], "升级到最新版本")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python manage_db.py init              # 初始化迁移环境")
        print("  python manage_db.py migrate [message] # 创建新迁移")
        print("  python manage_db.py upgrade [revision] # 升级数据库")
        print("  python manage_db.py downgrade [revision] # 降级数据库")
        print("  python manage_db.py current           # 查看当前版本")
        print("  python manage_db.py history           # 查看迁移历史")
        print("  python manage_db.py reset             # 重置数据库")
        return
    
    command = sys.argv[1]
    
    # 检查数据库连接
    print(f"🔗 数据库连接: {settings.DATABASE_URL}")
    
    if command == "init":
        init_migrations()
    elif command == "migrate":
        message = sys.argv[2] if len(sys.argv) > 2 else None
        create_migration(message)
    elif command == "upgrade":
        revision = sys.argv[2] if len(sys.argv) > 2 else "head"
        upgrade_database(revision)
    elif command == "downgrade":
        revision = sys.argv[2] if len(sys.argv) > 2 else "-1"
        downgrade_database(revision)
    elif command == "current":
        show_current_revision()
    elif command == "history":
        show_history()
    elif command == "reset":
        reset_database()
    else:
        print(f"❌ 未知命令: {command}")


if __name__ == "__main__":
    main()