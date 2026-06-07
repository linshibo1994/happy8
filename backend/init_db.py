"""数据库初始化脚本"""

import sys
from pathlib import Path

# 添加app目录到Python路径
sys.path.append(str(Path(__file__).parent))

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from db_manager import DatabaseManager

# 设置日志
setup_logging()
logger = get_logger("app.database")


def initialize_database():
    """初始化数据库"""
    logger.info("开始初始化数据库...")
    
    db_manager = DatabaseManager()
    
    # 1. 检查数据库连接
    logger.info("1. 检查数据库连接...")
    if not db_manager.check_database_connection():
        logger.info("数据库连接失败，尝试创建数据库...")
        if not db_manager.create_database_if_not_exists():
            logger.error("创建数据库失败")
            return False
    
    # 2. 初始化迁移环境
    logger.info("2. 初始化迁移环境...")
    if not db_manager.init_migrations():
        logger.error("初始化迁移环境失败")
        return False
    
    # 3. 执行数据库迁移
    logger.info("3. 执行数据库迁移...")
    if not db_manager.upgrade_database():
        logger.error("数据库迁移失败")
        return False
    
    # 4. 验证迁移
    logger.info("4. 验证迁移...")
    if not db_manager.validate_migrations():
        logger.error("迁移验证失败")
        return False
    
    # 5. 显示当前版本
    current_rev = db_manager.get_current_revision()
    logger.info(f"5. 当前数据库版本: {current_rev}")
    
    logger.info("数据库初始化完成！")
    return True


def insert_initial_data():
    """插入初始数据"""
    logger.info("开始插入初始数据...")
    
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from app.models.membership import MembershipPlan
        from app.models.prediction import AlgorithmConfig
        from datetime import datetime
        
        engine = create_engine(str(settings.DATABASE_URL))
        SessionLocal = sessionmaker(bind=engine)
        
        with SessionLocal() as session:
            # 插入会员套餐
            logger.info("插入默认会员套餐...")
            
            # 免费套餐
            free_plan = MembershipPlan(
                name="免费会员",
                level="free",
                duration_days=365,  # 免费一年
                price=0,
                features='["基础预测", "历史查询", "简单统计"]',
                max_predictions_per_day=5,
                available_algorithms='["frequency", "hot_cold", "missing"]',
                is_active=True,
                sort_order=1,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # VIP套餐
            vip_plan = MembershipPlan(
                name="VIP会员",
                level="vip",
                duration_days=30,
                price=2980,  # 29.80元
                original_price=3980,
                features='["所有预测算法", "无限历史查询", "高级统计", "趋势分析"]',
                max_predictions_per_day=50,
                available_algorithms='["frequency", "hot_cold", "missing", "markov", "markov_2nd", "ensemble", "clustering", "monte_carlo", "ml_ensemble"]',
                is_active=True,
                sort_order=2,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # 尊享套餐
            premium_plan = MembershipPlan(
                name="尊享会员",
                level="premium",
                duration_days=365,
                price=19800,  # 198元
                original_price=29800,
                features='["所有预测算法", "无限预测次数", "专属算法", "API接口", "数据导出"]',
                max_predictions_per_day=None,  # 无限制
                available_algorithms='["all"]',
                is_active=True,
                sort_order=3,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            session.add_all([free_plan, vip_plan, premium_plan])
            
            # 插入算法配置
            logger.info("插入默认算法配置...")
            
            # 使用统一配置生成完整17种算法
            from app.utils.algorithm_config_updater import update_algorithm_configs
            algorithms = update_algorithm_configs()
            
            for algo_data in algorithms:
                algorithm = AlgorithmConfig(
                    algorithm_name=algo_data["algorithm_name"],
                    display_name=algo_data["display_name"],
                    description=algo_data["description"],
                    default_params=algo_data["default_params"],
                    required_level=algo_data["required_level"],
                    is_active=algo_data.get("is_active", True),
                    sort_order=algo_data["sort_order"],
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                session.add(algorithm)
            
            # 提交事务
            session.commit()
            logger.info("初始数据插入完成")
            return True
            
    except Exception as e:
        logger.error(f"插入初始数据失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 Happy8小程序数据库初始化")
    print(f"数据库URL: {settings.DATABASE_URL}")
    print()
    
    # 初始化数据库
    if not initialize_database():
        print("❌ 数据库初始化失败")
        sys.exit(1)
    
    # 插入初始数据
    if not insert_initial_data():
        print("❌ 初始数据插入失败")
        sys.exit(1)
    
    print("✅ 数据库初始化完成！")
    print()
    print("接下来可以启动API服务:")
    print("  python -m uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
