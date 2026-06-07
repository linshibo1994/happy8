#!/usr/bin/env python3
"""简化测试脚本"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

try:
    print("✅ 测试开始...")
    
    # 测试基础配置导入
    print("1. 测试配置模块...")
    from app.core.config import settings
    print(f"   数据库URL: {settings.DATABASE_URL}")
    
    # 测试异常处理模块
    print("2. 测试异常处理模块...")
    from app.core.exceptions import BusinessException, ErrorCode
    
    # 测试数据模型
    print("3. 测试数据模型...")
    from app.models.user import User
    from app.models.membership import Membership
    from app.models.prediction import LotteryResult
    
    # 测试API模型
    print("4. 测试API模型...")
    from app.api.schemas.user_schemas import WeChatLoginRequest
    
    print("✅ 基础模块测试通过！")
    print("\n接下来需要：")
    print("1. 安装依赖: pip install -r requirements.txt")
    print("2. 配置数据库连接")
    print("3. 运行数据库初始化: python init_db.py")
    print("4. 启动API服务: python start.py")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请检查模块依赖和路径设置")
except Exception as e:
    print(f"❌ 其他错误: {e}")
    sys.exit(1)

print("\n🎉 应用结构验证完成！")