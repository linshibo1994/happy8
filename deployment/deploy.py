#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8预测系统部署脚本
Happy8 Prediction System Deployment Script
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path

class Happy8Deployer:
    """快乐8系统部署器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent  # 项目根目录
        self.deployment_dir = Path(__file__).parent       # 部署目录
        self.config = self._load_config()

        # 添加src目录到Python路径
        sys.path.insert(0, str(self.project_root / "src"))
    
    def _load_config(self):
        """加载部署配置"""
        config_file = self.deployment_dir / "deploy_config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return self._create_default_config()
    
    def _create_default_config(self):
        """创建默认配置"""
        config = {
            "app": {
                "name": "快乐8智能预测系统",
                "version": "1.0.0",
                "port": 8501,
                "host": "0.0.0.0"
            },
            "data": {
                "auto_update": True,
                "update_interval": 300,  # 5分钟
                "backup_enabled": True,
                "max_periods": 1000
            },
            "performance": {
                "cache_enabled": True,
                "parallel_processing": True,
                "gpu_enabled": False
            },
            "security": {
                "rate_limit": 100,  # 每分钟请求数
                "auth_required": False
            }
        }
        
        # 保存配置文件
        config_file = self.project_root / "deploy_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return config
    
    def check_environment(self):
        """检查部署环境"""
        print("🔍 检查部署环境...")
        
        # 检查Python版本
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            print("❌ Python版本过低，需要Python 3.8+")
            return False
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}")
        
        # 检查依赖包
        try:
            import pandas, numpy, sklearn, streamlit, requests
            print("✅ 核心依赖包已安装")
        except ImportError as e:
            print(f"❌ 缺少依赖包: {e}")
            return False
        
        # 检查数据文件
        data_file = self.project_root / "data" / "happy8_results.csv"
        if data_file.exists():
            print("✅ 数据文件存在")
        else:
            print("⚠️ 数据文件不存在，将自动爬取")
        
        # 检查核心模块
        try:
            from happy8_analyzer import Happy8Analyzer
            from happy8_app import main
            print("✅ 核心模块导入成功")
        except ImportError as e:
            print(f"❌ 核心模块导入失败: {e}")
            return False
        
        return True
    
    def install_dependencies(self):
        """安装依赖包"""
        print("📦 安装依赖包...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print("❌ requirements.txt文件不存在")
            return False
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            print("✅ 依赖包安装完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖包安装失败: {e}")
            return False
    
    def initialize_data(self):
        """初始化数据"""
        print("📊 初始化数据...")
        
        try:
            from happy8_analyzer import Happy8Analyzer
            analyzer = Happy8Analyzer()
            
            # 检查现有数据
            data = analyzer.load_data()
            if len(data) < 10:
                print("数据量不足，开始爬取最新数据...")
                new_data = analyzer.crawl_latest_data(limit=50)
                print(f"✅ 成功爬取 {len(new_data)} 期数据")
            else:
                print(f"✅ 数据充足，共 {len(data)} 期")
            
            return True
        except Exception as e:
            print(f"❌ 数据初始化失败: {e}")
            return False
    
    def run_tests(self):
        """运行测试"""
        print("🧪 运行系统测试...")
        
        try:
            result = subprocess.run([
                sys.executable, "test_system.py"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print("✅ 系统测试通过")
                return True
            else:
                print(f"❌ 系统测试失败: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ 测试执行失败: {e}")
            return False
    
    def start_web_app(self):
        """启动Web应用"""
        print("🚀 启动Web应用...")
        
        app_config = self.config["app"]
        
        try:
            # 构建启动命令
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(self.project_root / "src" / "happy8_app.py"),
                "--server.port", str(app_config["port"]),
                "--server.address", app_config["host"],
                "--server.headless", "true"
            ]
            
            print(f"启动命令: {' '.join(cmd)}")
            print(f"访问地址: http://{app_config['host']}:{app_config['port']}")
            
            # 启动应用
            subprocess.run(cmd, cwd=self.project_root)
            
        except KeyboardInterrupt:
            print("\n👋 应用已停止")
        except Exception as e:
            print(f"❌ 应用启动失败: {e}")
    
    def deploy(self, skip_tests=False):
        """完整部署流程"""
        print("🎯 开始部署快乐8预测系统...")
        print("=" * 50)
        
        # 1. 检查环境
        if not self.check_environment():
            print("❌ 环境检查失败，部署终止")
            return False
        
        # 2. 安装依赖
        if not self.install_dependencies():
            print("❌ 依赖安装失败，部署终止")
            return False
        
        # 3. 初始化数据
        if not self.initialize_data():
            print("❌ 数据初始化失败，部署终止")
            return False
        
        # 4. 运行测试
        if not skip_tests and not self.run_tests():
            print("❌ 系统测试失败，部署终止")
            return False
        
        print("=" * 50)
        print("🎉 部署完成！系统已准备就绪")
        print("=" * 50)
        
        # 5. 启动应用
        self.start_web_app()
        
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="快乐8预测系统部署脚本")
    parser.add_argument("--skip-tests", action="store_true", help="跳过系统测试")
    parser.add_argument("--check-only", action="store_true", help="仅检查环境")
    parser.add_argument("--install-deps", action="store_true", help="仅安装依赖")
    
    args = parser.parse_args()
    
    deployer = Happy8Deployer()
    
    if args.check_only:
        deployer.check_environment()
    elif args.install_deps:
        deployer.install_dependencies()
    else:
        deployer.deploy(skip_tests=args.skip_tests)

if __name__ == "__main__":
    main()
