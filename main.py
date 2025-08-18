#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8智能预测系统 - 主启动文件
Happy8 Prediction System - Main Entry Point

统一的系统入口，支持多种启动方式

作者: CodeBuddy
版本: v1.0
创建时间: 2025-08-18
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def start_web_app():
    """启动Web应用"""
    print("🚀 启动快乐8预测系统Web界面...")
    print("访问地址: http://localhost:8501")
    print("按 Ctrl+C 停止服务")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/happy8_app.py",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Web服务已停止")
    except Exception as e:
        print(f"❌ Web服务启动失败: {e}")

def start_cli():
    """启动命令行界面"""
    print("🖥️ 启动快乐8预测系统命令行界面...")
    
    try:
        subprocess.run([sys.executable, "scripts/start.py", "cli"])
    except Exception as e:
        print(f"❌ 命令行界面启动失败: {e}")

def run_demo():
    """运行演示"""
    print("🎯 运行快乐8预测系统演示...")
    
    try:
        subprocess.run([sys.executable, "scripts/demo.py"])
    except Exception as e:
        print(f"❌ 演示运行失败: {e}")

def deploy_system():
    """部署系统"""
    print("📦 部署快乐8预测系统...")
    
    try:
        subprocess.run([sys.executable, "deployment/deploy.py"])
    except Exception as e:
        print(f"❌ 系统部署失败: {e}")

def show_help():
    """显示帮助信息"""
    print("""
🎯 快乐8智能预测系统

使用方法:
  python main.py [命令]

可用命令:
  web      启动Web界面 (默认)
  cli      启动命令行界面  
  demo     运行系统演示
  deploy   部署系统
  help     显示此帮助信息

示例:
  python main.py          # 启动Web界面
  python main.py web      # 启动Web界面
  python main.py cli      # 启动命令行界面
  python main.py demo     # 运行演示
  python main.py deploy   # 部署系统

更多信息请查看 README.md
    """)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="快乐8智能预测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'command', 
        nargs='?', 
        default='web',
        choices=['web', 'cli', 'demo', 'deploy', 'help'],
        help='要执行的命令 (默认: web)'
    )
    
    args = parser.parse_args()
    
    # 检查项目结构
    required_dirs = ['src', 'data', 'deployment', 'scripts']
    missing_dirs = [d for d in required_dirs if not Path(d).exists()]
    
    if missing_dirs:
        print(f"❌ 缺少必要目录: {', '.join(missing_dirs)}")
        print("请确保项目结构完整")
        return
    
    # 执行对应命令
    if args.command == 'web':
        start_web_app()
    elif args.command == 'cli':
        start_cli()
    elif args.command == 'demo':
        run_demo()
    elif args.command == 'deploy':
        deploy_system()
    elif args.command == 'help':
        show_help()
    else:
        show_help()

if __name__ == "__main__":
    main()
