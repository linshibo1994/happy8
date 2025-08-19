#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8智能预测系统 - 主启动文件
Happy8 Prediction System - Main Entry Point

统一的系统入口，支持多种启动方式：
- Web界面模式: python main.py web
- 命令行预测: python main.py predict
- 批量测试: python main.py batch-test
- 性能测试: python main.py performance-test

作者: linshibo
开发者: linshibo
版本: v1.4.0
创建时间: 2025-08-18
最后更新: 2025-08-19
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
    print("使用方法:")
    print("  python -c \"from src.happy8_analyzer import Happy8Analyzer; analyzer = Happy8Analyzer(); print('系统已初始化，可以开始使用')\"")
    print("或者直接在Python中导入使用:")
    print("  from src.happy8_analyzer import Happy8Analyzer")

def run_demo():
    """运行演示"""
    print("🎯 运行快乐8预测系统演示...")

    try:
        # 直接运行演示代码
        from src.happy8_analyzer import Happy8Analyzer

        print("初始化分析器...")
        analyzer = Happy8Analyzer()

        print("加载数据...")
        data = analyzer.load_data()
        print(f"成功加载 {len(data)} 期历史数据")

        print("执行预测演示...")
        result = analyzer.predict_with_smart_mode('2025999', 30, 5, 'frequency')
        numbers = result['prediction_result'].predicted_numbers
        print(f"预测结果: {numbers}")

        print("✅ 演示完成！")

    except Exception as e:
        print(f"❌ 演示运行失败: {e}")
        import traceback
        traceback.print_exc()

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
🎯 快乐8智能预测系统 v1.4.0
作者: linshibo

使用方法:
  python main.py [命令]

可用命令:
  web      启动Web界面 (默认)
  cli      显示命令行使用说明
  demo     运行系统演示
  deploy   部署系统
  help     显示此帮助信息

示例:
  python main.py          # 启动Web界面
  python main.py web      # 启动Web界面
  python main.py cli      # 显示命令行使用说明
  python main.py demo     # 运行演示
  python main.py deploy   # 部署系统

🌟 系统特性:
  - 17种预测算法 (统计学+机器学习+深度学习+贝叶斯推理)
  - 智能模式切换 (历史验证+未来预测)
  - Web界面 + 命令行双模式
  - 完整的质量控制体系

📚 更多信息请查看:
  - README.md (项目概述)
  - docs/用户使用指南.md (详细教程)
  - docs/部署指南.md (部署方案)
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
    required_dirs = ['src', 'data']
    missing_dirs = [d for d in required_dirs if not Path(d).exists()]

    if missing_dirs:
        print(f"❌ 缺少必要目录: {', '.join(missing_dirs)}")
        print("请确保项目结构完整")
        return

    # 检查关键文件
    required_files = ['src/happy8_analyzer.py', 'src/happy8_app.py']
    missing_files = [f for f in required_files if not Path(f).exists()]

    if missing_files:
        print(f"❌ 缺少关键文件: {', '.join(missing_files)}")
        print("请确保项目文件完整")
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
