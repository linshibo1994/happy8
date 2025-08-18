#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8智能预测系统 - 启动脚本
Happy8 Prediction System - Startup Script

提供多种启动方式：命令行、Web界面等

作者: CodeBuddy
版本: v1.0
创建时间: 2025-08-17
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_dependencies():
    """检查依赖包"""
    try:
        import pandas
        import numpy
        import sklearn
        print("✓ 核心依赖包检查通过")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖包: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def start_web_interface():
    """启动Web界面"""
    print("启动快乐8智能预测系统Web界面...")
    
    # 检查streamlit是否安装
    try:
        import streamlit
    except ImportError:
        print("✗ Streamlit未安装，请运行: pip install streamlit")
        return False
    
    # 启动streamlit应用
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "happy8_app.py", 
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n程序已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        return False
    
    return True

def start_cli():
    """启动命令行界面"""
    print("启动快乐8智能预测系统命令行界面...")
    print("使用 --help 查看帮助信息")
    
    # 导入并运行CLI
    try:
        from happy8_analyzer import Happy8CLI
        cli = Happy8CLI()
        cli.run()
    except Exception as e:
        print(f"启动失败: {e}")
        return False
    
    return True

def show_demo():
    """显示演示"""
    print("快乐8智能预测系统演示")
    print("=" * 50)
    
    try:
        from happy8_analyzer import Happy8Analyzer
        
        # 初始化分析器
        analyzer = Happy8Analyzer()
        
        # 加载数据
        print("正在加载数据...")
        data = analyzer.load_data(periods=100)
        print(f"成功加载 {len(data)} 期数据")
        
        # 执行预测
        print("\n正在执行频率分析预测...")
        result = analyzer.predict(
            target_issue="20250817001",
            periods=100,
            count=30,
            method="frequency"
        )
        
        print(f"\n预测结果:")
        print(f"目标期号: {result.target_issue}")
        print(f"预测方法: {result.method}")
        print(f"执行时间: {result.execution_time:.2f}秒")
        print(f"预测号码: {result.predicted_numbers[:10]}...")  # 显示前10个
        
        # 显示可用方法
        methods = analyzer.get_available_methods()
        print(f"\n可用预测方法: {', '.join(methods)}")
        
    except Exception as e:
        print(f"演示失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="快乐8智能预测系统启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
启动方式:
  python start.py web          # 启动Web界面 (推荐)
  python start.py cli          # 启动命令行界面
  python start.py demo         # 运行演示
  python start.py check        # 检查环境
        """
    )
    
    parser.add_argument('mode', 
                       choices=['web', 'cli', 'demo', 'check'],
                       help='启动模式')
    
    args = parser.parse_args()
    
    # 检查依赖
    if not check_dependencies():
        return False
    
    # 根据模式启动
    if args.mode == 'web':
        return start_web_interface()
    elif args.mode == 'cli':
        return start_cli()
    elif args.mode == 'demo':
        return show_demo()
    elif args.mode == 'check':
        print("✓ 环境检查通过，系统可以正常运行")
        return True
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)