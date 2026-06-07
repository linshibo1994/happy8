#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8智能预测系统 - 主启动文件
"""

import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

def start_cli():
    """启动命令行界面"""
    print("命令行模式说明:")
    print("  python -c \"from engine.happy8_analyzer import Happy8Analyzer; analyzer = Happy8Analyzer(); print('系统已初始化')\"")
    print("或者直接导入:")
    print("  from engine.happy8_analyzer import Happy8Analyzer")

def start_api():
    """启动后端 API 服务"""
    subprocess.run([sys.executable, str(PROJECT_ROOT / "backend" / "start.py")], check=True)

def run_demo():
    """运行演示"""
    print("运行快乐8预测系统演示...")

    try:
        from happy8_analyzer import Happy8Analyzer

        analyzer = Happy8Analyzer()
        data = analyzer.load_data()
        print(f"成功加载 {len(data)} 期历史数据")
        result = analyzer.predict_with_smart_mode('2025999', 30, 5, 'frequency')
        numbers = result['prediction_result'].predicted_numbers
        print(f"预测结果: {numbers}")
    except Exception as e:
        print(f"演示运行失败: {e}")
        import traceback
        traceback.print_exc()

def show_help():
    """显示帮助信息"""
    print("""
快乐8智能预测系统

使用方法:
  python main.py [命令]

可用命令:
  api      启动 FastAPI 后端
  cli      显示命令行使用说明
  demo     运行系统演示
  help     显示此帮助信息

示例:
  python main.py api
  python main.py demo
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
        default='cli',
        choices=['api', 'cli', 'demo', 'help'],
        help='要执行的命令 (默认: cli)'
    )
    
    args = parser.parse_args()
    
    # 检查项目结构
    required_dirs = ['engine', 'data', 'backend']
    missing_dirs = [d for d in required_dirs if not Path(d).exists()]

    if missing_dirs:
        print(f"❌ 缺少必要目录: {', '.join(missing_dirs)}")
        print("请确保项目结构完整")
        return

    # 检查关键文件
    required_files = ['engine/happy8_analyzer.py', 'backend/start.py']
    missing_files = [f for f in required_files if not Path(f).exists()]

    if missing_files:
        print(f"❌ 缺少关键文件: {', '.join(missing_files)}")
        print("请确保项目文件完整")
        return
    
    # 执行对应命令
    if args.command == 'api':
        start_api()
    elif args.command == 'cli':
        start_cli()
    elif args.command == 'demo':
        run_demo()
    elif args.command == 'help':
        show_help()
    else:
        show_help()

if __name__ == "__main__":
    main()
