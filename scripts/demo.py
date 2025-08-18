#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8智能预测系统 - 演示脚本
Happy8 Prediction System - Demo Script

展示系统的核心功能和使用方法

作者: CodeBuddy
版本: v1.0
创建时间: 2025-08-17
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def demo_basic_usage():
    """演示基本使用"""
    print("🎯 快乐8智能预测系统演示")
    print("=" * 50)
    
    try:
        # 导入核心模块
        from happy8_analyzer import Happy8Analyzer, Happy8Result
        
        print("✓ 核心模块导入成功")
        
        # 创建分析器
        analyzer = Happy8Analyzer()
        print("✓ 分析器初始化成功")
        
        # 演示数据模型
        print("\n📊 数据模型演示:")
        sample_numbers = [1, 5, 12, 18, 23, 29, 34, 41, 47, 52, 58, 63, 67, 71, 75, 78, 2, 8, 15, 25]
        result = Happy8Result(
            issue="20250817001",
            date="2025-08-17", 
            time="09:05:00",
            numbers=sample_numbers
        )
        
        print(f"  期号: {result.issue}")
        print(f"  开奖号码: {result.numbers}")
        print(f"  号码总和: {result.number_sum}")
        print(f"  号码平均值: {result.number_avg:.2f}")
        print(f"  号码跨度: {result.number_range}")
        print(f"  奇数个数: {result.odd_count}")
        print(f"  大号个数: {result.big_count}")
        print(f"  区域分布: {result.zone_distribution}")
        
        # 演示数据管理
        print("\n📁 数据管理演示:")
        print("  正在获取真实数据...")
        analyzer.data_manager.crawl_initial_data(50)  # 获取50期真实数据
        
        data = analyzer.load_data()
        print(f"  ✓ 成功加载 {len(data)} 期数据")
        
        # 演示预测功能
        print("\n🎯 预测功能演示:")
        methods = ['frequency', 'hot_cold', 'markov']
        
        for method in methods:
            print(f"\n  执行 {method} 预测...")
            start_time = time.time()
            
            try:
                prediction = analyzer.predict(
                    target_issue="20250817100",
                    periods=30,
                    count=20,
                    method=method
                )
                
                execution_time = time.time() - start_time
                
                print(f"    ✓ 预测完成，耗时: {execution_time:.2f}秒")
                print(f"    预测号码: {prediction.predicted_numbers[:10]}...")  # 显示前10个
                print(f"    置信度: {prediction.confidence_scores[:5] if prediction.confidence_scores else 'N/A'}...")
                
            except Exception as e:
                print(f"    ✗ 预测失败: {e}")
        
        # 演示对比功能
        print("\n📈 对比功能演示:")
        try:
            # 使用数据中存在的期号
            data = analyzer.load_data()
            if len(data) > 0:
                test_issue = data.iloc[-1]['issue']  # 使用最后一期
                
                prediction_result, comparison_result = analyzer.analyze_and_predict(
                    target_issue=test_issue,
                    periods=30,
                    count=20,
                    method="frequency"
                )
            
                print(f"  ✓ 预测和对比完成")
                print(f"  命中数量: {comparison_result.hit_count}/{comparison_result.total_predicted}")
                print(f"  命中率: {comparison_result.hit_rate:.2%}")
                print(f"  命中号码: {sorted(comparison_result.hit_numbers)}")
            else:
                print("  ⚠️ 没有数据，跳过对比功能演示")
            
        except Exception as e:
            print(f"  ✗ 对比功能失败: {e}")
        
        # 演示性能统计
        print("\n📊 性能统计演示:")
        try:
            performance = analyzer.get_performance_summary()
            if performance:
                for method, stats in performance.items():
                    print(f"  {method}: 平均耗时 {stats.get('avg_execution_time', 0):.2f}秒")
            else:
                print("  暂无性能数据")
        except Exception as e:
            print(f"  性能统计获取失败: {e}")
        
        print("\n🎉 演示完成！")
        print("\n💡 使用提示:")
        print("  1. 运行 'python3 start.py web' 启动Web界面")
        print("  2. 运行 'python3 start.py cli' 启动命令行界面") 
        print("  3. 运行 'python3 test_system.py' 执行完整测试")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        print("请先安装依赖: pip3 install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"✗ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_web_interface():
    """演示Web界面启动"""
    print("\n🌐 Web界面启动演示:")
    print("=" * 30)
    
    try:
        import streamlit
        print("✓ Streamlit已安装")
        print("💡 运行以下命令启动Web界面:")
        print("   python3 start.py web")
        print("   然后在浏览器中访问: http://localhost:8501")
        return True
        
    except ImportError:
        print("✗ Streamlit未安装")
        print("请运行: pip3 install streamlit")
        return False

def demo_docker_deployment():
    """演示Docker部署"""
    print("\n🐳 Docker部署演示:")
    print("=" * 30)
    
    print("💡 使用Docker部署系统:")
    print("1. 构建镜像:")
    print("   docker build -t happy8-system .")
    print("\n2. 运行容器:")
    print("   docker run -p 8501:8501 happy8-system")
    print("\n3. 或使用docker-compose:")
    print("   docker-compose up -d")
    print("\n4. 访问系统:")
    print("   http://localhost:8501 (直接访问)")
    print("   http://localhost (通过Nginx代理)")

def main():
    """主函数"""
    print(f"快乐8智能预测系统演示 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 基本功能演示
    if not demo_basic_usage():
        print("\n❌ 基本功能演示失败，请检查系统配置")
        return False
    
    # Web界面演示
    demo_web_interface()
    
    # Docker部署演示
    demo_docker_deployment()
    
    print("\n" + "=" * 50)
    print("🎯 快乐8智能预测系统演示完成")
    print("感谢使用！如有问题请查看README.md文档")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)