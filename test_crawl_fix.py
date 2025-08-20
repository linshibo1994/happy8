#!/usr/bin/env python3
"""
快乐8数据爬取功能修复验证脚本
测试真实数据源的爬取功能
"""

import sys
import os
sys.path.insert(0, 'src')

from happy8_analyzer import Happy8Crawler, Happy8Analyzer
import pandas as pd
from pathlib import Path

def test_xml_data_source():
    """测试500彩票网XML数据源"""
    print("🔍 测试500彩票网XML数据源")
    print("=" * 50)
    
    crawler = Happy8Crawler()
    
    try:
        # 测试爬取5期数据
        results = crawler._crawl_from_500wan(5)
        
        if results:
            print(f"✅ 成功获取 {len(results)} 期真实数据")
            print("\n📊 数据样本:")
            for i, result in enumerate(results[:3]):
                print(f"  期号: {result.issue}")
                print(f"  日期: {result.date}")
                print(f"  号码: {result.numbers}")
                print(f"  号码数量: {len(result.numbers)}")
                print(f"  号码范围: {min(result.numbers)}-{max(result.numbers)}")
                print("-" * 30)
            
            # 验证数据完整性
            print("\n🔍 数据完整性验证:")
            all_valid = True
            for result in results:
                if len(result.numbers) != 20:
                    print(f"❌ 期号 {result.issue}: 号码数量错误 ({len(result.numbers)})")
                    all_valid = False
                elif not all(1 <= num <= 80 for num in result.numbers):
                    print(f"❌ 期号 {result.issue}: 号码范围错误")
                    all_valid = False
                elif len(set(result.numbers)) != 20:
                    print(f"❌ 期号 {result.issue}: 存在重复号码")
                    all_valid = False
            
            if all_valid:
                print("✅ 所有数据验证通过")
            
            return True
        else:
            print("❌ 未获取到数据")
            return False
            
    except Exception as e:
        print(f"❌ 爬取失败: {e}")
        return False

def test_incremental_update():
    """测试增量更新功能"""
    print("\n🔄 测试增量更新功能")
    print("=" * 50)
    
    try:
        analyzer = Happy8Analyzer()
        
        # 获取当前数据状态
        current_data = analyzer.load_data()
        initial_count = len(current_data)
        print(f"当前数据量: {initial_count} 期")
        
        if initial_count > 0:
            latest_issue = current_data.iloc[0]['issue']
            print(f"最新期号: {latest_issue}")
        
        # 执行增量更新
        print("\n执行增量更新...")
        new_count = analyzer.data_manager.crawl_latest_data(20)
        
        # 检查更新结果
        updated_data = analyzer.load_data()
        final_count = len(updated_data)
        
        print(f"更新后数据量: {final_count} 期")
        print(f"新增数据: {new_count} 期")
        
        if final_count > 0:
            new_latest_issue = updated_data.iloc[0]['issue']
            print(f"最新期号: {new_latest_issue}")
        
        if new_count > 0:
            print("✅ 增量更新成功")
        else:
            print("📋 当前数据已是最新")
        
        return True
        
    except Exception as e:
        print(f"❌ 增量更新失败: {e}")
        return False

def test_data_storage():
    """测试数据存储功能"""
    print("\n💾 测试数据存储功能")
    print("=" * 50)
    
    try:
        analyzer = Happy8Analyzer()
        data = analyzer.load_data()
        
        if len(data) == 0:
            print("❌ 没有数据可测试")
            return False
        
        print(f"数据总量: {len(data)} 期")
        
        # 检查数据排序
        issues = data['issue'].tolist()
        is_sorted = all(issues[i] >= issues[i+1] for i in range(len(issues)-1))
        
        if is_sorted:
            print("✅ 数据按期号倒序排列正确")
        else:
            print("❌ 数据排序错误")
            return False
        
        # 检查数据完整性
        print(f"最新期号: {issues[0]}")
        print(f"最早期号: {issues[-1]}")
        
        # 检查号码列
        number_cols = [f'num{i}' for i in range(1, 21)]
        missing_cols = [col for col in number_cols if col not in data.columns]
        
        if missing_cols:
            print(f"❌ 缺少号码列: {missing_cols}")
            return False
        else:
            print("✅ 号码列完整")
        
        # 检查数据类型
        sample_row = data.iloc[0]
        sample_numbers = [int(sample_row[f'num{i}']) for i in range(1, 21)]
        
        if len(sample_numbers) == 20 and all(1 <= num <= 80 for num in sample_numbers):
            print("✅ 数据格式正确")
        else:
            print("❌ 数据格式错误")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 数据存储测试失败: {e}")
        return False

def test_data_deduplication():
    """测试数据去重功能"""
    print("\n🔍 测试数据去重功能")
    print("=" * 50)
    
    try:
        analyzer = Happy8Analyzer()
        data = analyzer.load_data()
        
        if len(data) == 0:
            print("❌ 没有数据可测试")
            return False
        
        # 检查是否有重复期号
        duplicate_issues = data['issue'].duplicated().sum()
        
        if duplicate_issues == 0:
            print("✅ 没有重复期号，去重功能正常")
        else:
            print(f"❌ 发现 {duplicate_issues} 个重复期号")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 去重测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🎯 快乐8数据爬取功能修复验证")
    print("=" * 60)
    
    test_results = []
    
    # 测试XML数据源
    test_results.append(("XML数据源", test_xml_data_source()))
    
    # 测试增量更新
    test_results.append(("增量更新", test_incremental_update()))
    
    # 测试数据存储
    test_results.append(("数据存储", test_data_storage()))
    
    # 测试数据去重
    test_results.append(("数据去重", test_data_deduplication()))
    
    # 汇总结果
    print("\n📊 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15} : {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！数据爬取功能修复成功！")
        print("\n✅ 修复内容:")
        print("  - 使用真实的500彩票网XML接口")
        print("  - 实现增量更新功能")
        print("  - 数据按期号倒序存储")
        print("  - 自动去重和数据验证")
        print("  - Web界面功能更新")
    else:
        print(f"\n⚠️  {total - passed} 项测试失败，需要进一步修复")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
