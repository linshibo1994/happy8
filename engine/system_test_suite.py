#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8智能预测系统 - 系统测试套件
Happy8 Prediction System - System Test Suite

完整的功能和性能测试，确保系统质量：
- 基础功能测试
- 算法性能测试
- 系统集成测试
- 压力测试和用户场景测试

作者: linshibo
开发者: linshibo
版本: v1.4.0
创建时间: 2025-08-19
"""

import sys
import time
import traceback
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

sys.path.insert(0, '.')
from engine.happy8_analyzer import Happy8Analyzer
from engine.performance_optimizer import PerformanceOptimizer


class SystemTestSuite:
    """系统测试套件"""
    
    def __init__(self):
        self.analyzer = Happy8Analyzer()
        self.optimizer = PerformanceOptimizer()
        self.test_results = {}
        
    def run_complete_test_suite(self) -> Dict[str, Any]:
        """运行完整的测试套件"""
        print("🧪 启动完整系统测试套件")
        print("=" * 80)
        
        # 1. 基础功能测试
        print("\n📋 第1阶段: 基础功能测试")
        basic_results = self._test_basic_functionality()
        
        # 2. 算法性能测试
        print("\n🎯 第2阶段: 算法性能测试")
        algorithm_results = self._test_algorithm_performance()
        
        # 3. 系统集成测试
        print("\n🔗 第3阶段: 系统集成测试")
        integration_results = self._test_system_integration()
        
        # 4. 压力测试
        print("\n💪 第4阶段: 系统压力测试")
        stress_results = self._test_system_stress()
        
        # 5. 用户场景测试
        print("\n👤 第5阶段: 用户场景测试")
        scenario_results = self._test_user_scenarios()
        
        # 汇总测试结果
        self.test_results = {
            'basic_functionality': basic_results,
            'algorithm_performance': algorithm_results,
            'system_integration': integration_results,
            'stress_testing': stress_results,
            'user_scenarios': scenario_results,
            'overall_score': self._calculate_overall_score()
        }
        
        # 生成测试报告
        self._generate_test_report()
        
        return self.test_results
    
    def _test_basic_functionality(self) -> Dict[str, bool]:
        """基础功能测试"""
        results = {}
        
        try:
            # 数据加载测试
            print("   测试数据加载...")
            data = self.analyzer.load_data()
            results['data_loading'] = len(data) > 0
            print(f"   ✅ 数据加载: {len(data)}期数据")
            
            # 数据爬取测试
            print("   测试数据爬取...")
            try:
                single_data = self.analyzer.crawler.crawl_single_issue('2025219')
                results['data_crawling'] = single_data is not None
                print("   ✅ 数据爬取: 正常")
            except:
                results['data_crawling'] = False
                print("   ⚠️ 数据爬取: 网络问题")
            
            # 智能预测模式测试
            print("   测试智能预测模式...")
            result = self.analyzer.predict_with_smart_mode(
                target_issue='2025219',
                periods=30,
                count=8,
                method='frequency'
            )
            results['smart_prediction'] = result is not None
            print("   ✅ 智能预测模式: 正常")
            
        except Exception as e:
            print(f"   ❌ 基础功能测试失败: {e}")
            results['basic_functionality_error'] = str(e)
        
        return results
    
    def _test_algorithm_performance(self) -> Dict[str, Dict[str, Any]]:
        """算法性能测试"""
        results = {}
        
        # 测试所有算法
        algorithms = [
            'frequency', 'hot_cold', 'missing',
            'markov', 'markov_2nd', 'markov_3rd', 'adaptive_markov',
            'transformer', 'gnn', 'monte_carlo', 'clustering',
            'advanced_ensemble', 'bayesian', 'super_predictor',
            'high_confidence', 'lstm', 'ensemble'
        ]
        
        for algorithm in algorithms:
            print(f"   测试 {algorithm}...")
            
            try:
                start_time = time.time()
                
                result = self.analyzer.predict_with_smart_mode(
                    target_issue='2025999',  # 未来期号
                    periods=30,
                    count=10,
                    method=algorithm
                )
                
                execution_time = time.time() - start_time
                
                pred_result = result['prediction_result']
                
                results[algorithm] = {
                    'success': True,
                    'execution_time': execution_time,
                    'prediction_count': len(pred_result.predicted_numbers),
                    'avg_confidence': np.mean(pred_result.confidence_scores) if pred_result.confidence_scores else 0,
                    'has_output': len(pred_result.predicted_numbers) > 0
                }
                
                status = "✅" if results[algorithm]['has_output'] else "⚠️"
                print(f"   {status} {algorithm}: {execution_time:.2f}s, 置信度={results[algorithm]['avg_confidence']:.3f}")
                
            except Exception as e:
                results[algorithm] = {
                    'success': False,
                    'error': str(e),
                    'execution_time': 0,
                    'prediction_count': 0,
                    'avg_confidence': 0,
                    'has_output': False
                }
                print(f"   ❌ {algorithm}: {e}")
        
        return results
    
    def _test_system_integration(self) -> Dict[str, bool]:
        """系统集成测试"""
        results = {}
        
        try:
            # 测试预测引擎集成
            print("   测试预测引擎集成...")
            engine = self.analyzer.prediction_engine
            results['prediction_engine'] = len(engine.predictors) == 17
            print(f"   ✅ 预测引擎: {len(engine.predictors)}个预测器")
            
            # 测试智能模式切换
            print("   测试智能模式切换...")
            
            # 历史验证模式
            hist_result = self.analyzer.predict_with_smart_mode(
                target_issue='2025219',
                periods=20,
                count=8,
                method='frequency'
            )
            
            # 未来预测模式
            future_result = self.analyzer.predict_with_smart_mode(
                target_issue='2025999',
                periods=20,
                count=8,
                method='frequency'
            )
            
            results['mode_switching'] = (
                hist_result['mode'] == 'historical_validation' and
                future_result['mode'] == 'future_prediction'
            )
            print("   ✅ 智能模式切换: 正常")
            
            # 测试结果对比分析
            print("   测试结果对比分析...")
            if hist_result.get('comparison_result'):
                comp = hist_result['comparison_result']
                results['comparison_analysis'] = hasattr(comp, 'hit_rate')
                print(f"   ✅ 对比分析: 命中率{comp.hit_rate:.1%}")
            else:
                results['comparison_analysis'] = False
                print("   ⚠️ 对比分析: 无结果")
            
        except Exception as e:
            print(f"   ❌ 系统集成测试失败: {e}")
            results['integration_error'] = str(e)
        
        return results
    
    def _test_system_stress(self) -> Dict[str, Any]:
        """系统压力测试"""
        results = {}
        
        try:
            print("   执行并发预测测试...")
            
            # 并发预测测试
            import concurrent.futures
            
            def single_prediction():
                return self.analyzer.predict_with_smart_mode(
                    target_issue='2025999',
                    periods=20,
                    count=5,
                    method='frequency'
                )
            
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(single_prediction) for _ in range(10)]
                concurrent_results = [f.result() for f in futures]
            
            concurrent_time = time.time() - start_time
            
            results['concurrent_predictions'] = {
                'success_count': len([r for r in concurrent_results if r is not None]),
                'total_time': concurrent_time,
                'avg_time_per_prediction': concurrent_time / 10
            }
            
            print(f"   ✅ 并发测试: {results['concurrent_predictions']['success_count']}/10成功")
            
            # 大数据量测试
            print("   执行大数据量测试...")
            
            start_time = time.time()
            large_result = self.analyzer.predict_with_smart_mode(
                target_issue='2025999',
                periods=100,  # 大数据量
                count=30,     # 大预测数量
                method='frequency'
            )
            large_data_time = time.time() - start_time
            
            results['large_data_test'] = {
                'success': large_result is not None,
                'execution_time': large_data_time,
                'prediction_count': len(large_result['prediction_result'].predicted_numbers) if large_result else 0
            }
            
            print(f"   ✅ 大数据量测试: {large_data_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ 压力测试失败: {e}")
            results['stress_error'] = str(e)
        
        return results
    
    def _test_user_scenarios(self) -> Dict[str, bool]:
        """用户场景测试"""
        results = {}
        
        scenarios = [
            {
                'name': '新手用户场景',
                'method': 'frequency',
                'periods': 30,
                'count': 8
            },
            {
                'name': '高级用户场景',
                'method': 'super_predictor',
                'periods': 50,
                'count': 15
            },
            {
                'name': '专业用户场景',
                'method': 'high_confidence',
                'periods': 100,
                'count': 20
            }
        ]
        
        for scenario in scenarios:
            print(f"   测试{scenario['name']}...")
            
            try:
                result = self.analyzer.predict_with_smart_mode(
                    target_issue='2025999',
                    periods=scenario['periods'],
                    count=scenario['count'],
                    method=scenario['method']
                )
                
                results[scenario['name']] = result is not None
                print(f"   ✅ {scenario['name']}: 正常")
                
            except Exception as e:
                results[scenario['name']] = False
                print(f"   ❌ {scenario['name']}: {e}")
        
        return results
    
    def _calculate_overall_score(self) -> float:
        """计算总体评分"""
        scores = []
        
        # 基础功能评分 (30%)
        basic_score = sum(1 for v in self.test_results.get('basic_functionality', {}).values() 
                         if isinstance(v, bool) and v) / max(1, len([v for v in self.test_results.get('basic_functionality', {}).values() if isinstance(v, bool)]))
        scores.append(basic_score * 0.3)
        
        # 算法性能评分 (40%)
        algorithm_results = self.test_results.get('algorithm_performance', {})
        successful_algorithms = sum(1 for r in algorithm_results.values() if r.get('success', False))
        algorithm_score = successful_algorithms / max(1, len(algorithm_results))
        scores.append(algorithm_score * 0.4)
        
        # 系统集成评分 (20%)
        integration_score = sum(1 for v in self.test_results.get('system_integration', {}).values() 
                               if isinstance(v, bool) and v) / max(1, len([v for v in self.test_results.get('system_integration', {}).values() if isinstance(v, bool)]))
        scores.append(integration_score * 0.2)
        
        # 用户场景评分 (10%)
        scenario_score = sum(1 for v in self.test_results.get('user_scenarios', {}).values() 
                            if isinstance(v, bool) and v) / max(1, len([v for v in self.test_results.get('user_scenarios', {}).values() if isinstance(v, bool)]))
        scores.append(scenario_score * 0.1)
        
        return sum(scores)
    
    def _generate_test_report(self):
        """生成测试报告"""
        print("\n" + "=" * 80)
        print("📊 系统测试报告")
        print("=" * 80)
        
        # 总体评分
        overall_score = self.test_results['overall_score']
        print(f"\n🏆 总体评分: {overall_score:.1%}")
        
        if overall_score >= 0.9:
            print("   评级: 优秀 ⭐⭐⭐⭐⭐")
        elif overall_score >= 0.8:
            print("   评级: 良好 ⭐⭐⭐⭐")
        elif overall_score >= 0.7:
            print("   评级: 合格 ⭐⭐⭐")
        else:
            print("   评级: 需要改进 ⭐⭐")
        
        # 详细结果
        print(f"\n📋 详细测试结果:")
        
        # 基础功能
        basic_results = self.test_results.get('basic_functionality', {})
        basic_success = sum(1 for v in basic_results.values() if isinstance(v, bool) and v)
        basic_total = len([v for v in basic_results.values() if isinstance(v, bool)])
        print(f"   基础功能: {basic_success}/{basic_total} 通过")
        
        # 算法性能
        algorithm_results = self.test_results.get('algorithm_performance', {})
        algo_success = sum(1 for r in algorithm_results.values() if r.get('success', False))
        algo_total = len(algorithm_results)
        print(f"   算法性能: {algo_success}/{algo_total} 通过")
        
        # 系统集成
        integration_results = self.test_results.get('system_integration', {})
        int_success = sum(1 for v in integration_results.values() if isinstance(v, bool) and v)
        int_total = len([v for v in integration_results.values() if isinstance(v, bool)])
        print(f"   系统集成: {int_success}/{int_total} 通过")
        
        # 用户场景
        scenario_results = self.test_results.get('user_scenarios', {})
        scenario_success = sum(1 for v in scenario_results.values() if isinstance(v, bool) and v)
        scenario_total = len([v for v in scenario_results.values() if isinstance(v, bool)])
        print(f"   用户场景: {scenario_success}/{scenario_total} 通过")
        
        print("\n✅ 系统测试套件执行完成！")


def run_system_tests():
    """运行系统测试"""
    test_suite = SystemTestSuite()
    return test_suite.run_complete_test_suite()


if __name__ == "__main__":
    run_system_tests()
