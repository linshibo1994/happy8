"""Happy8原始算法完整集成适配器"""

import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path

# 添加原始Happy8项目路径到系统路径
HAPPY8_ROOT = Path(__file__).parent.parent.parent.parent.parent
HAPPY8_SRC = HAPPY8_ROOT / "src"
sys.path.insert(0, str(HAPPY8_SRC))

print(f"🔍 Happy8根目录: {HAPPY8_ROOT}")
print(f"🔍 Happy8源码目录: {HAPPY8_SRC}")
print(f"🔍 源码目录是否存在: {HAPPY8_SRC.exists()}")

# 尝试导入原始Happy8的所有组件
try:
    # 导入原始数据模型
    from happy8_analyzer import (
        Happy8Result, PredictionResult, Happy8Analyzer,
        FrequencyPredictor, HotColdPredictor, MissingPredictor,
        MarkovPredictor, Markov2ndPredictor, Markov3rdPredictor,
        AdaptiveMarkovPredictor, TransformerPredictor, 
        GraphNeuralNetworkPredictor, MonteCarloPredictor,
        ClusteringPredictor, AdvancedEnsemblePredictor,
        BayesianPredictor, SuperPredictor, HighConfidencePredictor,
        LSTMPredictor, EnsemblePredictor, DataManager
    )
    ORIGINAL_HAPPY8_AVAILABLE = True
    print("✅ 成功导入原始Happy8的所有预测器")
except ImportError as e:
    print(f"⚠️ 无法导入原始Happy8组件: {e}")
    ORIGINAL_HAPPY8_AVAILABLE = False


class Happy8AlgorithmAdapter:
    """Happy8原始算法适配器 - 完整集成17种算法"""
    
    def __init__(self):
        self.original_analyzer = None
        self.data_manager = None
        
        if ORIGINAL_HAPPY8_AVAILABLE:
            try:
                # 确保数据目录存在
                data_dir = HAPPY8_ROOT / "data"
                data_dir.mkdir(exist_ok=True)
                
                # 初始化原始分析器
                self.original_analyzer = Happy8Analyzer(str(data_dir))
                self.data_manager = DataManager(str(data_dir))
                
                print(f"✅ 原始Happy8分析器初始化成功")
                print(f"📊 可用算法: {list(self.original_analyzer.prediction_engine.predictors.keys())}")
                
            except Exception as e:
                print(f"❌ 原始Happy8分析器初始化失败: {e}")
                self.original_analyzer = None
        else:
            print("❌ 原始Happy8不可用，无法提供完整功能")
    
    def is_original_available(self) -> bool:
        """检查原始分析器是否可用"""
        return self.original_analyzer is not None
    
    def get_all_available_algorithms(self) -> List[str]:
        """获取所有可用的算法"""
        if self.original_analyzer:
            return list(self.original_analyzer.prediction_engine.predictors.keys())
        else:
            return []
    
    def convert_db_to_happy8_format(self, historical_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """将数据库格式转换为Happy8原始格式"""
        if not historical_data:
            return pd.DataFrame()
        
        # 构建符合原始Happy8格式的DataFrame
        data_list = []
        for item in historical_data:
            row = {
                'issue': item['issue'],
                'date': item['date']
            }
            
            # 将numbers列表转换为num1, num2, ..., num20列
            numbers = item['numbers']
            for i, num in enumerate(numbers[:20], 1):
                row[f'num{i}'] = num
            
            # 如果号码不足20个，用0填充（通常不会发生）
            for i in range(len(numbers) + 1, 21):
                row[f'num{i}'] = 0
            
            # 添加其他字段
            if 'sum_value' in item:
                row['sum_value'] = item['sum_value']
            else:
                row['sum_value'] = sum(numbers)
            
            if 'odd_count' in item:
                row['odd_count'] = item['odd_count']
            else:
                row['odd_count'] = sum(1 for n in numbers if n % 2 == 1)
            
            if 'big_count' in item:
                row['big_count'] = item['big_count']
            else:
                row['big_count'] = sum(1 for n in numbers if n >= 41)
            
            data_list.append(row)
        
        df = pd.DataFrame(data_list)
        # 确保按期号排序（最新的在前面，符合原始系统的期望）
        df = df.sort_values('issue', ascending=False).reset_index(drop=True)
        
        return df
    
    def convert_original_result(self, predicted_numbers: List[int], confidence_scores: List[float], algorithm: str) -> Dict[str, Any]:
        """将原始算法结果转换为我们的API格式"""
        
        # 计算综合置信度
        if confidence_scores:
            overall_confidence = float(np.mean(confidence_scores))
        else:
            overall_confidence = 0.5
        
        # 确保置信度在合理范围内
        overall_confidence = max(0.1, min(0.99, overall_confidence))
        
        return {
            "predicted_numbers": predicted_numbers,
            "confidence_score": overall_confidence,
            "analysis_data": {
                "algorithm": algorithm,
                "engine": "original_happy8",
                "predictor_scores": dict(zip(predicted_numbers, confidence_scores)) if confidence_scores else {},
                "total_candidates": len(predicted_numbers),
                "confidence_distribution": {
                    "min": float(min(confidence_scores)) if confidence_scores else 0,
                    "max": float(max(confidence_scores)) if confidence_scores else 0,
                    "std": float(np.std(confidence_scores)) if confidence_scores else 0
                }
            }
        }
    
    async def execute_original_algorithm(
        self, 
        algorithm: str, 
        historical_data: List[Dict[str, Any]], 
        count: int, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行原始Happy8算法"""
        
        if not self.original_analyzer:
            raise RuntimeError(f"原始Happy8分析器不可用，无法执行算法: {algorithm}")
        
        # 转换数据格式
        df = self.convert_db_to_happy8_format(historical_data)
        if df.empty:
            raise ValueError("没有可用的历史数据")
        
        # 获取对应的预测器
        predictor = self.original_analyzer.prediction_engine.predictors.get(algorithm)
        if not predictor:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        try:
            # 执行原始算法
            print(f"🚀 执行原始Happy8算法: {algorithm}")
            predicted_numbers, confidence_scores = predictor.predict(
                data=df,
                count=count,
                **params
            )
            
            print(f"✅ 算法 {algorithm} 执行成功，返回 {len(predicted_numbers)} 个号码")
            
            # 转换结果格式
            result = self.convert_original_result(predicted_numbers, confidence_scores, algorithm)
            
            return result
            
        except Exception as e:
            print(f"❌ 原始算法 {algorithm} 执行失败: {e}")
            raise RuntimeError(f"算法执行失败: {e}")
    
    # 为每个具体算法提供专门的接口
    async def frequency_analysis(self, historical_data: List[Dict[str, Any]], count: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """频率分析 - 原始算法"""
        return await self.execute_original_algorithm("frequency", historical_data, count, params)
    
    async def hot_cold_analysis(self, historical_data: List[Dict[str, Any]], count: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """冷热分析 - 原始算法"""
        return await self.execute_original_algorithm("hot_cold", historical_data, count, params)
    
    async def missing_analysis(self, historical_data: List[Dict[str, Any]], count: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """遗漏分析 - 原始算法"""
        # 检查原始系统是否有missing算法
        if "missing" in self.get_all_available_algorithms():
            return await self.execute_original_algorithm("missing", historical_data, count, params)
        else:
            # 如果原始系统没有missing算法，我们需要基于原始框架创建一个
            return await self._create_missing_predictor(historical_data, count, params)
    
    async def markov_analysis(self, historical_data: List[Dict[str, Any]], count: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """马尔可夫链分析 - 原始算法"""
        # 尝试使用自适应马尔可夫，如果没有则使用普通马尔可夫
        for algo_name in ["adaptive_markov", "markov_3rd", "markov_2nd", "markov"]:
            if algo_name in self.get_all_available_algorithms():
                return await self.execute_original_algorithm(algo_name, historical_data, count, params)
        
        raise RuntimeError("没有可用的马尔可夫算法")
    
    async def ml_ensemble_analysis(self, historical_data: List[Dict[str, Any]], count: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """机器学习集成 - 原始算法"""
        # 尝试使用高级集成，如果没有则使用普通集成
        for algo_name in ["advanced_ensemble", "ensemble"]:
            if algo_name in self.get_all_available_algorithms():
                return await self.execute_original_algorithm(algo_name, historical_data, count, params)
        
        raise RuntimeError("没有可用的集成学习算法")
    
    async def deep_learning_analysis(self, historical_data: List[Dict[str, Any]], count: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """深度学习分析 - 原始算法"""
        # 尝试使用各种深度学习算法
        for algo_name in ["transformer", "lstm", "gnn"]:
            if algo_name in self.get_all_available_algorithms():
                try:
                    return await self.execute_original_algorithm(algo_name, historical_data, count, params)
                except Exception as e:
                    print(f"深度学习算法 {algo_name} 失败: {e}")
                    continue
        
        raise RuntimeError("没有可用的深度学习算法")
    
    async def super_predictor_analysis(self, historical_data: List[Dict[str, Any]], count: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """超级预测器 - 原始算法"""
        # 尝试使用超级预测器或高置信度预测器
        for algo_name in ["super_predictor", "high_confidence"]:
            if algo_name in self.get_all_available_algorithms():
                return await self.execute_original_algorithm(algo_name, historical_data, count, params)
        
        raise RuntimeError("没有可用的超级预测器算法")
    
    # 其他特殊算法
    async def bayesian_analysis(self, historical_data: List[Dict[str, Any]], count: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """贝叶斯推理 - 原始算法"""
        return await self.execute_original_algorithm("bayesian", historical_data, count, params)
    
    async def monte_carlo_analysis(self, historical_data: List[Dict[str, Any]], count: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """蒙特卡洛预测 - 原始算法"""
        return await self.execute_original_algorithm("monte_carlo", historical_data, count, params)
    
    async def clustering_analysis(self, historical_data: List[Dict[str, Any]], count: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """聚类预测 - 原始算法"""
        return await self.execute_original_algorithm("clustering", historical_data, count, params)
    
    async def _create_missing_predictor(self, historical_data: List[Dict[str, Any]], count: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """基于原始框架创建遗漏分析预测器"""
        
        if not self.original_analyzer:
            raise RuntimeError("原始分析器不可用")
        
        # 创建一个临时的遗漏分析预测器，符合原始框架的接口
        class MissingPredictor:
            def __init__(self, analyzer):
                self.analyzer = analyzer
            
            def predict(self, data: pd.DataFrame, count: int = 30, **kwargs) -> Tuple[List[int], List[float]]:
                """基于遗漏期数的预测"""
                print("执行遗漏分析预测...")
                
                # 统计每个号码的遗漏期数
                missing_periods = {}
                for num in range(1, 81):
                    missing_periods[num] = 0
                
                # 从最新数据开始计算遗漏
                for idx, (_, row) in enumerate(data.iterrows()):
                    current_numbers = [row[f'num{i}'] for i in range(1, 21) if row[f'num{i}'] > 0]
                    
                    for num in range(1, 81):
                        if num not in current_numbers:
                            missing_periods[num] += 1
                        else:
                            # 号码出现了，遗漏期数已确定
                            pass
                
                # 按遗漏期数排序
                sorted_missing = sorted(missing_periods.items(), key=lambda x: x[1], reverse=True)
                
                # 选择遗漏期数最长的号码
                predicted_numbers = [num for num, periods in sorted_missing[:count]]
                
                # 计算置信度（基于遗漏期数的合理性）
                confidence_scores = []
                max_missing = max(missing_periods.values()) if missing_periods.values() else 1
                
                for num in predicted_numbers:
                    # 遗漏期数越长，基础置信度越高，但要考虑合理性
                    missing_count = missing_periods[num]
                    base_confidence = missing_count / max_missing if max_missing > 0 else 0.5
                    
                    # 加入合理性调整（遗漏过长可能不太合理）
                    if missing_count > 50:  # 遗漏超过50期
                        base_confidence *= 0.8
                    elif missing_count > 30:  # 遗漏超过30期
                        base_confidence *= 0.9
                    
                    confidence_scores.append(max(0.1, min(0.9, base_confidence)))
                
                return predicted_numbers, confidence_scores
        
        # 创建临时预测器实例
        temp_predictor = MissingPredictor(self.original_analyzer)
        
        # 转换数据并执行预测
        df = self.convert_db_to_happy8_format(historical_data)
        predicted_numbers, confidence_scores = temp_predictor.predict(df, count, **params)
        
        # 转换结果格式
        result = self.convert_original_result(predicted_numbers, confidence_scores, "missing")
        
        return result
    
    async def get_algorithm_info(self, algorithm: str) -> Dict[str, Any]:
        """获取算法详细信息"""
        if not self.original_analyzer:
            return {"available": False, "error": "原始分析器不可用"}
        
        # missing支持适配器内置回退实现，即使原始引擎未注册也应视为可用
        if algorithm == "missing":
            return {
                "available": True,
                "algorithm": algorithm,
                "predictor_class": "MissingPredictor(Fallback)",
                "description": self._get_algorithm_description(algorithm),
                "complexity": self._get_algorithm_complexity(algorithm),
                "data_requirements": self._get_data_requirements(algorithm),
            }

        available_algorithms = self.get_all_available_algorithms()
        
        if algorithm not in available_algorithms:
            return {"available": False, "error": f"算法 {algorithm} 不存在"}
        
        # 获取预测器实例
        predictor = self.original_analyzer.prediction_engine.predictors.get(algorithm)
        
        return {
            "available": True,
            "algorithm": algorithm,
            "predictor_class": predictor.__class__.__name__ if predictor else "Unknown",
            "description": self._get_algorithm_description(algorithm),
            "complexity": self._get_algorithm_complexity(algorithm),
            "data_requirements": self._get_data_requirements(algorithm)
        }
    
    def _get_algorithm_description(self, algorithm: str) -> str:
        """获取算法描述"""
        descriptions = {
            "frequency": "基于历史频率统计的预测算法",
            "hot_cold": "基于号码冷热趋势的预测算法", 
            "missing": "基于号码遗漏期数的预测算法",
            "markov": "基于马尔可夫链状态转移的预测算法",
            "markov_2nd": "二阶马尔可夫链预测算法",
            "markov_3rd": "三阶马尔可夫链预测算法",
            "adaptive_markov": "自适应马尔可夫链预测算法",
            "transformer": "基于Transformer的深度学习预测算法",
            "lstm": "基于LSTM的深度学习预测算法",
            "gnn": "基于图神经网络的预测算法",
            "monte_carlo": "蒙特卡洛模拟预测算法",
            "clustering": "基于聚类分析的预测算法",
            "ensemble": "集成学习预测算法",
            "advanced_ensemble": "高级集成学习预测算法",
            "bayesian": "贝叶斯推理预测算法",
            "super_predictor": "超级预测器（融合多种算法）",
            "high_confidence": "高置信度预测器"
        }
        return descriptions.get(algorithm, "未知算法")
    
    def _get_algorithm_complexity(self, algorithm: str) -> str:
        """获取算法复杂度"""
        complexity_map = {
            "frequency": "low",
            "hot_cold": "low",
            "missing": "low",
            "markov": "medium",
            "markov_2nd": "medium",
            "markov_3rd": "high",
            "adaptive_markov": "high",
            "transformer": "very_high",
            "lstm": "high",
            "gnn": "very_high",
            "monte_carlo": "medium",
            "clustering": "medium",
            "ensemble": "high",
            "advanced_ensemble": "very_high",
            "bayesian": "medium",
            "super_predictor": "very_high",
            "high_confidence": "high"
        }
        return complexity_map.get(algorithm, "unknown")
    
    def _get_data_requirements(self, algorithm: str) -> Dict[str, Any]:
        """获取算法数据需求"""
        requirements = {
            "frequency": {"min_periods": 10, "recommended_periods": 100},
            "hot_cold": {"min_periods": 20, "recommended_periods": 150},
            "missing": {"min_periods": 30, "recommended_periods": 200},
            "markov": {"min_periods": 50, "recommended_periods": 200},
            "markov_2nd": {"min_periods": 100, "recommended_periods": 300},
            "markov_3rd": {"min_periods": 150, "recommended_periods": 400},
            "adaptive_markov": {"min_periods": 100, "recommended_periods": 300},
            "transformer": {"min_periods": 200, "recommended_periods": 500},
            "lstm": {"min_periods": 150, "recommended_periods": 400},
            "gnn": {"min_periods": 200, "recommended_periods": 500},
            "monte_carlo": {"min_periods": 100, "recommended_periods": 300},
            "clustering": {"min_periods": 100, "recommended_periods": 250},
            "ensemble": {"min_periods": 150, "recommended_periods": 300},
            "advanced_ensemble": {"min_periods": 200, "recommended_periods": 400},
            "bayesian": {"min_periods": 100, "recommended_periods": 250},
            "super_predictor": {"min_periods": 200, "recommended_periods": 500},
            "high_confidence": {"min_periods": 150, "recommended_periods": 350}
        }
        return requirements.get(algorithm, {"min_periods": 50, "recommended_periods": 200})
