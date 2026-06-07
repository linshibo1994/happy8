"""Happy8 原始算法完整集成适配器。"""

import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# 添加原始 Happy8 算法引擎目录到系统路径
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

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
except ImportError as e:
    ORIGINAL_HAPPY8_AVAILABLE = False


class Happy8AlgorithmAdapter:
    """Happy8原始算法适配器 - 完整集成17种算法"""

    def __init__(self):
        self.original_analyzer = None
        self.data_manager = None

        if ORIGINAL_HAPPY8_AVAILABLE:
            try:
                # 确保数据目录存在
                data_dir = PROJECT_ROOT / "data"
                data_dir.mkdir(exist_ok=True)

                # 初始化原始分析器
                self.original_analyzer = Happy8Analyzer(str(data_dir))
                self.data_manager = DataManager(str(data_dir))

            except Exception as e:
                self.original_analyzer = None

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
        """将原始算法排序结果转换为API格式。

        历史字段名仍保留 confidence_score 以兼容前端和数据库，但这里的值表示
        算法输出的平均排序质量分，不代表下一期真实命中概率。
        """

        # 计算综合排序质量分
        if confidence_scores:
            overall_confidence = float(np.mean(confidence_scores))
        else:
            overall_confidence = 0.5

        # 确保质量分在合理展示范围内
        overall_confidence = max(0.1, min(0.99, overall_confidence))

        return {
            "predicted_numbers": predicted_numbers,
            "confidence_score": overall_confidence,
            "analysis_data": {
                "algorithm": algorithm,
                "engine": "original_happy8",
                "score_semantics": "ranking_quality_score_not_hit_probability",
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
            predicted_numbers, confidence_scores = predictor.predict(
                data=df,
                count=count,
                **params
            )

            # 转换结果格式
            result = self.convert_original_result(predicted_numbers, confidence_scores, algorithm)

            return result

        except Exception as e:
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
            # 如果原始系统没有missing算法，使用适配器内置确定性回退实现
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
        """执行遗漏分析的内置回退实现。"""
        df = self.convert_db_to_happy8_format(historical_data)
        if df.empty:
            raise ValueError("没有可用的历史数据")

        predicted_numbers, confidence_scores = self._run_missing_fallback(df, count)

        # 转换结果格式
        result = self.convert_original_result(predicted_numbers, confidence_scores, "missing")

        return result

    def _run_missing_fallback(self, data: pd.DataFrame, count: int) -> Tuple[List[int], List[float]]:
        """基于当前遗漏和平均周期生成遗漏预测结果。"""
        if data is None or data.empty or count <= 0:
            return [], []

        missing_periods = self._calculate_current_missing_periods(data)
        avg_cycles = self._calculate_average_cycles(data)
        rebound_probs = self._calculate_rebound_probabilities(missing_periods, avg_cycles)

        sorted_probs = sorted(rebound_probs.items(), key=lambda item: (-item[1], item[0]))
        selected = sorted_probs[:count]
        predicted_numbers = [num for num, _ in selected]
        confidence_scores = [float(prob) for _, prob in selected]

        if confidence_scores:
            max_confidence = max(confidence_scores)
            if max_confidence > 0:
                confidence_scores = [score / max_confidence for score in confidence_scores]

        return predicted_numbers, confidence_scores

    def _calculate_current_missing_periods(self, data: pd.DataFrame) -> Dict[int, int]:
        """计算当前遗漏期数；DataFrame 第0行必须是最新期。"""
        missing_periods = {}

        for num in range(1, 81):
            missing_periods[num] = 0
            for _, row in data.iterrows():
                numbers = [
                    int(row[f"num{i}"])
                    for i in range(1, 21)
                    if int(row[f"num{i}"]) > 0
                ]
                if num in numbers:
                    break
                missing_periods[num] += 1

        return missing_periods

    def _calculate_average_cycles(self, data: pd.DataFrame) -> Dict[int, float]:
        """计算每个号码历史出现的平均间隔。"""
        default_cycle = 4.0

        if data is None or data.empty:
            return {num: default_cycle for num in range(1, 81)}

        avg_cycles = {}
        for num in range(1, 81):
            appearances = []
            for index, row in data.iterrows():
                numbers = [
                    int(row[f"num{i}"])
                    for i in range(1, 21)
                    if int(row[f"num{i}"]) > 0
                ]
                if num in numbers:
                    appearances.append(index)

            if len(appearances) > 1:
                intervals = [
                    appearances[i] - appearances[i - 1]
                    for i in range(1, len(appearances))
                ]
                avg_cycles[num] = sum(intervals) / len(intervals) if intervals else default_cycle
            else:
                avg_cycles[num] = default_cycle

        return avg_cycles

    def _calculate_rebound_probabilities(
        self,
        missing_periods: Dict[int, int],
        avg_cycles: Dict[int, float],
    ) -> Dict[int, float]:
        """根据当前遗漏和平均周期计算启发式排序分。"""
        rebound_probs = {}

        for num in range(1, 81):
            missing_count = missing_periods.get(num, 0)
            avg_cycle = max(float(avg_cycles.get(num, 4.0)), 1.0)

            if missing_count == 0:
                rebound_probs[num] = 0.1
            elif missing_count <= avg_cycle:
                rebound_probs[num] = 0.3 + (missing_count / avg_cycle) * 0.4
            else:
                excess_ratio = (missing_count - avg_cycle) / avg_cycle
                rebound_probs[num] = 0.7 + min(excess_ratio * 0.3, 0.3)

        return rebound_probs

    async def get_algorithm_info(self, algorithm: str) -> Dict[str, Any]:
        """获取算法详细信息"""
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

        if not self.original_analyzer:
            return {"available": False, "error": "原始分析器不可用"}

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
            "markov": "基于单号跨期状态转移的排序模型",
            "markov_2nd": "二阶单号跨期状态转移排序模型",
            "markov_3rd": "三阶单号跨期状态转移排序模型",
            "adaptive_markov": "融合一至三阶跨期状态转移的排序模型",
            "transformer": "基于时间窗口的Transformer多标签排序模型",
            "lstm": "基于时间窗口的LSTM多标签排序模型",
            "gnn": "基于号码共现图传播的确定性排序模型",
            "monte_carlo": "基于1-80无放回抽样的蒙特卡洛入选排序模型",
            "clustering": "基于KMeans历史结构相似性的排序模型",
            "ensemble": "固定权重融合多种排序分的模型",
            "advanced_ensemble": "使用时间后段验证集加权的多输出集成排序模型",
            "bayesian": "基于Dirichlet后验均值的贝叶斯排序评分模型",
            "super_predictor": "综合排序融合器",
            "high_confidence": "质量门控预测器"
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
