#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8智能预测系统 - 批量预测对比功能
Happy8 Prediction System - Batch Prediction Comparison Feature

实现批量预测对比功能，支持：
- 批量执行多轮预测
- 统计分析预测结果
- Excel数据导出
- 实时进度跟踪

作者: linshibo
开发者: linshibo
版本: v1.5.0
创建时间: 2025-09-12
"""

import os
import sys
import time
import uuid
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path

# 导入现有的预测系统组件
try:
    from happy8_analyzer import Happy8Analyzer, PredictionResult, ComparisonResult, Happy8Result
except ImportError:
    # 如果直接运行此文件，添加路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from happy8_analyzer import Happy8Analyzer, PredictionResult, ComparisonResult, Happy8Result

# ================================
# 数据模型定义
# ================================

@dataclass
class BatchConfig:
    """批量预测配置参数"""
    target_issue: str        # 目标期号
    analysis_periods: int    # 分析期数
    prediction_method: str   # 预测方法
    number_count: int        # 生成号码数量
    comparison_times: int    # 对比次数
    max_parallel: int = 4    # 最大并发数
    timeout_seconds: int = 30  # 单次预测超时时间

    def __post_init__(self):
        """参数验证"""
        if not self.target_issue or len(self.target_issue) != 7:
            raise ValueError("目标期号必须是7位数字")
        if not 10 <= self.analysis_periods <= 500:
            raise ValueError("分析期数必须在10-500之间")
        if not 1 <= self.number_count <= 30:
            raise ValueError("生成号码数量必须在1-30之间")
        if not 1 <= self.comparison_times <= 100:
            raise ValueError("对比次数必须在1-100之间")
        if not 1 <= self.max_parallel <= 8:
            raise ValueError("最大并发数必须在1-8之间")

@dataclass
class EnhancedPredictionResult:
    """增强的预测结果，包含批量预测所需的额外信息"""
    # 继承原有字段
    target_issue: str
    analysis_periods: int
    method: str
    predicted_numbers: List[int]
    confidence_scores: List[float] 
    generation_time: datetime
    execution_time: float
    parameters: Dict[str, Any]
    
    # 新增字段
    round_number: int           # 预测轮次
    hit_numbers: List[int]      # 命中的号码
    hit_count: int             # 命中数量
    hit_rate: float            # 命中率
    actual_numbers: List[int]   # 实际开奖号码
    session_id: str            # 会话ID
    success: bool              # 预测是否成功

    @classmethod
    def from_prediction_result(cls, pred_result: PredictionResult, round_number: int, 
                             session_id: str, actual_numbers: List[int] = None):
        """从原始PredictionResult创建增强版本"""
        # 计算命中信息
        hit_numbers = []
        hit_count = 0
        hit_rate = 0.0
        
        if actual_numbers:
            hit_numbers = [num for num in pred_result.predicted_numbers if num in actual_numbers]
            hit_count = len(hit_numbers)
            hit_rate = hit_count / len(pred_result.predicted_numbers) if pred_result.predicted_numbers else 0.0
        
        return cls(
            target_issue=pred_result.target_issue,
            analysis_periods=pred_result.analysis_periods,
            method=pred_result.method,
            predicted_numbers=pred_result.predicted_numbers,
            confidence_scores=pred_result.confidence_scores,
            generation_time=pred_result.generation_time,
            execution_time=pred_result.execution_time,
            parameters=pred_result.parameters,
            round_number=round_number,
            hit_numbers=hit_numbers,
            hit_count=hit_count,
            hit_rate=hit_rate,
            actual_numbers=actual_numbers or [],
            session_id=session_id,
            success=True
        )

@dataclass
class StatisticReport:
    """统计报告"""
    total_rounds: int                    # 总轮次
    success_rounds: int                  # 成功轮次
    avg_hit_rate: float                  # 平均命中率
    max_hit_rate: float                  # 最高命中率
    min_hit_rate: float                  # 最低命中率
    std_deviation: float                 # 标准差
    hit_rate_distribution: Dict[str, int] # 命中率分布
    confidence_interval: tuple           # 置信区间 (95%)
    quartiles: List[float]               # 四分位数
    percentiles: Dict[int, float]        # 百分位数
    outliers: List[int]                  # 异常值轮次
    avg_execution_time: float            # 平均执行时间
    total_execution_time: float          # 总执行时间

@dataclass
class BatchSession:
    """批量预测会话信息"""
    session_id: str
    config: BatchConfig
    status: str  # 'running', 'completed', 'failed', 'cancelled'
    current_round: int
    results: List[EnhancedPredictionResult]
    statistics: Optional[StatisticReport]
    start_time: datetime
    end_time: Optional[datetime]
    error_message: Optional[str]
    
    @property
    def progress(self) -> float:
        """计算进度百分比"""
        return (self.current_round / self.config.comparison_times) * 100 if self.config.comparison_times > 0 else 0

@dataclass
class BatchResult:
    """批量预测最终结果"""
    session: BatchSession
    predictions: List[EnhancedPredictionResult]
    statistics: StatisticReport
    execution_summary: Dict[str, Any]
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        def datetime_converter(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object {obj} is not JSON serializable")
        
        data = asdict(self)
        return json.dumps(data, default=datetime_converter, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BatchResult':
        """从JSON字符串创建BatchResult"""
        data = json.loads(json_str)
        
        # 重构datetime字段
        def convert_datetime_fields(obj, path=""):
            if isinstance(obj, dict):
                new_obj = {}
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if key in ['generation_time', 'start_time', 'end_time'] and isinstance(value, str):
                        try:
                            new_obj[key] = datetime.fromisoformat(value) if value else None
                        except (ValueError, TypeError):
                            new_obj[key] = value
                    else:
                        new_obj[key] = convert_datetime_fields(value, current_path)
                return new_obj
            elif isinstance(obj, list):
                return [convert_datetime_fields(item, f"{path}[{i}]") for i, item in enumerate(obj)]
            else:
                return obj
        
        converted_data = convert_datetime_fields(data)
        
        # 重建对象结构
        session_data = converted_data['session']
        config = BatchConfig(**session_data['config'])
        
        predictions = []
        for pred_data in converted_data['predictions']:
            predictions.append(EnhancedPredictionResult(**pred_data))
        
        statistics = StatisticReport(**converted_data['statistics'])
        
        session = BatchSession(
            session_id=session_data['session_id'],
            config=config,
            status=session_data['status'],
            current_round=session_data['current_round'],
            results=predictions,
            statistics=statistics,
            start_time=session_data['start_time'],
            end_time=session_data['end_time'],
            error_message=session_data['error_message']
        )
        
        return cls(
            session=session,
            predictions=predictions,
            statistics=statistics,
            execution_summary=converted_data['execution_summary']
        )
    
    def save_to_file(self, filepath: str) -> str:
        """保存到文件"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.to_json())
            return filepath
        except Exception as e:
            raise Exception(f"保存BatchResult到文件失败: {e}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'BatchResult':
        """从文件加载"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return cls.from_json(f.read())
        except Exception as e:
            raise Exception(f"从文件加载BatchResult失败: {e}")

# ================================
# 异常类定义
# ================================

class BatchPredictionError(Exception):
    """批量预测基础异常类"""
    pass

class InvalidConfigError(BatchPredictionError):
    """无效配置参数异常"""
    pass

class PredictionExecutionError(BatchPredictionError):
    """预测执行异常"""
    pass

class BatchTimeoutError(BatchPredictionError):
    """批量预测超时异常"""
    pass

# ================================
# 统计分析引擎
# ================================

class StatisticEngine:
    """统计分析引擎"""
    
    @staticmethod
    def analyze_batch_results(results: List[EnhancedPredictionResult]) -> StatisticReport:
        """分析批量预测结果"""
        if not results:
            raise ValueError("预测结果列表为空")
        
        # 过滤成功的预测结果
        successful_results = [r for r in results if r.success]
        success_count = len(successful_results)
        
        if success_count == 0:
            # 如果没有成功的预测，返回空统计
            return StatisticReport(
                total_rounds=len(results),
                success_rounds=0,
                avg_hit_rate=0.0,
                max_hit_rate=0.0,
                min_hit_rate=0.0,
                std_deviation=0.0,
                hit_rate_distribution={},
                confidence_interval=(0.0, 0.0),
                quartiles=[0.0, 0.0, 0.0, 0.0, 0.0],
                percentiles={},
                outliers=[],
                avg_execution_time=0.0,
                total_execution_time=0.0
            )
        
        # 提取命中率数据
        hit_rates = [r.hit_rate for r in successful_results]
        execution_times = [r.execution_time for r in successful_results]
        
        # 基本统计
        avg_hit_rate = np.mean(hit_rates)
        max_hit_rate = np.max(hit_rates)
        min_hit_rate = np.min(hit_rates)
        std_deviation = np.std(hit_rates)
        
        # 命中率分布
        hit_rate_distribution = StatisticEngine._calculate_hit_rate_distribution(hit_rates)
        
        # 置信区间 (95%)
        confidence_interval = StatisticEngine._calculate_confidence_interval(hit_rates)
        
        # 四分位数
        quartiles = [
            np.percentile(hit_rates, 0),    # 最小值
            np.percentile(hit_rates, 25),   # Q1
            np.percentile(hit_rates, 50),   # 中位数
            np.percentile(hit_rates, 75),   # Q3
            np.percentile(hit_rates, 100)   # 最大值
        ]
        
        # 百分位数
        percentiles = {
            p: np.percentile(hit_rates, p) 
            for p in [5, 10, 25, 50, 75, 90, 95]
        }
        
        # 异常值检测 (使用IQR方法)
        outliers = StatisticEngine._detect_outliers(successful_results, hit_rates)
        
        # 执行时间统计
        avg_execution_time = np.mean(execution_times)
        total_execution_time = np.sum(execution_times)
        
        return StatisticReport(
            total_rounds=len(results),
            success_rounds=success_count,
            avg_hit_rate=avg_hit_rate,
            max_hit_rate=max_hit_rate,
            min_hit_rate=min_hit_rate,
            std_deviation=std_deviation,
            hit_rate_distribution=hit_rate_distribution,
            confidence_interval=confidence_interval,
            quartiles=quartiles,
            percentiles=percentiles,
            outliers=outliers,
            avg_execution_time=avg_execution_time,
            total_execution_time=total_execution_time
        )
    
    @staticmethod
    def _calculate_hit_rate_distribution(hit_rates: List[float]) -> Dict[str, int]:
        """计算命中率分布"""
        ranges = [
            ("0-10%", 0.0, 0.1),
            ("10-20%", 0.1, 0.2),
            ("20-30%", 0.2, 0.3),
            ("30-40%", 0.3, 0.4),
            ("40-50%", 0.4, 0.5),
            ("50-60%", 0.5, 0.6),
            ("60-70%", 0.6, 0.7),
            ("70-80%", 0.7, 0.8),
            ("80-90%", 0.8, 0.9),
            ("90-100%", 0.9, 1.0)
        ]
        
        distribution = {}
        for range_name, min_val, max_val in ranges:
            count = sum(1 for rate in hit_rates if min_val <= rate < max_val)
            # 处理100%的特殊情况
            if range_name == "90-100%":
                count += sum(1 for rate in hit_rates if rate == 1.0)
            distribution[range_name] = count
        
        return distribution
    
    @staticmethod
    def _calculate_confidence_interval(hit_rates: List[float], confidence: float = 0.95) -> tuple:
        """计算置信区间"""
        if len(hit_rates) < 2:
            return (0.0, 0.0)
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(hit_rates, lower_percentile)
        upper_bound = np.percentile(hit_rates, upper_percentile)
        
        return (lower_bound, upper_bound)
    
    @staticmethod
    def _detect_outliers(results: List[EnhancedPredictionResult], hit_rates: List[float]) -> List[int]:
        """检测异常值"""
        if len(hit_rates) < 4:
            return []
        
        Q1 = np.percentile(hit_rates, 25)
        Q3 = np.percentile(hit_rates, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = []
        for i, rate in enumerate(hit_rates):
            if rate < lower_bound or rate > upper_bound:
                outliers.append(results[i].round_number)
        
        return outliers

# ================================
# 批量预测器
# ================================

class BatchPredictor:
    """批量预测器 - 核心预测引擎"""
    
    def __init__(self, analyzer: Happy8Analyzer):
        """初始化批量预测器
        
        Args:
            analyzer: Happy8Analyzer实例
        """
        self.analyzer = analyzer
        self._current_session: Optional[BatchSession] = None
        self._cancel_requested = False
        self._lock = threading.Lock()
    
    def execute_batch_prediction(self, config: BatchConfig, 
                               progress_callback: Optional[Callable[[BatchSession], None]] = None) -> BatchResult:
        """执行批量预测
        
        Args:
            config: 批量预测配置
            progress_callback: 进度回调函数
            
        Returns:
            批量预测结果
            
        Raises:
            InvalidConfigError: 配置参数无效
            PredictionExecutionError: 预测执行失败
        """
        try:
            # 验证配置
            config.__post_init__()
        except ValueError as e:
            raise InvalidConfigError(f"配置参数无效: {e}")
        
        # 创建会话
        session = BatchSession(
            session_id=str(uuid.uuid4()),
            config=config,
            status='running',
            current_round=0,
            results=[],
            statistics=None,
            start_time=datetime.now(),
            end_time=None,
            error_message=None
        )
        
        with self._lock:
            self._current_session = session
            self._cancel_requested = False
        
        # 获取实际开奖号码（用于历史期号验证）
        actual_numbers = self._get_actual_numbers(config.target_issue)
        
        try:
            # 执行批量预测
            if config.max_parallel > 1:
                results = self._execute_parallel_predictions(config, session, actual_numbers, progress_callback)
            else:
                results = self._execute_sequential_predictions(config, session, actual_numbers, progress_callback)
            
            # 生成统计报告
            statistics = StatisticEngine.analyze_batch_results(results)
            
            # 更新会话状态
            session.status = 'completed'
            session.end_time = datetime.now()
            session.results = results
            session.statistics = statistics
            
            # 最终回调
            if progress_callback:
                progress_callback(session)
            
            # 创建批量结果
            return BatchResult(
                session=session,
                predictions=results,
                statistics=statistics,
                execution_summary=self._create_execution_summary(session, statistics)
            )
            
        except Exception as e:
            session.status = 'failed'
            session.end_time = datetime.now()
            session.error_message = str(e)
            
            if progress_callback:
                progress_callback(session)
            
            raise PredictionExecutionError(f"批量预测执行失败: {e}")
        
        finally:
            with self._lock:
                self._current_session = None
    
    def _execute_sequential_predictions(self, config: BatchConfig, session: BatchSession,
                                      actual_numbers: List[int], 
                                      progress_callback: Optional[Callable]) -> List[EnhancedPredictionResult]:
        """执行顺序预测"""
        results = []
        
        for round_num in range(1, config.comparison_times + 1):
            if self._cancel_requested:
                break
            
            session.current_round = round_num
            
            try:
                # 执行单次预测
                pred_result = self._execute_single_prediction(config, round_num)
                
                # 创建增强结果
                enhanced_result = EnhancedPredictionResult.from_prediction_result(
                    pred_result, round_num, session.session_id, actual_numbers
                )
                
                results.append(enhanced_result)
                
                # 进度回调
                if progress_callback:
                    progress_callback(session)
                
            except Exception as e:
                # 记录失败的预测
                failed_result = self._create_failed_result(config, round_num, session.session_id, str(e))
                results.append(failed_result)
                
                print(f"第{round_num}轮预测失败: {e}")
                
                # 如果连续失败超过3次，停止预测
                if len([r for r in results[-3:] if not r.success]) >= 3:
                    raise PredictionExecutionError(f"连续预测失败，停止批量预测")
        
        return results
    
    def _execute_parallel_predictions(self, config: BatchConfig, session: BatchSession,
                                    actual_numbers: List[int],
                                    progress_callback: Optional[Callable]) -> List[EnhancedPredictionResult]:
        """执行并行预测"""
        results = [None] * config.comparison_times
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=config.max_parallel) as executor:
            # 提交所有预测任务
            future_to_round = {
                executor.submit(self._execute_single_prediction, config, round_num): round_num 
                for round_num in range(1, config.comparison_times + 1)
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_round, timeout=config.timeout_seconds * config.comparison_times):
                if self._cancel_requested:
                    # 取消所有未完成的任务
                    for f in future_to_round:
                        f.cancel()
                    break
                
                round_num = future_to_round[future]
                
                try:
                    pred_result = future.result()
                    enhanced_result = EnhancedPredictionResult.from_prediction_result(
                        pred_result, round_num, session.session_id, actual_numbers
                    )
                    results[round_num - 1] = enhanced_result
                    
                except Exception as e:
                    failed_result = self._create_failed_result(config, round_num, session.session_id, str(e))
                    results[round_num - 1] = failed_result
                    print(f"第{round_num}轮预测失败: {e}")
                
                completed_count += 1
                session.current_round = completed_count
                
                # 进度回调
                if progress_callback:
                    progress_callback(session)
        
        # 过滤None值（被取消的任务）
        return [r for r in results if r is not None]
    
    def _execute_single_prediction(self, config: BatchConfig, round_num: int) -> PredictionResult:
        """执行单次预测"""
        return self.analyzer.predict_with_smart_mode(
            target_issue=config.target_issue,
            periods=config.analysis_periods,
            count=config.number_count,
            method=config.prediction_method
        )['prediction_result']
    
    def _get_actual_numbers(self, target_issue: str) -> List[int]:
        """获取实际开奖号码"""
        try:
            data = self.analyzer.load_data()
            target_data = data[data['issue'] == target_issue]
            
            if not target_data.empty:
                numbers_str = target_data.iloc[0]['numbers']
                if isinstance(numbers_str, str):
                    return [int(x.strip()) for x in numbers_str.split(',')]
                elif isinstance(numbers_str, list):
                    return numbers_str
            
            return []  # 如果是未来期号，返回空列表
        
        except Exception as e:
            print(f"获取实际开奖号码失败: {e}")
            return []
    
    def _create_failed_result(self, config: BatchConfig, round_num: int, session_id: str, 
                            error_msg: str) -> EnhancedPredictionResult:
        """创建失败的预测结果"""
        return EnhancedPredictionResult(
            target_issue=config.target_issue,
            analysis_periods=config.analysis_periods,
            method=config.prediction_method,
            predicted_numbers=[],
            confidence_scores=[],
            generation_time=datetime.now(),
            execution_time=0.0,
            parameters={'error': error_msg},
            round_number=round_num,
            hit_numbers=[],
            hit_count=0,
            hit_rate=0.0,
            actual_numbers=[],
            session_id=session_id,
            success=False
        )
    
    def _create_execution_summary(self, session: BatchSession, statistics: StatisticReport) -> Dict[str, Any]:
        """创建执行摘要"""
        total_time = (session.end_time - session.start_time).total_seconds() if session.end_time else 0
        
        return {
            'session_id': session.session_id,
            'total_time_seconds': total_time,
            'average_time_per_round': total_time / session.config.comparison_times if session.config.comparison_times > 0 else 0,
            'success_rate': statistics.success_rounds / statistics.total_rounds if statistics.total_rounds > 0 else 0,
            'config': asdict(session.config),
            'final_status': session.status
        }
    
    def get_current_session(self) -> Optional[BatchSession]:
        """获取当前会话"""
        with self._lock:
            return self._current_session
    
    def cancel_prediction(self) -> bool:
        """取消当前预测"""
        with self._lock:
            if self._current_session and self._current_session.status == 'running':
                self._cancel_requested = True
                self._current_session.status = 'cancelled'
                return True
            return False

# ================================
# 导出引擎
# ================================

class ExportEngine:
    """数据导出引擎"""
    
    @staticmethod
    def export_to_excel(batch_result: BatchResult, filepath: str) -> str:
        """导出到Excel文件
        
        Args:
            batch_result: 批量预测结果
            filepath: 文件路径
            
        Returns:
            实际保存的文件路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 工作表1: 预测结果详情
                predictions_data = []
                for result in batch_result.predictions:
                    predictions_data.append({
                        '轮次': result.round_number,
                        '目标期号': result.target_issue,
                        '预测方法': result.method,
                        '分析期数': result.analysis_periods,
                        '预测号码': ','.join(map(str, result.predicted_numbers)),
                        '实际号码': ','.join(map(str, result.actual_numbers)) if result.actual_numbers else '未知',
                        '命中号码': ','.join(map(str, result.hit_numbers)),
                        '命中数量': result.hit_count,
                        '命中率': f"{result.hit_rate:.2%}",
                        '执行时间(秒)': f"{result.execution_time:.3f}",
                        '生成时间': result.generation_time.strftime('%Y-%m-%d %H:%M:%S'),
                        '是否成功': '成功' if result.success else '失败'
                    })
                
                predictions_df = pd.DataFrame(predictions_data)
                predictions_df.to_excel(writer, sheet_name='预测结果详情', index=False)
                
                # 工作表2: 统计摘要
                stats = batch_result.statistics
                summary_data = [
                    ['统计项目', '数值'],
                    ['总轮次', stats.total_rounds],
                    ['成功轮次', stats.success_rounds],
                    ['成功率', f"{stats.success_rounds / stats.total_rounds:.2%}" if stats.total_rounds > 0 else "0%"],
                    ['平均命中率', f"{stats.avg_hit_rate:.2%}"],
                    ['最高命中率', f"{stats.max_hit_rate:.2%}"],
                    ['最低命中率', f"{stats.min_hit_rate:.2%}"],
                    ['标准差', f"{stats.std_deviation:.4f}"],
                    ['95%置信区间', f"[{stats.confidence_interval[0]:.2%}, {stats.confidence_interval[1]:.2%}]"],
                    ['平均执行时间(秒)', f"{stats.avg_execution_time:.3f}"],
                    ['总执行时间(秒)', f"{stats.total_execution_time:.3f}"]
                ]
                
                summary_df = pd.DataFrame(summary_data[1:], columns=summary_data[0])
                summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
                
                # 工作表3: 命中率分布
                distribution_data = [
                    ['命中率区间', '频次', '百分比']
                ]
                total_success = stats.success_rounds
                for range_name, count in stats.hit_rate_distribution.items():
                    percentage = f"{count / total_success:.1%}" if total_success > 0 else "0%"
                    distribution_data.append([range_name, count, percentage])
                
                distribution_df = pd.DataFrame(distribution_data[1:], columns=distribution_data[0])
                distribution_df.to_excel(writer, sheet_name='命中率分布', index=False)
                
                # 工作表4: 配置信息
                config = batch_result.session.config
                config_data = [
                    ['配置项', '设置值'],
                    ['目标期号', config.target_issue],
                    ['分析期数', config.analysis_periods],
                    ['预测方法', config.prediction_method],
                    ['生成号码数', config.number_count],
                    ['对比次数', config.comparison_times],
                    ['最大并发数', config.max_parallel],
                    ['会话ID', batch_result.session.session_id],
                    ['开始时间', batch_result.session.start_time.strftime('%Y-%m-%d %H:%M:%S')],
                    ['结束时间', batch_result.session.end_time.strftime('%Y-%m-%d %H:%M:%S') if batch_result.session.end_time else '未完成']
                ]
                
                config_df = pd.DataFrame(config_data[1:], columns=config_data[0])
                config_df.to_excel(writer, sheet_name='配置信息', index=False)
            
            return filepath
            
        except Exception as e:
            raise Exception(f"导出Excel文件失败: {e}")
    
    @staticmethod
    def export_to_csv(batch_result: BatchResult, filepath: str) -> str:
        """导出到CSV文件
        
        Args:
            batch_result: 批量预测结果
            filepath: 文件路径
            
        Returns:
            实际保存的文件路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 准备数据
            csv_data = []
            for result in batch_result.predictions:
                csv_data.append({
                    '轮次': result.round_number,
                    '目标期号': result.target_issue,
                    '预测方法': result.method,
                    '分析期数': result.analysis_periods,
                    '预测号码': ','.join(map(str, result.predicted_numbers)),
                    '实际号码': ','.join(map(str, result.actual_numbers)) if result.actual_numbers else '',
                    '命中号码': ','.join(map(str, result.hit_numbers)),
                    '命中数量': result.hit_count,
                    '命中率': result.hit_rate,
                    '执行时间': result.execution_time,
                    '生成时间': result.generation_time.isoformat(),
                    '是否成功': result.success
                })
            
            df = pd.DataFrame(csv_data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            return filepath
            
        except Exception as e:
            raise Exception(f"导出CSV文件失败: {e}")

    @staticmethod
    def generate_download_filename(batch_result: BatchResult, format_type: str = 'excel') -> str:
        """生成下载文件名
        
        Args:
            batch_result: 批量预测结果
            format_type: 格式类型 ('excel' 或 'csv')
            
        Returns:
            文件名
        """
        config = batch_result.session.config
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        extension = 'xlsx' if format_type == 'excel' else 'csv'
        
        filename = f"批量预测_{config.target_issue}_{config.prediction_method}_{config.comparison_times}次_{timestamp}.{extension}"
        
        return filename

if __name__ == "__main__":
    # 测试代码
    print("批量预测对比功能模块已加载")
    print("包含以下主要组件:")
    print("- BatchConfig: 批量预测配置")
    print("- EnhancedPredictionResult: 增强预测结果")  
    print("- StatisticReport: 统计报告")
    print("- BatchPredictor: 批量预测器")
    print("- StatisticEngine: 统计分析引擎")
    print("- ExportEngine: 数据导出引擎")