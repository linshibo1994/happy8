"""预测系统API模型"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

# 统一算法白名单，避免多处维护导致不一致
SUPPORTED_ALGORITHMS = [
    "frequency",
    "hot_cold",
    "missing",
    "markov",
    "adaptive_markov",
    "transformer",
    "advanced_ensemble",
    "ensemble",
    "ml_ensemble",
    "deep_learning",
    "super_predictor",
    "markov_basic",
    "markov_2nd",
    "markov_3rd",
    "lstm",
    "gnn",
    "bayesian",
    "monte_carlo",
    "clustering",
    "ensemble_basic",
    "high_confidence",
]


class PredictionRequest(BaseModel):
    """预测请求"""
    algorithm: str = Field(..., description="预测算法")
    target_issue: str = Field(..., description="目标期号")
    periods: int = Field(30, description="分析期数", ge=10, le=200)
    count: int = Field(5, description="预测号码数量", ge=1, le=20)
    params: Optional[Dict[str, Any]] = Field(None, description="算法参数")
    
    @field_validator("algorithm")
    def validate_algorithm(cls, v):
        if v not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"不支持的算法: {v}")
        return v
    
    @field_validator("target_issue")
    def validate_target_issue(cls, v):
        if not v or len(v) < 6:
            raise ValueError("无效的期号格式")
        return v


class PredictionResponse(BaseModel):
    """预测响应"""
    predicted_numbers: List[int] = Field(..., description="预测号码")
    confidence_score: float = Field(..., description="置信度")
    analysis_data: Dict[str, Any] = Field(..., description="分析数据")
    algorithm: str = Field(..., description="使用的算法")
    target_issue: str = Field(..., description="目标期号")
    periods: int = Field(..., description="分析期数")
    execution_time: float = Field(..., description="执行时间(秒)")
    is_cached: bool = Field(..., description="是否来自缓存")


class AlgorithmInfo(BaseModel):
    """算法信息"""
    id: int = Field(..., description="算法ID")
    algorithm_name: str = Field(..., description="算法名称")
    display_name: str = Field(..., description="显示名称")
    description: str = Field(..., description="算法描述")
    required_level: str = Field(..., description="所需会员等级")
    has_permission: bool = Field(..., description="是否有权限")
    default_params: Dict[str, Any] = Field(..., description="默认参数")
    avg_execution_time: Optional[float] = Field(None, description="平均执行时间")
    success_rate: Optional[float] = Field(None, description="成功率")
    usage_count: int = Field(0, description="使用次数")


class PredictionHistoryResponse(BaseModel):
    """预测历史响应"""
    id: int = Field(..., description="记录ID")
    algorithm: str = Field(..., description="算法名称")
    target_issue: str = Field(..., description="目标期号")
    periods: int = Field(..., description="分析期数")
    count: int = Field(..., description="预测数量")
    predicted_numbers: List[int] = Field(..., description="预测号码")
    confidence_score: Optional[float] = Field(None, description="置信度")
    actual_numbers: Optional[List[int]] = Field(None, description="实际开奖号码")
    hit_count: Optional[int] = Field(None, description="命中数量")
    hit_rate: Optional[float] = Field(None, description="命中率")
    is_hit: Optional[bool] = Field(None, description="是否命中")
    execution_time: float = Field(..., description="执行时间")
    is_cached: bool = Field(..., description="是否来自缓存")
    created_at: str = Field(..., description="创建时间")


class LotteryResultResponse(BaseModel):
    """开奖结果响应"""
    id: int = Field(..., description="结果ID")
    issue: str = Field(..., description="期号")
    draw_date: str = Field(..., description="开奖日期")
    numbers: List[int] = Field(..., description="开奖号码")
    sum_value: int = Field(..., description="和值")
    odd_count: int = Field(..., description="奇数个数")
    even_count: int = Field(..., description="偶数个数")
    big_count: int = Field(..., description="大数个数")
    small_count: int = Field(..., description="小数个数")
    zone_distribution: Dict[str, int] = Field(..., description="区间分布")


class BatchPredictionRequest(BaseModel):
    """批量预测请求"""
    algorithms: List[str] = Field(..., description="算法列表")
    target_issue: str = Field(..., description="目标期号")
    periods: int = Field(30, description="分析期数", ge=10, le=200)
    count: int = Field(5, description="预测号码数量", ge=1, le=20)
    
    @field_validator("algorithms")
    def validate_algorithms(cls, v):
        if not v or len(v) == 0:
            raise ValueError("算法列表不能为空")
        
        for algo in v:
            if algo not in SUPPORTED_ALGORITHMS:
                raise ValueError(f"不支持的算法: {algo}")
        
        return v


class BatchPredictionResponse(BaseModel):
    """批量预测响应"""
    results: List[PredictionResponse] = Field(..., description="预测结果列表")
    total_count: int = Field(..., description="总数量")
    success_count: int = Field(..., description="成功数量")
    failed_count: int = Field(..., description="失败数量")
    total_execution_time: float = Field(..., description="总执行时间")


class PredictionStatsResponse(BaseModel):
    """预测统计响应"""
    user_id: int = Field(..., description="用户ID")
    total_predictions: int = Field(..., description="总预测次数")
    today_predictions: int = Field(..., description="今日预测次数")
    hit_predictions: int = Field(..., description="命中预测次数")
    overall_hit_rate: float = Field(..., description="总体命中率")
    favorite_algorithm: Optional[str] = Field(None, description="最常用算法")
    algorithm_stats: Dict[str, Dict[str, Any]] = Field(..., description="算法统计")
    recent_performance: List[Dict[str, Any]] = Field(..., description="近期表现")


class UserPredictionLimitResponse(BaseModel):
    """用户预测限制响应"""
    can_predict: bool = Field(..., description="是否可以预测")
    predictions_today: int = Field(..., description="今日已预测次数")
    daily_limit: Optional[int] = Field(None, description="日限制次数")
    membership_level: str = Field(..., description="会员等级")
    remaining_predictions: Optional[int] = Field(None, description="剩余次数")
    next_reset_time: Optional[str] = Field(None, description="下次重置时间")
