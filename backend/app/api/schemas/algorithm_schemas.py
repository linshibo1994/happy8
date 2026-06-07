from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class AlgorithmInfo(BaseModel):
    """算法信息"""
    algorithm_name: str = Field(..., description="算法名称")
    display_name: str = Field(..., description="显示名称") 
    description: str = Field(..., description="算法描述")
    required_level: str = Field(..., description="需要的会员等级")
    complexity: str = Field(..., description="复杂度")
    success_rate: float = Field(..., description="成功率")
    usage_count: int = Field(default=0, description="使用次数")
    has_permission: bool = Field(default=False, description="用户是否有权限")
    is_recommended: bool = Field(default=False, description="是否推荐")
    default_params: Optional[Dict[str, Any]] = Field(default=None, description="默认参数")

class LotteryResultResponse(BaseModel):
    """开奖结果响应"""
    id: int
    issue: str = Field(..., description="期号")
    draw_date: datetime = Field(..., description="开奖日期")
    numbers: List[int] = Field(..., description="开奖号码")
    sum_value: int = Field(..., description="和值")
    odd_count: int = Field(..., description="奇数个数")
    even_count: int = Field(..., description="偶数个数")
    big_count: int = Field(..., description="大数个数")
    small_count: int = Field(..., description="小数个数")
    created_at: datetime

class LotteryStatsResponse(BaseModel):
    """开奖统计响应"""
    type: str = Field(..., description="统计类型")
    periods: int = Field(..., description="统计期数")
    data: Dict[str, Any] = Field(..., description="统计数据")

class LotteryTrendsResponse(BaseModel):
    """开奖走势响应"""
    type: str = Field(..., description="走势类型")
    periods: int = Field(..., description="统计期数")
    trends: Dict[str, Any] = Field(..., description="走势数据")

class AlgorithmStatsResponse(BaseModel):
    """算法统计响应"""
    algorithm_name: str = Field(..., description="算法名称")
    usage_count: int = Field(..., description="使用次数")
    success_rate: float = Field(..., description="成功率")
    avg_confidence: float = Field(..., description="平均置信度")
    last_used: Optional[datetime] = Field(default=None, description="最后使用时间")

class UserStatsResponse(BaseModel):
    """用户统计响应"""
    total_predictions: int = Field(default=0, description="总预测次数")
    overall_hit_rate: float = Field(default=0.0, description="总命中率")
    today_predictions: int = Field(default=0, description="今日预测次数")
    this_month_predictions: int = Field(default=0, description="本月预测次数")
    favorite_algorithm: Optional[str] = Field(default=None, description="最常用算法")
    best_hit_rate: float = Field(default=0.0, description="最佳命中率")