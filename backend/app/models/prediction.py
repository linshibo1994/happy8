"""预测和彩票数据相关模型"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Text, 
    ForeignKey, Float, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.base import Base


class LotteryResult(Base):
    """彩票开奖结果表"""
    
    __tablename__ = "lottery_results"

    id = Column(Integer, primary_key=True, index=True, comment="结果ID")
    issue = Column(
        String(20), 
        unique=True, 
        nullable=False, 
        index=True, 
        comment="期号"
    )
    draw_date = Column(DateTime, nullable=False, index=True, comment="开奖日期")
    
    # 开奖号码 JSON格式存储 [1,2,3,...,20]
    numbers = Column(JSON, nullable=False, comment="开奖号码(JSON)")
    
    # 统计信息
    sum_value = Column(Integer, nullable=False, comment="号码总和")
    odd_count = Column(Integer, nullable=False, comment="奇数个数")
    even_count = Column(Integer, nullable=False, comment="偶数个数")
    big_count = Column(Integer, nullable=False, comment="大号个数(41-80)")
    small_count = Column(Integer, nullable=False, comment="小号个数(1-40)")
    
    # 区间分布 JSON格式
    zone_distribution = Column(JSON, nullable=True, comment="区间分布统计(JSON)")
    
    # 时间字段
    created_at = Column(
        DateTime, 
        default=func.now(), 
        nullable=False, 
        comment="创建时间"
    )

    def __repr__(self):
        return f"<LotteryResult(id={self.id}, issue='{self.issue}', draw_date='{self.draw_date}')>"


class PredictionHistory(Base):
    """预测历史记录表"""
    
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True, comment="预测ID")
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="用户ID"
    )
    
    # 预测参数
    algorithm = Column(String(50), nullable=False, index=True, comment="预测算法")
    target_issue = Column(String(20), nullable=False, index=True, comment="目标期号")
    periods = Column(Integer, default=30, comment="分析期数")
    count = Column(Integer, default=5, comment="预测号码个数")
    
    # 预测结果 JSON格式
    predicted_numbers = Column(JSON, nullable=False, comment="预测号码(JSON)")
    confidence_score = Column(Float, nullable=True, comment="置信度")
    algorithm_params = Column(JSON, nullable=True, comment="算法参数(JSON)")
    
    # 命中情况
    actual_numbers = Column(JSON, nullable=True, comment="实际开奖号码(JSON)")
    hit_count = Column(Integer, default=0, comment="命中个数")
    hit_rate = Column(Float, nullable=True, comment="命中率")
    is_hit = Column(Boolean, default=False, comment="是否命中")
    
    # 分析数据
    analysis_data = Column(JSON, nullable=True, comment="分析过程数据(JSON)")
    execution_time = Column(Float, nullable=True, comment="执行时间(秒)")
    
    # 缓存标识
    is_cached = Column(Boolean, default=False, comment="是否来自缓存")
    cache_key = Column(String(100), nullable=True, index=True, comment="缓存键")
    
    # 时间字段
    created_at = Column(
        DateTime, 
        default=func.now(), 
        nullable=False, 
        comment="创建时间"
    )

    # 关联关系
    user = relationship("User", back_populates="predictions")

    def __repr__(self):
        return f"<PredictionHistory(id={self.id}, algorithm='{self.algorithm}', target_issue='{self.target_issue}')>"


class AlgorithmConfig(Base):
    """算法配置表"""
    
    __tablename__ = "algorithm_configs"

    id = Column(Integer, primary_key=True, index=True, comment="配置ID")
    algorithm_name = Column(
        String(50), 
        unique=True, 
        nullable=False, 
        comment="算法名称"
    )
    display_name = Column(String(100), nullable=False, comment="显示名称")
    description = Column(Text, nullable=True, comment="算法描述")
    
    # 配置参数 JSON格式
    default_params = Column(JSON, nullable=True, comment="默认参数(JSON)")
    
    # 权限设置
    required_level = Column(
        String(20), 
        default="free", 
        comment="所需会员等级"
    )
    
    # 状态
    is_active = Column(Boolean, default=True, comment="是否启用")
    sort_order = Column(Integer, default=0, comment="排序权重")
    
    # 性能统计
    avg_execution_time = Column(Float, nullable=True, comment="平均执行时间")
    success_rate = Column(Float, nullable=True, comment="成功率")
    usage_count = Column(Integer, default=0, comment="使用次数")
    
    # 时间字段
    created_at = Column(
        DateTime, 
        default=func.now(), 
        nullable=False, 
        comment="创建时间"
    )
    updated_at = Column(
        DateTime, 
        default=func.now(), 
        onupdate=func.now(), 
        nullable=False, 
        comment="更新时间"
    )

    def __repr__(self):
        return f"<AlgorithmConfig(id={self.id}, algorithm_name='{self.algorithm_name}')>"