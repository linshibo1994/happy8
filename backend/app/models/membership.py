"""会员系统相关数据模型"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Text, 
    ForeignKey, Float, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.models.base import Base


class MembershipLevel(str, enum.Enum):
    """会员等级枚举"""
    FREE = "free"
    VIP = "vip"  
    PREMIUM = "premium"


class OrderStatus(str, enum.Enum):
    """订单状态枚举"""
    PENDING = "pending"      # 待支付
    PAID = "paid"           # 已支付
    CANCELLED = "cancelled"  # 已取消
    REFUNDED = "refunded"   # 已退款
    EXPIRED = "expired"     # 已过期


class PaymentMethod(str, enum.Enum):
    """支付方式枚举"""
    WECHAT_PAY = "wechat_pay"
    ALIPAY = "alipay"


class MembershipPlan(Base):
    """会员套餐表"""
    
    __tablename__ = "membership_plans"

    id = Column(Integer, primary_key=True, index=True, comment="套餐ID")
    name = Column(String(50), nullable=False, comment="套餐名称")
    level = Column(
        SQLEnum(MembershipLevel), 
        nullable=False, 
        comment="会员等级"
    )
    duration_days = Column(Integer, nullable=False, comment="有效期天数")
    price = Column(Integer, nullable=False, comment="价格(分)")
    original_price = Column(Integer, nullable=True, comment="原价(分)")
    
    # 特权配置 JSON格式
    features = Column(Text, nullable=False, comment="特权列表(JSON)")
    
    # 使用限制
    max_predictions_per_day = Column(
        Integer, 
        nullable=True, 
        comment="每日预测次数限制"
    )
    available_algorithms = Column(
        Text, 
        nullable=True, 
        comment="可用算法列表(JSON)"
    )
    
    # 状态和排序
    is_active = Column(Boolean, default=True, nullable=False, comment="是否启用")
    sort_order = Column(Integer, default=0, comment="排序权重")
    
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

    # 关联关系
    orders = relationship("MembershipOrder", back_populates="plan")

    def __repr__(self):
        return f"<MembershipPlan(id={self.id}, name='{self.name}', level='{self.level}')>"


class Membership(Base):
    """用户会员信息表"""
    
    __tablename__ = "memberships"

    id = Column(Integer, primary_key=True, index=True, comment="会员ID")
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False, 
        unique=True,
        comment="用户ID"
    )
    level = Column(
        SQLEnum(MembershipLevel), 
        nullable=False, 
        default=MembershipLevel.FREE, 
        comment="当前会员等级"
    )
    expire_date = Column(DateTime, nullable=True, comment="到期时间")
    auto_renew = Column(Boolean, default=False, comment="是否自动续费")
    
    # 使用统计
    predictions_today = Column(Integer, default=0, comment="今日已用预测次数")
    predictions_total = Column(Integer, default=0, comment="总预测次数")
    
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

    # 关联关系
    user = relationship("User", back_populates="membership")

    def __repr__(self):
        return f"<Membership(id={self.id}, user_id={self.user_id}, level='{self.level}')>"


class MembershipOrder(Base):
    """会员订单表"""
    
    __tablename__ = "membership_orders"

    id = Column(Integer, primary_key=True, index=True, comment="订单ID")
    order_no = Column(
        String(32), 
        unique=True, 
        nullable=False, 
        index=True, 
        comment="订单号"
    )
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        comment="用户ID"
    )
    plan_id = Column(
        Integer, 
        ForeignKey("membership_plans.id"), 
        nullable=False,
        comment="套餐ID"
    )
    
    # 价格信息
    amount = Column(Integer, nullable=False, comment="实际支付金额(分)")
    original_amount = Column(Integer, nullable=True, comment="原价(分)")
    discount_amount = Column(Integer, default=0, comment="优惠金额(分)")
    
    # 订单状态
    status = Column(
        SQLEnum(OrderStatus), 
        default=OrderStatus.PENDING, 
        nullable=False,
        comment="订单状态"
    )
    
    # 支付信息
    pay_method = Column(
        SQLEnum(PaymentMethod), 
        default=PaymentMethod.WECHAT_PAY,
        comment="支付方式"
    )
    transaction_id = Column(String(64), nullable=True, comment="第三方交易号")
    
    # 时间信息
    expire_at = Column(DateTime, nullable=True, comment="订单过期时间")
    paid_at = Column(DateTime, nullable=True, comment="支付时间")
    
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

    # 关联关系
    user = relationship("User", back_populates="orders")
    plan = relationship("MembershipPlan", back_populates="orders")

    def __repr__(self):
        return f"<MembershipOrder(id={self.id}, order_no='{self.order_no}', status='{self.status}')>"