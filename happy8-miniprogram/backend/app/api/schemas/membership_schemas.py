"""会员系统API模型"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum


class MembershipLevelEnum(str, Enum):
    """会员等级枚举"""
    FREE = "free"
    VIP = "vip"
    PREMIUM = "premium"


class OrderStatusEnum(str, Enum):
    """订单状态枚举"""
    PENDING = "pending"
    PAID = "paid"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    EXPIRED = "expired"


class MembershipPlanResponse(BaseModel):
    """会员套餐响应"""
    id: int = Field(..., description="套餐ID")
    name: str = Field(..., description="套餐名称")
    level: MembershipLevelEnum = Field(..., description="会员等级")
    duration_days: int = Field(..., description="有效期天数")
    price: int = Field(..., description="价格(分)")
    original_price: Optional[int] = Field(None, description="原价(分)")
    features: str = Field(..., description="特权列表(JSON)")
    max_predictions_per_day: Optional[int] = Field(None, description="每日预测次数限制")
    available_algorithms: Optional[str] = Field(None, description="可用算法列表(JSON)")
    is_active: bool = Field(..., description="是否启用")
    sort_order: Optional[int] = Field(None, description="排序权重")
    
    model_config = ConfigDict(from_attributes=True)


class MembershipResponse(BaseModel):
    """会员信息响应"""
    id: int = Field(..., description="会员ID")
    user_id: int = Field(..., description="用户ID")
    level: MembershipLevelEnum = Field(..., description="当前会员等级")
    expire_date: Optional[datetime] = Field(None, description="到期时间")
    auto_renew: Optional[bool] = Field(None, description="是否自动续费")
    predictions_today: Optional[int] = Field(None, description="今日已用预测次数")
    predictions_total: Optional[int] = Field(None, description="总预测次数")
    is_valid: bool = Field(..., description="会员是否有效")
    days_remaining: Optional[int] = Field(None, description="剩余天数")
    
    model_config = ConfigDict(from_attributes=True)


class CreateOrderRequest(BaseModel):
    """创建订单请求"""
    plan_id: int = Field(..., description="套餐ID")
    
    @field_validator("plan_id")
    def validate_plan_id(cls, v):
        if v <= 0:
            raise ValueError("套餐ID无效")
        return v


class MembershipOrderResponse(BaseModel):
    """会员订单响应"""
    id: int = Field(..., description="订单ID")
    order_no: str = Field(..., description="订单号")
    user_id: int = Field(..., description="用户ID")
    plan_id: int = Field(..., description="套餐ID")
    plan_name: Optional[str] = Field(None, description="套餐名称")
    amount: int = Field(..., description="实际支付金额(分)")
    original_amount: Optional[int] = Field(None, description="原价(分)")
    discount_amount: Optional[int] = Field(None, description="优惠金额(分)")
    status: OrderStatusEnum = Field(..., description="订单状态")
    pay_method: Optional[str] = Field(None, description="支付方式")
    transaction_id: Optional[str] = Field(None, description="第三方交易号")
    expire_at: Optional[datetime] = Field(None, description="订单过期时间")
    paid_at: Optional[datetime] = Field(None, description="支付时间")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    
    model_config = ConfigDict(from_attributes=True)


class OrderListResponse(BaseModel):
    """订单列表响应"""
    orders: List[MembershipOrderResponse] = Field(..., description="订单列表")
    total: int = Field(..., description="总数量")
    has_more: bool = Field(..., description="是否有更多")


class MembershipStatusResponse(BaseModel):
    """会员状态响应"""
    membership: MembershipResponse = Field(..., description="会员信息")
    permissions: dict = Field(..., description="权限信息")
    limits: dict = Field(..., description="使用限制")


class UpgradeMembershipRequest(BaseModel):
    """升级会员请求"""
    order_no: str = Field(..., description="订单号")
    
    @field_validator("order_no")
    def validate_order_no(cls, v):
        if not v or len(v) < 10:
            raise ValueError("无效的订单号")
        return v
