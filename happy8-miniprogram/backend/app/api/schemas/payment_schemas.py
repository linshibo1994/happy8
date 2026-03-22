"""支付系统API模型"""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class CreatePaymentRequest(BaseModel):
    """创建支付请求"""
    order_no: str = Field(..., description="订单号")
    payment_method: str = Field("wechat_pay", description="支付方式")
    
    @field_validator("order_no")
    def validate_order_no(cls, v):
        if not v or len(v) < 10:
            raise ValueError("无效的订单号")
        return v
    
    @field_validator("payment_method")
    def validate_payment_method(cls, v):
        allowed_methods = ["wechat_pay"]
        if v not in allowed_methods:
            raise ValueError("不支持的支付方式")
        return v


class PaymentResponse(BaseModel):
    """支付响应"""
    prepay_id: Optional[str] = Field(None, description="预支付ID")
    order_no: str = Field(..., description="订单号")
    amount: int = Field(..., description="支付金额(分)")
    currency: str = Field("CNY", description="货币类型")
    expire_time: Optional[str] = Field(None, description="过期时间")
    payment_params: Optional[Dict[str, Any]] = Field(None, description="小程序支付参数")


class PaymentStatusResponse(BaseModel):
    """支付状态响应"""
    order_no: str = Field(..., description="订单号")
    status: str = Field(..., description="支付状态")
    transaction_id: Optional[str] = Field(None, description="交易号")
    paid_at: Optional[str] = Field(None, description="支付时间")
    created_at: Optional[str] = Field(None, description="创建时间")
    expire_at: Optional[str] = Field(None, description="过期时间")
    reason: Optional[str] = Field(None, description="失败原因")
    trade_state: Optional[str] = Field(None, description="微信交易状态")


class CancelPaymentRequest(BaseModel):
    """取消支付请求"""
    order_no: str = Field(..., description="订单号")
    reason: Optional[str] = Field(None, description="取消原因")
    
    @field_validator("order_no")
    def validate_order_no(cls, v):
        if not v or len(v) < 10:
            raise ValueError("无效的订单号")
        return v


class PaymentMethodResponse(BaseModel):
    """支付方式响应"""
    methods: list = Field(..., description="支付方式列表")
    default: str = Field(..., description="默认支付方式")


class WeChatNotifyRequest(BaseModel):
    """微信支付回调请求"""
    # 微信回调的请求体会直接处理，不通过Pydantic验证
    pass
