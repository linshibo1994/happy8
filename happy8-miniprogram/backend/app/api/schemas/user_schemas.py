"""用户API Pydantic模型"""

import json
from typing import Optional, Dict, Any, ForwardRef
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, field_validator


# 前向引用
UserInfoRef = ForwardRef('UserInfo')


class WeChatLoginRequest(BaseModel):
    """微信登录请求"""
    js_code: str = Field(..., description="微信授权码")
    user_info: Optional[Dict[str, Any]] = Field(None, description="用户信息")
    
    @field_validator("js_code")
    def validate_js_code(cls, v):
        if not v or len(v) < 10:
            raise ValueError("无效的授权码")
        return v


class UserInfo(BaseModel):
    """用户信息"""
    id: int = Field(..., description="用户ID")
    openid: str = Field(..., description="微信OpenID")
    nickname: str = Field(..., description="昵称")
    avatar_url: Optional[str] = Field(None, description="头像URL")
    phone: Optional[str] = Field(None, description="手机号")
    email: Optional[str] = Field(None, description="邮箱")
    is_new: bool = Field(False, description="是否新用户")
    
    model_config = ConfigDict(from_attributes=True)


class LoginResponse(BaseModel):
    """登录响应"""
    access_token: str = Field(..., description="访问令牌")
    refresh_token: str = Field(..., description="刷新令牌")
    token_type: str = Field("bearer", description="令牌类型")
    expires_in: int = Field(..., description="过期时间（秒）")
    user: UserInfo = Field(..., description="用户信息")


class UserProfileRequest(BaseModel):
    """用户资料更新请求"""
    nickname: Optional[str] = Field(None, max_length=50, description="昵称")
    phone: Optional[str] = Field(None, pattern=r'^1[3-9]\d{9}$', description="手机号")
    email: Optional[str] = Field(None, pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', description="邮箱")
    real_name: Optional[str] = Field(None, max_length=20, description="真实姓名")
    gender: Optional[str] = Field(None, pattern=r'^(male|female|unknown)$', description="性别")
    birthday: Optional[datetime] = Field(None, description="生日")
    address: Optional[str] = Field(None, max_length=200, description="地址")
    preferences: Optional[str] = Field(None, description="偏好设置（JSON字符串）")
    
    @field_validator("nickname")
    def validate_nickname(cls, v):
        if v and (len(v) < 2 or len(v) > 50):
            raise ValueError("昵称长度应在2-50个字符之间")
        return v
    
    @field_validator("preferences")
    def validate_preferences(cls, v):
        if v:
            try:
                json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("偏好设置必须是有效的JSON字符串")
        return v


class UserProfileResponse(BaseModel):
    """用户资料响应"""
    id: int = Field(..., description="用户ID")
    openid: str = Field(..., description="微信OpenID")
    nickname: str = Field(..., description="昵称")
    avatar_url: Optional[str] = Field(None, description="头像URL")
    phone: Optional[str] = Field(None, description="手机号")
    email: Optional[str] = Field(None, description="邮箱")
    real_name: Optional[str] = Field(None, description="真实姓名")
    gender: Optional[str] = Field(None, description="性别")
    birthday: Optional[datetime] = Field(None, description="生日")
    address: Optional[str] = Field(None, description="地址")
    preferences: Optional[str] = Field(None, description="偏好设置")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    
    model_config = ConfigDict(from_attributes=True)


class RefreshTokenRequest(BaseModel):
    """刷新令牌请求"""
    refresh_token: str = Field(..., description="刷新令牌")


class RefreshTokenResponse(BaseModel):
    """刷新令牌响应"""
    access_token: str = Field(..., description="新的访问令牌")
    token_type: str = Field("bearer", description="令牌类型")
    expires_in: int = Field(..., description="过期时间（秒）")


class UserStatisticsResponse(BaseModel):
    """用户统计响应"""
    user_id: int = Field(..., description="用户ID")
    join_date: str = Field(..., description="加入日期")
    prediction_count: int = Field(..., description="预测次数")
    order_count: int = Field(..., description="订单数量")
    membership: Dict[str, Any] = Field(..., description="会员信息")
