from pydantic import BaseModel, Field
from typing import Dict, Optional, Any
from datetime import datetime

class WeChatLoginRequest(BaseModel):
    """微信登录请求"""
    code: str = Field(..., description="微信授权码")
    user_info: Optional[Dict[str, Any]] = Field(None, description="微信用户信息")

class RefreshTokenRequest(BaseModel):
    """刷新令牌请求"""
    refresh_token: str = Field(..., description="刷新令牌")

class TokenResponse(BaseModel):
    """令牌响应"""
    access_token: str = Field(..., description="访问令牌")
    refresh_token: str = Field(..., description="刷新令牌")
    token_type: str = Field(default="bearer", description="令牌类型")

class UserInfoResponse(BaseModel):
    """用户信息响应"""
    id: int
    wechat_openid: str
    nickname: Optional[str] = None
    avatar_url: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    created_at: datetime

class WeChatLoginResponse(BaseModel):
    """微信登录响应"""
    access_token: str = Field(..., description="访问令牌")
    refresh_token: str = Field(..., description="刷新令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    user_info: UserInfoResponse = Field(..., description="用户信息")

class LogoutResponse(BaseModel):
    """登出响应"""
    message: str = Field(default="登出成功", description="响应消息")
