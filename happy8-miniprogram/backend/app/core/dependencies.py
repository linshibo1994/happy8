"""认证依赖项"""

from typing import Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.cache import cache_service
from app.core.auth import jwt_manager, get_token_blacklist
from app.core.exceptions import BusinessException
from app.models.user import User


# HTTP Bearer认证
security = HTTPBearer()

async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> int:
    """获取当前用户ID"""
    token = credentials.credentials
    
    # 检查令牌是否在黑名单中
    token_blacklist = get_token_blacklist()
    if await token_blacklist.is_blacklisted(token):
        raise BusinessException.token_invalid("令牌已失效")
    
    # 验证令牌并获取用户ID
    try:
        user_id = jwt_manager.get_user_id_from_token(token)
        return user_id
    except Exception as e:
        raise BusinessException.auth_failed("身份验证失败")


async def get_current_user(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
) -> User:
    """获取当前用户"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise BusinessException.user_not_found("用户不存在")
    
    if not user.is_active:
        raise BusinessException.user_inactive("用户已被禁用")
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """获取当前活跃用户"""
    return current_user


def require_membership_level(required_level: str):
    """要求特定会员等级的装饰器"""
    def dependency(current_user: User = Depends(get_current_active_user)):
        from app.models.membership import MembershipLevel
        
        # 获取用户会员信息
        membership = current_user.membership
        if not membership:
            raise BusinessException.insufficient_permission("需要会员权限")
        
        # 检查会员等级
        level_hierarchy = {
            MembershipLevel.FREE: 0,
            MembershipLevel.VIP: 1,
            MembershipLevel.PREMIUM: 2
        }
        
        user_level = level_hierarchy.get(membership.level, -1)
        normalized_required_level = str(required_level or MembershipLevel.FREE.value).lower()
        try:
            required_level_enum = MembershipLevel(normalized_required_level)
        except ValueError:
            raise BusinessException.validation_error(f"无效的会员等级配置: {required_level}")
        required_level_value = level_hierarchy.get(required_level_enum, 999)
        
        if user_level < required_level_value:
            raise BusinessException.insufficient_permission(f"需要{required_level}会员权限")
        
        return current_user
    
    return dependency


def require_vip_membership():
    """要求VIP会员权限"""
    return require_membership_level("vip")


def require_premium_membership():
    """要求尊享会员权限"""
    return require_membership_level("premium")


async def get_optional_user(
    request: Request,
    db: Session = Depends(get_db)
) -> Optional[User]:
    """获取可选的当前用户（用于可选认证的接口）"""
    authorization = request.headers.get("Authorization")
    
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    token = authorization.split(" ")[1]
    
    try:
        # 检查令牌是否在黑名单中
        token_blacklist = get_token_blacklist()
        if await token_blacklist.is_blacklisted(token):
            return None
        
        # 验证令牌
        user_id = jwt_manager.get_user_id_from_token(token)
        user = db.query(User).filter(User.id == user_id).first()
        
        if user and user.is_active:
            return user
    except:
        pass
    
    return None


class RateLimitDependency:
    """频率限制依赖"""
    
    def __init__(self, max_requests: int, window_seconds: int, action: str = "default"):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.action = action
    
    async def __call__(self, request: Request, current_user: User = Depends(get_current_active_user)):
        from app.core.cache import RateLimiter
        
        rate_limiter = RateLimiter(cache_service)
        identifier = f"user:{current_user.id}"
        
        is_allowed, remaining_time = await rate_limiter.is_allowed(
            identifier, self.action, self.max_requests, self.window_seconds
        )
        
        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"请求过于频繁，请{remaining_time}秒后重试",
                headers={"Retry-After": str(remaining_time)}
            )
        
        return current_user


def rate_limit(max_requests: int, window_seconds: int, action: str = "default"):
    """频率限制装饰器"""
    return RateLimitDependency(max_requests, window_seconds, action)


# 预定义的频率限制
prediction_rate_limit = rate_limit(max_requests=10, window_seconds=60, action="prediction")
login_rate_limit = rate_limit(max_requests=5, window_seconds=300, action="login")
sms_rate_limit = rate_limit(max_requests=3, window_seconds=300, action="sms")
