"""用户API路由"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.cache import get_cache, CacheService
from app.core.dependencies import (
    get_current_active_user, 
    get_optional_user,
    login_rate_limit
)
from app.core.exceptions import create_success_response
from app.core.auth import get_token_blacklist, jwt_manager
from app.models.user import User
from app.services.user_service import UserService
from app.api.schemas.user_schemas import (
    WeChatLoginRequest,
    LoginResponse,
    UserProfileRequest,
    UserProfileResponse,
    RefreshTokenRequest,
    RefreshTokenResponse,
    UserStatisticsResponse
)

router = APIRouter(prefix="/users", tags=["用户管理"])
bearer_scheme = HTTPBearer()


@router.post("/login/wechat", response_model=Dict[str, Any])
async def wechat_login(
    request: WeChatLoginRequest,
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """微信小程序登录"""
    user_service = UserService(db, cache)
    
    result = await user_service.login_with_wechat(
        js_code=request.js_code,
        user_info=request.user_info
    )
    
    return create_success_response(data=result, message="登录成功")


@router.post("/refresh", response_model=Dict[str, Any])
async def refresh_token(
    request: RefreshTokenRequest,
    cache: CacheService = Depends(get_cache)
):
    """刷新访问令牌"""
    # 检查刷新令牌是否在黑名单中
    token_blacklist = get_token_blacklist()
    if await token_blacklist.is_blacklisted(request.refresh_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="刷新令牌已失效"
        )
    
    # 生成新的访问令牌
    new_access_token = jwt_manager.refresh_access_token(request.refresh_token)
    
    response_data = {
        "access_token": new_access_token,
        "token_type": "bearer",
        "expires_in": jwt_manager.access_token_expire_minutes * 60
    }
    
    return create_success_response(data=response_data, message="令牌刷新成功")


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_active_user),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """用户登出"""
    token_blacklist_instance = get_token_blacklist()
    await token_blacklist_instance.add_to_blacklist(credentials.credentials)

    return create_success_response(message="登出成功")


@router.get("/profile", response_model=Dict[str, Any])
async def get_user_profile(
    current_user: User = Depends(get_current_active_user)
):
    """获取用户资料"""
    user_data = {
        "id": current_user.id,
        "openid": current_user.openid,
        "nickname": current_user.nickname,
        "avatar_url": current_user.avatar_url,
        "phone": current_user.phone,
        "email": current_user.email,
        "real_name": current_user.profile.real_name if current_user.profile else None,
        "gender": current_user.profile.gender if current_user.profile else None,
        "birthday": current_user.profile.birthday if current_user.profile else None,
        "address": current_user.profile.address if current_user.profile else None,
        "preferences": current_user.profile.preferences if current_user.profile else None,
        "created_at": current_user.created_at,
        "updated_at": current_user.updated_at
    }
    
    return create_success_response(data=user_data, message="获取用户资料成功")


@router.put("/profile", response_model=Dict[str, Any])
async def update_user_profile(
    request: UserProfileRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """更新用户资料"""
    user_service = UserService(db, cache)
    
    # 转换请求数据
    profile_data = request.dict(exclude_unset=True)
    
    updated_user = await user_service.update_user_profile(
        user_id=current_user.id,
        profile_data=profile_data
    )
    
    user_data = {
        "id": updated_user.id,
        "openid": updated_user.openid,
        "nickname": updated_user.nickname,
        "avatar_url": updated_user.avatar_url,
        "phone": updated_user.phone,
        "email": updated_user.email,
        "real_name": updated_user.profile.real_name if updated_user.profile else None,
        "gender": updated_user.profile.gender if updated_user.profile else None,
        "birthday": updated_user.profile.birthday if updated_user.profile else None,
        "address": updated_user.profile.address if updated_user.profile else None,
        "preferences": updated_user.profile.preferences if updated_user.profile else None,
        "created_at": updated_user.created_at,
        "updated_at": updated_user.updated_at
    }
    
    return create_success_response(data=user_data, message="用户资料更新成功")


@router.get("/statistics", response_model=Dict[str, Any])
async def get_user_statistics(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """获取用户统计信息"""
    user_service = UserService(db, cache)
    
    statistics = await user_service.get_user_statistics(current_user.id)
    
    return create_success_response(data=statistics, message="获取用户统计成功")


@router.get("/me", response_model=Dict[str, Any])
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """获取当前用户基本信息"""
    user_data = {
        "id": current_user.id,
        "openid": current_user.openid,
        "nickname": current_user.nickname,
        "avatar_url": current_user.avatar_url,
        "is_active": current_user.is_active,
        "created_at": current_user.created_at
    }
    
    return create_success_response(data=user_data, message="获取用户信息成功")


@router.delete("/account")
async def deactivate_account(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """停用账户"""
    user_service = UserService(db, cache)
    
    success = await user_service.deactivate_user(current_user.id)
    
    if success:
        return create_success_response(message="账户已停用")
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="账户停用失败"
        )
