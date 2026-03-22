from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.api.schemas.auth_schemas import (
    RefreshTokenRequest,
    WeChatLoginRequest,
)
from app.core.auth import (
    create_access_token,
    create_refresh_token,
    get_token_blacklist,
)
from app.core.cache import CacheService, get_cache
from app.core.dependencies import get_current_user, get_db
from app.core.exceptions import create_success_response
from app.models.user import User
from app.services.user_service import UserService

import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["认证"])
bearer_scheme = HTTPBearer()


@router.post("/wechat-login")
async def wechat_login(
    login_data: WeChatLoginRequest,
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache),
):
    """微信登录"""
    try:
        user_service = UserService(db, cache)
        login_result = await user_service.login_with_wechat(
            js_code=login_data.code,
            user_info=login_data.user_info,
        )

        response_data = {
            "access_token": login_result["access_token"],
            "refresh_token": login_result["refresh_token"],
            "token_type": login_result.get("token_type", "bearer"),
            "expires_in": login_result.get("expires_in"),
            "user_info": {
                "id": login_result["user"]["id"],
                "wechat_openid": login_result["user"].get("wechat_openid"),
                "nickname": login_result["user"].get("nickname"),
                "avatar_url": login_result["user"].get("avatar_url"),
                "phone": login_result["user"].get("phone"),
                "email": login_result["user"].get("email"),
                "is_new": login_result["user"].get("is_new", False),
            },
        }

        return create_success_response(data=response_data, message="登录成功")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"微信登录失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登录服务暂时不可用",
        )


@router.post("/refresh")
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    db: Session = Depends(get_db),
):
    """刷新访问令牌"""
    try:
        from app.core.auth import verify_refresh_token

        payload = verify_refresh_token(refresh_data.refresh_token)
        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的刷新令牌",
            )

        user = db.query(User).filter(User.id == int(user_id)).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户不存在",
            )

        data = {
            "access_token": create_access_token({"sub": user_id}),
            "refresh_token": create_refresh_token({"sub": user_id}),
            "token_type": "bearer",
        }

        return create_success_response(data=data, message="令牌刷新成功")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"刷新令牌失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="刷新令牌失败",
        )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """登出"""
    try:
        token_blacklist = get_token_blacklist()
        await token_blacklist.add_to_blacklist(credentials.credentials)
        return create_success_response(message="登出成功")
    except Exception as e:
        logger.error(f"登出失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登出失败",
        )
