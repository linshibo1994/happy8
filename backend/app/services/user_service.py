"""用户服务"""

from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.user import User, UserProfile
from app.models.membership import Membership, MembershipLevel
from app.core.exceptions import BusinessException
from app.core.auth import create_token_pair
from app.core.cache import CacheService, CacheKeyManager
from app.core.logging import service_logger as logger
from app.utils.wechat import WeChatAPI


class UserService:
    """用户服务"""
    
    def __init__(self, db: Session, cache: CacheService):
        self.db = db
        self.cache = cache
        self.wechat_api = WeChatAPI(cache)
    
    async def login_with_wechat(self, js_code: str, user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """微信小程序登录"""
        try:
            # 获取微信session信息
            session_data = await self.wechat_api.code_to_session(js_code)
            openid = session_data["openid"]
            unionid = session_data.get("unionid")
            
            # 查找或创建用户
            user = self.db.query(User).filter(User.openid == openid).first()
            
            if not user:
                # 创建新用户
                user = await self._create_new_user(openid, unionid, user_info)
                logger.info(f"新用户注册: {user.id}, openid: {openid}")
            else:
                # 更新用户信息
                if user_info:
                    await self._update_user_info(user, user_info)
                logger.info(f"用户登录: {user.id}, openid: {openid}")
            
            # 生成令牌
            tokens = create_token_pair(user.id)
            
            # 缓存用户信息
            await self._cache_user_info(user)
            
            user_payload = {
                "id": user.id,
                "wechat_openid": user.openid,
                "nickname": user.nickname,
                "avatar_url": user.avatar_url,
                "phone": user.phone,
                "email": user.email,
                "is_new": user.created_at == user.updated_at,
            }

            return {
                **tokens,
                "user": user_payload,
            }
            
        except Exception as e:
            logger.error(f"微信登录失败: {e}")
            if isinstance(e, BusinessException):
                raise
            raise BusinessException.auth_failed("登录失败")
    
    async def _create_new_user(
        self, 
        openid: str, 
        unionid: Optional[str], 
        user_info: Optional[Dict[str, Any]]
    ) -> User:
        """创建新用户"""
        try:
            # 创建用户
            user = User(
                openid=openid,
                unionid=unionid,
                nickname=user_info.get("nickName", "微信用户") if user_info else "微信用户",
                avatar_url=user_info.get("avatarUrl", "") if user_info else "",
                is_active=True,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.db.add(user)
            self.db.flush()  # 获取用户ID
            
            # 创建用户资料
            profile = UserProfile(
                user_id=user.id,
                gender=user_info.get("gender") if user_info else None,
                preferences='{}',
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.db.add(profile)
            
            # 创建免费会员
            membership = Membership(
                user_id=user.id,
                level=MembershipLevel.FREE,
                expire_date=None,  # 免费会员不过期
                auto_renew=False,
                predictions_today=0,
                predictions_total=0,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.db.add(membership)
            self.db.commit()
            
            # 刷新关联对象
            self.db.refresh(user)
            
            return user
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"创建用户失败: {e}")
            raise BusinessException.validation_error("用户创建失败")
    
    async def _update_user_info(self, user: User, user_info: Dict[str, Any]):
        """更新用户信息"""
        try:
            # 更新基本信息
            if "nickName" in user_info:
                user.nickname = user_info["nickName"]
            if "avatarUrl" in user_info:
                user.avatar_url = user_info["avatarUrl"]
            
            user.updated_at = datetime.now()
            
            # 更新用户资料
            if user.profile and "gender" in user_info:
                user.profile.gender = user_info["gender"]
                user.profile.updated_at = datetime.now()
            
            self.db.commit()
            await self._cache_user_info(user)
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新用户信息失败: {e}")
    
    async def _cache_user_info(self, user: User):
        """缓存用户信息"""
        try:
            user_data = {
                "id": user.id,
                "openid": user.openid,
                "nickname": user.nickname,
                "avatar_url": user.avatar_url,
                "phone": user.phone,
                "email": user.email,
                "is_active": user.is_active
            }
            
            cache_key = CacheKeyManager.user_key(user.id)
            await self.cache.set(cache_key, user_data, expire=3600)  # 1小时
            
        except Exception as e:
            logger.warning(f"缓存用户信息失败: {e}")
    
    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """根据ID获取用户"""
        # 先从缓存获取
        cache_key = CacheKeyManager.user_key(user_id)
        cached_user = await self.cache.get(cache_key)
        
        if cached_user:
            # 从数据库获取完整信息
            user = self.db.query(User).filter(User.id == user_id).first()
            return user
        
        # 从数据库获取
        user = self.db.query(User).filter(User.id == user_id).first()
        if user:
            await self._cache_user_info(user)
        
        return user
    
    async def update_user_profile(
        self, 
        user_id: int, 
        profile_data: Dict[str, Any]
    ) -> User:
        """更新用户资料"""
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                raise BusinessException.user_not_found("用户不存在")
            
            # 更新用户基本信息
            if "nickname" in profile_data:
                user.nickname = profile_data["nickname"]
            if "phone" in profile_data:
                # 检查手机号是否已存在
                existing_user = self.db.query(User).filter(
                    and_(User.phone == profile_data["phone"], User.id != user_id)
                ).first()
                if existing_user:
                    raise BusinessException.validation_error("手机号已被使用")
                user.phone = profile_data["phone"]
            if "email" in profile_data:
                # 检查邮箱是否已存在
                existing_user = self.db.query(User).filter(
                    and_(User.email == profile_data["email"], User.id != user_id)
                ).first()
                if existing_user:
                    raise BusinessException.validation_error("邮箱已被使用")
                user.email = profile_data["email"]
            
            user.updated_at = datetime.now()
            
            # 更新用户资料
            if user.profile:
                profile = user.profile
            else:
                profile = UserProfile(user_id=user_id)
                self.db.add(profile)
            
            if "real_name" in profile_data:
                profile.real_name = profile_data["real_name"]
            if "gender" in profile_data:
                profile.gender = profile_data["gender"]
            if "birthday" in profile_data:
                profile.birthday = profile_data["birthday"]
            if "address" in profile_data:
                profile.address = profile_data["address"]
            if "preferences" in profile_data:
                profile.preferences = profile_data["preferences"]
            
            profile.updated_at = datetime.now()
            
            self.db.commit()
            self.db.refresh(user)
            
            # 更新缓存
            await self._cache_user_info(user)
            
            logger.info(f"用户资料更新成功: {user_id}")
            return user
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新用户资料失败: {e}")
            if isinstance(e, BusinessException):
                raise
            raise BusinessException.validation_error("资料更新失败")
    
    async def deactivate_user(self, user_id: int) -> bool:
        """停用用户"""
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                raise BusinessException.user_not_found("用户不存在")
            
            user.is_active = False
            user.updated_at = datetime.now()
            
            self.db.commit()
            
            # 清除缓存
            cache_key = CacheKeyManager.user_key(user_id)
            await self.cache.delete(cache_key)
            
            logger.info(f"用户已停用: {user_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"停用用户失败: {e}")
            if isinstance(e, BusinessException):
                raise
            raise BusinessException.validation_error("用户停用失败")
    
    async def get_user_statistics(self, user_id: int) -> Dict[str, Any]:
        """获取用户统计信息"""
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                raise BusinessException.user_not_found("用户不存在")
            
            # 获取预测统计
            prediction_count = len(user.predictions) if user.predictions else 0
            
            # 获取订单统计
            order_count = len(user.orders) if user.orders else 0
            
            # 获取会员信息
            membership = user.membership
            membership_info = {
                "level": membership.level.value if membership else "free",
                "expire_date": membership.expire_date.isoformat() if membership and membership.expire_date else None,
                "predictions_today": membership.predictions_today if membership else 0,
                "predictions_total": membership.predictions_total if membership else 0
            }
            
            return {
                "user_id": user_id,
                "join_date": user.created_at.isoformat(),
                "prediction_count": prediction_count,
                "order_count": order_count,
                "membership": membership_info
            }
            
        except Exception as e:
            logger.error(f"获取用户统计失败: {e}")
            if isinstance(e, BusinessException):
                raise
            raise BusinessException.validation_error("获取统计信息失败")
