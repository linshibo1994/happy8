"""会员服务"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.models.user import User
from app.models.membership import (
    MembershipPlan, Membership, MembershipOrder, 
    MembershipLevel, OrderStatus
)
from app.core.exceptions import BusinessException
from app.core.cache import CacheService, CacheKeyManager
from app.core.logging import service_logger as logger


class MembershipService:
    """会员服务"""
    
    def __init__(self, db: Session, cache: CacheService):
        self.db = db
        self.cache = cache
    
    async def get_all_plans(self, include_inactive: bool = False) -> List[MembershipPlan]:
        """获取所有会员套餐"""
        try:
            # 从缓存获取
            cache_key = f"membership_plans:{'all' if include_inactive else 'active'}"
            cached_plans = await self.cache.get(cache_key)
            
            if cached_plans:
                logger.debug("从缓存获取会员套餐")
                # 返回数据库对象
                query = self.db.query(MembershipPlan)
                if not include_inactive:
                    query = query.filter(MembershipPlan.is_active == True)
                return query.order_by(MembershipPlan.sort_order).all()
            
            # 从数据库获取
            query = self.db.query(MembershipPlan)
            if not include_inactive:
                query = query.filter(MembershipPlan.is_active == True)
            
            plans = query.order_by(MembershipPlan.sort_order).all()
            
            # 缓存结果
            plans_data = [
                {
                    "id": plan.id,
                    "name": plan.name,
                    "level": plan.level.value,
                    "price": plan.price,
                    "is_active": plan.is_active
                }
                for plan in plans
            ]
            await self.cache.set(cache_key, plans_data, expire=3600)  # 1小时
            
            return plans
            
        except Exception as e:
            logger.error(f"获取会员套餐失败: {e}")
            raise BusinessException.data_not_found("获取套餐信息失败")
    
    async def get_plan_by_id(self, plan_id: int) -> Optional[MembershipPlan]:
        """根据ID获取会员套餐"""
        try:
            plan = self.db.query(MembershipPlan).filter(
                MembershipPlan.id == plan_id
            ).first()
            
            if not plan:
                raise BusinessException.data_not_found("套餐不存在")
            
            if not plan.is_active:
                raise BusinessException.validation_error("套餐已停用")
            
            return plan
            
        except Exception as e:
            logger.error(f"获取套餐详情失败: {e}")
            if isinstance(e, BusinessException):
                raise
            raise BusinessException.data_not_found("获取套餐详情失败")
    
    async def get_user_membership(self, user_id: int) -> Optional[Membership]:
        """获取用户会员信息"""
        try:
            # 先从缓存获取
            cache_key = CacheKeyManager.user_membership_key(user_id)
            cached_membership = await self.cache.get(cache_key)
            
            if cached_membership:
                # 从数据库获取最新状态
                membership = self.db.query(Membership).filter(
                    Membership.user_id == user_id
                ).first()
                return membership
            
            # 从数据库获取
            membership = self.db.query(Membership).filter(
                Membership.user_id == user_id
            ).first()
            
            if membership:
                # 缓存会员信息
                membership_data = {
                    "user_id": membership.user_id,
                    "level": membership.level.value,
                    "expire_date": membership.expire_date.isoformat() if membership.expire_date else None,
                    "predictions_today": membership.predictions_today,
                    "predictions_total": membership.predictions_total
                }
                await self.cache.set(cache_key, membership_data, expire=1800)  # 30分钟
            
            return membership
            
        except Exception as e:
            logger.error(f"获取用户会员信息失败: {e}")
            return None
    
    async def check_membership_validity(self, user_id: int) -> bool:
        """检查用户会员是否有效"""
        try:
            membership = await self.get_user_membership(user_id)
            if not membership:
                return False
            
            # 免费会员永远有效
            if membership.level == MembershipLevel.FREE:
                return True
            
            # 检查是否过期
            if membership.expire_date and membership.expire_date < datetime.now():
                # 过期，降级为免费会员
                await self._downgrade_to_free(membership)
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"检查会员有效性失败: {e}")
            return False
    
    async def _downgrade_to_free(self, membership: Membership):
        """降级为免费会员"""
        try:
            membership.level = MembershipLevel.FREE
            membership.expire_date = None
            membership.auto_renew = False
            membership.updated_at = datetime.now()
            
            self.db.commit()
            
            # 清除缓存
            cache_key = CacheKeyManager.user_membership_key(membership.user_id)
            await self.cache.delete(cache_key)
            
            logger.info(f"用户会员已降级为免费: {membership.user_id}")
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"降级会员失败: {e}")
    
    async def check_permission(self, user_id: int, required_level: str) -> bool:
        """检查用户权限"""
        try:
            membership = await self.get_user_membership(user_id)
            if not membership:
                return False
            
            # 检查会员是否有效
            if not await self.check_membership_validity(user_id):
                return required_level == "free"
            
            # 权限等级映射
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
                logger.warning(f"未知会员等级配置: {required_level}")
                return False
            required_level_value = level_hierarchy.get(required_level_enum, 999)
            
            return user_level >= required_level_value
            
        except Exception as e:
            logger.error(f"检查用户权限失败: {e}")
            return False
    
    async def create_order(
        self, 
        user_id: int, 
        plan_id: int,
        user_info: Dict[str, Any] = None
    ) -> MembershipOrder:
        """创建会员订单"""
        try:
            # 获取套餐信息
            plan = await self.get_plan_by_id(plan_id)
            
            # 检查用户是否存在
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                raise BusinessException.user_not_found("用户不存在")
            
            # 生成订单号
            import uuid
            order_no = f"MB{datetime.now().strftime('%Y%m%d%H%M%S')}{str(uuid.uuid4())[:8].upper()}"
            
            # 创建订单
            order = MembershipOrder(
                order_no=order_no,
                user_id=user_id,
                plan_id=plan_id,
                amount=plan.price,
                original_amount=plan.original_price or plan.price,
                discount_amount=(plan.original_price or plan.price) - plan.price,
                status=OrderStatus.PENDING,
                expire_at=datetime.now() + timedelta(minutes=30),  # 30分钟过期
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.db.add(order)
            self.db.commit()
            self.db.refresh(order)
            
            logger.info(f"会员订单创建成功: {order_no}, 用户: {user_id}, 套餐: {plan_id}")
            return order
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"创建会员订单失败: {e}")
            if isinstance(e, BusinessException):
                raise
            raise BusinessException.validation_error("订单创建失败")
    
    async def upgrade_membership(
        self, 
        user_id: int, 
        plan: MembershipPlan,
        order: MembershipOrder
    ) -> Membership:
        """升级用户会员"""
        try:
            # 获取或创建会员记录
            membership = await self.get_user_membership(user_id)
            if not membership:
                # 创建新的会员记录
                membership = Membership(
                    user_id=user_id,
                    level=MembershipLevel.FREE,
                    predictions_today=0,
                    predictions_total=0,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                self.db.add(membership)
            
            # 计算新的过期时间
            now = datetime.now()
            current_expire = membership.expire_date
            
            if current_expire and current_expire > now:
                # 当前会员未过期，延长时间
                new_expire = current_expire + timedelta(days=plan.duration_days)
            else:
                # 当前会员已过期或首次购买
                new_expire = now + timedelta(days=plan.duration_days)
            
            # 更新会员信息
            membership.level = plan.level
            membership.expire_date = new_expire
            membership.updated_at = now
            
            # 更新订单状态
            order.status = OrderStatus.PAID
            order.paid_at = now
            order.updated_at = now
            
            self.db.commit()
            self.db.refresh(membership)
            
            # 清除缓存
            cache_key = CacheKeyManager.user_membership_key(user_id)
            await self.cache.delete(cache_key)
            
            logger.info(f"用户会员升级成功: {user_id}, 等级: {plan.level.value}, 过期时间: {new_expire}")
            return membership
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"升级会员失败: {e}")
            raise BusinessException.validation_error("会员升级失败")
    
    async def get_user_orders(
        self, 
        user_id: int, 
        status: Optional[OrderStatus] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[MembershipOrder]:
        """获取用户订单列表"""
        try:
            query = self.db.query(MembershipOrder).filter(
                MembershipOrder.user_id == user_id
            )
            
            if status:
                query = query.filter(MembershipOrder.status == status)
            
            orders = query.order_by(
                MembershipOrder.created_at.desc()
            ).offset(offset).limit(limit).all()
            
            return orders
            
        except Exception as e:
            logger.error(f"获取用户订单失败: {e}")
            raise BusinessException.data_not_found("获取订单列表失败")
    
    async def get_order_by_no(self, order_no: str) -> Optional[MembershipOrder]:
        """根据订单号获取订单"""
        try:
            order = self.db.query(MembershipOrder).filter(
                MembershipOrder.order_no == order_no
            ).first()
            
            return order
            
        except Exception as e:
            logger.error(f"获取订单详情失败: {e}")
            return None
    
    async def cancel_expired_orders(self) -> int:
        """取消过期订单"""
        try:
            now = datetime.now()
            expired_orders = self.db.query(MembershipOrder).filter(
                and_(
                    MembershipOrder.status == OrderStatus.PENDING,
                    MembershipOrder.expire_at < now
                )
            ).all()
            
            count = 0
            for order in expired_orders:
                order.status = OrderStatus.EXPIRED
                order.updated_at = now
                count += 1
            
            if count > 0:
                self.db.commit()
                logger.info(f"已取消 {count} 个过期订单")
            
            return count
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"取消过期订单失败: {e}")
            return 0
    
    async def update_daily_prediction_count(self, user_id: int, count: int = 1) -> bool:
        """更新用户今日预测次数"""
        try:
            membership = await self.get_user_membership(user_id)
            if not membership:
                return False
            
            membership.predictions_today += count
            membership.predictions_total += count
            membership.updated_at = datetime.now()
            
            self.db.commit()
            
            # 更新缓存
            cache_key = CacheKeyManager.user_membership_key(user_id)
            await self.cache.delete(cache_key)
            
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新预测次数失败: {e}")
            return False
    
    async def reset_daily_predictions(self) -> int:
        """重置所有用户的日预测次数（定时任务）"""
        try:
            result = self.db.query(Membership).update(
                {Membership.predictions_today: 0},
                synchronize_session=False
            )
            
            self.db.commit()
            
            # 清除所有用户会员缓存
            await self.cache.clear_pattern("user:*:membership")
            
            logger.info(f"已重置 {result} 个用户的日预测次数")
            return result
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"重置日预测次数失败: {e}")
            return 0
