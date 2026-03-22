"""会员系统API路由"""

import json
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.cache import get_cache, CacheService
from app.core.dependencies import (
    get_current_active_user,
    require_vip_membership,
    require_premium_membership
)
from app.core.exceptions import create_success_response
from app.models.user import User
from app.models.membership import OrderStatus
from app.services.membership_service import MembershipService
from app.api.schemas.membership_schemas import (
    MembershipPlanResponse,
    MembershipResponse,
    CreateOrderRequest,
    MembershipOrderResponse,
    OrderListResponse,
    MembershipStatusResponse,
    UpgradeMembershipRequest
)

router = APIRouter(prefix="/memberships", tags=["会员系统"])


@router.get("/plans", response_model=Dict[str, Any])
async def get_membership_plans(
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """获取所有会员套餐"""
    membership_service = MembershipService(db, cache)
    
    plans = await membership_service.get_all_plans()
    
    plans_data = []
    for plan in plans:
        plan_data = {
            "id": plan.id,
            "name": plan.name,
            "level": plan.level.value,
            "duration_days": plan.duration_days,
            "price": plan.price,
            "original_price": plan.original_price,
            "features": json.loads(plan.features) if plan.features else [],
            "max_predictions_per_day": plan.max_predictions_per_day,
            "available_algorithms": json.loads(plan.available_algorithms) if plan.available_algorithms else [],
            "is_active": plan.is_active,
            "sort_order": plan.sort_order
        }
        plans_data.append(plan_data)
    
    return create_success_response(data=plans_data, message="获取套餐列表成功")


@router.get("/plans/{plan_id}", response_model=Dict[str, Any])
async def get_membership_plan(
    plan_id: int,
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """获取指定会员套餐详情"""
    membership_service = MembershipService(db, cache)
    
    plan = await membership_service.get_plan_by_id(plan_id)
    
    plan_data = {
        "id": plan.id,
        "name": plan.name,
        "level": plan.level.value,
        "duration_days": plan.duration_days,
        "price": plan.price,
        "original_price": plan.original_price,
        "features": json.loads(plan.features) if plan.features else [],
        "max_predictions_per_day": plan.max_predictions_per_day,
        "available_algorithms": json.loads(plan.available_algorithms) if plan.available_algorithms else [],
        "is_active": plan.is_active,
        "sort_order": plan.sort_order,
        "description": plan.description if hasattr(plan, 'description') else ""
    }
    
    return create_success_response(data=plan_data, message="获取套餐详情成功")


@router.get("/status", response_model=Dict[str, Any])
async def get_membership_status(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """获取当前用户会员状态"""
    membership_service = MembershipService(db, cache)
    
    # 获取会员信息
    membership = await membership_service.get_user_membership(current_user.id)
    if not membership:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户会员信息不存在"
        )
    
    # 检查会员有效性
    is_valid = await membership_service.check_membership_validity(current_user.id)
    
    # 计算剩余天数
    days_remaining = None
    if membership.expire_date:
        from datetime import datetime
        delta = membership.expire_date - datetime.now()
        days_remaining = max(0, delta.days)
    
    # 构造响应数据
    membership_data = {
        "id": membership.id,
        "user_id": membership.user_id,
        "level": membership.level.value,
        "expire_date": membership.expire_date.isoformat() if membership.expire_date else None,
        "auto_renew": membership.auto_renew,
        "predictions_today": membership.predictions_today,
        "predictions_total": membership.predictions_total,
        "is_valid": is_valid,
        "days_remaining": days_remaining
    }
    
    # 权限信息
    permissions = {
        "can_use_vip_algorithms": await membership_service.check_permission(current_user.id, "vip"),
        "can_use_premium_algorithms": await membership_service.check_permission(current_user.id, "premium"),
        "can_export_data": await membership_service.check_permission(current_user.id, "premium")
    }
    
    # 使用限制
    limits = {
        "daily_predictions_limit": 5,  # 默认免费限制
        "daily_predictions_used": membership.predictions_today,
        "total_predictions": membership.predictions_total
    }
    
    # 根据会员等级调整限制
    if membership.level.value == "vip":
        limits["daily_predictions_limit"] = 50
    elif membership.level.value == "premium":
        limits["daily_predictions_limit"] = None  # 无限制
    
    response_data = {
        "membership": membership_data,
        "permissions": permissions,
        "limits": limits
    }
    
    return create_success_response(data=response_data, message="获取会员状态成功")


@router.post("/orders", response_model=Dict[str, Any])
async def create_membership_order(
    request: CreateOrderRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """创建会员购买订单"""
    membership_service = MembershipService(db, cache)
    
    # 创建订单
    order = await membership_service.create_order(
        user_id=current_user.id,
        plan_id=request.plan_id
    )
    
    order_data = {
        "id": order.id,
        "order_no": order.order_no,
        "user_id": order.user_id,
        "plan_id": order.plan_id,
        "amount": order.amount,
        "original_amount": order.original_amount,
        "discount_amount": order.discount_amount,
        "status": order.status.value,
        "expire_at": order.expire_at.isoformat() if order.expire_at else None,
        "created_at": order.created_at.isoformat()
    }
    
    return create_success_response(data=order_data, message="订单创建成功")


@router.get("/orders", response_model=Dict[str, Any])
async def get_user_orders(
    status: Optional[str] = Query(None, description="订单状态过滤"),
    limit: int = Query(20, ge=1, le=100, description="每页数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """获取用户会员订单列表"""
    membership_service = MembershipService(db, cache)
    
    # 转换状态参数
    order_status = None
    if status:
        try:
            order_status = OrderStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="无效的订单状态"
            )
    
    # 获取订单列表
    orders = await membership_service.get_user_orders(
        user_id=current_user.id,
        status=order_status,
        limit=limit + 1,  # 多查询1个用于判断是否有更多
        offset=offset
    )
    
    has_more = len(orders) > limit
    if has_more:
        orders = orders[:limit]
    
    # 构造响应数据
    orders_data = []
    for order in orders:
        order_data = {
            "id": order.id,
            "order_no": order.order_no,
            "user_id": order.user_id,
            "plan_id": order.plan_id,
            "plan_name": order.plan.name if order.plan else "",
            "amount": order.amount,
            "original_amount": order.original_amount,
            "discount_amount": order.discount_amount,
            "status": order.status.value,
            "pay_method": order.pay_method.value if order.pay_method else None,
            "transaction_id": order.transaction_id,
            "expire_at": order.expire_at.isoformat() if order.expire_at else None,
            "paid_at": order.paid_at.isoformat() if order.paid_at else None,
            "created_at": order.created_at.isoformat(),
            "updated_at": order.updated_at.isoformat()
        }
        orders_data.append(order_data)
    
    response_data = {
        "orders": orders_data,
        "total": len(orders_data),
        "has_more": has_more
    }
    
    return create_success_response(data=response_data, message="获取订单列表成功")


@router.get("/orders/{order_no}", response_model=Dict[str, Any])
async def get_order_detail(
    order_no: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """获取订单详情"""
    membership_service = MembershipService(db, cache)
    
    order = await membership_service.get_order_by_no(order_no)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="订单不存在"
        )
    
    # 检查订单所有权
    if order.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权访问此订单"
        )
    
    order_data = {
        "id": order.id,
        "order_no": order.order_no,
        "user_id": order.user_id,
        "plan_id": order.plan_id,
        "plan_name": order.plan.name if order.plan else "",
        "amount": order.amount,
        "original_amount": order.original_amount,
        "discount_amount": order.discount_amount,
        "status": order.status.value,
        "pay_method": order.pay_method.value if order.pay_method else None,
        "transaction_id": order.transaction_id,
        "expire_at": order.expire_at.isoformat() if order.expire_at else None,
        "paid_at": order.paid_at.isoformat() if order.paid_at else None,
        "created_at": order.created_at.isoformat(),
        "updated_at": order.updated_at.isoformat()
    }
    
    return create_success_response(data=order_data, message="获取订单详情成功")


@router.post("/upgrade", response_model=Dict[str, Any])
async def upgrade_membership(
    request: UpgradeMembershipRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """升级会员（支付成功后调用）"""
    membership_service = MembershipService(db, cache)
    
    # 获取订单
    order = await membership_service.get_order_by_no(request.order_no)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="订单不存在"
        )
    
    # 检查订单所有权
    if order.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权操作此订单"
        )
    
    # 检查订单状态
    if order.status != OrderStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="订单状态不允许升级"
        )
    
    # 获取套餐信息
    plan = await membership_service.get_plan_by_id(order.plan_id)
    
    # 升级会员
    membership = await membership_service.upgrade_membership(
        user_id=current_user.id,
        plan=plan,
        order=order
    )
    
    membership_data = {
        "id": membership.id,
        "user_id": membership.user_id,
        "level": membership.level.value,
        "expire_date": membership.expire_date.isoformat() if membership.expire_date else None,
        "auto_renew": membership.auto_renew,
        "predictions_today": membership.predictions_today,
        "predictions_total": membership.predictions_total,
        "updated_at": membership.updated_at.isoformat()
    }
    
    return create_success_response(data=membership_data, message="会员升级成功")


@router.get("/permissions", response_model=Dict[str, Any])
async def check_permissions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """检查用户权限"""
    membership_service = MembershipService(db, cache)
    
    permissions = {
        "free": await membership_service.check_permission(current_user.id, "free"),
        "vip": await membership_service.check_permission(current_user.id, "vip"),
        "premium": await membership_service.check_permission(current_user.id, "premium")
    }
    
    return create_success_response(data=permissions, message="权限检查完成")