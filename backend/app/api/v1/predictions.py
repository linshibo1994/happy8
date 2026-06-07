"""预测系统API路由"""

import asyncio
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.cache import get_cache, CacheService
from app.core.dependencies import get_current_active_user
from app.core.exceptions import create_success_response, BusinessException
from app.models.user import User
from app.services.prediction_service import PredictionService
from app.api.schemas.prediction_schemas import (
    PredictionRequest,
    PredictionResponse,
    AlgorithmInfo,
    PredictionHistoryResponse,
    LotteryResultResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionStatsResponse,
    UserPredictionLimitResponse
)

router = APIRouter(prefix="/predictions", tags=["预测系统"])


@router.get("/algorithms", response_model=Dict[str, Any])
async def get_available_algorithms(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """获取用户可用的算法列表"""
    prediction_service = PredictionService(db, cache)
    
    algorithms = await prediction_service.get_available_algorithms(current_user.id)
    
    return create_success_response(
        data={"algorithms": algorithms}, 
        message="获取算法列表成功"
    )


@router.get("/limit", response_model=Dict[str, Any])
async def check_prediction_limit(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """检查用户预测限制"""
    prediction_service = PredictionService(db, cache)
    
    can_predict = await prediction_service.check_prediction_limit(current_user.id)
    
    # 获取详细限制信息
    from app.services.membership_service import MembershipService
    membership_service = MembershipService(db, cache)
    membership = await membership_service.get_user_membership(current_user.id)
    
    if membership:
        daily_limit = None
        remaining = None
        
        if membership.level == "FREE":
            daily_limit = 5
            remaining = max(0, 5 - membership.predictions_today)
        elif membership.level == "VIP":
            daily_limit = 50
            remaining = max(0, 50 - membership.predictions_today)
        else:  # PREMIUM
            daily_limit = None
            remaining = None
        
        limit_info = {
            "can_predict": can_predict,
            "predictions_today": membership.predictions_today,
            "daily_limit": daily_limit,
            "membership_level": membership.level,
            "remaining_predictions": remaining,
            "next_reset_time": "00:00:00"  # 每日重置
        }
    else:
        limit_info = {
            "can_predict": False,
            "predictions_today": 0,
            "daily_limit": 0,
            "membership_level": "NONE",
            "remaining_predictions": 0,
            "next_reset_time": None
        }
    
    return create_success_response(
        data=limit_info,
        message="获取预测限制信息成功"
    )


@router.post("/predict", response_model=Dict[str, Any])
async def predict_numbers(
    request: PredictionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """执行号码预测"""
    prediction_service = PredictionService(db, cache)
    
    try:
        result = await prediction_service.predict_numbers(
            user_id=current_user.id,
            algorithm=request.algorithm,
            target_issue=request.target_issue,
            periods=request.periods,
            count=request.count,
            params=request.params
        )
        
        return create_success_response(
            data=result,
            message="预测成功"
        )
        
    except BusinessException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail
        )


@router.post("/batch-predict", response_model=Dict[str, Any])
async def batch_predict_numbers(
    request: BatchPredictionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """批量预测（多算法同时预测）"""
    prediction_service = PredictionService(db, cache)
    
    # 检查用户是否有足够的预测次数（避免并发下超额）
    from app.services.membership_service import MembershipService
    membership_service = MembershipService(db, cache)
    membership = await membership_service.get_user_membership(current_user.id)

    if not membership:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="会员信息不存在或已失效"
        )

    membership_level = (
        membership.level.value
        if hasattr(membership.level, "value")
        else str(membership.level).lower()
    )

    used_today = membership.predictions_today or 0
    if membership_level == "free":
        remaining = max(0, 5 - used_today)
    elif membership_level == "vip":
        remaining = max(0, 50 - used_today)
    else:
        remaining = None

    if remaining is not None and len(request.algorithms) > remaining:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"剩余预测次数不足，当前剩余 {remaining} 次，本次请求 {len(request.algorithms)} 次"
        )

    can_predict = await prediction_service.check_prediction_limit(current_user.id)
    if not can_predict:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="预测次数已达上限"
        )
    
    coroutines = [
        prediction_service.predict_numbers(
            user_id=current_user.id,
            algorithm=algorithm,
            target_issue=request.target_issue,
            periods=request.periods,
            count=request.count,
            params={}
        )
        for algorithm in request.algorithms
    ]

    execution_results = await asyncio.gather(*coroutines, return_exceptions=True)
    results = []
    failed_count = 0
    total_execution_time = 0.0
    failures = []

    for algorithm, result in zip(request.algorithms, execution_results):
        if isinstance(result, Exception):
            failed_count += 1
            failures.append({"algorithm": algorithm, "error": str(result)})
            continue
        results.append(result)
        total_execution_time += result.get("execution_time", 0)
    
    batch_result = {
        "results": results,
        "total_count": len(request.algorithms),
        "success_count": len(results),
        "failed_count": failed_count,
        "total_execution_time": total_execution_time,
        "failures": failures
    }
    
    return create_success_response(
        data=batch_result,
        message=f"批量预测完成，成功{len(results)}个，失败{failed_count}个"
    )


@router.get("/history", response_model=Dict[str, Any])
async def get_prediction_history(
    algorithm: Optional[str] = Query(None, description="筛选算法"),
    limit: int = Query(20, description="返回数量", ge=1, le=100),
    offset: int = Query(0, description="偏移量", ge=0),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """获取用户预测历史"""
    prediction_service = PredictionService(db, cache)
    
    history = await prediction_service.get_prediction_history(
        user_id=current_user.id,
        algorithm=algorithm,
        limit=limit,
        offset=offset
    )
    
    return create_success_response(
        data={
            "history": history,
            "total": len(history),
            "limit": limit,
            "offset": offset
        },
        message="获取预测历史成功"
    )


@router.get("/lottery-results", response_model=Dict[str, Any])
async def get_latest_lottery_results(
    limit: int = Query(10, description="返回数量", ge=1, le=50),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """获取最新开奖结果（无需登录）"""
    prediction_service = PredictionService(db, cache)
    
    results = await prediction_service.get_latest_lottery_results(limit)
    
    return create_success_response(
        data={
            "results": results,
            "total": len(results)
        },
        message="获取开奖结果成功"
    )


@router.get("/stats", response_model=Dict[str, Any])
async def get_prediction_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """获取用户预测统计"""
    from app.models.prediction import PredictionHistory
    from sqlalchemy import func, desc
    from collections import defaultdict
    import json
    
    try:
        # 基础统计
        total_predictions = db.query(func.count(PredictionHistory.id)).filter(
            PredictionHistory.user_id == current_user.id
        ).scalar() or 0
        
        # 今日预测次数
        from datetime import datetime, date
        today = date.today()
        today_predictions = db.query(func.count(PredictionHistory.id)).filter(
            PredictionHistory.user_id == current_user.id,
            func.date(PredictionHistory.created_at) == today
        ).scalar() or 0
        
        # 命中统计
        hit_predictions = db.query(func.count(PredictionHistory.id)).filter(
            PredictionHistory.user_id == current_user.id,
            PredictionHistory.is_hit == True
        ).scalar() or 0
        
        overall_hit_rate = (hit_predictions / total_predictions * 100) if total_predictions > 0 else 0.0
        
        # 最常用算法
        favorite_result = db.query(
            PredictionHistory.algorithm,
            func.count(PredictionHistory.algorithm).label('count')
        ).filter(
            PredictionHistory.user_id == current_user.id
        ).group_by(PredictionHistory.algorithm).order_by(desc('count')).first()
        
        favorite_algorithm = favorite_result.algorithm if favorite_result else None
        
        # 算法统计
        algorithm_stats = defaultdict(lambda: {"count": 0, "hit_count": 0, "hit_rate": 0.0})
        
        algo_results = db.query(
            PredictionHistory.algorithm,
            func.count(PredictionHistory.id).label('total_count'),
            func.sum(func.case([(PredictionHistory.is_hit == True, 1)], else_=0)).label('hit_count')
        ).filter(
            PredictionHistory.user_id == current_user.id
        ).group_by(PredictionHistory.algorithm).all()
        
        for algo_result in algo_results:
            algorithm_stats[algo_result.algorithm] = {
                "count": algo_result.total_count,
                "hit_count": algo_result.hit_count or 0,
                "hit_rate": (algo_result.hit_count or 0) / algo_result.total_count * 100 if algo_result.total_count > 0 else 0.0
            }
        
        # 近期表现（最近10次预测）
        recent_history = db.query(PredictionHistory).filter(
            PredictionHistory.user_id == current_user.id
        ).order_by(desc(PredictionHistory.created_at)).limit(10).all()
        
        recent_performance = []
        for history in recent_history:
            recent_performance.append({
                "date": history.created_at.strftime("%Y-%m-%d"),
                "algorithm": history.algorithm,
                "target_issue": history.target_issue,
                "is_hit": history.is_hit,
                "hit_count": history.hit_count,
                "confidence_score": history.confidence_score
            })
        
        stats = {
            "user_id": current_user.id,
            "total_predictions": total_predictions,
            "today_predictions": today_predictions,
            "hit_predictions": hit_predictions,
            "overall_hit_rate": round(overall_hit_rate, 2),
            "favorite_algorithm": favorite_algorithm,
            "algorithm_stats": dict(algorithm_stats),
            "recent_performance": recent_performance
        }
        
        return create_success_response(
            data=stats,
            message="获取预测统计成功"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取统计信息失败"
        )


@router.get("/diagnostics", response_model=Dict[str, Any])
async def get_algorithm_diagnostics(
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """获取算法诊断信息（管理员功能）"""
    prediction_service = PredictionService(db, cache)
    
    diagnostics = await prediction_service.get_algorithm_diagnostics()
    
    return create_success_response(
        data=diagnostics,
        message="获取算法诊断信息成功"
    )


@router.get("/test/predict", response_model=Dict[str, Any])
async def test_prediction(
    algorithm: str = Query("frequency", description="测试算法"),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """测试预测功能（仅开发环境）"""
    from app.core.config import settings
    
    if not settings.DEBUG:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="接口不存在"
        )
    
    # 创建临时测试用户
    from app.models.user import User
    test_user = db.query(User).filter(User.phone == "test_user").first()
    if not test_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请先创建测试用户"
        )
    
    prediction_service = PredictionService(db, cache)
    
    try:
        # 执行测试预测
        result = await prediction_service.predict_numbers(
            user_id=test_user.id,
            algorithm=algorithm,
            target_issue="2025001",
            periods=30,
            count=5,
            params={}
        )
        
        return create_success_response(
            data={
                **result,
                "test_mode": True,
                "test_user_id": test_user.id
            },
            message="测试预测成功"
        )
        
    except BusinessException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail
        )
