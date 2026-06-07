from datetime import date, datetime
import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.cache import get_cache, CacheService
from app.core.dependencies import get_current_active_user
from app.core.exceptions import create_success_response
from app.models.user import User
from app.models.prediction import AlgorithmConfig, PredictionHistory
from app.services.prediction_service import PredictionService
from app.services.happy8_algorithm_adapter import Happy8AlgorithmAdapter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/algorithms", tags=["算法"])


@router.get("/")
async def get_available_algorithms(
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache),
    current_user: User = Depends(get_current_active_user),
):
    """获取用户可用的算法列表"""
    try:
        prediction_service = PredictionService(db, cache)
        algorithms = await prediction_service.get_available_algorithms(current_user.id)
        return create_success_response(data=algorithms, message="获取算法列表成功")
    except Exception as exc:
        logger.error("获取算法列表失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取算法列表失败",
        )


@router.get("/user-stats")
async def get_user_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """获取用户预测统计信息"""
    try:
        total_predictions = (
            db.query(func.count(PredictionHistory.id))
            .filter(PredictionHistory.user_id == current_user.id)
            .scalar()
            or 0
        )

        today = date.today()
        first_day = datetime(today.year, today.month, 1)

        today_predictions = (
            db.query(func.count(PredictionHistory.id))
            .filter(
                PredictionHistory.user_id == current_user.id,
                func.date(PredictionHistory.created_at) == today,
            )
            .scalar()
            or 0
        )

        this_month_predictions = (
            db.query(func.count(PredictionHistory.id))
            .filter(
                PredictionHistory.user_id == current_user.id,
                PredictionHistory.created_at >= first_day,
            )
            .scalar()
            or 0
        )

        hit_predictions = (
            db.query(
                func.sum(
                    func.case(
                        [(PredictionHistory.is_hit == True, 1)], else_=0  # noqa: E712
                    )
                )
            )
            .filter(PredictionHistory.user_id == current_user.id)
            .scalar()
            or 0
        )

        overall_hit_rate = (
            (hit_predictions / total_predictions) * 100 if total_predictions else 0.0
        )

        favorite_result = (
            db.query(
                PredictionHistory.algorithm,
                func.count(PredictionHistory.id).label("count"),
            )
            .filter(PredictionHistory.user_id == current_user.id)
            .group_by(PredictionHistory.algorithm)
            .order_by(func.count(PredictionHistory.id).desc())
            .first()
        )
        favorite_algorithm = favorite_result.algorithm if favorite_result else None

        algorithm_stats_rows = (
            db.query(
                PredictionHistory.algorithm,
                func.count(PredictionHistory.id).label("total"),
                func.sum(
                    func.case(
                        [(PredictionHistory.is_hit == True, 1)], else_=0  # noqa: E712
                    )
                ).label("hits"),
            )
            .filter(PredictionHistory.user_id == current_user.id)
            .group_by(PredictionHistory.algorithm)
            .all()
        )

        algorithm_stats = {}
        best_hit_rate = 0.0
        for row in algorithm_stats_rows:
            total = row.total or 0
            hits = row.hits or 0
            hit_rate = (hits / total * 100) if total else 0.0
            algorithm_stats[row.algorithm] = {
                "count": total,
                "hit_count": hits,
                "hit_rate": round(hit_rate, 2),
            }
            best_hit_rate = max(best_hit_rate, hit_rate)

        recent_history = (
            db.query(PredictionHistory)
            .filter(PredictionHistory.user_id == current_user.id)
            .order_by(PredictionHistory.created_at.desc())
            .limit(10)
            .all()
        )

        recent_performance = [
            {
                "date": history.created_at.strftime("%Y-%m-%d"),
                "algorithm": history.algorithm,
                "target_issue": history.target_issue,
                "is_hit": history.is_hit,
                "hit_count": history.hit_count,
                "confidence_score": history.confidence_score,
            }
            for history in recent_history
        ]

        data = {
            "user_id": current_user.id,
            "total_predictions": total_predictions,
            "today_predictions": today_predictions,
            "hit_predictions": hit_predictions,
            "overall_hit_rate": round(overall_hit_rate, 2),
            "favorite_algorithm": favorite_algorithm,
            "this_month_predictions": this_month_predictions,
            "best_hit_rate": round(best_hit_rate, 2),
            "algorithm_stats": algorithm_stats,
            "recent_performance": recent_performance,
        }

        return create_success_response(data=data, message="获取用户统计成功")
    except Exception as exc:
        logger.error("获取用户统计失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户统计失败",
        )


@router.get("/{algorithm_name}")
async def get_algorithm_detail(
    algorithm_name: str,
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache),
    current_user: User = Depends(get_current_active_user),
):
    """获取算法详细信息"""
    try:
        prediction_service = PredictionService(db, cache)
        algorithms = await prediction_service.get_available_algorithms(current_user.id)
        for algorithm in algorithms:
            if algorithm.get("algorithm_name") == algorithm_name:
                return create_success_response(data=algorithm, message="获取算法详情成功")

        # 如果在可用列表中找不到，尝试从适配器获取基础信息
        adapter = Happy8AlgorithmAdapter()
        mapped_algorithm = prediction_service.algorithm_mapping.get(
            algorithm_name, algorithm_name
        )
        adapter_info = await adapter.get_algorithm_info(mapped_algorithm)
        if not adapter_info.get("available", False):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="算法不存在"
            )

        detail = {
            "algorithm_name": algorithm_name,
            "display_name": adapter_info.get("description", algorithm_name),
            "description": adapter_info.get("description") or "",
            "required_level": "free",
            "complexity": adapter_info.get("complexity", "unknown"),
            "success_rate": 0.0,
            "usage_count": 0,
            "has_permission": False,
            "is_recommended": False,
            "default_params": {},
            "original_algorithm": mapped_algorithm,
        }
        return create_success_response(data=detail, message="获取算法详情成功")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("获取算法详情失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取算法详情失败",
        )


@router.get("/{algorithm_name}/stats")
async def get_algorithm_stats(
    algorithm_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """获取算法统计信息"""
    try:
        algorithm = (
            db.query(AlgorithmConfig)
            .filter(AlgorithmConfig.algorithm_name == algorithm_name)
            .first()
        )
        if not algorithm:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="算法不存在"
            )

        total_predictions = (
            db.query(func.count(PredictionHistory.id))
            .filter(PredictionHistory.algorithm == algorithm_name)
            .scalar()
            or 0
        )
        hit_predictions = (
            db.query(
                func.sum(
                    func.case(
                        [(PredictionHistory.is_hit == True, 1)], else_=0  # noqa: E712
                    )
                )
            )
            .filter(PredictionHistory.algorithm == algorithm_name)
            .scalar()
            or 0
        )
        avg_confidence = (
            db.query(func.avg(PredictionHistory.confidence_score))
            .filter(PredictionHistory.algorithm == algorithm_name)
            .scalar()
            or 0.0
        )
        last_used = (
            db.query(func.max(PredictionHistory.created_at))
            .filter(PredictionHistory.algorithm == algorithm_name)
            .scalar()
        )

        success_rate = algorithm.success_rate or (
            (hit_predictions / total_predictions) if total_predictions else 0.0
        )

        stats = {
            "algorithm_name": algorithm_name,
            "usage_count": algorithm.usage_count or total_predictions,
            "success_rate": float(success_rate),
            "avg_confidence": float(avg_confidence),
            "last_used": last_used.isoformat() if last_used else None,
            "predictions_total": total_predictions,
            "hit_predictions": hit_predictions,
        }

        return create_success_response(data=stats, message="获取算法统计成功")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("获取算法统计失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取算法统计失败",
        )


@router.put("/{algorithm_name}/params")
async def update_algorithm_params(
    algorithm_name: str,
    params: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """更新算法参数（仅限管理员）"""
    try:
        if not getattr(current_user, "is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="权限不足"
            )

        algorithm = (
            db.query(AlgorithmConfig)
            .filter(AlgorithmConfig.algorithm_name == algorithm_name)
            .first()
        )
        if not algorithm:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="算法不存在"
            )

        algorithm.default_params = params
        db.commit()
        return create_success_response(message="算法参数更新成功")
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        logger.error("更新算法参数失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新算法参数失败",
        )
