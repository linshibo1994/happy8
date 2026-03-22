from datetime import date
import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.core.dependencies import get_db, get_current_active_user
from app.core.exceptions import create_success_response
from app.models.user import User
from app.models.prediction import LotteryResult
from app.services.lottery_service import LotteryService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/lottery", tags=["开奖"])


def serialize_lottery_result(result: Any) -> Dict[str, Any]:
    """将开奖结果序列化为可返回的字典（兼容ORM对象和字典）。"""
    if not result:
        return {}
    if isinstance(result, dict):
        return result

    return {
        "id": result.id,
        "issue": result.issue,
        "draw_date": result.draw_date.isoformat() if result.draw_date else None,
        "numbers": result.numbers,
        "sum_value": result.sum_value,
        "odd_count": result.odd_count,
        "even_count": result.even_count,
        "big_count": result.big_count,
        "small_count": result.small_count,
        "zone_distribution": result.zone_distribution,
    }


@router.get("/latest")
async def get_latest_results(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """获取最新开奖结果"""
    try:
        lottery_service = LotteryService(db)
        results = await lottery_service.get_latest_results(limit)
        serialized = [serialize_lottery_result(item) for item in results]
        data = {"results": serialized, "total": len(serialized)}
        return create_success_response(data=data, message="获取开奖结果成功")
    except Exception as exc:
        logger.error("获取最新开奖结果失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取开奖结果失败",
        )


@router.get("/history")
async def get_historical_results(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    issue: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """获取历史开奖结果"""
    try:
        lottery_service = LotteryService(db)
        results = await lottery_service.get_historical_results(
            limit=limit,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
            issue=issue,
        )
        serialized = [serialize_lottery_result(item) for item in results]
        data = {"results": serialized, "total": len(serialized)}
        return create_success_response(data=data, message="获取历史开奖结果成功")
    except Exception as exc:
        logger.error("获取历史开奖结果失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取历史开奖结果失败",
        )


@router.get("/results/{issue}")
async def get_result_by_issue(
    issue: str,
    db: Session = Depends(get_db),
):
    """根据期号获取开奖结果"""
    try:
        lottery_service = LotteryService(db)
        result = await lottery_service.get_result_by_issue(issue)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="开奖结果不存在"
            )

        return create_success_response(
            data=serialize_lottery_result(result), message="获取开奖结果成功"
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("获取开奖结果失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取开奖结果失败",
        )


@router.get("/statistics")
async def get_statistics(
    periods: int = Query(100, ge=10, le=1000),
    stat_type: str = Query("frequency", pattern="^(frequency|hot_cold|missing|zone)$"),
    db: Session = Depends(get_db),
):
    """获取开奖统计信息"""
    try:
        lottery_service = LotteryService(db)
        stats = await lottery_service.get_statistics(periods, stat_type)
        return create_success_response(data=stats, message="获取统计信息成功")
    except Exception as exc:
        logger.error("获取统计信息失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取统计信息失败",
        )


@router.get("/trends")
async def get_trends(
    periods: int = Query(50, ge=10, le=200),
    numbers: Optional[List[int]] = Query(None),
    db: Session = Depends(get_db),
):
    """获取号码走势"""
    try:
        lottery_service = LotteryService(db)
        trends = await lottery_service.get_trends(periods, numbers)
        return create_success_response(data=trends, message="获取走势数据成功")
    except Exception as exc:
        logger.error("获取走势数据失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取走势数据失败",
        )


@router.get("/search")
async def search_results(
    numbers: Optional[List[int]] = Query(None),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """搜索开奖结果"""
    try:
        lottery_service = LotteryService(db)
        results = await lottery_service.search_results(
            numbers=numbers,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )
        serialized = [serialize_lottery_result(item) for item in results]
        data = {"results": serialized, "total": len(serialized)}
        return create_success_response(data=data, message="搜索开奖结果成功")
    except Exception as exc:
        logger.error("搜索开奖结果失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="搜索开奖结果失败",
        )


@router.post("/sync")
async def sync_latest_data(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """同步最新开奖数据（需要管理员权限）"""
    try:
        if not getattr(current_user, "is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="权限不足"
            )

        lottery_service = LotteryService(db)
        updated_count = await lottery_service.sync_latest_data()

        return create_success_response(
            data={"updated_count": updated_count}, message="数据同步成功"
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("同步数据失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="同步数据失败",
        )
