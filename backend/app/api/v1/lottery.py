from datetime import date
import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.core.dependencies import get_db, get_current_active_user
from app.core.exceptions import create_success_response
from app.models.user import User
from app.models.prediction import LotteryResult
from app.services.lottery_sandbox_service import LotterySandboxService
from app.services.lottery_service import LotteryService
from app.services.lottery_sync_service import lottery_sync_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/lottery", tags=["开奖"])


def parse_zone_query(zones: Optional[str]) -> Optional[List[int]]:
    """解析八区查询参数，支持逗号分隔格式。"""
    if not zones:
        return None
    parsed = []
    for item in zones.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            zone = int(item)
        except ValueError:
            continue
        if 1 <= zone <= 8 and zone not in parsed:
            parsed.append(zone)
    return parsed or None


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
        total = await lottery_service.count_historical_results(
            start_date=start_date,
            end_date=end_date,
            issue=issue,
        )
        data = {"results": serialized, "total": total}
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
    current_user: User = Depends(get_current_active_user),
):
    """同步最新开奖数据（需要管理员权限）"""
    try:
        if not getattr(current_user, "is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="权限不足"
            )

        sync_summary = await lottery_sync_service.sync_latest(force=True, trigger="admin")

        return create_success_response(
            data=sync_summary, message="数据同步成功"
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("同步数据失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="同步数据失败",
        )


@router.post("/auto-sync")
async def auto_sync_latest_data():
    """页面加载或刷新时自动同步最新开奖数据。"""
    try:
        sync_summary = await lottery_sync_service.sync_latest(force=False, trigger="page")
        return create_success_response(data=sync_summary, message="自动同步检查完成")
    except Exception as exc:
        logger.error("自动同步开奖数据失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="自动同步开奖数据失败",
        )


@router.get("/sandbox/filter")
async def filter_sandbox_results(
    rules: Optional[List[str]] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    periods: int = Query(100, ge=1, le=1000),
    event_type: str = Query("consecutive", pattern="^(consecutive|gap|mixed|interval)$"),
    level: int = Query(3, ge=2, le=4),
    scope: str = Query("global", pattern="^(global|zone)$"),
    zones: Optional[str] = Query(None),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    issue: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """数据沙盘规则过滤。"""
    try:
        service = LotterySandboxService(db)
        data = await service.filter_results(
            rules=rules,
            limit=limit,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
            issue=issue,
            periods=periods,
            event_type=event_type,
            level=level,
            scope=scope,
            zones=parse_zone_query(zones),
        )
        return create_success_response(data=data, message="数据沙盘过滤成功")
    except Exception as exc:
        logger.error("数据沙盘过滤失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="数据沙盘过滤失败",
        )


@router.get("/sandbox/intervals")
async def analyze_sandbox_intervals(
    rules: Optional[List[str]] = Query(None),
    periods: int = Query(100, ge=1, le=1000),
    event_type: str = Query("consecutive", pattern="^(consecutive|gap|mixed|interval)$"),
    level: int = Query(3, ge=2, le=4),
    scope: str = Query("global", pattern="^(global|zone)$"),
    zones: Optional[str] = Query(None),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    issue: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """数据沙盘间隔分析。"""
    try:
        service = LotterySandboxService(db)
        data = await service.analyze_intervals(
            rules=rules,
            start_date=start_date,
            end_date=end_date,
            issue=issue,
            periods=periods,
            event_type=event_type,
            level=level,
            scope=scope,
            zones=parse_zone_query(zones),
        )
        return create_success_response(data=data, message="数据沙盘间隔分析成功")
    except Exception as exc:
        logger.error("数据沙盘间隔分析失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="数据沙盘间隔分析失败",
        )


@router.get("/sandbox/summary")
async def summarize_sandbox_patterns(
    rules: Optional[List[str]] = Query(None),
    periods: int = Query(100, ge=1, le=1000),
    event_type: str = Query("consecutive", pattern="^(consecutive|gap|mixed|interval)$"),
    level: int = Query(3, ge=2, le=4),
    scope: str = Query("global", pattern="^(global|zone)$"),
    zones: Optional[str] = Query(None),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    issue: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """数据沙盘规律总结。"""
    try:
        service = LotterySandboxService(db)
        data = await service.summarize_patterns(
            rules=rules,
            periods=periods,
            start_date=start_date,
            end_date=end_date,
            issue=issue,
            event_type=event_type,
            level=level,
            scope=scope,
            zones=parse_zone_query(zones),
        )
        return create_success_response(data=data, message="数据沙盘规律总结成功")
    except Exception as exc:
        logger.error("数据沙盘规律总结失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="数据沙盘规律总结失败",
        )
