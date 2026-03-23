import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from backend.api.services.analyzer import get_analyzer_instance
from backend.utils.formatter import build_response, format_pair_frequency_result


router = APIRouter()
logger = logging.getLogger(__name__)


def _frequency_from_data(data: Any) -> Dict[int, int]:
    frequency: Dict[int, int] = {num: 0 for num in range(1, 81)}
    for _, row in data.iterrows():
        for col_idx in range(1, 21):
            frequency[int(row[f"num{col_idx}"])] += 1
    return frequency


def _missing_from_data(data: Any) -> Dict[int, int]:
    missing: Dict[int, int] = {}
    for num in range(1, 81):
        miss_count = 0
        found = False
        for _, row in data.iterrows():
            numbers = [int(row[f"num{i}"]) for i in range(1, 21)]
            if num in numbers:
                found = True
                break
            miss_count += 1
        missing[num] = miss_count if found else len(data)
    return missing


def _normalize_issue(issue: str) -> str:
    return "".join(ch for ch in str(issue) if ch.isdigit())


@router.get("/api/analysis/frequency")
def analyze_frequency(
    periods: int = Query(300, ge=10, le=5000),
    top_n: int = Query(20, ge=1, le=80),
) -> dict:
    try:
        analyzer = get_analyzer_instance()
        data = analyzer.load_data(periods)
        if data.empty:
            return build_response(False, None, "暂无数据可分析")

        frequency = _frequency_from_data(data)
        total_slots = len(data) * 20
        sorted_items = sorted(frequency.items(), key=lambda x: x[1], reverse=True)

        result = [
            {
                "number": num,
                "count": count,
                "rate": round(count / total_slots, 6) if total_slots else 0.0,
            }
            for num, count in sorted_items
        ]
        payload = {
            "periods": int(len(data)),
            "top_n": top_n,
            "top_numbers": result[:top_n],
            "all_numbers": result,
        }
        return build_response(True, payload, "频率分析完成")
    except Exception as exc:
        logger.exception(f"操作失败: {exc}")
        return build_response(False, None, f"频率分析失败: {exc}")


@router.get("/api/analysis/hot-cold")
def analyze_hot_cold(
    periods: int = Query(300, ge=10, le=5000),
    hot_count: int = Query(10, ge=1, le=40),
    cold_count: int = Query(10, ge=1, le=40),
) -> dict:
    try:
        analyzer = get_analyzer_instance()
        data = analyzer.load_data(periods)
        if data.empty:
            return build_response(False, None, "暂无数据可分析")

        frequency = _frequency_from_data(data)
        sorted_desc = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        sorted_asc = sorted(frequency.items(), key=lambda x: x[1])

        hot_numbers = [{"number": num, "count": count} for num, count in sorted_desc[:hot_count]]
        cold_numbers = [{"number": num, "count": count} for num, count in sorted_asc[:cold_count]]

        payload = {
            "periods": int(len(data)),
            "hot_numbers": hot_numbers,
            "cold_numbers": cold_numbers,
        }
        return build_response(True, payload, "冷热号分析完成")
    except Exception as exc:
        logger.exception(f"操作失败: {exc}")
        return build_response(False, None, f"冷热号分析失败: {exc}")


@router.get("/api/analysis/missing")
def analyze_missing(periods: int = Query(300, ge=10, le=5000), top_n: int = Query(20, ge=1, le=80)) -> dict:
    try:
        analyzer = get_analyzer_instance()
        data = analyzer.load_data(periods)
        if data.empty:
            return build_response(False, None, "暂无数据可分析")

        missing = _missing_from_data(data)
        sorted_items = sorted(missing.items(), key=lambda x: x[1], reverse=True)
        items = [{"number": num, "missing_periods": miss} for num, miss in sorted_items]

        payload = {
            "periods": int(len(data)),
            "top_n": top_n,
            "top_missing": items[:top_n],
            "all_numbers": items,
        }
        return build_response(True, payload, "遗漏分析完成")
    except Exception as exc:
        logger.exception(f"操作失败: {exc}")
        return build_response(False, None, f"遗漏分析失败: {exc}")


@router.get("/api/analysis/pair-frequency")
def analyze_pair_frequency(
    target_issue: Optional[str] = Query(None, description="目标期号"),
    period_count: int = Query(20, ge=1, le=100, description="统计期数"),
    top_n: int = Query(50, ge=1, le=200),
) -> dict:
    try:
        analyzer = get_analyzer_instance()
        if not target_issue:
            data = analyzer.load_data(1)
            if data.empty:
                return build_response(False, None, "暂无数据可分析")
            target_issue = str(data.iloc[0]["issue"])

        target_issue = _normalize_issue(target_issue)
        result = analyzer.analyze_pair_frequency(target_issue=target_issue, period_count=period_count)
        payload = format_pair_frequency_result(result, top_n=top_n)
        return build_response(True, payload, "数字对频率分析完成")
    except Exception as exc:
        logger.exception(f"操作失败: {exc}")
        return build_response(False, None, f"数字对频率分析失败: {exc}")
