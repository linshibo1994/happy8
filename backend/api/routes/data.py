from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.api.services.analyzer import get_analyzer_instance, refresh_data_from_network, reload_local_data
from backend.utils.formatter import build_response


router = APIRouter()


class HistoryRequest(BaseModel):
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(50, ge=1, le=500, description="每页条数")
    issue: Optional[str] = Field(None, description="精确期号筛选")
    start_issue: Optional[str] = Field(None, description="起始期号")
    end_issue: Optional[str] = Field(None, description="结束期号")


class RefreshRequest(BaseModel):
    reload_from_local: bool = Field(False, description="仅重新加载本地数据")


def _issue_to_int(issue: str) -> int:
    digits = "".join(ch for ch in str(issue) if ch.isdigit())
    return int(digits) if digits else -1


def _serialize_record(row: Dict[str, Any]) -> Dict[str, Any]:
    numbers = [int(row[f"num{i}"]) for i in range(1, 21)]
    return {
        "issue": str(row["issue"]),
        "date": str(row.get("date", "")),
        "numbers": numbers,
        "sum": int(sum(numbers)),
        "avg": round(sum(numbers) / 20, 4),
        "range": int(max(numbers) - min(numbers)),
        "odd_count": int(sum(1 for x in numbers if x % 2 == 1)),
        "big_count": int(sum(1 for x in numbers if x >= 41)),
    }


@router.get("/api/data/latest")
async def get_latest_data() -> dict:
    try:
        analyzer = get_analyzer_instance()
        data = analyzer.load_data(1)
        if data.empty:
            return build_response(False, None, "暂无开奖数据")

        latest = _serialize_record(data.iloc[0].to_dict())
        return build_response(True, latest, "获取成功")
    except Exception as exc:
        return build_response(False, None, f"获取最新数据失败: {exc}")


@router.post("/api/data/history")
async def get_history_data(payload: HistoryRequest) -> dict:
    try:
        analyzer = get_analyzer_instance()
        data = analyzer.load_data()

        if payload.issue:
            data = data[data["issue"].astype(str) == str(payload.issue)]

        if payload.start_issue:
            start_val = _issue_to_int(payload.start_issue)
            data = data[data["issue"].astype(str).map(_issue_to_int) >= start_val]

        if payload.end_issue:
            end_val = _issue_to_int(payload.end_issue)
            data = data[data["issue"].astype(str).map(_issue_to_int) <= end_val]

        total = len(data)
        start_idx = (payload.page - 1) * payload.page_size
        end_idx = start_idx + payload.page_size
        paged = data.iloc[start_idx:end_idx]

        items: List[Dict[str, Any]] = [_serialize_record(row.to_dict()) for _, row in paged.iterrows()]

        response_data = {
            "items": items,
            "pagination": {
                "page": payload.page,
                "page_size": payload.page_size,
                "total": int(total),
                "total_pages": int((total + payload.page_size - 1) // payload.page_size),
            },
            "filters": {
                "issue": payload.issue,
                "start_issue": payload.start_issue,
                "end_issue": payload.end_issue,
            },
        }
        return build_response(True, response_data, "获取成功")
    except Exception as exc:
        return build_response(False, None, f"获取历史数据失败: {exc}")


@router.post("/api/data/refresh")
async def refresh_data(payload: RefreshRequest) -> dict:
    try:
        if payload.reload_from_local:
            total_records = reload_local_data()
            return build_response(
                True,
                {"total_records": total_records, "mode": "local_reload"},
                "本地数据重载完成",
            )

        refreshed = refresh_data_from_network()
        return build_response(True, refreshed, "数据刷新完成")
    except Exception as exc:
        return build_response(False, None, f"刷新数据失败: {exc}")


@router.get("/api/data/statistics")
async def get_data_statistics(periods: int = 300) -> dict:
    try:
        analyzer = get_analyzer_instance()
        data = analyzer.load_data(periods)
        if data.empty:
            return build_response(False, None, "暂无数据可分析")

        number_cols = [f"num{i}" for i in range(1, 21)]
        all_numbers = data[number_cols].values.flatten()
        total_slots = len(data) * 20

        frequency_map: Dict[int, int] = {num: 0 for num in range(1, 81)}
        for num in all_numbers:
            frequency_map[int(num)] += 1

        sorted_numbers = sorted(frequency_map.items(), key=lambda x: x[1], reverse=True)
        frequency_list = [
            {
                "number": num,
                "count": count,
                "rate": round(count / total_slots, 6) if total_slots else 0.0,
            }
            for num, count in sorted_numbers
        ]

        response_data = {
            "periods": int(len(data)),
            "total_records": int(len(data)),
            "latest_issue": str(data.iloc[0]["issue"]) if len(data) > 0 else None,
            "frequency": frequency_list,
            "top_10": frequency_list[:10],
            "bottom_10": frequency_list[-10:],
        }
        return build_response(True, response_data, "统计成功")
    except Exception as exc:
        return build_response(False, None, f"统计失败: {exc}")

