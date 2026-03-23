import asyncio
import json
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.api.services.analyzer import get_analyzer_instance
from backend.config import METHOD_MAPPING
from backend.utils.formatter import build_response, format_sse_event, to_jsonable


router = APIRouter()
logger = logging.getLogger(__name__)


class BatchComparisonRequest(BaseModel):
    target_issue: str = Field(..., description="目标期号")
    method: str = Field(..., description="预测方法")
    periods: int = Field(300, ge=10, le=500, description="分析期数")
    count: int = Field(20, ge=1, le=30, description="生成号码数")
    comparison_times: int = Field(20, ge=1, le=100, description="批量对比次数")
    max_parallel: int = Field(1, ge=1, le=8, description="最大并发数")
    timeout_seconds: int = Field(30, ge=5, le=300, description="单次超时秒数")


METHOD_ALIASES = {
    "super": "super_predictor",
    "graph_nn": "gnn",
    "dynamic_bayes": "bayesian",
    "adaptive_ensemble": "advanced_ensemble",
    "high_confidence_full": "high_confidence",
    "high_confidence_advanced": "high_confidence",
    "high_confidence_lite": "high_confidence",
    "high_confidence_complete": "high_confidence",
    "hybrid": "super_predictor",
    "hybrid_v2": "super_predictor",
    "stats": "frequency",
    "probability": "bayesian",
    "decision_tree": "advanced_ensemble",
    "patterns": "clustering",
    "frequency_cons2": "frequency",
    "consensus_halving": "ensemble",
}


def _load_batch_components():
    project_root = Path(__file__).resolve().parents[3]
    src_root = project_root / "src"
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

    from batch_predictor import BatchConfig, BatchPredictor  # noqa: E402

    class ApiBatchPredictor(BatchPredictor):
        def _get_actual_numbers(self, target_issue: str):  # type: ignore[override]
            try:
                data = self.analyzer.load_data()
                target_data = data[data["issue"].astype(str) == str(target_issue)]
                if target_data.empty:
                    return []
                row = target_data.iloc[0]
                return [int(row[f"num{i}"]) for i in range(1, 21)]
            except Exception as exc:
                logger.exception(f"操作失败: {exc}")
                return []

    return BatchConfig, ApiBatchPredictor


def _resolve_method(method: str) -> str:
    method = METHOD_ALIASES.get(method, method)
    if method in METHOD_MAPPING:
        return method
    reverse_map = {value: key for key, value in METHOD_MAPPING.items()}
    if method in reverse_map:
        return reverse_map[method]
    raise ValueError(f"不支持的算法: {method}")


def _build_progress_payload(session: Any) -> Dict[str, Any]:
    total = int(session.config.comparison_times)
    current = int(session.current_round)
    progress = round((current / total) * 100, 2) if total else 0.0
    return {
        # 兼容前端字段
        "round": current,
        "total": total,
        "percentage": progress,
        # 新字段
        "session_id": session.session_id,
        "status": session.status,
        "current_round": current,
        "total_rounds": total,
        "progress": progress,
        "timestamp": datetime.now().isoformat(),
    }


def _batch_result_to_dict(batch_result: Any) -> Dict[str, Any]:
    try:
        if hasattr(batch_result, "to_json") and callable(batch_result.to_json):
            return json.loads(batch_result.to_json())
    except Exception as exc:
        logger.exception(f"操作失败: {exc}")
        pass
    return to_jsonable(batch_result)


def _to_int_list(values: Any) -> List[int]:
    if not isinstance(values, list):
        return []
    items: List[int] = []
    for value in values:
        try:
            items.append(int(value))
        except Exception:
            continue
    return items


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_round_result(prediction: Dict[str, Any], actual_numbers_set: set[int]) -> Dict[str, Any]:
    predicted_numbers = _to_int_list(prediction.get("predicted_numbers"))
    hit_numbers = _to_int_list(prediction.get("hit_numbers"))

    if not hit_numbers and actual_numbers_set:
        hit_numbers = [num for num in predicted_numbers if num in actual_numbers_set]

    if hit_numbers:
        predicted_set = set(predicted_numbers)
        ordered_hits: List[int] = []
        seen = set()
        for num in hit_numbers:
            if num in predicted_set and num not in seen:
                ordered_hits.append(num)
                seen.add(num)
        hit_numbers = ordered_hits

    hit_count = _safe_int(prediction.get("hit_count"), len(hit_numbers))
    if hit_numbers:
        hit_count = len(hit_numbers)
    hit_count = max(hit_count, 0)

    default_hit_rate = (hit_count / len(predicted_numbers)) if predicted_numbers else 0.0
    hit_rate = _safe_float(prediction.get("hit_rate"), default_hit_rate)
    hit_rate = max(0.0, min(hit_rate, 1.0))

    success = bool(prediction.get("success", True)) and bool(predicted_numbers)
    if not success:
        predicted_numbers = []
        hit_numbers = []
        hit_count = 0
        hit_rate = 0.0

    return {
        "round": _safe_int(prediction.get("round_number") or prediction.get("round"), 0),
        "analysis_periods": _safe_int(prediction.get("analysis_periods"), 0),
        "predicted_numbers": predicted_numbers,
        "hit_numbers": hit_numbers,
        "hit_count": hit_count,
        "hit_rate": hit_rate,
        "success": success,
    }


def _build_comparison_summary(raw_data: Dict[str, Any], stream_meta: Dict[str, Any]) -> Dict[str, Any]:
    predictions = raw_data.get("predictions")
    predictions_list = predictions if isinstance(predictions, list) else []

    actual_numbers: List[int] = []
    for pred in predictions_list:
        if isinstance(pred, dict):
            actual_numbers = _to_int_list(pred.get("actual_numbers"))
            if actual_numbers:
                break

    actual_numbers_set = set(actual_numbers)
    round_results = [
        _to_round_result(pred, actual_numbers_set)
        for pred in predictions_list
        if isinstance(pred, dict)
    ]

    success_rows = [row for row in round_results if row.get("success")]
    success_predictions = len(success_rows)
    total_predictions = len(round_results)
    success_rate = (success_predictions / total_predictions) if total_predictions > 0 else 0.0

    hit_counts = [int(row.get("hit_count", 0)) for row in success_rows]
    hit_rates = [float(row.get("hit_rate", 0.0)) for row in success_rows]
    avg_hit_count = (sum(hit_counts) / len(hit_counts)) if hit_counts else 0.0
    avg_hit_rate = (sum(hit_rates) / len(hit_rates)) if hit_rates else 0.0
    best_hit_count = max(hit_counts) if hit_counts else 0
    best_hit_rate = max(hit_rates) if hit_rates else 0.0

    used_periods = sorted(
        {
            int(item.get("analysis_periods") or 0)
            for item in round_results
            if int(item.get("analysis_periods") or 0) > 0
        }
    )
    fixed_periods = stream_meta.get("periods_value")
    if isinstance(fixed_periods, str):
        try:
            fixed_periods = int(fixed_periods)
        except Exception:
            fixed_periods = None
    fixed_periods = fixed_periods if isinstance(fixed_periods, int) else None

    return {
        "success": True,
        "target_issue": stream_meta.get("target_issue"),
        "actual_result": {
            "issue": stream_meta.get("target_issue"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "numbers": actual_numbers,
        },
        "method_name": stream_meta.get("method_name"),
        "comparison_times": total_predictions,
        "success_predictions": success_predictions,
        "success_rate": success_rate,
        "avg_hit_count": round(avg_hit_count, 4),
        "avg_hit_rate": round(avg_hit_rate, 6),
        "best_hit_count": best_hit_count,
        "best_hit_rate": round(best_hit_rate, 6),
        "periods_stats": {
            "mode": stream_meta.get("periods_mode", "fixed"),
            "fixed_value": fixed_periods,
            "used_periods": used_periods,
            "min_periods": min(used_periods) if used_periods else 0,
            "max_periods": max(used_periods) if used_periods else 0,
            "avg_periods": (sum(used_periods) / len(used_periods)) if used_periods else 0,
        },
        "detailed_results": round_results,
        "generated_time": datetime.now().isoformat(),
    }


def _run_batch_job(payload: BatchComparisonRequest) -> Dict[str, Any]:
    resolved_method = _resolve_method(payload.method)
    analyzer = get_analyzer_instance()
    BatchConfig, ApiBatchPredictor = _load_batch_components()
    predictor = ApiBatchPredictor(analyzer)
    config = BatchConfig(
        target_issue=payload.target_issue,
        analysis_periods=payload.periods,
        prediction_method=resolved_method,
        number_count=payload.count,
        comparison_times=payload.comparison_times,
        max_parallel=payload.max_parallel,
        timeout_seconds=payload.timeout_seconds,
    )
    batch_result = predictor.execute_batch_prediction(config)
    return _batch_result_to_dict(batch_result)


async def _stream_batch_comparison(
    payload: BatchComparisonRequest,
    request: Request,
    stream_meta: Optional[Dict[str, Any]] = None,
) -> StreamingResponse:
    stream_meta = stream_meta or {
        "target_issue": payload.target_issue,
        "method_name": payload.method,
        "periods_mode": "fixed",
        "periods_value": payload.periods,
    }

    async def event_generator():
        queue: "Queue[tuple[str, Dict[str, Any]]]" = Queue()

        def worker() -> None:
            try:
                resolved_method = _resolve_method(payload.method)
                analyzer = get_analyzer_instance()
                BatchConfig, ApiBatchPredictor = _load_batch_components()
                predictor = ApiBatchPredictor(analyzer)
                config = BatchConfig(
                    target_issue=payload.target_issue,
                    analysis_periods=payload.periods,
                    prediction_method=resolved_method,
                    number_count=payload.count,
                    comparison_times=payload.comparison_times,
                    max_parallel=payload.max_parallel,
                    timeout_seconds=payload.timeout_seconds,
                )

                def on_progress(session: Any) -> None:
                    queue.put(("progress", _build_progress_payload(session)))

                batch_result = predictor.execute_batch_prediction(config, progress_callback=on_progress)
                result_payload = build_response(
                    True,
                    _batch_result_to_dict(batch_result),
                    "批量对比完成",
                )
                queue.put(("result", result_payload))
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"操作失败: {exc}")
                queue.put(("error", build_response(False, None, f"批量对比失败: {exc}")))

        yield format_sse_event(
            "progress",
            {
                "status": "starting",
                "progress": 0,
                "round": 0,
                "total": int(payload.comparison_times),
                "percentage": 0,
                "message": "已接收批量对比请求",
                "timestamp": datetime.now().isoformat(),
            },
        )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            if await request.is_disconnected():
                return

            try:
                event_type, event_payload = queue.get_nowait()
            except Empty:
                await asyncio.sleep(0.3)
                continue

            if event_type == "progress":
                yield format_sse_event("progress", event_payload)
                continue

            if event_type == "error":
                yield format_sse_event("error", event_payload)
                yield format_sse_event(
                    "complete",
                    {
                        "success": False,
                        "message": event_payload.get("message", "批量对比失败"),
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                break

            raw_data = event_payload.get("data") if isinstance(event_payload, dict) else None
            raw_data = raw_data if isinstance(raw_data, dict) else {}
            summary = _build_comparison_summary(raw_data, stream_meta)

            for row in summary.get("detailed_results", []):
                if isinstance(row, dict):
                    yield format_sse_event("result", row)

            yield format_sse_event(
                "complete",
                {
                    "success": True,
                    "summary": summary,
                    "message": "批量对比流程结束",
                    "timestamp": datetime.now().isoformat(),
                },
            )
            break

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@router.get("/api/comparison/issues")
def get_available_issues(limit: int = Query(100, ge=1, le=1000)) -> Dict[str, Any]:
    try:
        analyzer = get_analyzer_instance()
        data = analyzer.load_data(limit)
        issues = [str(row_issue) for row_issue in data["issue"].astype(str).tolist()]
        return build_response(True, {"issues": issues, "total": len(issues)}, "获取可用期号成功")
    except Exception as exc:
        logger.exception(f"操作失败: {exc}")
        return build_response(False, None, f"获取可用期号失败: {exc}")


@router.post("/api/comparison/start")
async def start_batch_comparison(payload: BatchComparisonRequest, request: Request) -> StreamingResponse:
    return await _stream_batch_comparison(
        payload,
        request,
        stream_meta={
            "target_issue": payload.target_issue,
            "method_name": payload.method,
            "periods_mode": "fixed",
            "periods_value": payload.periods,
        },
    )


@router.post("/api/comparison/batch")
def run_batch_comparison(payload: BatchComparisonRequest) -> Dict[str, Any]:
    try:
        result_data = _run_batch_job(payload)
        summary = _build_comparison_summary(
            result_data,
            {
                "target_issue": payload.target_issue,
                "method_name": payload.method,
                "periods_mode": "fixed",
                "periods_value": payload.periods,
            },
        )
        return build_response(True, summary, "批量对比完成")
    except Exception as exc:
        logger.exception(f"操作失败: {exc}")
        return build_response(False, None, f"批量对比失败: {exc}")


@router.get("/api/comparison/batch/stream")
async def batch_comparison_stream(
    request: Request,
    target_issue: str = Query(..., description="目标期号"),
    method_name: str = Query(..., description="预测方法"),
    periods_mode: str = Query("fixed", description="期数模式"),
    periods_value: int = Query(100, ge=10, le=500, description="分析期数"),
    comparison_times: int = Query(50, ge=1, le=100, description="批量对比次数"),
) -> StreamingResponse:
    payload = BatchComparisonRequest(
        target_issue=target_issue,
        method=method_name,
        periods=periods_value,
        count=20,
        comparison_times=comparison_times,
        max_parallel=1,
        timeout_seconds=30,
    )
    return await _stream_batch_comparison(
        payload,
        request,
        stream_meta={
            "target_issue": target_issue,
            "method_name": method_name,
            "periods_mode": periods_mode,
            "periods_value": periods_value,
        },
    )
