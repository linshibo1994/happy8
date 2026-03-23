import asyncio
import logging
import threading
import time
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.api.services.analyzer import get_analyzer_instance
from backend.api.services.cache import cache_manager
from backend.config import METHOD_MAPPING
from backend.utils.formatter import build_response, format_prediction_result, format_sse_event


router = APIRouter()
logger = logging.getLogger(__name__)

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


class PredictionRequest(BaseModel):
    method: str = Field(..., description="算法名称")
    periods: int = Field(300, ge=10, le=5000, description="分析期数")
    count: int = Field(20, ge=1, le=30, description="生成号码数")
    target_issue: Optional[str] = Field(None, description="目标期号")


def _resolve_method(method: str) -> str:
    alias_target = METHOD_ALIASES.get(method)
    if alias_target:
        method = alias_target

    if method in METHOD_MAPPING:
        return method

    reverse_map = {value: key for key, value in METHOD_MAPPING.items()}
    if method in reverse_map:
        return reverse_map[method]

    raise ValueError(f"不支持的算法: {method}")


def _next_issue_from_latest(analyzer: Any) -> str:
    data = analyzer.load_data(1)
    if data.empty:
        raise ValueError("无可用历史数据，无法推导目标期号")

    latest_issue = str(data.iloc[0]["issue"])
    digits = "".join(ch for ch in latest_issue if ch.isdigit())
    if not digits:
        raise ValueError(f"无效期号格式: {latest_issue}")

    next_issue = str(int(digits) + 1).zfill(len(digits))
    return next_issue


def _build_cache_key(payload: PredictionRequest, resolved_method: str, target_issue: str) -> str:
    return (
        f"predict:{payload.method}:{resolved_method}:"
        f"{payload.periods}:{payload.count}:{target_issue}"
    )


def _execute_prediction(payload: PredictionRequest) -> Dict[str, Any]:
    analyzer = get_analyzer_instance()
    resolved_method = _resolve_method(payload.method)
    target_issue = payload.target_issue or _next_issue_from_latest(analyzer)
    cache_key = _build_cache_key(payload, resolved_method, target_issue)

    cached = cache_manager.get(cache_key)
    if cached is not None:
        return cached

    start_time = time.time()
    smart_result = analyzer.predict_with_smart_mode(
        target_issue=target_issue,
        periods=payload.periods,
        count=payload.count,
        method=resolved_method,
    )
    response_data = format_prediction_result(
        smart_result=smart_result,
        request_method=payload.method,
        resolved_method=resolved_method,
    )
    response_data["target_issue"] = target_issue
    response_data["wall_time"] = round(time.time() - start_time, 6)

    response = build_response(True, response_data, "预测完成")
    cache_manager.set(cache_key, response, ttl_seconds=300)
    return response


@router.post("/api/predict")
def predict(payload: PredictionRequest) -> dict:
    try:
        return _execute_prediction(payload)
    except Exception as exc:
        logger.exception(f"操作失败: {exc}")
        return build_response(False, None, f"预测失败: {exc}")


@router.get("/api/predict/stream")
async def predict_stream(
    request: Request,
    method: str = Query(..., description="算法名称"),
    periods: int = Query(300, ge=10, le=5000, description="分析期数"),
    count: int = Query(20, ge=1, le=30, description="生成号码数"),
    target_issue: Optional[str] = Query(None, description="目标期号"),
) -> StreamingResponse:
    payload = PredictionRequest(method=method, periods=periods, count=count, target_issue=target_issue)

    async def event_generator():
        queue: "Queue[tuple[str, Dict[str, Any]]]" = Queue()

        def worker() -> None:
            try:
                result = _execute_prediction(payload)
                queue.put(("result", result))
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"操作失败: {exc}")
                queue.put(
                    (
                        "error",
                        build_response(False, None, f"预测失败: {exc}"),
                    )
                )

        yield format_sse_event(
            "progress",
            {"step": "INIT", "progress": 5, "message": "请求已接收", "timestamp": datetime.now().isoformat()},
        )
        yield format_sse_event(
            "progress",
            {"step": "LOAD_DATA", "progress": 20, "message": "准备加载数据", "timestamp": datetime.now().isoformat()},
        )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        progress = 40
        tick = 0
        yield format_sse_event(
            "progress",
            {"step": "INFERENCE", "progress": progress, "message": "模型计算中", "timestamp": datetime.now().isoformat()},
        )

        while True:
            if await request.is_disconnected():
                break

            try:
                event_type, event_payload = queue.get_nowait()
            except Empty:
                tick += 1
                if tick % 2 == 0 and progress < 90:
                    progress = min(90, progress + 5)
                    yield format_sse_event(
                        "progress",
                        {
                            "step": "INFERENCE",
                            "progress": progress,
                            "message": "模型计算中",
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                await asyncio.sleep(0.5)
                continue

            if event_type == "error":
                yield format_sse_event("error", event_payload)
                yield format_sse_event(
                    "complete",
                    {
                        "success": False,
                        "message": event_payload.get("message", "预测失败"),
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                break

            yield format_sse_event(
                "progress",
                {"step": "DONE", "progress": 100, "message": "预测完成", "timestamp": datetime.now().isoformat()},
            )
            yield format_sse_event("result", event_payload)
            yield format_sse_event(
                "complete",
                {"success": True, "message": "预测流程结束", "timestamp": datetime.now().isoformat()},
            )
            break

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)
