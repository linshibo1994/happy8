import asyncio
import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.api.services.analyzer import get_analyzer_instance
from backend.config import METHOD_MAPPING
from backend.utils.formatter import build_response, format_sse_event, to_jsonable


router = APIRouter()


class BatchComparisonRequest(BaseModel):
    target_issue: str = Field(..., description="目标期号")
    method: str = Field(..., description="预测方法")
    periods: int = Field(300, ge=10, le=500, description="分析期数")
    count: int = Field(20, ge=1, le=30, description="生成号码数")
    comparison_times: int = Field(20, ge=1, le=100, description="批量对比次数")
    max_parallel: int = Field(1, ge=1, le=8, description="最大并发数")
    timeout_seconds: int = Field(30, ge=5, le=300, description="单次超时秒数")


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
            except Exception:
                return []

    return BatchConfig, ApiBatchPredictor


def _resolve_method(method: str) -> str:
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
    except Exception:
        pass
    return to_jsonable(batch_result)


@router.post("/api/comparison/start")
async def start_batch_comparison(payload: BatchComparisonRequest, request: Request) -> StreamingResponse:
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
                queue.put(("error", build_response(False, None, f"批量对比失败: {exc}")))

        yield format_sse_event(
            "progress",
            {
                "status": "starting",
                "progress": 0,
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
                yield format_sse_event("error_event", event_payload)
                yield format_sse_event(
                    "complete",
                    {
                        "success": False,
                        "message": event_payload.get("message", "批量对比失败"),
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                break

            yield format_sse_event("result", event_payload)
            yield format_sse_event(
                "complete",
                {"success": True, "message": "批量对比流程结束", "timestamp": datetime.now().isoformat()},
            )
            break

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)
