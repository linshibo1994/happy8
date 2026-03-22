import dataclasses
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict


def now_iso() -> str:
    return datetime.now().isoformat()


def to_jsonable(data: Any) -> Any:
    if dataclasses.is_dataclass(data):
        return to_jsonable(dataclasses.asdict(data))

    if isinstance(data, dict):
        return {str(k): to_jsonable(v) for k, v in data.items()}

    if isinstance(data, (list, tuple, set)):
        return [to_jsonable(item) for item in data]

    if isinstance(data, (datetime, date)):
        return data.isoformat()

    if isinstance(data, Path):
        return str(data)

    if hasattr(data, "item") and callable(getattr(data, "item")):
        try:
            return data.item()
        except Exception:
            pass

    if hasattr(data, "to_dict") and callable(getattr(data, "to_dict")):
        try:
            return data.to_dict(orient="records")
        except Exception:
            try:
                return data.to_dict()
            except Exception:
                pass

    return data


def build_response(success: bool, data: Any = None, message: str = "") -> Dict[str, Any]:
    return {
        "success": success,
        "data": to_jsonable(data) if data is not None else None,
        "message": message,
        "timestamp": now_iso(),
    }


def format_prediction_result(
    smart_result: Dict[str, Any],
    request_method: str,
    resolved_method: str,
) -> Dict[str, Any]:
    prediction_result = smart_result.get("prediction_result")
    comparison_result = smart_result.get("comparison_result")

    if not prediction_result:
        return {}

    predicted_numbers = getattr(prediction_result, "predicted_numbers", None)
    if predicted_numbers is None:
        predicted_numbers = getattr(prediction_result, "numbers", [])

    confidence_scores = getattr(prediction_result, "confidence_scores", [])
    avg_confidence = float(sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0.0

    payload = {
        "target_issue": getattr(prediction_result, "target_issue", smart_result.get("target_issue")),
        "analysis_periods": getattr(prediction_result, "analysis_periods", 0),
        "request_method": request_method,
        "resolved_method": resolved_method,
        "engine_method": getattr(prediction_result, "method", resolved_method),
        "numbers": predicted_numbers or [],
        "confidence_scores": confidence_scores,
        "confidence": round(avg_confidence, 6),
        "execution_time": getattr(prediction_result, "execution_time", 0.0),
        "generation_time": getattr(prediction_result, "generation_time", now_iso()),
        "parameters": getattr(prediction_result, "parameters", {}),
        "mode": smart_result.get("mode"),
        "mode_description": smart_result.get("mode_description"),
    }

    if comparison_result is not None:
        payload["comparison"] = {
            "target_issue": getattr(comparison_result, "target_issue", ""),
            "actual_numbers": getattr(comparison_result, "actual_numbers", []),
            "hit_numbers": getattr(comparison_result, "hit_numbers", []),
            "miss_numbers": getattr(comparison_result, "miss_numbers", []),
            "hit_count": getattr(comparison_result, "hit_count", 0),
            "total_predicted": getattr(comparison_result, "total_predicted", 0),
            "hit_rate": getattr(comparison_result, "hit_rate", 0.0),
            "hit_distribution": getattr(comparison_result, "hit_distribution", {}),
            "comparison_time": getattr(comparison_result, "comparison_time", None),
        }
    else:
        payload["comparison"] = None

    return to_jsonable(payload)


def format_pair_frequency_result(result: Any, top_n: int = 50) -> Dict[str, Any]:
    frequency_items = getattr(result, "frequency_items", []) or []
    top_items = frequency_items[:top_n]

    return to_jsonable(
        {
            "target_issue": getattr(result, "target_issue", ""),
            "requested_periods": getattr(result, "requested_periods", 0),
            "actual_periods": getattr(result, "actual_periods", 0),
            "start_issue": getattr(result, "start_issue", ""),
            "end_issue": getattr(result, "end_issue", ""),
            "total_pairs": getattr(result, "total_pairs", 0),
            "execution_time": getattr(result, "execution_time", 0.0),
            "analysis_time": getattr(result, "analysis_time", None),
            "top_pairs": top_items,
        }
    )


def format_sse_event(event: str, payload: Dict[str, Any]) -> str:
    safe_payload = to_jsonable(payload)
    return f"event: {event}\ndata: {json.dumps(safe_payload, ensure_ascii=False)}\n\n"

