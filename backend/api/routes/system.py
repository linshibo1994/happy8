import platform
import sys

from fastapi import APIRouter

from backend.api.services.analyzer import get_analyzer_instance, get_analyzer_status
from backend.api.services.cache import cache_manager
from backend.config import METHOD_GROUPS, METHOD_MAPPING, settings
from backend.utils.formatter import build_response


router = APIRouter()


@router.get("/api/system/info")
async def get_system_info() -> dict:
    try:
        analyzer = get_analyzer_instance()
        data = analyzer.load_data()
        latest_issue = str(data.iloc[0]["issue"]) if len(data) > 0 else None

        payload = {
            "app": {
                "name": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "debug": settings.DEBUG,
            },
            "runtime": {
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "algorithms": {
                "total": len(METHOD_MAPPING),
                "groups": METHOD_GROUPS,
                "methods": list(METHOD_MAPPING.keys()),
            },
            "data_status": {
                "total_records": int(len(data)),
                "latest_issue": latest_issue,
            },
            "analyzer_status": get_analyzer_status(),
            "cache": cache_manager.stats(),
        }
        return build_response(True, payload, "获取系统信息成功")
    except Exception as exc:
        return build_response(False, None, f"获取系统信息失败: {exc}")


@router.post("/api/system/cache/clear")
async def clear_cache() -> dict:
    try:
        cleared_count = cache_manager.clear()
        analyzer = get_analyzer_instance(ensure_loaded=False)
        if hasattr(analyzer, "clear_pair_frequency_cache"):
            analyzer.clear_pair_frequency_cache()

        payload = {
            "cleared_cache_items": cleared_count,
            "cache_stats": cache_manager.stats(),
        }
        return build_response(True, payload, "缓存已清除")
    except Exception as exc:
        return build_response(False, None, f"清除缓存失败: {exc}")

