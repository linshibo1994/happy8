from fastapi import APIRouter

from backend.api.services.analyzer import get_analyzer_status
from backend.config import settings
from backend.utils.formatter import build_response


router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    data = {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "analyzer": get_analyzer_status(),
    }
    return build_response(True, data, "服务运行正常")

