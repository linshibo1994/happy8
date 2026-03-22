from fastapi import APIRouter

from backend.config import METHOD_GROUPS, METHOD_MAPPING
from backend.utils.formatter import build_response


router = APIRouter()


@router.get("/api/methods")
async def get_methods() -> dict:
    groups = []
    flat = []

    for category, methods in METHOD_GROUPS.items():
        group_items = []
        for method in methods:
            item = {
                "method": method,
                "display_name": method,
                "mapped_function": METHOD_MAPPING.get(method),
                "category": category,
            }
            group_items.append(item)
            flat.append(item)

        groups.append({"category": category, "methods": group_items, "count": len(group_items)})

    data = {
        "groups": groups,
        "methods": flat,
        "total": len(flat),
    }
    return build_response(True, data, "获取算法列表成功")

