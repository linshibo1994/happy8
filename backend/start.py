#!/usr/bin/env python3
"""启动脚本"""

import sys
import uvicorn
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from app.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD and settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )