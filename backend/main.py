import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)


app = FastAPI(
    title=settings.APP_NAME,
    description="Happy8 lottery prediction backend service",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if settings.CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict:
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


from backend.api.routes import analysis, comparison, data, health, methods, prediction, system  # noqa: E402

app.include_router(health.router, tags=["health"])
app.include_router(data.router, tags=["data"])
app.include_router(prediction.router, tags=["prediction"])
app.include_router(analysis.router, tags=["analysis"])
app.include_router(methods.router, tags=["methods"])
app.include_router(comparison.router, tags=["comparison"])
app.include_router(system.router, tags=["system"])


def run_server(host: Optional[str] = None, port: Optional[int] = None) -> None:
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=host or settings.API_HOST,
        port=port or settings.API_PORT,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    run_server()
