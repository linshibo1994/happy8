"""API v1路由聚合"""

from fastapi import APIRouter
from app.api.v1 import users, memberships, payments, predictions, auth, algorithms, lottery

api_router = APIRouter()

# 注册各模块路由
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(memberships.router)
api_router.include_router(payments.router)
api_router.include_router(predictions.router)
api_router.include_router(algorithms.router)
api_router.include_router(lottery.router)