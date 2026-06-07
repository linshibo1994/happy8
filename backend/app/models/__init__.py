"""模型导入初始化文件"""

# 导入所有模型以确保Alembic能够发现它们
from app.models.base import Base
from app.models.user import User, UserProfile
from app.models.membership import MembershipPlan, Membership, MembershipOrder
from app.models.prediction import LotteryResult, PredictionHistory, AlgorithmConfig

# 确保所有模型都被注册到Base.metadata
__all__ = [
    "Base",
    "User", 
    "UserProfile",
    "MembershipPlan",
    "Membership", 
    "MembershipOrder",
    "LotteryResult",
    "PredictionHistory",
    "AlgorithmConfig",
]