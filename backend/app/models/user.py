"""用户相关数据模型"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.base import Base


class User(Base):
    """用户基础信息表"""
    
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, comment="用户ID")
    openid = Column(
        String(64), 
        unique=True, 
        nullable=False, 
        index=True, 
        comment="微信openid"
    )
    unionid = Column(
        String(64), 
        unique=True, 
        nullable=True, 
        index=True, 
        comment="微信unionid"
    )
    nickname = Column(String(100), nullable=False, comment="用户昵称")
    avatar_url = Column(String(500), nullable=True, comment="头像URL")
    phone = Column(String(20), nullable=True, comment="手机号")
    email = Column(String(100), nullable=True, comment="邮箱")
    is_active = Column(Boolean, default=True, nullable=False, comment="是否激活")
    
    # 时间字段
    created_at = Column(
        DateTime, 
        default=func.now(), 
        nullable=False, 
        comment="创建时间"
    )
    updated_at = Column(
        DateTime, 
        default=func.now(), 
        onupdate=func.now(), 
        nullable=False, 
        comment="更新时间"
    )

    # 关联关系
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    membership = relationship("Membership", back_populates="user", uselist=False)
    predictions = relationship("PredictionHistory", back_populates="user")
    orders = relationship("MembershipOrder", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, nickname='{self.nickname}', openid='{self.openid}')>"


class UserProfile(Base):
    """用户详细资料表"""
    
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True, comment="资料ID")
    user_id = Column(
        Integer, 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False, 
        unique=True,
        comment="用户ID"
    )
    
    # 个人信息
    real_name = Column(String(50), nullable=True, comment="真实姓名")
    id_card = Column(String(200), nullable=True, comment="身份证号(加密)")
    gender = Column(String(10), nullable=True, comment="性别")
    birthday = Column(DateTime, nullable=True, comment="生日")
    address = Column(Text, nullable=True, comment="地址")
    
    # 偏好设置 JSON格式
    preferences = Column(Text, nullable=True, comment="用户偏好设置(JSON)")
    
    # 时间字段
    created_at = Column(
        DateTime, 
        default=func.now(), 
        nullable=False, 
        comment="创建时间"
    )
    updated_at = Column(
        DateTime, 
        default=func.now(), 
        onupdate=func.now(), 
        nullable=False, 
        comment="更新时间"
    )

    # 关联关系
    user = relationship("User", back_populates="profile")

    def __repr__(self):
        return f"<UserProfile(id={self.id}, user_id={self.user_id}, real_name='{self.real_name}')>"