# Happy8彩票预测小程序 - Design Document

## Overview

Happy8彩票预测小程序采用现代化的前后端分离架构，前端使用uni-app框架确保跨平台兼容性，后端使用FastAPI提供高性能API服务。系统设计重点关注可扩展性、安全性和用户体验，通过微服务架构实现功能解耦，使用Redis缓存提升性能，集成微信支付V3实现商业化。

### 设计原则
- **安全优先**：所有用户数据加密传输和存储，支付流程严格验证
- **性能优化**：多级缓存、异步处理、连接池管理
- **用户体验**：界面简洁美观、响应迅速、操作便捷
- **可维护性**：模块化设计、清晰的代码结构、完善的文档

## Architecture

### 整体架构图
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   微信小程序      │    │     Web管理后台   │    │   H5移动端       │
│   (uni-app)     │    │   (Vue3+Vite)   │    │  (uni-app编译)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │ HTTPS/WSS
                ┌─────────────────▼─────────────────┐
                │          API网关 (Nginx)          │
                │     负载均衡 + SSL终端 + 限流      │
                └─────────────────┬─────────────────┘
                                 │
                ┌─────────────────▼─────────────────┐
                │        FastAPI应用集群            │
                │   ┌─────────┬─────────┬─────────┐ │
                │   │用户服务  │会员服务  │支付服务  │ │
                │   └─────────┼─────────┼─────────┘ │
                │   ┌─────────┴─────────┴─────────┐ │
                │   │    预测服务 + 数据服务       │ │
                │   └─────────────────────────────┘ │
                └─────────────────┬─────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                       │                        │
┌───────▼────────┐    ┌─────────▼──────────┐    ┌────────▼────────┐
│ Redis Cluster  │    │   MySQL Master     │    │ Happy8分析器     │
│   (缓存层)      │    │    (主数据库)       │    │ (Python算法)    │
│                │    │                    │    │                │
│ - 会话管理      │    │ - 用户数据          │    │ - 17种算法      │
│ - 预测缓存      │    │ - 会员信息          │    │ - 历史数据      │
│ - 限流计数      │    │ - 订单记录          │    │ - 预测计算      │
└────────────────┘    └────────────────────┘    └─────────────────┘
                                │
                      ┌─────────▼──────────┐
                      │   MySQL Slave      │
                      │   (只读副本)        │
                      │ - 数据分析          │
                      │ - 报表查询          │
                      └────────────────────┘
```

### 技术栈详情

**前端技术栈**：
- **uni-app 3.x** + Vue 3 + TypeScript
- **Wot Design Uni** - UI组件库
- **Pinia** - 状态管理
- **uni-request** - HTTP请求封装
- **uCharts** - 图表展示

**后端技术栈**：
- **FastAPI 0.100+** - Web框架
- **SQLAlchemy 2.0** - ORM框架
- **Alembic** - 数据库迁移
- **Redis** - 缓存和会话存储
- **Celery** - 异步任务队列
- **Pydantic** - 数据验证

**数据存储**：
- **MySQL 8.0** - 主数据库
- **Redis 6.0** - 缓存数据库
- **Nginx** - 反向代理和负载均衡

## Components and Interfaces

### 前端组件架构

```typescript
// 页面层级结构
src/
├── pages/                    // 页面组件
│   ├── index/               // 首页 - 展示最新开奖和快速预测
│   ├── predict/             // 预测页面 - 算法选择和结果展示
│   ├── history/             // 历史数据 - 开奖记录和走势图
│   ├── member/              // 会员中心 - 套餐购买和权益管理
│   └── profile/             // 个人中心 - 用户信息和设置
├── components/              // 公共组件
│   ├── NumberBall/          // 号码球组件
│   ├── PredictCard/         // 预测卡片组件
│   ├── ChartView/           // 图表组件
│   ├── MembershipCard/      // 会员卡组件
│   └── PaymentModal/        // 支付弹窗组件
├── store/                   // 状态管理
│   ├── user.ts              // 用户状态
│   ├── member.ts            // 会员状态
│   └── predict.ts           // 预测状态
└── utils/                   // 工具函数
    ├── api.ts               // API接口封装
    ├── auth.ts              // 认证工具
    └── format.ts            // 数据格式化
```

### 后端服务接口

**用户认证服务 (`/api/auth`)**：
```python
class AuthService:
    POST /auth/wechat-login      # 微信登录
    POST /auth/refresh-token     # 刷新token
    POST /auth/logout           # 登出
    GET  /auth/profile          # 获取用户信息
    PUT  /auth/profile          # 更新用户信息
```

**会员管理服务 (`/api/membership`)**：
```python
class MembershipService:
    GET  /membership/plans      # 获取会员套餐
    GET  /membership/status     # 获取会员状态
    POST /membership/purchase   # 购买会员
    GET  /membership/orders     # 订单历史
    POST /membership/renew      # 续费会员
```

**支付服务 (`/api/payment`)**：
```python
class PaymentService:
    POST /payment/create-order  # 创建支付订单
    POST /payment/wechat-pay   # 微信支付
    POST /payment/callback     # 支付回调
    GET  /payment/status/{order_id}  # 查询支付状态
    POST /payment/refund       # 申请退款
```

**预测服务 (`/api/predict`)**：
```python
class PredictService:
    GET  /predict/algorithms    # 获取算法列表
    POST /predict/execute      # 执行预测
    GET  /predict/history      # 预测历史
    GET  /predict/statistics   # 预测统计
    DELETE /predict/history/{id}  # 删除预测记录
```

**数据服务 (`/api/data`)**：
```python
class DataService:
    GET  /data/latest-result   # 最新开奖结果
    GET  /data/history         # 历史开奖数据
    GET  /data/statistics      # 数据统计
    GET  /data/trends          # 走势分析
    POST /data/export          # 数据导出
```

## Data Models

### 用户相关数据模型

```python
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    openid = Column(String(64), unique=True, nullable=False, index=True)
    unionid = Column(String(64), unique=True, nullable=True)
    nickname = Column(String(100), nullable=False)
    avatar_url = Column(String(500), nullable=True)
    phone = Column(String(20), nullable=True)
    email = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class UserProfile(Base):
    __tablename__ = 'user_profiles'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    real_name = Column(String(50), nullable=True)
    id_card = Column(String(18), nullable=True)  # 加密存储
    address = Column(Text, nullable=True)
    preferences = Column(Text, nullable=True)  # JSON格式
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
```

### 会员系统数据模型

```python
class MembershipPlan(Base):
    __tablename__ = 'membership_plans'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)  # 月会员、年会员
    level = Column(String(20), nullable=False)  # vip, premium
    duration_days = Column(Integer, nullable=False)
    price = Column(Integer, nullable=False)  # 分为单位
    original_price = Column(Integer, nullable=True)  # 原价
    features = Column(Text, nullable=False)  # JSON格式特权列表
    is_active = Column(Boolean, default=True)
    sort_order = Column(Integer, default=0)
    created_at = Column(DateTime, nullable=False)

class Membership(Base):
    __tablename__ = 'memberships'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    level = Column(String(20), nullable=False, default='free')
    expire_date = Column(DateTime, nullable=True)
    auto_renew = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

class MembershipOrder(Base):
    __tablename__ = 'membership_orders'
    
    id = Column(Integer, primary_key=True)
    order_no = Column(String(32), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    plan_id = Column(Integer, ForeignKey('membership_plans.id'), nullable=False)
    amount = Column(Integer, nullable=False)  # 实际支付金额(分)
    status = Column(String(20), default='pending')  # pending, paid, cancelled, refunded
    pay_method = Column(String(20), default='wechat_pay')
    transaction_id = Column(String(64), nullable=True)  # 微信支付交易号
    paid_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False)
```

### 预测相关数据模型

```python
class PredictionHistory(Base):
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    algorithm = Column(String(50), nullable=False)
    target_issue = Column(String(20), nullable=False)
    predicted_numbers = Column(Text, nullable=False)  # JSON格式
    confidence_score = Column(Float, nullable=True)
    actual_numbers = Column(Text, nullable=True)  # 实际开奖号码
    hit_count = Column(Integer, default=0)  # 命中个数
    is_hit = Column(Boolean, default=False)  # 是否命中
    created_at = Column(DateTime, nullable=False)

class LotteryResult(Base):
    __tablename__ = 'lottery_results'
    
    id = Column(Integer, primary_key=True)
    issue = Column(String(20), unique=True, nullable=False, index=True)
    draw_date = Column(DateTime, nullable=False)
    numbers = Column(Text, nullable=False)  # JSON格式，20个号码
    sum_value = Column(Integer, nullable=False)  # 号码总和
    odd_count = Column(Integer, nullable=False)  # 奇数个数
    even_count = Column(Integer, nullable=False)  # 偶数个数
    created_at = Column(DateTime, nullable=False)
```

### API响应数据模型

```python
from pydantic import BaseModel
from typing import List, Optional

class UserResponse(BaseModel):
    id: int
    nickname: str
    avatar_url: Optional[str]
    membership_level: str
    membership_expire: Optional[datetime]

class PredictRequest(BaseModel):
    algorithm: str
    target_issue: str
    periods: int = 30
    count: int = 5

class PredictResponse(BaseModel):
    success: bool
    data: dict
    predicted_numbers: List[int]
    confidence_score: float
    algorithm_info: dict
    cached: bool = False

class PaymentRequest(BaseModel):
    plan_id: int
    payment_method: str = "wechat_pay"

class PaymentResponse(BaseModel):
    order_no: str
    payment_params: dict  # 微信支付参数
    amount: int
    expire_time: datetime
```

## Error Handling

### 统一错误处理机制

```python
from fastapi import HTTPException
from enum import Enum

class ErrorCode(Enum):
    # 用户相关错误 (1000-1999)
    USER_NOT_FOUND = (1001, "用户不存在")
    USER_NOT_AUTHENTICATED = (1002, "用户未认证")
    USER_PERMISSION_DENIED = (1003, "权限不足")
    
    # 会员相关错误 (2000-2999)
    MEMBERSHIP_EXPIRED = (2001, "会员已过期")
    MEMBERSHIP_LIMIT_EXCEEDED = (2002, "会员权限已用完")
    INVALID_MEMBERSHIP_PLAN = (2003, "无效的会员套餐")
    
    # 支付相关错误 (3000-3999)
    PAYMENT_FAILED = (3001, "支付失败")
    ORDER_NOT_FOUND = (3002, "订单不存在")
    ORDER_ALREADY_PAID = (3003, "订单已支付")
    REFUND_FAILED = (3004, "退款失败")
    
    # 预测相关错误 (4000-4999)
    ALGORITHM_NOT_FOUND = (4001, "算法不存在")
    PREDICTION_FAILED = (4002, "预测失败")
    PREDICTION_LIMIT_EXCEEDED = (4003, "预测次数已用完")
    INVALID_TARGET_ISSUE = (4004, "无效的期号")
    
    # 系统相关错误 (5000-5999)
    SYSTEM_ERROR = (5001, "系统错误")
    DATABASE_ERROR = (5002, "数据库错误")
    CACHE_ERROR = (5003, "缓存错误")
    EXTERNAL_API_ERROR = (5004, "外部API调用失败")

class APIException(HTTPException):
    def __init__(self, error_code: ErrorCode, detail: str = None):
        code, default_message = error_code.value
        super().__init__(
            status_code=400,
            detail={
                "error_code": code,
                "message": detail or default_message,
                "timestamp": datetime.now().isoformat()
            }
        )
```

### 前端错误处理

```typescript
// uni-app错误处理
class ErrorHandler {
    static handle(error: any) {
        const { error_code, message } = error.data || {};
        
        switch (error_code) {
            case 1002: // 用户未认证
                uni.navigateTo({ url: '/pages/login/index' });
                break;
            case 2001: // 会员过期
                uni.showModal({
                    title: '会员已过期',
                    content: '请续费后继续使用',
                    confirmText: '去续费',
                    success: (res) => {
                        if (res.confirm) {
                            uni.navigateTo({ url: '/pages/member/index' });
                        }
                    }
                });
                break;
            case 3001: // 支付失败
                uni.showToast({ title: '支付失败，请重试', icon: 'error' });
                break;
            default:
                uni.showToast({ title: message || '操作失败', icon: 'error' });
        }
    }
}
```

## Testing Strategy

### 测试架构

```
测试金字塔结构：
                    ┌─────────────┐
                    │  E2E测试     │  <- 少量关键流程测试
                    │  (Playwright)│
                ┌───┴─────────────┴───┐
                │   集成测试           │  <- 中等数量API测试
                │  (FastAPI TestClient)│
            ┌───┴─────────────────────┴───┐
            │      单元测试                │  <- 大量单元测试
            │   (pytest + Jest)           │
            └─────────────────────────────┘
```

### 后端测试策略

```python
# pytest配置 - conftest.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///./test.db")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    yield TestingSessionLocal()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client(test_db):
    def get_test_db():
        return test_db
    app.dependency_overrides[get_db] = get_test_db
    return TestClient(app)

# 单元测试示例
class TestUserService:
    def test_create_user(self, test_db):
        user_data = {
            "openid": "test_openid",
            "nickname": "测试用户",
            "avatar_url": "https://example.com/avatar.jpg"
        }
        user = UserService.create_user(test_db, user_data)
        assert user.openid == "test_openid"
        assert user.nickname == "测试用户"

# 集成测试示例
class TestAuthAPI:
    def test_wechat_login(self, client):
        response = client.post("/api/auth/wechat-login", json={
            "code": "mock_wechat_code"
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "user_info" in data
```

### 前端测试策略

```typescript
// Jest配置 - jest.config.js
module.exports = {
    preset: '@vue/cli-plugin-unit-jest/presets/typescript-and-babel',
    testEnvironment: 'jsdom',
    setupFilesAfterEnv: ['<rootDir>/tests/setup.ts'],
    collectCoverageFrom: [
        'src/**/*.{ts,vue}',
        '!src/main.ts',
        '!src/**/*.d.ts'
    ],
    coverageThreshold: {
        global: {
            branches: 80,
            functions: 80,
            lines: 80,
            statements: 80
        }
    }
};

// 组件测试示例
import { mount } from '@vue/test-utils';
import NumberBall from '@/components/NumberBall.vue';

describe('NumberBall.vue', () => {
    it('renders number correctly', () => {
        const wrapper = mount(NumberBall, {
            props: { number: 15, type: 'predicted' }
        });
        expect(wrapper.text()).toBe('15');
        expect(wrapper.classes()).toContain('predicted-number');
    });
});
```

### E2E测试策略

```typescript
// Playwright配置
import { test, expect } from '@playwright/test';

test.describe('用户注册登录流程', () => {
    test('微信登录成功', async ({ page }) => {
        await page.goto('/pages/login/index');
        await page.click('.wechat-login-btn');
        
        // Mock微信授权回调
        await page.route('**/api/auth/wechat-login', async route => {
            await route.fulfill({
                status: 200,
                body: JSON.stringify({
                    access_token: 'mock_token',
                    user_info: { nickname: '测试用户' }
                })
            });
        });
        
        await expect(page.locator('.user-nickname')).toContainText('测试用户');
    });
});

test.describe('会员购买流程', () => {
    test('购买VIP会员成功', async ({ page }) => {
        // 登录 -> 选择套餐 -> 支付 -> 确认
        await page.goto('/pages/member/index');
        await page.click('[data-plan="vip_monthly"]');
        await page.click('.pay-btn');
        
        // Mock微信支付
        await page.route('**/api/payment/wechat-pay', async route => {
            await route.fulfill({
                status: 200,
                body: JSON.stringify({ success: true, order_no: 'test_order' })
            });
        });
        
        await expect(page.locator('.payment-success')).toBeVisible();
    });
});
```

### 性能测试

```python
# Locust性能测试
from locust import HttpUser, task, between

class Happy8User(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # 模拟登录
        response = self.client.post("/api/auth/wechat-login", json={
            "code": "mock_code"
        })
        self.token = response.json()["access_token"]
        self.client.headers.update({"Authorization": f"Bearer {self.token}"})
    
    @task(3)
    def get_latest_result(self):
        self.client.get("/api/data/latest-result")
    
    @task(2)
    def execute_prediction(self):
        self.client.post("/api/predict/execute", json={
            "algorithm": "frequency",
            "target_issue": "2025999",
            "periods": 30,
            "count": 5
        })
    
    @task(1)
    def get_membership_status(self):
        self.client.get("/api/membership/status")
```

### CI/CD测试流水线

```yaml
# .github/workflows/test.yml
name: Test Pipeline

on: [push, pull_request]

jobs:
  backend-test:
    runs-on: ubuntu-latest
    services:
      mysql:
        image: mysql:8.0
        env:
          MYSQL_ROOT_PASSWORD: password
        options: --health-cmd="mysqladmin ping" --health-interval=10s
      redis:
        image: redis:6.0
    
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pytest tests/ --cov=app --cov-report=xml
          coverage report --fail-under=80
  
  frontend-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
        with:
          node-version: '16'
      
      - name: Install dependencies
        run: npm install
      
      - name: Run tests
        run: npm run test:unit
      
      - name: E2E tests
        run: npm run test:e2e
```
