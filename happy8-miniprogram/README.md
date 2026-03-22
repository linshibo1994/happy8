# Happy8彩票预测小程序

基于现有Happy8彩票预测系统开发的微信小程序，采用前后端分离架构，支持17种预测算法、会员体系和微信支付功能。

## 项目结构

```
happy8-miniprogram/
├── frontend/                    # 小程序前端 (uni-app)
│   ├── pages/                  # 页面组件
│   │   ├── index/             # 首页
│   │   ├── predict/           # 预测页面
│   │   ├── history/           # 历史数据
│   │   ├── member/            # 会员中心
│   │   └── profile/           # 个人中心
│   ├── components/            # 公共组件
│   │   ├── NumberBall/        # 号码球组件
│   │   ├── PredictCard/       # 预测卡片组件
│   │   ├── ChartView/         # 图表组件
│   │   ├── MembershipCard/    # 会员卡组件
│   │   └── PaymentModal/      # 支付弹窗组件
│   ├── store/                 # 状态管理 (Pinia)
│   ├── utils/                 # 工具函数
│   └── static/                # 静态资源
├── backend/                   # 后端服务 (FastAPI)
│   ├── app/
│   │   ├── api/               # API路由
│   │   ├── models/            # 数据模型 (SQLAlchemy)
│   │   ├── services/          # 业务逻辑
│   │   ├── core/              # 核心配置
│   │   └── utils/             # 工具函数
│   ├── tests/                 # 测试文件
│   └── scripts/               # 脚本文件
├── database/                  # 数据库相关
│   ├── init/                  # 初始化脚本
│   ├── migrations/            # 数据库迁移
│   └── seeds/                 # 测试数据
├── deployment/                # 部署配置
│   ├── docker/                # Docker配置
│   ├── nginx/                 # Nginx配置
│   └── scripts/               # 部署脚本
└── docs/                      # 项目文档
```

## 技术栈

### 前端
- **uni-app** + Vue 3 + TypeScript
- **Wot Design Uni** - UI组件库
- **Pinia** - 状态管理
- **uCharts** - 图表展示

### 后端  
- **FastAPI** + Python 3.9+
- **SQLAlchemy** - ORM框架
- **Alembic** - 数据库迁移
- **Redis** - 缓存和会话存储
- **Celery** - 异步任务队列

### 数据库
- **MySQL 8.0** - 主数据库
- **Redis 6.0** - 缓存数据库

### 部署
- **Docker** + Docker Compose
- **Nginx** - 反向代理和负载均衡

## 核心功能

1. **用户认证系统** - 微信小程序登录、JWT认证
2. **会员体系管理** - 三级会员制度、权限控制
3. **微信支付集成** - 支付V3 API、订单管理
4. **预测引擎** - 17种算法、结果缓存
5. **数据管理** - 历史数据、可视化图表

## 开发计划

详细的开发计划请参考：`../happy8-miniprogram-specs/tasks.md`

- **总开发时间**: 316小时 (约40个工作日)
- **推荐周期**: 8-10周
- **5个阶段**: 基础搭建 → 后端开发 → 前端开发 → 集成优化 → 部署上线

## 快速开始

### 环境要求
- Node.js 16+
- Python 3.9+
- MySQL 8.0+
- Redis 6.0+
- Docker (可选)

### 后端启动
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 前端启动
```bash
cd frontend  
npm install
npm run dev:mp-weixin
```

## 文档链接

- [需求文档](../happy8-miniprogram-specs/requirements.md)
- [设计文档](../happy8-miniprogram-specs/design.md) 
- [任务计划](../happy8-miniprogram-specs/tasks.md)

## 开发团队

基于现有Happy8彩票预测系统进行开发，复用17种预测算法。

## License

MIT License