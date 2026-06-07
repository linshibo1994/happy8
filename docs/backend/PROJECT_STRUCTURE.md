# 项目结构说明

这是Happy8彩票预测小程序的完整项目结构，采用前后端分离架构。

## 目录结构详解

### frontend/ - 小程序前端
基于uni-app框架，支持Vue3+TypeScript，可编译为微信小程序。

**pages/**: 页面组件
- `index/`: 首页 - 最新开奖结果、快速预测入口
- `predict/`: 预测页面 - 算法选择、参数配置、结果展示
- `history/`: 历史数据 - 开奖记录、走势图表
- `member/`: 会员中心 - 套餐购买、会员管理
- `profile/`: 个人中心 - 用户信息、设置

**components/**: 可复用组件
- `NumberBall/`: 彩票号码球组件，支持不同状态显示
- `PredictCard/`: 预测结果卡片组件
- `ChartView/`: 数据图表组件，基于uCharts
- `MembershipCard/`: 会员等级卡片组件
- `PaymentModal/`: 支付弹窗组件

**store/**: Pinia状态管理
- 用户状态管理
- 会员信息管理
- 预测数据管理

### backend/ - 后端服务
基于FastAPI框架，提供RESTful API服务。

**app/api/**: API路由模块
- 用户认证API
- 会员管理API
- 支付服务API
- 预测功能API
- 数据管理API

**app/models/**: SQLAlchemy数据模型
- User, UserProfile
- Membership, MembershipPlan, MembershipOrder
- PredictionHistory, LotteryResult

**app/services/**: 业务逻辑层
- AuthService: 认证服务
- MembershipService: 会员服务
- PaymentService: 支付服务
- PredictService: 预测服务

### database/ - 数据库管理
**init/**: 数据库初始化
- 创建数据库和用户
- 基础配置

**migrations/**: Alembic数据库迁移
- 版本控制
- 结构变更

**seeds/**: 测试数据
- 示例用户数据
- 会员套餐配置
- 历史彩票数据

### deployment/ - 部署配置
**docker/**: 容器化配置
- Dockerfile
- docker-compose.yml

**nginx/**: 反向代理配置
- SSL配置
- 负载均衡

**scripts/**: 部署脚本
- 自动化部署
- 环境配置

## 开发环境设置

1. 克隆项目后，分别进入frontend和backend目录
2. 安装依赖并启动服务
3. 配置数据库连接
4. 运行迁移脚本初始化数据库

## 注意事项

- 前后端采用不同端口运行，通过API通信
- 所有敏感配置使用环境变量
- 遵循RESTful API设计规范
- 代码严格按照TypeScript/Python类型提示