# Happy8 智能预测小程序项目

## 项目简介

Happy8 是一个基于人工智能和机器学习的彩票预测系统，集成了17种不同的预测算法，为用户提供智能化的预测服务。项目采用前后端分离架构，包含微信小程序前端、FastAPI后端服务和完整的会员支付体系。

## 🎯 核心特性

### 智能预测系统
- **17种预测算法**: 从基础统计分析到深度学习模型
- **智能模式切换**: 自动识别历史验证模式和未来预测模式
- **置信度评估**: 每个预测结果都包含置信度评分
- **实时数据同步**: 自动获取最新开奖数据

### 会员体系
- **三级会员制度**: FREE（免费）/ VIP / PREMIUM
- **差异化权限**: 不同等级享受不同算法和预测次数
- **微信支付集成**: 便捷的会员升级和续费
- **使用统计分析**: 详细的预测历史和命中率统计

### 用户体验
- **微信登录**: 一键登录，无需注册
- **响应式设计**: 完美适配各种屏幕尺寸
- **实时同步**: 多端数据实时同步
- **离线缓存**: 关键数据本地缓存

## 🏗️ 系统架构

```
Happy8 项目架构
├── Frontend (微信小程序)
│   ├── uni-app + Vue 3 + TypeScript
│   ├── Wot Design Uni (UI组件库)
│   ├── Pinia (状态管理)
│   └── SCSS (样式处理)
├── Backend (API服务)
│   ├── FastAPI + Python 3.9+
│   ├── SQLAlchemy 2.0 (ORM)
│   ├── MySQL 8.0 (主数据库)
│   ├── Redis 6.0 (缓存)
│   └── Docker (容器化)
└── Algorithm Engine (原始Happy8系统)
    ├── 17种预测算法
    ├── 数据处理引擎
    └── 机器学习模型
```

## 📦 项目结构

```
happy8-miniprogram/
├── frontend/                 # 微信小程序前端
│   ├── src/
│   │   ├── pages/            # 页面文件
│   │   ├── components/       # 自定义组件
│   │   ├── stores/           # 状态管理
│   │   ├── services/         # API服务层
│   │   ├── types/            # TypeScript类型
│   │   └── constants/        # 常量配置
│   ├── pages.json           # 页面配置
│   ├── manifest.json        # 应用配置
│   └── package.json         # 依赖管理
├── backend/                  # 后端API服务
│   ├── app/
│   │   ├── api/             # API路由
│   │   ├── core/            # 核心配置
│   │   ├── db/              # 数据库模型
│   │   ├── services/        # 业务逻辑
│   │   └── utils/           # 工具函数
│   ├── alembic/             # 数据库迁移
│   ├── requirements.txt     # Python依赖
│   └── docker-compose.yml   # Docker配置
└── docs/                    # 项目文档
    ├── requirements.md      # 需求文档
    ├── design.md           # 设计文档
    └── tasks.md            # 任务规划
```

## 🚀 快速开始

### 环境要求

- **前端开发**:
  - Node.js >= 16.0.0
  - npm >= 7.0.0
  - 微信开发者工具
  - HBuilderX (推荐)

- **后端开发**:
  - Python >= 3.9
  - Docker & Docker Compose
  - MySQL 8.0
  - Redis 6.0

### 安装部署

#### 1. 克隆项目

```bash
git clone https://github.com/linshibo1994/happy8.git
cd happy8/happy8-miniprogram
```

#### 2. 后端服务部署

```bash
cd backend

# 使用Docker一键启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f backend
```

后端服务将在以下端口启动：
- API服务: http://localhost:8000
- MySQL: localhost:3306
- Redis: localhost:6379

#### 3. 前端开发

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev:mp-weixin

# 构建生产版本
npm run build:mp-weixin
```

使用微信开发者工具打开 `frontend/dist/dev/mp-weixin` 目录进行预览和调试。

#### 4. 配置微信小程序

1. 在微信公众平台申请小程序AppID
2. 配置服务器域名（API地址）
3. 配置微信支付商户号
4. 更新 `frontend/manifest.json` 中的AppID

## 🔧 核心功能模块

### 预测算法集成

项目完整集成了原始Happy8系统的17种预测算法：

- **基础统计类**: 频率分析、热冷号分析、遗漏分析
- **马尔可夫链**: 一阶、二阶、三阶、自适应马尔可夫链
- **机器学习**: 蒙特卡罗模拟、聚类分析、集成学习
- **深度学习**: Transformer模型、LSTM网络、图神经网络
- **高级算法**: 贝叶斯推理、超级预测器、高置信度预测

### 数据库设计

主要数据表：
- `users`: 用户基本信息
- `user_profiles`: 用户详细资料
- `memberships`: 会员信息
- `membership_plans`: 会员套餐
- `orders`: 订单记录
- `predictions`: 预测记录
- `lottery_results`: 开奖结果
- `algorithms`: 算法配置

### API接口设计

RESTful API设计，主要端点：
- `/auth/*`: 用户认证相关
- `/user/*`: 用户管理相关
- `/membership/*`: 会员体系相关
- `/predictions/*`: 预测功能相关
- `/lottery/*`: 开奖数据相关
- `/algorithms/*`: 算法管理相关

## 📱 小程序页面

### 主要页面

1. **首页** (`pages/index/index`): 
   - 最新开奖信息
   - 推荐算法展示
   - 用户统计概览
   - 快速功能入口

2. **预测页面** (`pages/predict/predict`):
   - 算法选择界面
   - 预测参数设置
   - 实时预测生成
   - 结果展示分析

3. **历史页面** (`pages/history/history`):
   - 预测历史记录
   - 筛选和统计
   - 命中率分析
   - 详细结果查看

4. **会员中心** (`pages/member/member`):
   - 会员状态展示
   - 套餐选择购买
   - 订单管理
   - 特权说明

5. **个人中心** (`pages/profile/profile`):
   - 用户信息管理
   - 设置和帮助
   - 统计数据
   - 功能服务

## 🎨 设计规范

### 色彩系统
- 主色: #d32f2f (红色，符合彩票主题)
- 辅助色: #666666 (灰色)
- 成功色: #4caf50 (绿色)
- 警告色: #ff9800 (橙色)
- 错误色: #f44336 (红色)

### 字体规范
- 大标题: 48rpx
- 标题: 36rpx  
- 正文: 32rpx
- 小字: 28rpx
- 标注: 24rpx

### 间距规范
- 小间距: 20rpx
- 中间距: 30rpx
- 大间距: 40rpx
- 超大间距: 50rpx

## 🔐 安全考虑

### 数据安全
- JWT token认证
- 密码加密存储
- 敏感数据传输加密
- SQL注入防护

### 隐私保护
- 用户数据最小化收集
- 隐私政策明确告知
- 数据访问权限控制
- 定期数据清理

### 支付安全
- 微信支付官方SDK
- 订单数据验证
- 支付结果校验
- 异常监控告警

## 📊 性能优化

### 前端优化
- 组件懒加载
- 图片压缩优化
- 本地缓存策略
- 网络请求优化

### 后端优化
- Redis缓存加速
- 数据库索引优化
- API响应压缩
- 异步任务处理

### 算法优化
- 模型缓存机制
- 并行计算优化
- 内存使用优化
- 计算结果缓存

## 🧪 测试策略

### 单元测试
- 业务逻辑测试
- API接口测试
- 数据库操作测试
- 算法准确性测试

### 集成测试
- 端到端流程测试
- 支付流程测试
- 数据同步测试
- 性能压力测试

### 用户测试
- 真机兼容性测试
- 用户体验测试
- 网络环境测试
- 边界情况测试

## 📈 监控运维

### 应用监控
- 接口响应时间监控
- 错误率统计
- 用户行为分析
- 性能指标监控

### 服务监控
- 服务器资源监控
- 数据库性能监控
- Redis缓存监控
- 网络状态监控

### 业务监控
- 预测准确率统计
- 用户活跃度分析
- 支付成功率监控
- 会员转化率分析

## 🤝 贡献指南

### 开发流程
1. Fork项目到个人仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

### 代码规范
- 遵循ESLint配置
- 使用TypeScript类型检查
- 编写单元测试
- 添加必要注释

## 📄 许可证

本项目仅供学习和研究使用，不得用于商业用途。

## 📞 联系方式

- 项目维护者: [linshibo1994](https://github.com/linshibo1994)
- 项目地址: https://github.com/linshibo1994/happy8
- 问题反馈: [Issues](https://github.com/linshibo1994/happy8/issues)

## 🎉 致谢

感谢所有为这个项目做出贡献的开发者和测试用户！

---

**注意**: 本项目是一个彩票预测系统，用于学习和研究目的。彩票具有随机性，任何预测算法都不能保证100%的准确性。请理性对待预测结果，合理安排个人财务。