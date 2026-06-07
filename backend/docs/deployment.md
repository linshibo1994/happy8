# Happy8小程序后端部署指南

## 🚀 快速开始

### 1. 环境准备

**系统要求:**
- Python 3.9+
- MySQL 8.0+
- Redis 6.0+
- Docker & Docker Compose (可选)

**必需服务:**
```bash
# MySQL数据库
# Redis缓存
# 微信小程序配置
```

### 2. 项目部署

#### 方式一：Docker部署（推荐）

```bash
# 1. 克隆项目
cd /Users/linshibo/GithubProject/Happy8/happy8-miniprogram

# 2. 配置环境变量
cp backend/.env.example backend/.env
# 编辑 .env 文件，配置数据库和微信参数

# 3. 启动所有服务
docker-compose up -d

# 4. 初始化数据库
docker-compose exec backend python init_db.py
```

#### 方式二：本地部署

```bash
# 1. 进入后端目录
cd backend

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件

# 5. 初始化数据库
python init_db.py

# 6. 启动服务
python start.py
```

### 3. 数据库管理

```bash
# 完整初始化（首次部署）
./db.sh setup

# 日常管理命令
./db.sh check           # 检查连接
./db.sh migrate "描述"   # 创建迁移
./db.sh upgrade         # 升级数据库
./db.sh backup          # 备份数据
./db.sh current         # 查看版本
./db.sh history         # 迁移历史
```

## 📝 配置说明

### 环境变量配置

在 `backend/.env` 文件中配置以下参数：

```bash
# 数据库配置
DATABASE_URL=mysql://happy8_user:happy8_pass_2025@localhost:3306/happy8_miniprogram

# Redis配置
REDIS_URL=redis://:happy8_redis_2025@localhost:6379/0

# JWT配置
SECRET_KEY=happy8_jwt_secret_key_2025_very_long_and_secure
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# 应用配置
ENVIRONMENT=production
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000

# 微信小程序配置
WECHAT_APP_ID=你的小程序AppID
WECHAT_APP_SECRET=你的小程序AppSecret

# 微信支付配置
WECHAT_PAY_MCHID=你的商户号
WECHAT_PAY_PRIVATE_KEY_PATH=/path/to/private_key.pem
WECHAT_PAY_CERT_SERIAL=证书序列号
WECHAT_PAY_APIV3_KEY=APIv3密钥
WECHAT_PAY_NOTIFY_URL=https://your-domain.com/api/v1/payments/notify
```

### 微信小程序配置

1. **获取AppID和AppSecret:**
   - 登录微信公众平台
   - 在"开发"->"开发设置"中获取

2. **配置服务器域名:**
   - 在"开发"->"开发设置"->"服务器域名"中添加
   - request合法域名: `https://your-domain.com`

3. **微信支付配置:**
   - 在微信商户平台下载证书
   - 配置支付回调URL

## 🔧 服务管理

### 启动服务

```bash
# 开发模式
python start.py

# 生产模式
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# 使用systemd管理（推荐）
sudo systemctl start happy8-api
sudo systemctl enable happy8-api
```

### 健康检查

```bash
# API健康检查
curl http://localhost:8000/health

# 响应示例
{
  "status": "healthy",
  "database": "connected",
  "redis": "connected",
  "version": "1.0.0"
}
```

### 日志管理

```bash
# 查看日志
tail -f logs/info.log
tail -f logs/error.log

# 日志轮转已自动配置
# 单个文件最大10MB，保留5个历史文件
```

## 🏗️ 架构说明

### 项目结构

```
backend/
├── app/                    # 应用代码
│   ├── api/               # API路由
│   │   ├── v1/           # API v1版本
│   │   └── schemas/      # 请求/响应模型
│   ├── core/             # 核心模块
│   │   ├── auth.py       # JWT认证
│   │   ├── cache.py      # Redis缓存
│   │   ├── config.py     # 配置管理
│   │   ├── database.py   # 数据库连接
│   │   ├── exceptions.py # 异常处理
│   │   └── logging.py    # 日志配置
│   ├── models/           # 数据模型
│   ├── services/         # 业务逻辑
│   ├── utils/            # 工具类
│   └── main.py          # 应用入口
├── alembic/             # 数据库迁移
├── docs/                # 项目文档
├── logs/                # 日志文件
├── requirements.txt     # Python依赖
├── docker-compose.yml   # Docker配置
├── db.sh               # 数据库管理脚本
├── init_db.py          # 数据库初始化
└── start.py            # 启动脚本
```

### 技术栈

- **后端框架:** FastAPI
- **数据库:** MySQL 8.0 + SQLAlchemy 2.0
- **缓存:** Redis 6.0
- **认证:** JWT + 微信OAuth
- **迁移:** Alembic
- **容器化:** Docker + Docker Compose

## 🔍 监控与维护

### 性能监控

- **连接池监控:** 数据库和Redis连接池状态
- **API响应时间:** 中间件自动记录
- **错误率统计:** 异常处理器统计
- **业务指标:** 用户、预测、订单等关键指标

### 安全措施

- **JWT认证:** 访问令牌 + 刷新令牌机制
- **令牌黑名单:** 登出时令牌失效
- **频率限制:** API调用频率控制
- **参数验证:** Pydantic严格验证
- **SQL注入防护:** SQLAlchemy ORM
- **CORS配置:** 跨域访问控制

### 备份策略

```bash
# 自动备份脚本
./db.sh backup backup_$(date +%Y%m%d_%H%M%S).sql

# 定时备份（添加到crontab）
0 2 * * * cd /path/to/backend && ./db.sh backup
```

## 🚨 故障排除

### 常见问题

1. **数据库连接失败**
   ```bash
   # 检查数据库服务
   systemctl status mysql

   # 检查配置
   ./db.sh check
   ```

2. **Redis连接失败**
   ```bash
   # 检查Redis服务
   systemctl status redis

   # 测试连接
   redis-cli ping
   ```

3. **微信API调用失败**
   - 检查AppID和AppSecret配置
   - 确认服务器域名已配置
   - 查看微信开发者工具网络请求

4. **权限问题**
   ```bash
   # 检查文件权限
   chmod +x db.sh
   chmod +x start.py

   # 检查日志目录权限
   mkdir -p logs
   chmod 755 logs
   ```

### 日志分析

```bash
# 查看错误日志
grep "ERROR" logs/error.log | tail -20

# 查看API访问日志
grep "API请求" logs/info.log | tail -20

# 查看特定用户操作
grep "用户: 123" logs/info.log
```

## 📚 API文档

- **开发环境:** http://localhost:8000/docs
- **生产环境:** 文档已禁用（安全考虑）

### 主要接口

- `POST /api/v1/users/login/wechat` - 微信登录
- `GET /api/v1/users/profile` - 获取用户资料
- `PUT /api/v1/users/profile` - 更新用户资料
- `POST /api/v1/users/refresh` - 刷新令牌
- `GET /api/v1/users/statistics` - 用户统计

## 🔄 升级指南

### 版本升级流程

1. **备份数据**
   ```bash
   ./db.sh backup upgrade_backup_$(date +%Y%m%d).sql
   ```

2. **更新代码**
   ```bash
   git pull origin main
   ```

3. **更新依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **数据库迁移**
   ```bash
   ./db.sh upgrade
   ```

5. **重启服务**
   ```bash
   systemctl restart happy8-api
   ```

6. **验证服务**
   ```bash
   curl http://localhost:8000/health
   ```

---

## 📞 技术支持

如需技术支持，请查看：
- 项目文档目录 `/docs`
- 错误日志 `/logs`
- GitHub Issues