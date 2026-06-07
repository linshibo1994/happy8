# 环境配置说明

## 开发环境配置

### 1. 系统要求
- **操作系统**: macOS 10.15+ / Windows 10+ / Ubuntu 18.04+
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Node.js**: 16.0+ (前端开发)
- **Python**: 3.9+ (后端开发)

### 2. Docker Compose服务

#### MySQL 8.0
- **端口**: 3306
- **数据库**: happy8_miniprogram
- **用户**: happy8_user
- **密码**: happy8_pass_2025
- **数据持久化**: mysql_data卷

#### Redis 6.0
- **端口**: 6379
- **密码**: happy8_redis_2025
- **数据持久化**: redis_data卷
- **配置**: AOF持久化

#### FastAPI后端
- **端口**: 8000
- **环境**: development
- **自动重载**: 启用
- **依赖**: MySQL, Redis

#### Nginx反向代理
- **HTTP端口**: 80
- **HTTPS端口**: 443
- **功能**: 负载均衡、SSL终端、限流

### 3. 快速启动

```bash
# 克隆项目
git clone <repository-url>
cd happy8-miniprogram

# 启动开发环境
./deployment/scripts/start-dev.sh

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 4. 环境变量配置

复制 `backend/.env.example` 到 `backend/.env` 并配置：

```bash
# 数据库配置
DATABASE_URL=mysql://happy8_user:happy8_pass_2025@localhost:3306/happy8_miniprogram

# Redis配置
REDIS_URL=redis://:happy8_redis_2025@localhost:6379/0

# JWT密钥（生产环境请使用随机生成的密钥）
SECRET_KEY=happy8_jwt_secret_key_2025_very_long_and_secure

# 微信小程序配置
WECHAT_APP_ID=your_wechat_app_id
WECHAT_APP_SECRET=your_wechat_app_secret

# 微信支付配置
WECHAT_PAY_MCHID=your_merchant_id
WECHAT_PAY_PRIVATE_KEY_PATH=/path/to/private_key.pem
WECHAT_PAY_CERT_SERIAL=your_cert_serial
WECHAT_PAY_APIV3_KEY=your_apiv3_key
```

### 5. 开发工具配置

#### VS Code
- 使用提供的 `happy8-miniprogram.code-workspace` 工作区
- 自动安装推荐的扩展
- 配置了Python、TypeScript、Docker等开发环境

#### 调试配置
- **FastAPI调试**: 使用VS Code调试配置
- **Docker调试**: 支持容器内调试
- **日志查看**: `docker-compose logs -f [service]`

### 6. 数据库管理

#### 数据库迁移
```bash
# 创建新迁移
docker-compose run --rm backend alembic revision --autogenerate -m "描述"

# 执行迁移
docker-compose run --rm backend alembic upgrade head

# 回滚迁移
docker-compose run --rm backend alembic downgrade -1
```

#### 数据备份
```bash
# 备份数据库
docker-compose exec mysql mysqldump -u happy8_user -p happy8_miniprogram > backup.sql

# 恢复数据库
docker-compose exec -i mysql mysql -u happy8_user -p happy8_miniprogram < backup.sql
```

### 7. 性能监控

#### 服务健康检查
- **后端健康检查**: http://localhost:8000/health
- **数据库连接**: docker-compose exec mysql mysql -u happy8_user -p
- **Redis连接**: docker-compose exec redis redis-cli

#### 资源监控
```bash
# 查看容器资源使用
docker stats

# 查看容器日志
docker-compose logs -f [service]

# 查看数据库状态
docker-compose exec mysql mysqladmin -u root -p status
```

## 生产环境配置

### 1. 环境要求
- **服务器**: 2核4GB+
- **存储**: 50GB+ SSD
- **带宽**: 5Mbps+
- **域名**: HTTPS证书

### 2. 部署步骤
```bash
# 设置域名
export DOMAIN=your-domain.com

# 运行部署脚本
./deployment/scripts/deploy-prod.sh
```

### 3. SSL证书
- 使用Let's Encrypt自动申请
- 自动续期配置
- HTTPS强制重定向

### 4. 安全配置
- 防火墙配置
- API限流
- 数据库访问控制
- SSL/TLS加密

## 故障排除

### 常见问题
1. **容器启动失败**: 检查端口占用、权限设置
2. **数据库连接失败**: 检查网络配置、用户权限
3. **Redis连接超时**: 检查密码配置、网络连通性
4. **Nginx配置错误**: 检查配置文件语法、证书路径

### 日志查看
```bash
# 查看所有服务日志
docker-compose logs

# 查看特定服务日志
docker-compose logs backend
docker-compose logs mysql
docker-compose logs redis
docker-compose logs nginx
```