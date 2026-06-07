# 安装说明

## 环境要求

### 系统要求
- macOS 10.15+ / Windows 10+ / Ubuntu 18.04+
- Node.js 16.0+
- Python 3.9+
- MySQL 8.0+
- Redis 6.0+

### 开发工具
- **前端**: HBuilderX 或 VS Code + uni-app插件
- **后端**: PyCharm 或 VS Code + Python插件
- **数据库**: MySQL Workbench 或 Navicat

## 快速安装

### 1. 克隆项目
```bash
git clone <repository-url>
cd happy8-miniprogram
```

### 2. 后端环境设置
```bash
cd backend

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 环境配置
cp .env.example .env
# 编辑.env文件，配置数据库等信息
```

### 3. 数据库初始化
```bash
# 启动MySQL服务
sudo service mysql start

# 创建数据库
mysql -u root -p
CREATE DATABASE happy8_miniprogram;
CREATE USER 'happy8_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON happy8_miniprogram.* TO 'happy8_user'@'localhost';
exit;

# 运行数据库迁移
cd backend
alembic upgrade head

# 导入测试数据
python scripts/seed_data.py
```

### 4. Redis启动
```bash
# macOS (使用Homebrew)
brew install redis
brew services start redis

# Ubuntu
sudo apt install redis-server
sudo service redis-server start

# 验证Redis运行
redis-cli ping
```

### 5. 后端服务启动
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. 前端环境设置
```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器（微信小程序）
npm run dev:mp-weixin

# 启动H5版本（用于调试）
npm run dev:h5
```

## Docker部署（推荐）

### 使用Docker Compose快速启动
```bash
# 构建并启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 手动Docker部署
```bash
# 构建后端镜像
cd backend
docker build -t happy8-backend .

# 构建前端镜像
cd frontend
docker build -t happy8-frontend .

# 启动MySQL容器
docker run -d \
  --name happy8-mysql \
  -e MYSQL_ROOT_PASSWORD=rootpassword \
  -e MYSQL_DATABASE=happy8_miniprogram \
  -e MYSQL_USER=happy8_user \
  -e MYSQL_PASSWORD=userpassword \
  -p 3306:3306 \
  mysql:8.0

# 启动Redis容器
docker run -d \
  --name happy8-redis \
  -p 6379:6379 \
  redis:6.0-alpine

# 启动后端容器
docker run -d \
  --name happy8-backend \
  --link happy8-mysql:mysql \
  --link happy8-redis:redis \
  -p 8000:8000 \
  happy8-backend

# 启动Nginx容器
docker run -d \
  --name happy8-nginx \
  --link happy8-backend:backend \
  -p 80:80 \
  -p 443:443 \
  -v ./deployment/nginx/nginx.conf:/etc/nginx/nginx.conf \
  nginx:alpine
```

## 微信小程序配置

### 1. 微信开发者工具
- 下载并安装微信开发者工具
- 导入frontend目录作为项目根目录
- 配置AppID和服务器域名

### 2. 服务器域名配置
在微信公众平台配置以下域名：
- **request合法域名**: https://your-domain.com
- **uploadFile合法域名**: https://your-domain.com
- **downloadFile合法域名**: https://your-domain.com

### 3. 微信支付配置
- 申请微信支付商户号
- 配置支付回调URL
- 下载API证书

## 生产环境部署

### 1. 服务器要求
- CPU: 2核+
- 内存: 4GB+
- 存储: 50GB+
- 带宽: 5Mbps+

### 2. 域名和SSL
```bash
# 安装Certbot
sudo apt install certbot python3-certbot-nginx

# 申请SSL证书
sudo certbot --nginx -d your-domain.com

# 自动续期
sudo crontab -e
# 添加：0 12 * * * /usr/bin/certbot renew --quiet
```

### 3. 性能优化
- 启用Nginx gzip压缩
- 配置Redis缓存
- 数据库连接池优化
- CDN静态资源加速

## 故障排除

### 常见问题

**1. 后端启动失败**
```bash
# 检查Python版本
python --version

# 检查依赖安装
pip list

# 检查数据库连接
mysql -u happy8_user -p -h localhost happy8_miniprogram
```

**2. 前端编译错误**
```bash
# 清除缓存
npm run clean

# 重新安装依赖
rm -rf node_modules package-lock.json
npm install

# 检查uni-app版本
npx @dcloudio/uvm
```

**3. 数据库连接问题**
```bash
# 检查MySQL服务状态
sudo service mysql status

# 检查端口占用
netstat -tlnp | grep 3306

# 重置MySQL密码
sudo mysql_secure_installation
```

**4. Redis连接问题**
```bash
# 检查Redis服务
redis-cli ping

# 查看Redis配置
redis-cli CONFIG GET "*"

# 重启Redis
sudo service redis-server restart
```

## 开发流程

1. **功能开发**: 在develop分支开发新功能
2. **测试验证**: 运行单元测试和集成测试
3. **代码审查**: 创建Pull Request进行代码审查
4. **部署发布**: 合并到main分支，自动部署

## 支持

如有问题，请查看：
- [项目文档](./PROJECT_STRUCTURE.md)
- [API文档](../backend/docs/api.md)
- [常见问题](./FAQ.md)