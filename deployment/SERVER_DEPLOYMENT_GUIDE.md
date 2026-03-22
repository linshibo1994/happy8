# Happy8 服务器部署完整指南

## 📋 目录
1. [部署前准备](#部署前准备)
2. [Docker 部署方案（推荐）](#docker-部署方案推荐)
3. [传统部署方案](#传统部署方案)
4. [Nginx 反向代理配置](#nginx-反向代理配置)
5. [SSL/HTTPS 配置](#sslhttps-配置)
6. [监控与维护](#监控与维护)
7. [故障排查](#故障排查)

---

## 部署前准备

### 1. 服务器要求

**最低配置**：
- CPU: 2核
- 内存: 4GB
- 硬盘: 20GB
- 操作系统: Ubuntu 20.04+ / CentOS 7+ / Debian 10+

**推荐配置**：
- CPU: 4核
- 内存: 8GB
- 硬盘: 50GB
- 操作系统: Ubuntu 22.04 LTS

### 2. 必需软件

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y git curl wget vim
```

---

## Docker 部署方案（推荐）

### 方案 A: Docker Compose（最简单）

#### 1. 安装 Docker 和 Docker Compose

```bash
# 安装 Docker
curl -fsSL https://get.docker.com | bash

# 启动 Docker 服务
sudo systemctl start docker
sudo systemctl enable docker

# 将当前用户添加到 docker 组（避免每次使用 sudo）
sudo usermod -aG docker $USER

# 重新登录或运行以下命令使组权限生效
newgrp docker

# 验证安装
docker --version
docker compose version
```

#### 2. 克隆项目到服务器

```bash
# 克隆项目
git clone https://github.com/你的用户名/Happy8.git
cd Happy8

# 或者如果已经在本地，使用 scp 上传
# scp -r Happy8 user@your-server:/path/to/destination
```

#### 3. 配置环境变量（可选）

```bash
# 创建 .env 文件
cat > .env <<EOF
HAPPY8_ENV=production
TZ=Asia/Shanghai
EOF
```

#### 4. 构建并启动服务

```bash
# 进入部署目录
cd deployment

# 构建并启动所有服务（后台运行）
docker compose up -d --build

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f happy8-app
```

#### 5. 访问应用

- 直接访问: `http://your-server-ip:8501`
- 通过 Nginx: `http://your-server-ip`

#### 6. 常用管理命令

```bash
# 停止服务
docker compose down

# 重启服务
docker compose restart

# 查看日志
docker compose logs -f

# 更新代码并重新部署
git pull
docker compose up -d --build

# 清理旧镜像
docker image prune -f
```

---

## 传统部署方案

### 方案 B: 直接在服务器上运行

#### 1. 安装 Python 环境

```bash
# 安装 Python 3.9+
sudo apt install -y python3.9 python3.9-venv python3-pip

# 验证安装
python3 --version
```

#### 2. 创建虚拟环境

```bash
cd Happy8

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 3. 配置 Systemd 服务

创建服务文件 `/etc/systemd/system/happy8.service`:

```ini
[Unit]
Description=Happy8 Prediction System
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/Happy8
Environment="PATH=/path/to/Happy8/venv/bin"
ExecStart=/path/to/Happy8/venv/bin/python main.py web
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务:

```bash
# 重载 systemd 配置
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start happy8

# 设置开机自启
sudo systemctl enable happy8

# 查看状态
sudo systemctl status happy8

# 查看日志
sudo journalctl -u happy8 -f
```

---

## Nginx 反向代理配置

### 安装 Nginx

```bash
sudo apt install -y nginx
```

### 配置 Nginx

创建配置文件 `/etc/nginx/sites-available/happy8`:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # 替换为你的域名或服务器IP

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    # 健康检查
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

启用配置:

```bash
# 创建软链接
sudo ln -s /etc/nginx/sites-available/happy8 /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重启 Nginx
sudo systemctl restart nginx
```

---

## SSL/HTTPS 配置

### 使用 Let's Encrypt 免费证书

```bash
# 安装 Certbot
sudo apt install -y certbot python3-certbot-nginx

# 获取证书并自动配置 Nginx
sudo certbot --nginx -d your-domain.com

# 测试自动续期
sudo certbot renew --dry-run
```

证书会自动续期，无需手动操作。

---

## 监控与维护

### 1. 日志管理

```bash
# Docker 方式查看日志
docker compose logs -f happy8-app

# Systemd 方式查看日志
sudo journalctl -u happy8 -f

# 查看 Nginx 日志
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 2. 性能监控

```bash
# 查看容器资源使用
docker stats

# 查看系统资源
htop
# 或
top
```

### 3. 数据备份

```bash
# 备份数据目录
tar -czf happy8_data_$(date +%Y%m%d).tar.gz data/

# 定期备份脚本（添加到 crontab）
# 每天凌晨 2 点备份
0 2 * * * cd /path/to/Happy8 && tar -czf /backup/happy8_data_$(date +\%Y\%m\%d).tar.gz data/
```

### 4. 自动更新

创建更新脚本 `update.sh`:

```bash
#!/bin/bash
cd /path/to/Happy8
git pull
docker compose down
docker compose up -d --build
```

---

## 故障排查

### 问题 1: 端口被占用

```bash
# 查看端口占用
sudo lsof -i :8501
sudo netstat -tulpn | grep 8501

# 修改端口（编辑 .streamlit/config.toml）
```

### 问题 2: 内存不足

```bash
# 查看内存使用
free -h

# 如果内存不足，注释掉 requirements.txt 中的大型依赖
# tensorflow-cpu, torch 等
```

### 问题 3: Docker 容器无法启动

```bash
# 查看详细日志
docker compose logs happy8-app

# 检查容器状态
docker compose ps

# 重新构建
docker compose build --no-cache
docker compose up -d
```

### 问题 4: Streamlit 无法访问

```bash
# 检查防火墙
sudo ufw status
sudo ufw allow 8501
sudo ufw allow 80
sudo ufw allow 443

# 检查 Streamlit 配置
cat .streamlit/config.toml
```

---

## 安全建议

1. **防火墙配置**
```bash
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
```

2. **定期更新系统**
```bash
sudo apt update && sudo apt upgrade -y
```

3. **限制访问**
- 使用 Nginx 配置 IP 白名单
- 启用基本认证
- 使用 VPN 访问

4. **数据安全**
- 定期备份数据
- 使用环境变量存储敏感信息
- 不要将 .env 文件提交到 Git

---

## 快速部署命令总结

### Docker 方式（推荐）

```bash
# 一键部署
git clone https://github.com/你的用户名/Happy8.git
cd Happy8/deployment
docker compose up -d --build

# 访问: http://your-server-ip:8501
```

### 传统方式

```bash
# 一键部署
git clone https://github.com/你的用户名/Happy8.git
cd Happy8
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py web

# 访问: http://localhost:8501
```

---

## 生产环境检查清单

- [ ] 服务器配置满足最低要求
- [ ] Docker 和 Docker Compose 已安装
- [ ] 项目代码已克隆到服务器
- [ ] 环境变量已配置
- [ ] 服务已启动并运行正常
- [ ] Nginx 反向代理已配置
- [ ] SSL 证书已配置（如使用域名）
- [ ] 防火墙规则已设置
- [ ] 日志监控已配置
- [ ] 数据备份策略已实施
- [ ] 自动更新脚本已配置

---

## 技术支持

如遇到问题，请检查:
1. 项目日志
2. Docker 容器状态
3. Nginx 配置
4. 防火墙设置
5. 系统资源使用情况
