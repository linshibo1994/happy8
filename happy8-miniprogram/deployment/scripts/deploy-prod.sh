#!/bin/bash

# 生产环境部署脚本

set -e

echo "🚀 部署Happy8小程序到生产环境..."

# 检查环境变量
if [ -z "$DOMAIN" ]; then
    echo "❌ 请设置DOMAIN环境变量"
    exit 1
fi

# 备份数据库
echo "💾 备份数据库..."
BACKUP_FILE="database/backups/backup_$(date +%Y%m%d_%H%M%S).sql"
docker-compose exec mysql mysqldump -u happy8_user -p happy8_miniprogram > "$BACKUP_FILE"
echo "✅ 数据库备份完成: $BACKUP_FILE"

# 拉取最新代码
echo "📥 拉取最新代码..."
git pull origin main

# 构建镜像
echo "🔨 构建Docker镜像..."
docker-compose build

# 更新SSL证书
echo "🔒 检查SSL证书..."
if [ ! -f "deployment/nginx/ssl/cert.pem" ]; then
    echo "📜 申请SSL证书..."
    sudo certbot certonly --nginx -d "$DOMAIN"
    cp "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" deployment/nginx/ssl/cert.pem
    cp "/etc/letsencrypt/live/$DOMAIN/privkey.pem" deployment/nginx/ssl/private.key
fi

# 运行数据库迁移
echo "🔧 运行数据库迁移..."
docker-compose run --rm backend alembic upgrade head

# 重启服务
echo "🔄 重启服务..."
docker-compose down
docker-compose up -d

# 健康检查
echo "🏥 健康检查..."
sleep 30
if curl -f "https://$DOMAIN/health" > /dev/null 2>&1; then
    echo "✅ 部署成功！服务正常运行"
else
    echo "❌ 部署失败！请检查日志"
    docker-compose logs
    exit 1
fi

echo "🎉 生产环境部署完成！"
echo "🌐 访问地址: https://$DOMAIN"