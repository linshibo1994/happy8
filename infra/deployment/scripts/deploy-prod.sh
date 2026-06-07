#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"

echo "🚀 部署 Happy8 到生产环境..."

if [ -z "$DOMAIN" ]; then
    echo "❌ 请设置DOMAIN环境变量"
    exit 1
fi

mkdir -p "$PROJECT_ROOT/infra/database/backups"

echo "💾 备份数据库..."
BACKUP_FILE="$PROJECT_ROOT/infra/database/backups/backup_$(date +%Y%m%d_%H%M%S).sql"
docker compose -f "$COMPOSE_FILE" exec mysql mysqldump -u happy8_user -phappy8_pass_2025 happy8_miniprogram > "$BACKUP_FILE"
echo "✅ 数据库备份完成: $BACKUP_FILE"

echo "📥 拉取最新代码..."
git -C "$PROJECT_ROOT" pull origin main

echo "🔨 构建Docker镜像..."
docker compose -f "$COMPOSE_FILE" build

echo "🔒 检查SSL证书..."
if [ ! -f "$PROJECT_ROOT/infra/deployment/nginx/ssl/cert.pem" ]; then
    echo "📜 申请SSL证书..."
    sudo certbot certonly --nginx -d "$DOMAIN"
    cp "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" "$PROJECT_ROOT/infra/deployment/nginx/ssl/cert.pem"
    cp "/etc/letsencrypt/live/$DOMAIN/privkey.pem" "$PROJECT_ROOT/infra/deployment/nginx/ssl/private.key"
fi

echo "🔧 运行数据库迁移..."
docker compose -f "$COMPOSE_FILE" run --rm backend alembic upgrade head

echo "🔄 重启服务..."
docker compose -f "$COMPOSE_FILE" down
docker compose -f "$COMPOSE_FILE" up -d

echo "🏥 健康检查..."
sleep 30
if curl -f "https://$DOMAIN/health" > /dev/null 2>&1; then
    echo "✅ 部署成功！服务正常运行"
else
    echo "❌ 部署失败！请检查日志"
    docker compose -f "$COMPOSE_FILE" logs
    exit 1
fi

echo "🎉 生产环境部署完成！"
echo "🌐 访问地址: https://$DOMAIN"
