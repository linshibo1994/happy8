#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"

echo "🚀 启动 Happy8 开发环境..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose不可用，请先安装Docker Compose插件"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/backend/.env" ]; then
    echo "📝 创建环境变量文件..."
    cp "$PROJECT_ROOT/backend/.env.example" "$PROJECT_ROOT/backend/.env"
    echo "✅ 请编辑 backend/.env 文件配置你的环境变量"
fi

echo "📁 创建必要的目录..."
mkdir -p "$PROJECT_ROOT/infra/deployment/nginx/ssl"
mkdir -p "$PROJECT_ROOT/backend/uploads"
mkdir -p "$PROJECT_ROOT/infra/database/backups"

echo "🐳 启动数据库和缓存..."
docker compose -f "$COMPOSE_FILE" up -d mysql redis

echo "⏳ 等待MySQL启动..."
sleep 30

echo "🔧 运行数据库迁移..."
docker compose -f "$COMPOSE_FILE" run --rm backend alembic upgrade head

echo "🚀 启动所有服务..."
docker compose -f "$COMPOSE_FILE" up -d

echo "✅ 开发环境启动完成！"
echo "   - 后端API: http://localhost:8000"
echo "   - API文档: http://localhost:8000/docs"
echo "   - MySQL: localhost:3306"
echo "   - Redis: localhost:6379"
