#!/bin/bash

# 快速启动开发环境脚本

set -e

echo "🚀 启动Happy8小程序开发环境..."

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 创建环境变量文件（如果不存在）
if [ ! -f "backend/.env" ]; then
    echo "📝 创建环境变量文件..."
    cp backend/.env.example backend/.env
    echo "✅ 请编辑 backend/.env 文件配置你的环境变量"
fi

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p deployment/nginx/ssl
mkdir -p backend/uploads
mkdir -p database/backups

# 启动服务
echo "🐳 启动Docker容器..."
docker-compose up -d mysql redis

echo "⏳ 等待MySQL启动..."
sleep 30

echo "🔧 运行数据库迁移..."
docker-compose run --rm backend alembic upgrade head

echo "🌱 导入测试数据..."
docker-compose run --rm backend python scripts/seed_data.py

echo "🚀 启动所有服务..."
docker-compose up -d

echo "✅ 开发环境启动完成！"
echo ""
echo "📊 服务访问地址："
echo "   - 后端API: http://localhost:8000"
echo "   - API文档: http://localhost:8000/docs"
echo "   - 前端H5: http://localhost:3000"
echo "   - MySQL: localhost:3306"
echo "   - Redis: localhost:6379"
echo ""
echo "📝 查看日志: docker-compose logs -f"
echo "🛑 停止服务: docker-compose down"