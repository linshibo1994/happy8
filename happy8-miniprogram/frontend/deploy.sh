#!/bin/bash

# Happy8 小程序前端部署脚本

set -e

echo "🚀 开始部署 Happy8 小程序前端..."

# 检查环境
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安装，请先安装 Node.js"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ npm 未安装，请先安装 npm"
    exit 1
fi

# 获取部署环境参数
ENVIRONMENT=${1:-production}
echo "📦 部署环境: $ENVIRONMENT"

# 检查环境配置文件
ENV_FILE=".env.$ENVIRONMENT"
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ 环境配置文件 $ENV_FILE 不存在"
    exit 1
fi

# 安装依赖
echo "📥 安装依赖包..."
npm ci

# 类型检查
echo "🔍 进行类型检查..."
npm run type-check

# 代码检查
echo "🔍 进行代码检查..."
npm run lint

# 构建项目
echo "🔨 构建项目..."
if [ "$ENVIRONMENT" = "production" ]; then
    npm run build:mp-weixin --mode production
else
    npm run build:mp-weixin --mode $ENVIRONMENT
fi

# 检查构建结果
BUILD_DIR="dist/build/mp-weixin"
if [ ! -d "$BUILD_DIR" ]; then
    echo "❌ 构建失败，输出目录不存在"
    exit 1
fi

# 计算包大小
PACKAGE_SIZE=$(du -sh "$BUILD_DIR" | cut -f1)
echo "📦 小程序包大小: $PACKAGE_SIZE"

# 检查包大小限制 (微信小程序主包限制2MB)
PACKAGE_SIZE_KB=$(du -sk "$BUILD_DIR" | cut -f1)
if [ "$PACKAGE_SIZE_KB" -gt 2048 ]; then
    echo "⚠️  警告: 包大小超过2MB，可能需要分包处理"
fi

# 生成部署信息
DEPLOY_INFO="deploy-info.json"
cat > "$BUILD_DIR/$DEPLOY_INFO" << EOF
{
  "environment": "$ENVIRONMENT",
  "buildTime": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "version": "$(node -p "require('./package.json').version")",
  "commit": "$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')",
  "packageSize": "$PACKAGE_SIZE"
}
EOF

echo "✅ 部署完成!"
echo "📂 构建产物位置: $BUILD_DIR"
echo "📋 部署信息: $BUILD_DIR/$DEPLOY_INFO"
echo ""
echo "🎯 下一步操作:"
echo "1. 使用微信开发者工具打开 $BUILD_DIR 目录"
echo "2. 进行真机测试"
echo "3. 提交审核"

# 可选：自动打开微信开发者工具
if command -v "/Applications/wechatwebdevtools.app/Contents/MacOS/cli" &> /dev/null; then
    echo "🔧 检测到微信开发者工具，是否自动打开项目? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        "/Applications/wechatwebdevtools.app/Contents/MacOS/cli" -o "$BUILD_DIR"
    fi
fi