# 快乐8智能预测系统

当前仓库已清理前端相关代码，保留后端与核心预测引擎，作为后续重构基线。

## 当前保留内容

- `backend/`：主 FastAPI 后端服务
- `src/`：快乐8预测算法与分析引擎
- `data/`：本地数据文件
- `deployment/`：后端部署配置
- `docs/`：现有项目文档
- `happy8-miniprogram/backend/`：小程序后端服务
- `happy8-miniprogram/database/`：数据库初始化、迁移、种子数据
- `happy8-miniprogram/deployment/`：后端部署脚本与 Nginx 配置

## 已移除内容

- 根目录 `frontend/` SPA 前端
- `happy8-miniprogram/frontend/` 小程序前端
- `happy8-miniprogram/frontend-template/` 前端模板
- `happy8-miniprogram/happy8-new/` 前端试验工程
- 微信开发者工程壳与相关前端配置

## 快速开始

### 根目录后端

```bash
pip install -r requirements.txt
python main.py web
```

### 小程序后端

```bash
cd happy8-miniprogram/backend
pip install -r requirements.txt
python start.py
```

## 重构建议

建议后续按以下顺序推进：

1. 确认保留哪一套后端作为主线。
2. 收口重复文档与部署目录。
3. 统一数据模型、配置项和启动方式。
