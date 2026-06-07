# Happy8 小程序后端

当前目录已移除小程序前端代码，仅保留后端、数据库与部署相关内容，作为后续重构基础。

## 保留结构

```text
happy8-miniprogram/
├── backend/        # FastAPI 后端
├── database/       # 数据库初始化、迁移、种子数据
├── deployment/     # 部署脚本与 Nginx 配置
├── docs/           # 项目文档
└── docker-compose.yml
```

## 启动后端

```bash
cd backend
pip install -r requirements.txt
python start.py
```

## 说明

- 已删除 `frontend/`、`frontend-template/`、`happy8-new/` 以及微信开发者工程配置。
- `docker-compose.yml` 已去掉前端开发服务，仅保留后端相关服务。
