# Happy8 项目

当前仓库已经收口为单一后端主线，便于后续重新开发前端。

## 当前目录结构

```text
.
├── backend/          # 唯一主后端（FastAPI + SQLAlchemy + Alembic）
├── engine/           # 快乐8算法引擎与分析脚本
├── infra/            # 数据库初始化、Nginx、部署脚本
├── docs/             # 项目说明文档
├── specs/            # 需求、设计、任务拆解
├── data/             # 本地数据文件
├── docker-compose.yml
├── main.py
└── requirements.txt
```

## 模块职责

- `backend/`：未来前端统一对接的 API 服务。
- `engine/`：独立算法层，不直接承担 Web 路由职责。
- `infra/database/`：数据库初始化和种子相关文件。
- `infra/deployment/`：Nginx 和部署脚本。
- `specs/`：产品需求和重构任务依据。

## 本地开发

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动后端

```bash
python main.py api
```

或直接：

```bash
python backend/start.py
```

### 运行算法演示

```bash
python main.py demo
```

### Docker 启动

```bash
docker compose up -d
```

## 重构原则

后续前端重建时，只面向 `backend/` 的 API 契约开发，不再新增第二套后端目录。
