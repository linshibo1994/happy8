# 项目结构说明

## 顶层结构

```text
.
├── backend/
├── engine/
├── infra/
├── docs/
├── specs/
├── data/
├── docker-compose.yml
├── main.py
└── requirements.txt
```

## 说明

### `backend/`

唯一主后端目录，包含：

- `app/`：业务代码
- `alembic/`：数据库迁移
- `tests/`：后端测试
- `start.py`：本地启动入口
- `Dockerfile`：容器构建文件

### `engine/`

算法引擎目录，负责：

- 快乐8历史数据分析
- 号码预测核心逻辑
- 数据抓取与性能优化脚本

### `infra/`

基础设施目录，负责：

- `database/`：数据库初始化文件
- `deployment/`：Nginx 与部署脚本

### `docs/`

面向开发的说明文档。

### `specs/`

产品需求、设计和任务拆解文档。
