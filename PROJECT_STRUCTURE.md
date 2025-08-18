# 快乐8预测系统项目结构

## 📁 目录结构

```
Happy8/
├── main.py                     # 主启动文件
├── README.md                   # 项目说明文档
├── requirements.txt            # Python依赖包
├── PROJECT_STRUCTURE.md        # 项目结构说明
│
├── src/                        # 源代码目录
│   ├── happy8_analyzer.py      # 核心分析器
│   └── happy8_app.py          # Web应用界面
│
├── scripts/                    # 脚本目录
│   ├── start.py               # 启动脚本
│   └── demo.py                # 演示脚本
│
├── deployment/                 # 部署相关文件
│   ├── deploy.py              # 部署脚本
│   ├── deploy_config.json     # 部署配置
│   ├── production_config.py   # 生产环境配置
│   ├── performance_optimizer.py # 性能优化器
│   ├── Dockerfile             # Docker配置
│   ├── docker-compose.yml     # Docker Compose配置
│   └── nginx.conf             # Nginx配置
│
├── data/                       # 数据目录
│   └── happy8_results.csv     # 历史开奖数据
│
└── docs/                       # 文档目录
    ├── 快乐8预测系统需求文档.md
    ├── 快乐8预测系统设计文档.md
    ├── 快乐8预测系统任务文档.md
    ├── 技术总结文档.md
    └── 项目实现总结.md
```

## 📋 文件说明

### 🚀 启动文件
- **main.py**: 统一的系统入口，支持多种启动方式
  - `python main.py web` - 启动Web界面
  - `python main.py cli` - 启动命令行界面
  - `python main.py demo` - 运行演示
  - `python main.py deploy` - 部署系统

### 💻 源代码 (src/)
- **happy8_analyzer.py**: 核心分析器，包含所有预测算法
  - 数据管理器 (DataManager)
  - 预测引擎 (PredictionEngine)
  - 对比引擎 (ComparisonEngine)
  - 6种预测算法实现
  
- **happy8_app.py**: Streamlit Web应用界面
  - 数据管理页面
  - 智能预测页面
  - 结果对比页面
  - 系统设置页面

### 🔧 脚本 (scripts/)
- **start.py**: 多功能启动脚本
  - 命令行界面
  - 依赖检查
  - 系统初始化
  
- **demo.py**: 系统演示脚本
  - 功能展示
  - 使用示例
  - 性能测试

### 🚀 部署 (deployment/)
- **deploy.py**: 一键部署脚本
  - 环境检查
  - 依赖安装
  - 数据初始化
  - 系统测试
  
- **deploy_config.json**: 部署配置文件
- **production_config.py**: 生产环境配置
- **performance_optimizer.py**: 性能优化器
- **Dockerfile**: Docker容器配置
- **docker-compose.yml**: Docker Compose配置
- **nginx.conf**: Nginx反向代理配置

### 📊 数据 (data/)
- **happy8_results.csv**: 历史开奖数据
  - 自动去重和排序
  - 支持增量更新
  - 数据验证和清洗

### 📚 文档 (docs/)
- **需求文档**: 系统功能需求和用户需求
- **设计文档**: 系统架构和技术设计
- **任务文档**: 开发任务和进度管理
- **技术总结**: 技术实现和经验总结
- **实现总结**: 项目实现过程和成果

## 🎯 使用方式

### 快速启动
```bash
# 启动Web界面（推荐）
python main.py web

# 启动命令行界面
python main.py cli

# 运行演示
python main.py demo
```

### 部署系统
```bash
# 一键部署
python main.py deploy

# 或直接运行部署脚本
python deployment/deploy.py
```

### 开发调试
```bash
# 直接运行Web应用
streamlit run src/happy8_app.py

# 直接运行演示
python scripts/demo.py
```

## 🔧 配置说明

### 环境变量
```bash
export HAPPY8_ENV=production    # 运行环境
export PYTHONPATH=$PWD/src     # Python路径
```

### 配置文件
- `deployment/deploy_config.json`: 部署配置
- `deployment/production_config.py`: 生产环境配置

## 📝 开发指南

### 添加新功能
1. 在 `src/happy8_analyzer.py` 中添加核心逻辑
2. 在 `src/happy8_app.py` 中添加Web界面
3. 更新 `scripts/demo.py` 添加演示
4. 更新文档说明

### 添加新算法
1. 在 `happy8_analyzer.py` 中创建新的预测器类
2. 在 `PredictionEngine` 中注册新算法
3. 在Web界面中添加选项
4. 编写测试和文档

### 部署新版本
1. 更新版本号
2. 运行测试
3. 更新文档
4. 执行部署脚本

## 🛠️ 维护说明

### 日志位置
- 系统日志: `logs/happy8_system.log`
- 错误日志: `logs/error.log`

### 数据备份
- 数据文件: `data/happy8_results.csv`
- 备份目录: `data/backups/`

### 性能监控
- 使用 `deployment/performance_optimizer.py`
- 查看性能报告和缓存状态

---

**📁 项目结构优化完成，所有文件已按功能分类整理！**
