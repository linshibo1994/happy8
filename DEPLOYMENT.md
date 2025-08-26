# 快乐8智能预测系统 - 部署指南

## Streamlit Cloud 部署 (推荐)

### 前提条件
1. GitHub账号
2. Streamlit Cloud账号 (免费)

### 部署步骤

#### 1. 推送代码到GitHub
```bash
git add .
git commit -m "准备Streamlit Cloud部署"
git push origin main
```

#### 2. 部署到Streamlit Cloud
1. 访问 [share.streamlit.io](https://share.streamlit.io)
2. 使用GitHub账号登录
3. 点击 "New app"
4. 选择您的GitHub仓库: `happy8`
5. 主文件路径: `streamlit_app.py`
6. 点击 "Deploy!"

#### 3. 访问应用
部署完成后，您将获得一个类似这样的URL:
`https://your-username-happy8-streamlit-app-xxx.streamlit.app`

### 配置说明

- **入口文件**: `streamlit_app.py` - Streamlit Cloud的标准入口
- **配置文件**: `.streamlit/config.toml` - Streamlit应用配置
- **依赖文件**: `requirements.txt` - 已优化为云端部署版本

### 故障排除

1. **内存不足**: 已注释掉大型依赖包(torch, transformers等)
2. **导入错误**: 确保所有Python文件在正确的目录结构中
3. **数据文件**: 确保data目录中的CSV文件已提交到Git

## 替代部署方案

### Railway 部署
1. 访问 [railway.app](https://railway.app)
2. 连接GitHub仓库
3. 选择Python模板
4. 设置启动命令: `streamlit run streamlit_app.py --server.port $PORT`

### Render 部署
1. 访问 [render.com](https://render.com)
2. 创建新的Web Service
3. 连接GitHub仓库
4. 设置启动命令: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

## Cloudflare 集成

部署完成后，可以通过Cloudflare进行CDN加速:

1. 在Cloudflare中添加CNAME记录指向Streamlit应用
2. 启用Cloudflare代理
3. 配置缓存规则和安全设置

## 注意事项

- Streamlit Cloud免费版有资源限制
- 大型机器学习模型可能需要付费版本
- 定期检查应用状态和日志
