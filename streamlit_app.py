#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8智能预测系统 - Streamlit Cloud 入口文件
Happy8 Prediction System - Streamlit Cloud Entry Point

这是Streamlit Cloud的标准入口文件，用于部署到云端。

作者: linshibo
版本: v1.4.0
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 直接执行src/happy8_app.py的内容
exec(open('src/happy8_app.py').read())
