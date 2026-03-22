#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8智能预测系统 - Web界面
Happy8 Prediction System - Web Interface

基于Streamlit的Web操作界面，提供直观的预测和分析功能：
- 智能预测: 17种算法可选
- 历史数据分析: 数据统计和可视化
- 算法对比: 多算法性能对比
- 实时预测: 支持历史验证和未来预测

作者: linshibo
开发者: linshibo
版本: v1.4.0
创建时间: 2025-08-17
最后更新: 2025-08-19
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

# 导入核心分析器
try:
    from happy8_analyzer import Happy8Analyzer, PredictionResult, ComparisonResult, PairFrequencyResult, PairFrequencyItem
    from batch_predictor import BatchPredictor, BatchConfig, BatchResult, ExportEngine
except ImportError:
    st.error("无法导入核心分析器，请确保 happy8_analyzer.py 和 batch_predictor.py 文件存在")
    st.stop()

# 页面配置
st.set_page_config(
    page_title="快乐8智能预测系统",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .prediction-number {
        display: inline-block;
        width: 40px;
        height: 40px;
        line-height: 40px;
        text-align: center;
        margin: 2px;
        border-radius: 50%;
        font-weight: bold;
        color: white;
    }
    
    .predicted-number {
        background-color: #ff6b6b;
    }
    
    .actual-number {
        background-color: #4ecdc4;
    }
    
    .hit-number {
        background-color: #45b7d1;
        border: 3px solid #ffd93d;
    }
    
    .miss-number {
        background-color: #95a5a6;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

@st.cache_resource
def get_analyzer():
    """获取分析器实例（缓存）"""
    try:
        analyzer = Happy8Analyzer()
        return analyzer
    except Exception as e:
        st.error(f"初始化分析器失败: {e}")
        st.exception(e)
        return None

def clear_analyzer_cache():
    """清除分析器缓存"""
    if hasattr(st, 'cache_resource'):
        get_analyzer.clear()

def refresh_analyzer_data():
    """刷新分析器数据"""
    analyzer = get_analyzer()
    if analyzer:
        # 清除所有数据缓存
        analyzer.historical_data = None
        analyzer.data_manager._data_cache = None
        # 重新加载数据
        data = analyzer.load_data()
        return len(data)
    return 0

def create_number_display(numbers: List[int], number_type: str = "predicted", color_class: str = None) -> str:
    """创建号码显示HTML"""
    html = ""
    if color_class:
        css_class = f"{color_class} prediction-number"
    else:
        css_class = f"{number_type}-number prediction-number"

    for i, num in enumerate(numbers):
        if i > 0 and i % 10 == 0:
            html += "<br>"
        html += f'<span class="{css_class}">{num:02d}</span>'

    return html

def create_comparison_display(predicted: List[int], actual: List[int]) -> str:
    """创建对比显示HTML"""
    hit_numbers = set(predicted) & set(actual)
    
    html_predicted = "<div><strong>预测号码:</strong><br>"
    for i, num in enumerate(predicted):
        if i > 0 and i % 10 == 0:
            html_predicted += "<br>"
        
        css_class = "hit-number prediction-number" if num in hit_numbers else "miss-number prediction-number"
        html_predicted += f'<span class="{css_class}">{num:02d}</span>'
    html_predicted += "</div>"
    
    html_actual = "<div style='margin-top: 1rem;'><strong>开奖号码:</strong><br>"
    for i, num in enumerate(actual):
        if i > 0 and i % 10 == 0:
            html_actual += "<br>"
        html_actual += f'<span class="actual-number prediction-number">{num:02d}</span>'
    html_actual += "</div>"
    
    return html_predicted + html_actual

def show_homepage():
    """显示首页"""
    st.markdown('<h1 class="main-header">🎯 快乐8智能预测系统</h1>', unsafe_allow_html=True)
    
    # 系统介绍
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎲 游戏规则</h3>
            <ul>
                <li>号码范围: 1-80号</li>
                <li>开奖号码: 每期20个</li>
                <li>开奖频率: 每天一期</li>
                <li>期号格式: 如2025238</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 预测算法</h3>
            <ul>
                <li>频率分析预测</li>
                <li>冷热号分析预测</li>
                <li>马尔可夫链预测</li>
                <li>LSTM神经网络</li>
                <li>集成学习预测</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 系统特点</h3>
            <ul>
                <li>智能预测分析</li>
                <li>结果自动对比</li>
                <li>性能实时监控</li>
                <li>多种界面支持</li>
                <li>数据自动管理</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # 快速预测区域
    st.markdown("---")
    st.subheader("🚀 快速预测")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # 获取默认期号（基于当前日期的合理期号）
        def get_default_issue():
            try:
                analyzer = get_analyzer()
                if analyzer:
                    data = analyzer.load_data()
                    if not data.empty:
                        # 获取最新期号，并生成下一期号作为预测目标
                        latest_issue = data['issue'].iloc[0]
                        # 期号格式如2025238，提取年份和期数
                        year = int(str(latest_issue)[:4])
                        period = int(str(latest_issue)[4:])
                        # 生成下一期号
                        next_period = period + 1
                        return f"{year}{next_period:03d}"
            except:
                pass
            # 如果无法获取数据，使用基于当前日期的期号
            from datetime import datetime
            current_date = datetime.now()
            # 计算当年的天数作为期号
            day_of_year = current_date.timetuple().tm_yday
            return f"{current_date.year}{day_of_year:03d}"
        
        target_issue = st.text_input(
            "目标期号", 
            value=get_default_issue(),
            help="输入要预测的期号，格式如：2025238"
        )
    
    with col2:
        method = st.selectbox(
            "预测方法",
            ["frequency", "hot_cold", "missing", "markov", "markov_2nd", "markov_3rd",
             "adaptive_markov", "transformer", "gnn", "monte_carlo", "clustering",
             "advanced_ensemble", "bayesian", "super_predictor", "high_confidence",
             "lstm", "ensemble"],
            format_func=lambda x: {
                "frequency": "频率分析",
                "hot_cold": "冷热号分析",
                "missing": "遗漏分析",
                "markov": "1阶马尔可夫链",
                "markov_2nd": "2阶马尔可夫链",
                "markov_3rd": "3阶马尔可夫链",
                "adaptive_markov": "自适应马尔可夫链",
                "transformer": "Transformer模型",
                "gnn": "图神经网络",
                "monte_carlo": "蒙特卡洛模拟",
                "clustering": "聚类分析",
                "advanced_ensemble": "自适应集成学习",
                "bayesian": "贝叶斯推理",
                "super_predictor": "超级预测器",
                "high_confidence": "高置信度预测",
                "lstm": "LSTM神经网络",
                "ensemble": "集成学习"
            }[x],
            index=16
        )
    
    with col3:
        periods = st.number_input(
            "分析期数", 
            min_value=50, 
            max_value=2000, 
            value=300, 
            step=50,
            help="用于分析的历史期数（建议50-2000期）"
        )
    
    with col4:
        count = st.number_input(
            "生成号码数", 
            min_value=5, 
            max_value=80, 
            value=30, 
            step=5,
            help="生成的预测号码数量（建议20-50个）"
        )
    
    if st.button("🎯 开始预测", type="primary", use_container_width=True):
        with st.spinner("正在执行预测..."):
            try:
                analyzer = get_analyzer()
                if analyzer is None:
                    st.error("分析器初始化失败")
                    return
                
                # 执行预测
                result = analyzer.predict(
                    target_issue=target_issue,
                    periods=periods,
                    count=count,
                    method=method
                )
                
                # 显示结果
                st.success("预测完成!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("执行时间", f"{result.execution_time:.2f}秒")
                col2.metric("分析期数", f"{result.analysis_periods}期")
                col3.metric("预测方法", result.method)
                
                # 显示预测号码
                st.subheader("预测号码")
                numbers_html = create_number_display(result.predicted_numbers[:20])
                st.markdown(numbers_html, unsafe_allow_html=True)
                
                # 保存到历史记录
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'target_issue': target_issue,
                    'method': method,
                    'result': result
                })
                
            except Exception as e:
                st.error(f"预测失败: {str(e)}")

def show_data_management():
    """显示数据管理页面"""
    st.header("📊 数据管理")
    
    analyzer = get_analyzer()
    if analyzer is None:
        st.error("分析器未初始化")
        return
    
    # 数据状态
    col_title, col_refresh = st.columns([3, 1])
    with col_title:
        st.subheader("数据状态")
    with col_refresh:
        if st.button("🔄 刷新数据", help="重新加载数据，清除缓存"):
            with st.spinner("正在刷新数据..."):
                try:
                    # 清除缓存并刷新数据
                    clear_analyzer_cache()
                    data_count = refresh_analyzer_data()
                    st.success(f"✅ 数据刷新完成！当前数据量: {data_count} 期")
                    st.rerun()
                except Exception as e:
                    st.error(f"数据刷新失败: {e}")

    try:
        data = analyzer.load_data()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总期数", len(data))

        if len(data) > 0:
            latest_issue = data.iloc[0]['issue']  # 数据已按最新期号排序
            col2.metric("最新期号", latest_issue)

            earliest_issue = data.iloc[-1]['issue']
            col3.metric("最早期号", earliest_issue)

            col4.metric("数据完整性", "✓ 正常")
        
        # 数据预览
        st.subheader("数据预览")
        if len(data) > 0:
            st.dataframe(data.head(10), use_container_width=True)  # 显示最新的10期数据
        else:
            st.info("暂无数据，请先爬取数据")
        
    except Exception as e:
        st.error(f"加载数据失败: {e}")
    
    # 数据操作
    st.subheader("数据操作")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔄 增量更新数据", use_container_width=True):
            limit = st.number_input("检查期数", min_value=10, max_value=200, value=50, key="update_limit",
                                   help="检查最近N期是否有新数据")

            with st.spinner(f"正在检查最近 {limit} 期数据..."):
                try:
                    new_count = analyzer.data_manager.crawl_latest_data(limit)
                    if new_count > 0:
                        st.success(f"✅ 发现并更新了 {new_count} 期新数据!")
                    else:
                        st.info("📋 当前数据已是最新，无需更新")
                    st.rerun()
                except Exception as e:
                    st.error(f"数据更新失败: {e}")

    with col2:
        if st.button("🕷️ 初始数据爬取", use_container_width=True):
            count = st.number_input("爬取期数", min_value=100, max_value=2000, value=1000, key="crawl_count",
                                   help="首次使用时爬取历史数据")

            with st.spinner(f"正在爬取 {count} 期历史数据..."):
                try:
                    analyzer.data_manager.crawl_initial_data(count)
                    st.success(f"✅ 初始数据爬取完成！获取 {count} 期数据")
                    st.rerun()
                except Exception as e:
                    st.error(f"数据爬取失败: {e}")

    with col3:
        if st.button("✅ 验证数据", use_container_width=True):
            with st.spinner("正在验证数据..."):
                try:
                    data = analyzer.load_data()
                    validation_result = analyzer.data_manager.validator.validate_happy8_data(data)
                    
                    st.write("验证结果:")
                    st.json(validation_result)
                    
                    if not validation_result['errors']:
                        st.success("数据验证通过!")
                    else:
                        st.warning("数据存在问题，请检查")
                        
                except Exception as e:
                    st.error(f"数据验证失败: {e}")
    
    with col4:
        if st.button("🗑️ 清空数据", use_container_width=True):
            if st.checkbox("确认清空所有数据", key="confirm_clear"):
                try:
                    if analyzer.data_manager.data_file.exists():
                        analyzer.data_manager.data_file.unlink()
                        st.success("数据已清空")
                        st.rerun()
                    else:
                        st.info("没有数据需要清空")
                except Exception as e:
                    st.error(f"清空数据失败: {e}")

    # 添加数据源状态检查
    st.markdown("---")
    st.subheader("🔍 数据源状态")

    col5, col6 = st.columns(2)

    with col5:
        if st.button("检查数据源状态", use_container_width=True):
            with st.spinner("检查数据源状态..."):
                try:
                    st.info("🔗 **数据源状态检查**")

                    # 测试500彩票网XML接口
                    try:
                        from happy8_analyzer import Happy8Crawler
                        crawler = Happy8Crawler()
                        test_results = crawler._crawl_from_500wan(1)
                        if test_results:
                            st.success("✅ 500彩票网XML接口: 正常")
                            st.info(f"最新期号: {test_results[0].issue}")
                        else:
                            st.error("❌ 500彩票网XML接口: 异常")
                    except Exception as e:
                        st.error(f"❌ 500彩票网XML接口: {e}")

                except Exception as e:
                    st.error(f"状态检查失败: {e}")

    with col6:
        st.info("""
        **数据源说明:**
        - 🎯 主要数据源: 500彩票网XML接口
        - 📊 数据格式: 真实官方开奖数据
        - 🔄 更新方式: 增量更新，只获取新数据
        - ✅ 数据验证: 自动验证数据完整性
        """)

def show_prediction_interface():
    """显示预测界面"""
    st.header("🎯 智能预测")

    analyzer = get_analyzer()
    if analyzer is None:
        st.error("分析器未初始化")
        return

    # 显示当前数据状态
    try:
        data = analyzer.load_data()
        st.info(f"📊 当前数据量: {len(data)} 期 | 最新期号: {data.iloc[0]['issue'] if len(data) > 0 else '无'} | 💡 如数据显示不正确，请到数据管理页面点击'刷新数据'按钮")
    except:
        st.warning("⚠️ 数据加载异常，请到数据管理页面检查数据状态")
    
    # 预测参数配置
    st.subheader("预测参数配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 获取默认期号（基于最新数据的下一期）
        def get_smart_default_issue():
            try:
                if len(data) > 0:
                    # 获取最新期号，并生成下一期号作为预测目标
                    latest_issue = data.iloc[0]['issue']
                    # 期号格式如2025238，提取年份和期数
                    year = int(str(latest_issue)[:4])
                    period = int(str(latest_issue)[4:])
                    # 生成下一期号
                    next_period = period + 1
                    return f"{year}{next_period:03d}"
            except:
                pass
            # 如果无法获取数据，使用基于当前日期的期号
            from datetime import datetime
            current_date = datetime.now()
            day_of_year = current_date.timetuple().tm_yday
            return f"{current_date.year}{day_of_year:03d}"
        
        target_issue = st.text_input(
            "目标期号", 
            value=get_smart_default_issue(),
            help="输入要预测的期号，格式如：2025238"
        )
        
        analysis_periods = st.number_input(
            "分析期数",
            min_value=50,
            max_value=2000,
            value=300,
            step=50,
            help="用于分析的历史期数（建议50-2000期）"
        )
    
    with col2:
        predict_count = st.number_input(
            "生成号码数",
            min_value=5,
            max_value=80,
            value=30,
            step=5,
            help="生成的预测号码数量（建议20-50个）"
        )
        
        method = st.selectbox(
            "预测方法",
            ["frequency", "hot_cold", "missing", "markov", "markov_2nd", "markov_3rd",
             "adaptive_markov", "transformer", "gnn", "monte_carlo", "clustering",
             "advanced_ensemble", "bayesian", "super_predictor", "high_confidence",
             "lstm", "ensemble"],
            format_func=lambda x: {
                "frequency": "频率分析",
                "hot_cold": "冷热号分析",
                "missing": "遗漏分析",
                "markov": "1阶马尔可夫链",
                "markov_2nd": "2阶马尔可夫链",
                "markov_3rd": "3阶马尔可夫链",
                "adaptive_markov": "自适应马尔可夫链",
                "transformer": "Transformer模型",
                "gnn": "图神经网络",
                "monte_carlo": "蒙特卡洛模拟",
                "clustering": "聚类分析",
                "advanced_ensemble": "自适应集成学习",
                "bayesian": "贝叶斯推理",
                "super_predictor": "超级预测器",
                "high_confidence": "高置信度预测",
                "lstm": "LSTM神经网络",
                "ensemble": "集成学习"
            }[x],
            index=16
        )
    
    # 高级选项
    with st.expander("高级选项"):
        show_explain = st.checkbox("显示详细分析过程", value=False)
        smart_mode = st.checkbox("智能模式（自动判断历史验证/未来预测）", value=True,
                                help="开启后系统会自动判断目标期号是否为历史期号，并选择相应的预测模式")

    # 预测按钮
    if st.button("🚀 开始智能预测", type="primary", use_container_width=True):
        with st.spinner("正在执行智能预测分析..."):
            try:
                analyzer = get_analyzer()
                if analyzer is None:
                    st.error("分析器初始化失败")
                    return

                if smart_mode:
                    # 使用智能预测模式
                    try:
                        smart_result = analyzer.predict_with_smart_mode(
                            target_issue=target_issue,
                            periods=analysis_periods,
                            count=predict_count,
                            method=method
                        )

                        # 显示智能预测结果
                        display_smart_prediction_result(smart_result)
                    except Exception as e:
                        st.error(f"智能预测失败: {str(e)}")
                        if show_explain:
                            st.exception(e)

                    # 保存到历史记录
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now(),
                        'target_issue': target_issue,
                        'method': method,
                        'result': smart_result['prediction_result'],
                        'comparison': smart_result.get('comparison_result'),
                        'mode': smart_result['mode'],
                        'mode_description': smart_result['mode_description']
                    })

                else:
                    # 传统预测模式
                    try:
                        prediction_result = analyzer.predict(
                            target_issue=target_issue,
                            periods=analysis_periods,
                            count=predict_count,
                            method=method
                        )

                        # 显示结果
                        display_prediction_result(prediction_result)
                    except Exception as e:
                        st.error(f"传统预测失败: {str(e)}")
                        if show_explain:
                            st.exception(e)

                    # 保存到历史记录
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now(),
                        'target_issue': target_issue,
                        'method': method,
                        'result': prediction_result,
                        'comparison': None,
                        'mode': 'traditional',
                        'mode_description': '传统预测模式'
                    })

            except Exception as e:
                st.error(f"预测失败: {str(e)}")
                if show_explain:
                    st.exception(e)

def display_smart_prediction_result(smart_result: dict):
    """显示智能预测结果"""
    prediction_result = smart_result['prediction_result']
    comparison_result = smart_result.get('comparison_result')
    mode = smart_result['mode']
    mode_description = smart_result['mode_description']

    # 显示模式信息
    if mode == 'historical_validation':
        st.success("🎯 智能预测完成 - 历史验证模式")
        st.info(f"📊 {mode_description}")
    else:
        st.success("🔮 智能预测完成 - 未来预测模式")
        st.info(f"🔮 {mode_description}")

    # 预测参数回显
    st.subheader("📋 预测参数")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("目标期号", prediction_result.target_issue)
    col2.metric("分析期数", prediction_result.analysis_periods)
    col3.metric("预测方法", prediction_result.method)
    col4.metric("执行耗时", f"{prediction_result.execution_time:.2f}秒")
    col5.metric("预测模式", "历史验证" if mode == 'historical_validation' else "未来预测")

    # 预测号码展示
    st.subheader("🎯 预测号码")
    numbers_html = create_number_display(prediction_result.predicted_numbers)
    st.markdown(numbers_html, unsafe_allow_html=True)

    # 如果是历史验证模式且有对比结果，显示对比分析
    if comparison_result:
        st.subheader("📊 验证结果")

        # 命中率展示
        hit_rate = comparison_result.hit_rate
        col1, col2, col3 = st.columns(3)
        col1.metric("命中率", f"{hit_rate:.1%}",
                   delta=f"{hit_rate - 0.25:.1%}" if hit_rate > 0.25 else None)
        col2.metric("命中数量", f"{len(comparison_result.hit_numbers)}/{len(prediction_result.predicted_numbers)}")
        col3.metric("验证状态", "✅ 已验证" if hit_rate > 0 else "❌ 未命中")

        # 命中号码和未命中号码
        if comparison_result.hit_numbers:
            st.write("**✅ 命中号码:**")
            hit_html = create_number_display(comparison_result.hit_numbers, color_class="hit-number")
            st.markdown(hit_html, unsafe_allow_html=True)

        if comparison_result.miss_numbers:
            st.write("**❌ 未命中号码:**")
            miss_html = create_number_display(comparison_result.miss_numbers, color_class="miss-number")
            st.markdown(miss_html, unsafe_allow_html=True)

        # 获取实际开奖号码
        try:
            analyzer = get_analyzer()
            actual_result = analyzer.data_manager.get_issue_result(prediction_result.target_issue)
            if actual_result:
                st.write("**🎲 实际开奖号码:**")
                actual_html = create_number_display(actual_result.numbers, color_class="actual-number")
                st.markdown(actual_html, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"无法获取实际开奖号码: {e}")

    # 置信度图表
    if prediction_result.confidence_scores and len(prediction_result.confidence_scores) == len(prediction_result.predicted_numbers):
        st.subheader("📊 置信度分析")

        df_confidence = pd.DataFrame({
            '号码': prediction_result.predicted_numbers,
            '置信度': prediction_result.confidence_scores
        })

        fig = px.bar(df_confidence, x='号码', y='置信度',
                    title="预测号码置信度分布",
                    color='置信度',
                    color_continuous_scale='viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def display_prediction_result(result: PredictionResult):
    """显示传统预测结果"""
    st.success("预测完成!")

    # 预测参数回显
    st.subheader("📋 预测参数")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("目标期号", result.target_issue)
    col2.metric("分析期数", result.analysis_periods)
    col3.metric("预测方法", result.method)
    col4.metric("执行耗时", f"{result.execution_time:.2f}秒")
    
    # 预测号码展示
    st.subheader("🎯 预测号码")
    numbers_html = create_number_display(result.predicted_numbers)
    st.markdown(numbers_html, unsafe_allow_html=True)
    
    # 置信度图表
    if result.confidence_scores and len(result.confidence_scores) == len(result.predicted_numbers):
        st.subheader("📊 置信度分析")
        
        df_confidence = pd.DataFrame({
            '号码': result.predicted_numbers,
            '置信度': result.confidence_scores
        })
        
        fig = px.bar(df_confidence, x='号码', y='置信度', 
                    title='预测号码置信度分布',
                    color='置信度',
                    color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)

def display_prediction_and_comparison(prediction_result: PredictionResult, comparison_result: ComparisonResult):
    """显示预测和对比结果"""
    st.success("预测和对比完成!")
    
    # 预测参数回显
    st.subheader("📋 预测参数")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("目标期号", prediction_result.target_issue)
    col2.metric("分析期数", prediction_result.analysis_periods)
    col3.metric("预测方法", prediction_result.method)
    col4.metric("执行耗时", f"{prediction_result.execution_time:.2f}秒")
    
    # 号码对比展示
    st.subheader("🎯 号码对比")
    comparison_html = create_comparison_display(
        comparison_result.predicted_numbers, 
        comparison_result.actual_numbers
    )
    st.markdown(comparison_html, unsafe_allow_html=True)
    
    # 命中统计
    st.subheader("📊 命中统计")
    col1, col2, col3 = st.columns(3)
    col1.metric("命中数量", f"{comparison_result.hit_count}/{comparison_result.total_predicted}")
    col2.metric("命中率", f"{comparison_result.hit_rate:.2%}")
    col3.metric("命中号码", ", ".join(map(str, sorted(comparison_result.hit_numbers))))
    
    # 详细分析
    st.subheader("📈 详细分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 大小号分析
        distribution = comparison_result.hit_distribution
        small_hits = distribution.get('small_numbers', 0)
        big_hits = distribution.get('big_numbers', 0)
        
        fig_size = go.Figure(data=[
            go.Bar(name='小号(1-40)', x=['命中'], y=[small_hits]),
            go.Bar(name='大号(41-80)', x=['命中'], y=[big_hits])
        ])
        fig_size.update_layout(title='大小号命中分布', barmode='group')
        st.plotly_chart(fig_size, use_container_width=True)
    
    with col2:
        # 奇偶分析
        odd_hits = distribution.get('odd_numbers', 0)
        even_hits = distribution.get('even_numbers', 0)
        
        fig_parity = go.Figure(data=[
            go.Pie(labels=['奇数', '偶数'], values=[odd_hits, even_hits])
        ])
        fig_parity.update_layout(title='奇偶号命中分布')
        st.plotly_chart(fig_parity, use_container_width=True)
    
    # 区域分布
    zone_data = []
    for i in range(1, 9):
        zone_hits = distribution.get(f'zone_{i}', 0)
        zone_data.append({'区域': f'区域{i}', '命中数': zone_hits})
    
    if zone_data:
        df_zones = pd.DataFrame(zone_data)
        fig_zones = px.bar(df_zones, x='区域', y='命中数', 
                          title='各区域命中分布',
                          color='命中数',
                          color_continuous_scale='blues')
        st.plotly_chart(fig_zones, use_container_width=True)

def show_history_records():
    """显示历史记录"""
    st.header("📋 历史记录")
    
    if not st.session_state.prediction_history:
        st.info("暂无预测历史记录")
        return
    
    # 历史记录列表
    st.subheader("预测历史")
    
    history_data = []
    for record in reversed(st.session_state.prediction_history):
        row = {
            '时间': record['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            '期号': record['target_issue'],
            '方法': record['method'],
            '执行时间': f"{record['result'].execution_time:.2f}秒"
        }
        
        if record.get('comparison'):
            row['命中率'] = f"{record['comparison'].hit_rate:.2%}"
        else:
            row['命中率'] = "未对比"
        
        history_data.append(row)
    
    df_history = pd.DataFrame(history_data)
    st.dataframe(df_history, use_container_width=True)
    
    # 统计分析
    if len(st.session_state.prediction_history) > 1:
        st.subheader("统计分析")
        
        # 方法使用频率
        method_counts = {}
        hit_rates_by_method = {}
        
        for record in st.session_state.prediction_history:
            method = record['method']
            method_counts[method] = method_counts.get(method, 0) + 1
            
            if record.get('comparison'):
                if method not in hit_rates_by_method:
                    hit_rates_by_method[method] = []
                hit_rates_by_method[method].append(record['comparison'].hit_rate)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 方法使用频率
            fig_methods = px.pie(
                values=list(method_counts.values()),
                names=list(method_counts.keys()),
                title='预测方法使用频率'
            )
            st.plotly_chart(fig_methods, use_container_width=True)
        
        with col2:
            # 平均命中率
            if hit_rates_by_method:
                avg_hit_rates = {method: np.mean(rates) for method, rates in hit_rates_by_method.items()}
                
                fig_hit_rates = px.bar(
                    x=list(avg_hit_rates.keys()),
                    y=list(avg_hit_rates.values()),
                    title='各方法平均命中率',
                    labels={'x': '预测方法', 'y': '平均命中率'}
                )
                st.plotly_chart(fig_hit_rates, use_container_width=True)

def show_pair_frequency_analysis():
    """显示数字对频率分析页面"""
    st.header("🔢 数字对频率分析")
    
    # 页面说明
    st.markdown("""
    **功能说明：** 分析指定期号范围内两位数组合的出现频率，帮助识别历史数据中的数字组合模式。
    
    **使用方法：**
    1. 输入目标期号（如2025238）
    2. 选择统计期数（系统将从目标期号向前统计指定期数）
    3. 点击"开始分析"查看结果
    """)
    
    st.markdown("---")
    
    # 获取分析器
    analyzer = get_analyzer()
    if not analyzer:
        st.error("系统未初始化，请先在数据管理页面加载数据")
        return
    
    # 输入参数区域
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        target_issue = st.text_input(
            "目标期号",
            value="2025238",
            help="输入7位期号，如2025238"
        )
    
    with col2:
        period_count = st.number_input(
            "统计期数",
            min_value=1,
            max_value=100,
            value=20,
            help="从目标期号向前统计的期数"
        )
    
    with col3:
        use_cache = st.checkbox("使用缓存", value=True, help="启用缓存可提高重复查询的速度")
    
    # 高级选项
    with st.expander("🔧 高级选项"):
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            show_top_n = st.number_input("显示前N个结果", min_value=5, max_value=5000, value=20)
            export_format = st.selectbox("导出格式", ["CSV", "Excel", "JSON"])
        
        with col_adv2:
            min_frequency = st.slider("最小频率过滤(%)", 0.0, 100.0, 0.0, 0.1)
            show_charts = st.checkbox("显示图表", value=True)
    
    # 分析按钮
    if st.button("🚀 开始分析", type="primary"):
        if not target_issue or len(target_issue) != 7 or not target_issue.isdigit():
            st.error("请输入有效的7位期号，如2025238")
            return
        
        try:
            # 显示进度
            with st.spinner("正在分析数字对频率..."):
                # 执行分析
                result = analyzer.analyze_pair_frequency(
                    target_issue=target_issue,
                    period_count=period_count,
                    use_cache=use_cache
                )
            
            # 显示分析结果
            _display_pair_frequency_results(result, show_top_n, min_frequency, show_charts, export_format)
            
        except Exception as e:
            st.error(f"分析失败: {str(e)}")
    
    # 批量分析功能
    st.markdown("---")
    st.subheader("📊 批量分析")
    
    col_batch1, col_batch2 = st.columns(2)
    
    with col_batch1:
        st.markdown("**跨期数对比分析**")
        if st.button("执行跨期数分析"):
            if target_issue and len(target_issue) == 7 and target_issue.isdigit():
                _show_cross_period_analysis(analyzer, target_issue)
            else:
                st.error("请先输入有效的目标期号")
    
    with col_batch2:
        st.markdown("**一致高频数字对**")
        if st.button("查找一致高频对"):
            if target_issue and len(target_issue) == 7 and target_issue.isdigit():
                _show_consistent_pairs_analysis(analyzer, target_issue)
            else:
                st.error("请先输入有效的目标期号")
    
    # 缓存管理
    st.markdown("---")
    st.subheader("💾 缓存管理")
    
    col_cache1, col_cache2, col_cache3 = st.columns(3)
    
    with col_cache1:
        if st.button("查看缓存信息"):
            cache_info = analyzer.get_pair_frequency_cache_info()
            st.json(cache_info)
    
    with col_cache2:
        if st.button("清空缓存"):
            analyzer.clear_pair_frequency_cache()
            st.success("缓存已清空")
    
    with col_cache3:
        new_cache_size = st.number_input("调整缓存大小", min_value=10, max_value=500, value=100)
        if st.button("应用缓存设置"):
            analyzer.pair_frequency_analyzer.resize_cache(new_cache_size)
            st.success(f"缓存大小已调整为 {new_cache_size}")


def _display_pair_frequency_results(result: PairFrequencyResult, show_top_n: int, min_frequency: float, show_charts: bool, export_format: str):
    """显示数字对频率分析结果"""
    
    # 基本信息
    st.success("✅ 分析完成！")
    
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    
    with col_info1:
        st.metric("目标期号", result.target_issue)
    
    with col_info2:
        st.metric("实际期数", f"{result.actual_periods}期")
    
    with col_info3:
        st.metric("数字对总数", result.total_pairs)
    
    with col_info4:
        st.metric("执行时间", f"{result.execution_time:.3f}秒")
    
    # 统计摘要
    summary = result.get_summary()
    
    st.markdown("### 📈 统计摘要")
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    
    with col_sum1:
        st.metric("不同数字对", summary['total_unique_pairs'])
    
    with col_sum2:
        st.metric("最高频率", f"{summary['max_frequency']}次")
    
    with col_sum3:
        st.metric("最低频率", f"{summary['min_frequency']}次")
    
    with col_sum4:
        st.metric("平均频率", f"{summary['avg_frequency']:.1f}次")
    
    # 过滤结果
    filtered_items = [
        item for item in result.frequency_items 
        if item.percentage >= min_frequency
    ][:show_top_n]
    
    if not filtered_items:
        st.warning("没有符合条件的数字对")
        return
    
    # 结果表格
    st.markdown("### 📋 分析结果")
    
    # 创建DataFrame
    df_results = pd.DataFrame([
        {
            "排名": i + 1,
            "数字对": f"({item.pair[0]:02d}, {item.pair[1]:02d})",
            "数字1": item.pair[0],
            "数字2": item.pair[1],
            "出现次数": item.count,
            "出现频率(%)": round(item.percentage, 1)
        }
        for i, item in enumerate(filtered_items)
    ])
    
    # 显示表格
    st.dataframe(
        df_results,
        use_container_width=True,
        hide_index=True
    )
    
    # 图表展示
    if show_charts and len(filtered_items) > 0:
        st.markdown("### 📊 可视化图表")
        
        # 频率分布柱状图
        fig_bar = px.bar(
            df_results.head(15),  # 只显示前15个
            x="数字对",
            y="出现次数",
            title="数字对出现频率分布（前15名）",
            color="出现频率(%)",
            color_continuous_scale="viridis"
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 频率分布饼图（前10名）
        if len(filtered_items) >= 5:
            fig_pie = px.pie(
                df_results.head(10),
                values="出现次数",
                names="数字对",
                title="数字对频率占比（前10名）"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # 导出功能
    st.markdown("### 💾 导出结果")
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        # 生成导出数据
        export_df = result.to_dataframe()
        
        if export_format == "CSV":
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="📥 下载CSV文件",
                data=csv_data,
                file_name=f"pair_frequency_{result.target_issue}_{result.actual_periods}periods.csv",
                mime="text/csv"
            )
        elif export_format == "JSON":
            json_data = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 下载JSON文件",
                data=json_data,
                file_name=f"pair_frequency_{result.target_issue}_{result.actual_periods}periods.json",
                mime="application/json"
            )
    
    with col_export2:
        # 生成分析报告
        report = result.generate_report()
        st.download_button(
            label="📄 下载分析报告",
            data=report,
            file_name=f"pair_frequency_report_{result.target_issue}_{result.actual_periods}periods.txt",
            mime="text/plain"
        )


def _show_cross_period_analysis(analyzer: Happy8Analyzer, target_issue: str):
    """显示跨期数对比分析"""
    
    with st.spinner("正在执行跨期数分析..."):
        period_counts = [10, 20, 30, 50]
        top_pairs_data = analyzer.get_top_pairs_across_periods(target_issue, period_counts, 10)
    
    st.markdown("#### 🔄 跨期数对比分析结果")
    
    # 创建对比表格
    comparison_data = []
    
    for period in period_counts:
        pairs_data = top_pairs_data.get(period, [])
        for i, item in enumerate(pairs_data[:5]):  # 只显示前5个
            comparison_data.append({
                "期数": f"{period}期",
                "排名": i + 1,
                "数字对": f"({item.pair[0]:02d}, {item.pair[1]:02d})",
                "出现次数": item.count,
                "频率(%)": round(item.percentage, 1)
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        
        # 透视表显示
        pivot_df = df_comparison.pivot_table(
            index=["排名", "数字对"],
            columns="期数",
            values="频率(%)",
            fill_value=0
        )
        
        st.dataframe(pivot_df, use_container_width=True)
        
        # 趋势图
        fig_trend = px.line(
            df_comparison,
            x="期数",
            y="频率(%)",
            color="数字对",
            title="数字对频率趋势对比",
            markers=True
        )
        st.plotly_chart(fig_trend, use_container_width=True)


def _show_consistent_pairs_analysis(analyzer: Happy8Analyzer, target_issue: str):
    """显示一致高频数字对分析"""
    
    with st.spinner("正在查找一致高频数字对..."):
        period_counts = [10, 20, 30, 50]
        consistent_pairs = analyzer.find_consistent_pairs(target_issue, period_counts, 30.0)
    
    st.markdown("#### 🎯 一致高频数字对分析结果")
    
    if consistent_pairs:
        st.success(f"找到 {len(consistent_pairs)} 个一致高频数字对")
        
        # 显示一致高频数字对
        consistent_df = pd.DataFrame([
            {
                "数字对": f"({pair[0]:02d}, {pair[1]:02d})",
                "数字1": pair[0],
                "数字2": pair[1]
            }
            for pair in consistent_pairs
        ])
        
        st.dataframe(consistent_df, use_container_width=True, hide_index=True)
        
        # 详细分析每个一致数字对
        if st.checkbox("显示详细分析"):
            for pair in consistent_pairs[:5]:  # 只显示前5个
                with st.expander(f"数字对 ({pair[0]:02d}, {pair[1]:02d}) 详细分析"):
                    detail_data = []
                    for period in period_counts:
                        result = analyzer.analyze_pair_frequency(target_issue, period)
                        pair_item = result.find_pair(pair[0], pair[1])
                        if pair_item:
                            detail_data.append({
                                "期数": f"{period}期",
                                "出现次数": pair_item.count,
                                "频率(%)": round(pair_item.percentage, 1)
                            })
                    
                    if detail_data:
                        detail_df = pd.DataFrame(detail_data)
                        st.dataframe(detail_df, use_container_width=True, hide_index=True)
    else:
        st.warning("未找到在所有期数下都保持高频的数字对")


def show_batch_prediction_interface():
    """显示批量预测对比界面"""
    st.header("🔄 批量预测对比")
    st.markdown("对同一期号进行多次预测，统计分析算法的稳定性和准确性")
    
    # 初始化session_state
    if 'batch_session' not in st.session_state:
        st.session_state.batch_session = None
    if 'batch_result' not in st.session_state:
        st.session_state.batch_result = None
    
    analyzer = get_analyzer()
    if analyzer is None:
        st.error("分析器未初始化")
        return

    # 显示当前数据状态
    try:
        data = analyzer.load_data()
        st.info(f"📊 当前数据量: {len(data)} 期 | 最新期号: {data.iloc[0]['issue'] if len(data) > 0 else '无'}")
    except:
        st.warning("⚠️ 数据加载异常，请到数据管理页面检查数据状态")
        return

    # 参数配置区域
    st.subheader("📋 批量预测配置")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 获取智能默认期号
            def get_smart_default_issue():
                try:
                    if len(data) > 0:
                        latest_issue = data.iloc[0]['issue']
                        return str(latest_issue)
                except:
                    pass
                return "2025091"
            
            target_issue = st.text_input(
                "目标期号", 
                value=get_smart_default_issue(),
                help="输入要预测的期号，格式如：2025091"
            )
            
            analysis_periods = st.number_input(
                "分析期数",
                min_value=10,
                max_value=500, 
                value=100,
                step=10,
                help="用于分析的历史期数（建议50-200期）"
            )
            
        with col2:
            # 预测方法选择
            method_options = {
                "frequency": "📊 频率分析",
                "hot_cold": "🌡️ 冷热号分析", 
                "missing": "📉 遗漏分析",
                "markov_chain": "🔗 马尔可夫链",
                "markov_2nd": "🔗 2阶马尔可夫链",
                "markov_3rd": "🔗 3阶马尔可夫链",
                "adaptive_markov": "🤖 自适应马尔可夫链",
                "transformer": "🤖 Transformer模型",
                "gnn": "🕸️ 图神经网络",
                "monte_carlo": "🎲 蒙特卡洛模拟",
                "clustering": "🎯 聚类分析",
                "ensemble": "🚀 集成学习",
                "bayesian": "📈 贝叶斯推理",
                "super_predictor": "⭐ 超级预测器",
                "high_confidence": "💎 高置信度预测"
            }
            
            selected_method = st.selectbox(
                "预测方法",
                options=list(method_options.keys()),
                format_func=lambda x: method_options[x],
                index=3,  # 默认选择马尔可夫链
                help="选择预测算法"
            )
            
            number_count = st.number_input(
                "生成号码数量",
                min_value=1,
                max_value=30,
                value=20,
                step=1,
                help="每轮预测生成的号码数量"
            )
            
        with col3:
            comparison_times = st.number_input(
                "对比次数",
                min_value=1,
                max_value=100,
                value=20,
                step=1,
                help="重复预测的次数"
            )
            
            max_parallel = st.number_input(
                "最大并发数",
                min_value=1,
                max_value=8,
                value=4,
                step=1,
                help="并行执行的最大线程数"
            )
            
            timeout_seconds = st.number_input(
                "单次预测超时(秒)",
                min_value=10,
                max_value=120,
                value=30,
                step=5,
                help="单次预测的超时时间"
            )

    # 执行控制区域
    st.subheader("🎮 执行控制")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        start_button = st.button(
            "🚀 开始批量预测", 
            type="primary",
            disabled=(st.session_state.batch_session is not None and 
                     st.session_state.batch_session.get('status') == 'running'),
            help="点击开始执行批量预测"
        )
    
    with col2:
        cancel_button = st.button(
            "❌ 取消预测",
            disabled=(st.session_state.batch_session is None or 
                     st.session_state.batch_session.get('status') != 'running'),
            help="取消当前的批量预测"
        )
    
    with col3:
        clear_button = st.button(
            "🧹 清除结果",
            help="清除当前的预测结果"
        )
    
    # 进度显示区域
    progress_container = st.container()
    
    # 执行逻辑
    if start_button:
        try:
            # 参数验证
            if not target_issue or len(target_issue) != 7:
                st.error("期号必须是7位数字")
                return
                
            # 创建BatchConfig
            config = BatchConfig(
                target_issue=target_issue,
                analysis_periods=analysis_periods,
                prediction_method=selected_method,
                number_count=number_count,
                comparison_times=comparison_times,
                max_parallel=max_parallel,
                timeout_seconds=timeout_seconds
            )
            
            # 初始化批量预测器
            batch_predictor = BatchPredictor(analyzer)
            
            # 创建进度显示
            with progress_container:
                st.info("🔄 正在准备批量预测...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                current_round_text = st.empty()
                
                # 执行批量预测
                def progress_callback(session):
                    progress = session.progress / 100
                    progress_bar.progress(progress)
                    status_text.text(f"状态: {session.status} | 进度: {session.progress:.1f}%")
                    current_round_text.text(f"当前轮次: {session.current_round}/{config.comparison_times}")
                
                # 保存会话信息到session_state
                st.session_state.batch_session = {'status': 'running'}
                
                try:
                    # 执行批量预测
                    batch_result = batch_predictor.execute_batch_prediction(
                        config, 
                        progress_callback
                    )
                    
                    # 保存结果
                    st.session_state.batch_result = batch_result
                    st.session_state.batch_session = {'status': 'completed'}
                    
                    st.success("✅ 批量预测完成！")
                    
                except Exception as e:
                    st.session_state.batch_session = {'status': 'failed'}
                    st.error(f"❌ 批量预测失败: {str(e)}")
                    
        except Exception as e:
            st.error(f"❌ 配置错误: {str(e)}")
    
    if cancel_button:
        st.session_state.batch_session = {'status': 'cancelled'}
        st.warning("❌ 批量预测已取消")
    
    if clear_button:
        st.session_state.batch_session = None
        st.session_state.batch_result = None
        st.success("🧹 结果已清除")
        st.rerun()

    # 结果展示区域
    if st.session_state.batch_result:
        show_batch_prediction_results(st.session_state.batch_result)


def show_batch_prediction_results(batch_result: BatchResult):
    """显示批量预测结果"""
    st.subheader("📊 预测结果分析")
    
    stats = batch_result.statistics
    session = batch_result.session
    
    # 统计摘要卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "总轮次", 
            stats.total_rounds,
            help="总预测轮次"
        )
    
    with col2:
        st.metric(
            "成功轮次", 
            stats.success_rounds,
            delta=f"{(stats.success_rounds/stats.total_rounds*100):.1f}%" if stats.total_rounds > 0 else "0%",
            help="成功执行的预测轮次"
        )
    
    with col3:
        st.metric(
            "平均命中率",
            f"{stats.avg_hit_rate:.2%}",
            help="所有成功预测的平均命中率"
        )
    
    with col4:
        st.metric(
            "命中率区间",
            f"{stats.min_hit_rate:.1%} - {stats.max_hit_rate:.1%}",
            help="最低和最高命中率"
        )

    # 详细统计信息
    st.subheader("📈 详细统计")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**统计指标**")
        stats_data = {
            "指标": ["标准差", "95%置信区间", "平均执行时间", "总执行时间"],
            "数值": [
                f"{stats.std_deviation:.4f}",
                f"[{stats.confidence_interval[0]:.2%}, {stats.confidence_interval[1]:.2%}]",
                f"{stats.avg_execution_time:.3f}秒",
                f"{stats.total_execution_time:.3f}秒"
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)
    
    with col2:
        st.write("**四分位数**")
        quartile_data = None
        if len(stats.quartiles) >= 5:
            quartile_data = {
                "分位数": ["最小值", "Q1 (25%)", "中位数 (50%)", "Q3 (75%)", "最大值"],
                "命中率": [f"{q:.2%}" for q in stats.quartiles[:5]]
            }
        elif len(stats.quartiles) == 4:
            quartile_data = {
                "分位数": ["最小值", "Q1 (25%)", "中位数 (50%)", "Q3 (75%)"],
                "命中率": [f"{q:.2%}" for q in stats.quartiles[:4]]
            }
        if quartile_data:
            st.dataframe(pd.DataFrame(quartile_data), hide_index=True, use_container_width=True)

    # 命中率分布图
    if stats.hit_rate_distribution:
        st.subheader("📊 命中率分布")
        
        # 创建分布数据
        ranges = list(stats.hit_rate_distribution.keys())
        counts = list(stats.hit_rate_distribution.values())
        
        fig = px.bar(
            x=ranges, 
            y=counts,
            title="命中率区间分布",
            labels={'x': '命中率区间', 'y': '频次'},
            color=counts,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # 预测结果详情表
    st.subheader("📋 预测结果详情")
    
    if batch_result.predictions:
        # 准备表格数据
        detail_data = []
        for pred in batch_result.predictions:
            detail_data.append({
                "轮次": pred.round_number,
                "预测号码": ", ".join(map(str, pred.predicted_numbers[:10])) + ("..." if len(pred.predicted_numbers) > 10 else ""),
                "命中号码": ", ".join(map(str, pred.hit_numbers)) if pred.hit_numbers else "无",
                "命中数量": pred.hit_count,
                "命中率": f"{pred.hit_rate:.2%}",
                "执行时间": f"{pred.execution_time:.3f}s",
                "状态": "成功" if pred.success else "失败"
            })
        
        detail_df = pd.DataFrame(detail_data)
        
        # 使用颜色区分成功和失败
        def highlight_rows(row):
            if row['状态'] == '失败':
                return ['background-color: #ffebee'] * len(row)
            elif float(row['命中率'].replace('%', '')) > stats.avg_hit_rate * 100:
                return ['background-color: #e8f5e8'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = detail_df.style.apply(highlight_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # 导出功能
    st.subheader("📥 导出结果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 导出Excel文件", type="primary"):
            try:
                # 生成文件名
                filename = ExportEngine.generate_download_filename(batch_result, 'excel')
                
                # 创建临时文件
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                    excel_path = ExportEngine.export_to_excel(batch_result, tmp.name)
                    
                    # 读取文件内容
                    with open(excel_path, 'rb') as f:
                        excel_data = f.read()
                    
                    # 提供下载
                    st.download_button(
                        label="⬇️ 下载Excel文件",
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # 清理临时文件
                    import os
                    os.unlink(excel_path)
                    
                    st.success("Excel文件已生成，点击下载按钮获取文件")
                    
            except Exception as e:
                st.error(f"导出Excel失败: {str(e)}")
    
    with col2:
        if st.button("📄 导出CSV文件"):
            try:
                # 生成文件名
                filename = ExportEngine.generate_download_filename(batch_result, 'csv')
                
                # 创建临时文件
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                    csv_path = ExportEngine.export_to_csv(batch_result, tmp.name)
                    
                    # 读取文件内容
                    with open(csv_path, 'r', encoding='utf-8-sig') as f:
                        csv_data = f.read()
                    
                    # 提供下载
                    st.download_button(
                        label="⬇️ 下载CSV文件",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv"
                    )
                    
                    # 清理临时文件
                    import os
                    os.unlink(csv_path)
                    
                    st.success("CSV文件已生成，点击下载按钮获取文件")
                    
            except Exception as e:
                st.error(f"导出CSV失败: {str(e)}")

    # 命中率趋势图
    if len(batch_result.predictions) > 1:
        st.subheader("📈 命中率趋势")
        
        # 准备趋势数据
        trend_data = []
        for pred in batch_result.predictions:
            if pred.success:
                trend_data.append({
                    "轮次": pred.round_number,
                    "命中率": pred.hit_rate,
                    "平均线": stats.avg_hit_rate
                })
        
        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            
            fig = go.Figure()
            
            # 添加命中率折线
            fig.add_trace(go.Scatter(
                x=trend_df['轮次'],
                y=trend_df['命中率'],
                mode='lines+markers',
                name='命中率',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            # 添加平均线
            fig.add_trace(go.Scatter(
                x=trend_df['轮次'],
                y=trend_df['平均线'],
                mode='lines',
                name=f'平均命中率 ({stats.avg_hit_rate:.2%})',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # 添加置信区间
            fig.add_hline(
                y=stats.confidence_interval[0], 
                line_dash="dot", 
                line_color="orange",
                annotation_text=f"95%置信下限: {stats.confidence_interval[0]:.2%}"
            )
            fig.add_hline(
                y=stats.confidence_interval[1], 
                line_dash="dot", 
                line_color="orange",
                annotation_text=f"95%置信上限: {stats.confidence_interval[1]:.2%}"
            )
            
            fig.update_layout(
                title="命中率变化趋势",
                xaxis_title="预测轮次",
                yaxis_title="命中率",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)


def show_system_settings():
    """显示系统设置"""
    st.header("⚙️ 系统设置")
    
    # 系统信息
    st.subheader("系统信息")
    
    analyzer = get_analyzer()
    if analyzer:
        available_methods = analyzer.get_available_methods()
        performance_summary = analyzer.get_performance_summary()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**可用预测方法:**")
            for method in available_methods:
                st.write(f"- {method}")
        
        with col2:
            st.write("**性能统计:**")
            if performance_summary:
                for method, stats in performance_summary.items():
                    st.write(f"- {method}: 平均耗时 {stats['avg_execution_time']:.2f}秒")
            else:
                st.write("暂无性能数据")
    
    # 清理操作
    st.subheader("数据清理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ 清空历史记录", use_container_width=True):
            st.session_state.prediction_history = []
            st.success("历史记录已清空")
            st.rerun()
    
    with col2:
        if st.button("🔄 重置系统", use_container_width=True):
            # 清空所有session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("系统已重置")
            st.rerun()

def main():
    """主函数"""
    # 侧边栏导航 - 改为一级菜单
    with st.sidebar:
        st.title("📋 功能导航")

        # 使用radio按钮创建一级菜单
        page = st.radio("选择功能", [
            "🏠 首页",
            "📊 数据管理",
            "🎯 智能预测",
            "🔄 批量预测对比",  # 新增功能
            "🔢 数字对分析",
            "📋 历史记录",
            "⚙️ 系统设置"
        ], index=0)

        st.markdown("---")
        st.markdown("### 📈 系统状态")

        # 显示系统状态
        analyzer = get_analyzer()
        if analyzer:
            try:
                data = analyzer.load_data()
                st.success(f"✓ 数据已加载 ({len(data)}期)")
            except:
                st.warning("⚠️ 数据未加载")
        else:
            st.error("✗ 系统未初始化")
        
        st.markdown("---")
        st.markdown("### ℹ️ 使用说明")
        st.markdown("""
        1. **数据管理**: 爬取和验证历史数据
        2. **智能预测**: 执行各种预测算法
        3. **数字对分析**: 分析数字对出现频率
        4. **历史记录**: 查看预测历史和统计
        5. **系统设置**: 配置和维护系统
        """)
    
    # 主内容区域
    if page == "🏠 首页":
        show_homepage()
    elif page == "📊 数据管理":
        show_data_management()
    elif page == "🎯 智能预测":
        show_prediction_interface()
    elif page == "🔄 批量预测对比":
        show_batch_prediction_interface()
    elif page == "🔢 数字对分析":
        show_pair_frequency_analysis()
    elif page == "📋 历史记录":
        show_history_records()
    elif page == "⚙️ 系统设置":
        show_system_settings()

if __name__ == "__main__":
    main()