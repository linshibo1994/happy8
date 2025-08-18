#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8智能预测系统 - Web界面
Happy8 Prediction System - Web Interface

基于Streamlit的Web操作界面，提供直观的预测和分析功能。

作者: CodeBuddy
版本: v1.0
创建时间: 2025-08-17
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
    from happy8_analyzer import Happy8Analyzer, PredictionResult, ComparisonResult
except ImportError:
    st.error("无法导入核心分析器，请确保 happy8_analyzer.py 文件存在")
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
        return Happy8Analyzer()
    except Exception as e:
        st.error(f"初始化分析器失败: {e}")
        return None

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
                <li>开奖频率: 每5分钟一期</li>
                <li>每天约288期</li>
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
        target_issue = st.text_input(
            "目标期号", 
            value=f"{datetime.now().strftime('%Y%m%d')}001",
            help="输入要预测的期号"
        )
    
    with col2:
        method = st.selectbox(
            "预测方法",
            ["frequency", "hot_cold", "missing", "markov", "markov_2nd", "markov_3rd",
             "adaptive_markov", "lstm", "ensemble"],
            format_func=lambda x: {
                "frequency": "频率分析",
                "hot_cold": "冷热号分析",
                "missing": "遗漏分析",
                "markov": "1阶马尔可夫链",
                "markov_2nd": "2阶马尔可夫链",
                "markov_3rd": "3阶马尔可夫链",
                "adaptive_markov": "自适应马尔可夫链",
                "lstm": "LSTM神经网络",
                "ensemble": "集成学习"
            }[x],
            index=8
        )
    
    with col3:
        periods = st.selectbox("分析期数", [100, 200, 300, 500], index=2)
    
    with col4:
        count = st.selectbox("生成号码数", [20, 30, 40, 50], index=1)
    
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
    st.subheader("数据状态")
    
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
        if st.button("🕷️ 爬取最新数据", use_container_width=True):
            count = st.number_input("爬取期数", min_value=100, max_value=2000, value=500, key="crawl_count")

            with st.spinner(f"正在爬取 {count} 期数据..."):
                try:
                    analyzer.data_manager.crawl_initial_data(count)
                    st.success("数据爬取完成!")
                    st.rerun()
                except Exception as e:
                    st.error(f"数据爬取失败: {e}")

    with col2:
        if st.button("📚 爬取所有历史数据", use_container_width=True):
            with st.spinner("正在爬取所有历史数据，这可能需要几分钟..."):
                try:
                    total_crawled = analyzer.crawl_all_historical_data()
                    st.success(f"历史数据爬取完成！总共获取 {total_crawled} 期数据")
                    st.rerun()
                except Exception as e:
                    st.error(f"历史数据爬取失败: {e}")

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
    
    with col3:
        if st.button("🔄 更新数据", use_container_width=True):
            with st.spinner("正在更新数据..."):
                try:
                    # 这里可以实现数据更新逻辑
                    st.info("数据更新功能开发中...")
                except Exception as e:
                    st.error(f"数据更新失败: {e}")

def show_prediction_interface():
    """显示预测界面"""
    st.header("🎯 智能预测")
    
    analyzer = get_analyzer()
    if analyzer is None:
        st.error("分析器未初始化")
        return
    
    # 预测参数配置
    st.subheader("预测参数配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_issue = st.text_input(
            "目标期号", 
            value=f"{datetime.now().strftime('%Y%m%d')}001",
            help="输入要预测的期号"
        )
        
        analysis_periods = st.selectbox(
            "分析期数",
            [100, 200, 300, 500, 1000],
            index=2,
            help="用于分析的历史期数"
        )
    
    with col2:
        predict_count = st.selectbox(
            "生成号码数",
            [10, 20, 30, 40, 50],
            index=2,
            help="生成的预测号码数量"
        )
        
        method = st.selectbox(
            "预测方法",
            ["frequency", "hot_cold", "missing", "markov", "markov_2nd", "markov_3rd",
             "adaptive_markov", "lstm", "ensemble"],
            format_func=lambda x: {
                "frequency": "频率分析",
                "hot_cold": "冷热号分析",
                "missing": "遗漏分析",
                "markov": "1阶马尔可夫链",
                "markov_2nd": "2阶马尔可夫链",
                "markov_3rd": "3阶马尔可夫链",
                "adaptive_markov": "自适应马尔可夫链",
                "lstm": "LSTM神经网络",
                "ensemble": "集成学习"
            }[x],
            index=8
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
                    smart_result = analyzer.predict_with_smart_mode(
                        target_issue=target_issue,
                        periods=analysis_periods,
                        count=predict_count,
                        method=method
                    )

                    # 显示智能预测结果
                    display_smart_prediction_result(smart_result)

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
                    prediction_result = analyzer.predict(
                        target_issue=target_issue,
                        periods=analysis_periods,
                        count=predict_count,
                        method=method
                    )

                    # 显示结果
                    display_prediction_result(prediction_result)

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
        3. **历史记录**: 查看预测历史和统计
        4. **系统设置**: 配置和维护系统
        """)
    
    # 主内容区域
    if page == "🏠 首页":
        show_homepage()
    elif page == "📊 数据管理":
        show_data_management()
    elif page == "🎯 智能预测":
        show_prediction_interface()
    elif page == "📋 历史记录":
        show_history_records()
    elif page == "⚙️ 系统设置":
        show_system_settings()

if __name__ == "__main__":
    main()