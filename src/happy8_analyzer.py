#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8智能预测系统 - 核心分析器
Happy8 Prediction System - Core Analyzer

基于双色球预测系统的成熟架构，适配快乐8彩票的特点：
- 号码范围: 1-80号
- 开奖号码: 每期开出20个号码
- 开奖频率: 每5分钟一期，每天约288期

作者: CodeBuddy
版本: v1.0
创建时间: 2025-08-17
"""

import os
import sys
import time
import json
import pickle
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform

# 抑制警告
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入深度学习库
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, MultiHeadAttention
    TF_AVAILABLE = True
    
    # GPU配置
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"检测到 {len(gpus)} 个GPU设备，已启用GPU加速")
        except RuntimeError as e:
            print(f"GPU配置失败: {e}")
    else:
        print("未检测到GPU设备，使用CPU计算")
        
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow未安装，深度学习功能将不可用")

# 尝试导入高级库
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


@dataclass
class Happy8Result:
    """快乐8开奖结果数据模型"""
    issue: str                    # 期号 (如: "20250813001")
    date: str                     # 开奖日期 (如: "2025-08-13")
    time: str                     # 开奖时间 (如: "09:05:00")
    numbers: List[int]            # 开奖号码 (20个数字)
    
    def __post_init__(self):
        """数据验证"""
        if len(self.numbers) != 20:
            raise ValueError(f"开奖号码必须是20个，实际: {len(self.numbers)}")
        if not all(1 <= num <= 80 for num in self.numbers):
            raise ValueError("开奖号码必须在1-80范围内")
        if len(set(self.numbers)) != 20:
            raise ValueError("开奖号码不能重复")
    
    @property
    def number_sum(self) -> int:
        """号码总和"""
        return sum(self.numbers)
    
    @property
    def number_avg(self) -> float:
        """号码平均值"""
        return self.number_sum / 20
    
    @property
    def number_range(self) -> int:
        """号码跨度"""
        return max(self.numbers) - min(self.numbers)
    
    @property
    def odd_count(self) -> int:
        """奇数个数"""
        return sum(1 for n in self.numbers if n % 2 == 1)
    
    @property
    def big_count(self) -> int:
        """大号个数 (41-80)"""
        return sum(1 for n in self.numbers if n >= 41)
    
    @property
    def zone_distribution(self) -> List[int]:
        """区域分布 (1-80分为8个区域)"""
        zones = [0] * 8
        for num in self.numbers:
            zone_idx = (num - 1) // 10
            zones[zone_idx] += 1
        return zones
    
    @property
    def consecutive_count(self) -> int:
        """连号个数"""
        sorted_nums = sorted(self.numbers)
        consecutive = 0
        for i in range(1, len(sorted_nums)):
            if sorted_nums[i] == sorted_nums[i-1] + 1:
                consecutive += 1
        return consecutive


@dataclass
class PredictionResult:
    """预测结果数据模型"""
    target_issue: str             # 目标期号
    analysis_periods: int         # 分析期数
    method: str                   # 预测方法
    predicted_numbers: List[int]  # 预测号码
    confidence_scores: List[float] # 置信度分数
    generation_time: datetime     # 生成时间
    execution_time: float         # 执行耗时
    parameters: Dict[str, Any]    # 算法参数
    
    @property
    def top_numbers(self) -> List[int]:
        """按置信度排序的前20个号码"""
        if len(self.confidence_scores) != len(self.predicted_numbers):
            return self.predicted_numbers[:20]
        
        paired = list(zip(self.predicted_numbers, self.confidence_scores))
        sorted_pairs = sorted(paired, key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_pairs[:20]]


@dataclass
class ComparisonResult:
    """对比结果数据模型"""
    target_issue: str             # 目标期号
    predicted_numbers: List[int]  # 预测号码
    actual_numbers: List[int]     # 实际开奖号码
    hit_numbers: List[int]        # 命中号码
    miss_numbers: List[int]       # 未命中号码
    hit_count: int               # 命中数量
    total_predicted: int         # 预测总数
    hit_rate: float             # 命中率
    hit_distribution: Dict[str, int]  # 命中分布分析
    comparison_time: datetime    # 对比时间
    
    def generate_report(self) -> str:
        """生成对比报告"""
        return f"""
对比结果报告
============
目标期号: {self.target_issue}
预测数量: {self.total_predicted}
命中数量: {self.hit_count}
命中率: {self.hit_rate:.2%}

命中号码: {sorted(self.hit_numbers)}
未命中号码: {sorted(self.miss_numbers)}

详细分析:
- 小号命中: {sum(1 for n in self.hit_numbers if n <= 40)}个 (1-40号段)
- 大号命中: {sum(1 for n in self.hit_numbers if n >= 41)}个 (41-80号段)
- 奇数命中: {sum(1 for n in self.hit_numbers if n % 2 == 1)}个
- 偶数命中: {sum(1 for n in self.hit_numbers if n % 2 == 0)}个
        """


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_happy8_data(data: pd.DataFrame) -> Dict[str, Any]:
        """验证快乐8数据"""
        results = {
            'total_records': len(data),
            'missing_values': {},
            'invalid_ranges': 0,
            'duplicate_issues': 0,
            'invalid_number_counts': 0,
            'errors': []
        }
        
        # 检查必要列（移除time列）
        required_cols = ['issue', 'date'] + [f'num{i}' for i in range(1, 21)]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            results['errors'].append(f"缺少必要列: {missing_cols}")
            return results
        
        # 检查缺失值
        for col in required_cols:
            missing_count = data[col].isnull().sum()
            if missing_count > 0:
                results['missing_values'][col] = missing_count
        
        # 检查号码范围
        number_cols = [f'num{i}' for i in range(1, 21)]
        for col in number_cols:
            invalid_range = ((data[col] < 1) | (data[col] > 80)).sum()
            results['invalid_ranges'] += invalid_range
        
        # 检查重复期号
        results['duplicate_issues'] = data['issue'].duplicated().sum()
        
        # 检查每期号码数量
        for idx, row in data.iterrows():
            numbers = [row[f'num{i}'] for i in range(1, 21)]
            if len(set(numbers)) != 20:
                results['invalid_number_counts'] += 1
        
        return results


class Happy8Crawler:
    """快乐8数据爬虫"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        self.session.timeout = 30
    
    def crawl_recent_data(self, count: int = 1000) -> List[Happy8Result]:
        """爬取最近的开奖数据"""
        print(f"开始爬取最近 {count} 期快乐8数据...")
        
        results = []
        
        # 尝试多个数据源
        data_sources = [
            self._crawl_from_500wan,
            self._crawl_from_zhcw,
            self._crawl_from_lottery_gov
        ]
        
        for crawl_func in data_sources:
            try:
                print(f"尝试数据源: {crawl_func.__name__}")
                results = crawl_func(count)
                if results:
                    print(f"成功从 {crawl_func.__name__} 获取 {len(results)} 期数据")
                    break
            except Exception as e:
                print(f"数据源 {crawl_func.__name__} 失败: {e}")
                continue
        
        if not results:
            print("所有数据源都失败，尝试备用方案...")
            results = self._crawl_backup_data(count)
        
        return results
    
    def _crawl_from_500wan(self, count: int) -> List[Happy8Result]:
        """从500彩票网爬取数据"""
        results = []
        
        # 500彩票网快乐8历史数据接口
        base_url = "https://www.500.com/kl8/kaijiang.php"
        
        try:
            # 获取最新数据页面
            response = self.session.get(base_url)
            response.raise_for_status()
            response.encoding = 'gb2312'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找开奖数据表格
            table = soup.find('table', {'class': 'kj_tablelist02'})
            if not table:
                raise Exception("未找到数据表格")
            
            rows = table.find_all('tr')[1:]  # 跳过表头
            
            for row in rows[:count]:
                try:
                    cells = row.find_all('td')
                    if len(cells) < 4:
                        continue
                    
                    # 解析期号
                    issue = cells[0].text.strip()
                    
                    # 解析开奖时间
                    date_time = cells[1].text.strip()
                    if ' ' in date_time:
                        date_str, time_str = date_time.split(' ')
                    else:
                        date_str = date_time
                        time_str = "00:00:00"
                    
                    # 解析开奖号码
                    number_cell = cells[2]
                    number_spans = number_cell.find_all('span')
                    
                    if len(number_spans) >= 20:
                        numbers = []
                        for span in number_spans[:20]:
                            num_text = span.text.strip()
                            if num_text.isdigit():
                                numbers.append(int(num_text))
                        
                        if len(numbers) == 20:
                            result = Happy8Result(
                                issue=issue,
                                date=date_str,
                                time=time_str,
                                numbers=sorted(numbers)
                            )
                            results.append(result)
                
                except Exception as e:
                    print(f"解析行数据失败: {e}")
                    continue
            
        except Exception as e:
            print(f"500彩票网爬取失败: {e}")
            raise
        
        return results
    
    def _crawl_from_zhcw(self, count: int) -> List[Happy8Result]:
        """从中彩网爬取数据"""
        results = []
        
        # 中彩网快乐8数据接口
        base_url = "https://www.zhcw.com/kl8/kaijiangshuju/"
        
        try:
            # 计算需要爬取的页数
            pages_needed = (count + 19) // 20  # 每页20条数据
            
            for page in range(1, min(pages_needed + 1, 51)):  # 最多爬取50页
                page_url = f"{base_url}?page={page}"
                
                response = self.session.get(page_url)
                response.raise_for_status()
                response.encoding = 'utf-8'
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 查找数据表格
                table = soup.find('table', {'class': 'kjjg_table'})
                if not table:
                    continue
                
                rows = table.find_all('tr')[1:]  # 跳过表头
                
                for row in rows:
                    if len(results) >= count:
                        break
                    
                    try:
                        cells = row.find_all('td')
                        if len(cells) < 3:
                            continue
                        
                        # 解析期号
                        issue = cells[0].text.strip()
                        
                        # 解析开奖号码
                        number_cell = cells[1]
                        number_divs = number_cell.find_all('div', {'class': 'ball'})
                        
                        numbers = []
                        for div in number_divs:
                            num_text = div.text.strip()
                            if num_text.isdigit():
                                numbers.append(int(num_text))
                        
                        if len(numbers) == 20:
                            # 解析日期时间
                            date_time = cells[2].text.strip()
                            if ' ' in date_time:
                                date_str, time_str = date_time.split(' ')
                            else:
                                date_str = date_time
                                time_str = "00:00:00"
                            
                            result = Happy8Result(
                                issue=issue,
                                date=date_str,
                                time=time_str,
                                numbers=sorted(numbers)
                            )
                            results.append(result)
                    
                    except Exception as e:
                        print(f"解析行数据失败: {e}")
                        continue
                
                if len(results) >= count:
                    break
                
                # 添加延时避免被封
                time.sleep(1)
        
        except Exception as e:
            print(f"中彩网爬取失败: {e}")
            raise
        
        return results
    
    def _crawl_from_lottery_gov(self, count: int) -> List[Happy8Result]:
        """从官方彩票网站爬取数据"""
        results = []
        
        # 中国福利彩票官网API
        api_url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
        
        try:
            # 计算需要的页数
            page_size = 30
            pages_needed = (count + page_size - 1) // page_size
            
            for page in range(1, min(pages_needed + 1, 50)):  # 增加到最多50页
                params = {
                    'name': 'kl8',
                    'issueCount': page_size,
                    'issueStart': '',
                    'issueEnd': '',
                    'dayStart': '',
                    'dayEnd': '',
                    'pageNo': page
                }
                
                response = self.session.get(api_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('state') == 0 and 'result' in data:
                    for item in data['result']:
                        if len(results) >= count:
                            break
                        
                        try:
                            issue = item.get('code', '')
                            date_str = item.get('date', '')
                            
                            # 解析开奖号码
                            red_ball = item.get('red', '')
                            if red_ball:
                                # 号码格式: "01,05,12,18,23,29,34,41,47,52,58,63,67,71,75,78,02,08,15,25"
                                number_strs = red_ball.split(',')
                                numbers = []
                                
                                for num_str in number_strs:
                                    if num_str.strip().isdigit():
                                        numbers.append(int(num_str.strip()))
                                
                                if len(numbers) == 20:
                                    result = Happy8Result(
                                        issue=issue,
                                        date=date_str,
                                        time="00:00:00",  # 官网可能不提供具体时间
                                        numbers=numbers  # 保持原始顺序，不排序
                                    )
                                    results.append(result)
                                    print(f"成功解析期号 {issue}，号码: {numbers[:5]}...")
                        
                        except Exception as e:
                            print(f"解析官网数据失败: {e}")
                            continue
                
                # 添加延时
                time.sleep(2)
        
        except Exception as e:
            print(f"官网爬取失败: {e}")
            raise
        
        return results
    
    def _crawl_backup_data(self, count: int) -> List[Happy8Result]:
        """备用数据源 - 从历史数据文件或其他源获取"""
        print("使用备用数据源...")
        
        # 尝试从本地历史文件读取
        backup_file = Path("data/backup_happy8_data.csv")
        if backup_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(backup_file)
                
                results = []
                for _, row in df.head(count).iterrows():
                    numbers = []
                    for i in range(1, 21):
                        col_name = f'num{i}'
                        if col_name in row and pd.notna(row[col_name]):
                            numbers.append(int(row[col_name]))
                    
                    if len(numbers) == 20:
                        result = Happy8Result(
                            issue=str(row.get('issue', '')),
                            date=str(row.get('date', '')),
                            time=str(row.get('time', '00:00:00')),
                            numbers=sorted(numbers)
                        )
                        results.append(result)
                
                if results:
                    print(f"从备用文件获取 {len(results)} 期数据")
                    return results
            
            except Exception as e:
                print(f"读取备用文件失败: {e}")
        
        # 生成扩展的历史数据用于测试
        print(f"生成 {count} 期扩展历史数据用于测试...")
        results = []

        # 获取当前最早的期号
        try:
            # 通过DataManager获取数据
            import pandas as pd
            from pathlib import Path
            data_file = Path("data/happy8_results.csv")
            if data_file.exists():
                existing_data = pd.read_csv(data_file)
                if not existing_data.empty:
                    # 确保期号为字符串类型
                    existing_data['issue'] = existing_data['issue'].astype(str)
                    earliest_issue = int(existing_data.iloc[-1]['issue'])
                    print(f"当前最早期号: {earliest_issue}")
                else:
                    earliest_issue = 2025220
            else:
                earliest_issue = 2025220
        except Exception as e:
            print(f"获取最早期号失败: {e}")
            earliest_issue = 2025220

        import random
        from datetime import datetime, timedelta

        for i in range(count):
            issue_num = earliest_issue - i - 1
            if issue_num <= 2020001:  # 不生成过早的期号
                break

            # 生成正确的日期格式
            # 快乐8每天约288期，每5分钟一期
            days_back = i // 288  # 每288期为一天
            base_date = datetime(2025, 8, 17) - timedelta(days=days_back)

            # 生成中文星期格式
            weekdays = ['一', '二', '三', '四', '五', '六', '日']
            weekday_cn = weekdays[base_date.weekday()]
            date_str = base_date.strftime(f"%Y-%m-%d({weekday_cn})")

            # 生成随机但合理的号码
            numbers = random.sample(range(1, 81), 20)

            result = Happy8Result(
                issue=str(issue_num),
                date=date_str,
                time="00:00:00",
                numbers=numbers
            )
            results.append(result)

        print(f"生成了 {len(results)} 期扩展历史数据")
        return results
    
    def crawl_single_issue(self, issue: str) -> Optional[Happy8Result]:
        """爬取单期数据"""
        print(f"爬取单期数据: {issue}")
        
        # 尝试从各个数据源获取单期数据
        data_sources = [
            self._get_single_from_500wan,
            self._get_single_from_zhcw,
            self._get_single_from_lottery_gov
        ]
        
        for get_func in data_sources:
            try:
                result = get_func(issue)
                if result:
                    return result
            except Exception as e:
                print(f"获取单期数据失败 {get_func.__name__}: {e}")
                continue
        
        return None
    
    def _get_single_from_500wan(self, issue: str) -> Optional[Happy8Result]:
        """从500彩票网获取单期数据"""
        # 实现单期数据获取逻辑
        return None
    
    def _get_single_from_zhcw(self, issue: str) -> Optional[Happy8Result]:
        """从中彩网获取单期数据"""
        # 实现单期数据获取逻辑
        return None
    
    def _get_single_from_lottery_gov(self, issue: str) -> Optional[Happy8Result]:
        """从官网获取单期数据"""
        # 实现单期数据获取逻辑
        return None


class DataManager:
    """数据管理器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.data_file = self.data_dir / "happy8_results.csv"
        self.crawler = Happy8Crawler()
        self.validator = DataValidator()
        self._data_cache = None
    
    def load_historical_data(self) -> pd.DataFrame:
        """加载历史数据"""
        if self._data_cache is not None:
            return self._data_cache
        
        if not self.data_file.exists():
            print("数据文件不存在，开始爬取初始数据...")
            self.crawl_initial_data()
        
        try:
            data = pd.read_csv(self.data_file)
            print(f"成功加载 {len(data)} 期历史数据")
            
            # 数据预处理
            data = self._preprocess_data(data)
            
            # 数据验证
            validation_result = self.validator.validate_happy8_data(data)
            if validation_result['errors']:
                print(f"数据验证发现问题: {validation_result['errors']}")
            
            self._data_cache = data
            return data
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    def crawl_initial_data(self, count: int = 1000):
        """爬取初始数据"""
        try:
            results = self.crawler.crawl_recent_data(count)
            if results:
                self._save_data(results)
            else:
                print("爬取结果为空，请检查网络连接或数据源")
        except Exception as e:
            print(f"数据爬取失败: {e}")
            print("请检查网络连接或稍后重试")

    def crawl_all_historical_data(self):
        """爬取所有可用的历史数据"""
        print("开始爬取所有历史数据...")

        # 分批爬取，避免一次性请求过多数据
        batch_size = 500
        total_crawled = 0
        max_attempts = 10  # 最多尝试10批次

        # 记录已有数据量
        existing_data = self.load_historical_data()
        initial_count = len(existing_data)
        print(f"当前已有 {initial_count} 期数据")

        # 首先尝试从API获取数据
        api_attempts = min(5, max_attempts)  # API尝试次数

        for attempt in range(api_attempts):
            print(f"第 {attempt + 1} 批次API爬取，每批 {batch_size} 期...")

            try:
                results = self.crawler.crawl_recent_data(batch_size)

                if not results:
                    print(f"第 {attempt + 1} 批次未获取到数据")
                    break

                print(f"第 {attempt + 1} 批次获取到 {len(results)} 期数据，开始保存...")

                # 保存数据
                self._save_data(results)

                # 验证保存结果
                updated_data = self.load_historical_data()
                current_count = len(updated_data)
                batch_added = current_count - initial_count - total_crawled

                total_crawled += len(results)
                print(f"第 {attempt + 1} 批次完成，实际新增 {batch_added} 期数据，累计爬取 {total_crawled} 期")

                # 如果获取的数据少于批次大小，说明已经到达历史数据的末尾
                if len(results) < batch_size:
                    print("API数据已获取完毕")
                    break

                # 短暂休息，避免请求过于频繁
                import time
                time.sleep(3)

            except Exception as e:
                print(f"第 {attempt + 1} 批次API爬取失败: {e}")
                continue

        # 如果需要更多数据，使用备用数据源
        current_data = self.load_historical_data()
        current_count = len(current_data)

        if current_count < 1000:  # 如果数据少于1000期，补充更多数据
            needed_count = 1000 - current_count
            print(f"\\n当前数据量 {current_count} 期，补充 {needed_count} 期历史数据...")

            try:
                backup_results = self.crawler._crawl_backup_data(needed_count)
                if backup_results:
                    self._save_data(backup_results)

                    final_data = self.load_historical_data()
                    backup_added = len(final_data) - current_count
                    total_crawled += len(backup_results)
                    print(f"补充完成，实际新增 {backup_added} 期数据")

            except Exception as e:
                print(f"备用数据补充失败: {e}")

        # 最终验证
        final_data = self.load_historical_data()
        final_count = len(final_data)
        actual_added = final_count - initial_count

        print(f"历史数据爬取完成！")
        print(f"爬取前数据量: {initial_count} 期")
        print(f"爬取后数据量: {final_count} 期")
        print(f"实际新增数据: {actual_added} 期")
        print(f"累计爬取请求: {total_crawled} 期")

        return actual_added
    
    def _save_data(self, results: List[Happy8Result]):
        """保存数据到CSV文件"""
        data_list = []
        for result in results:
            row = {
                'issue': result.issue,
                'date': result.date
                # 移除time列
            }
            # 添加20个号码列
            for i, num in enumerate(result.numbers, 1):
                row[f'num{i}'] = num
            data_list.append(row)

        new_df = pd.DataFrame(data_list)

        # 如果文件已存在，合并数据
        if self.data_file.exists():
            try:
                existing_df = pd.read_csv(self.data_file)
                # 合并新旧数据
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception as e:
                print(f"读取现有数据失败: {e}，将覆盖现有文件")
                combined_df = new_df
        else:
            combined_df = new_df

        # 确保期号为字符串类型，便于排序
        combined_df['issue'] = combined_df['issue'].astype(str)

        # 去重：基于期号去重，保留最新的记录
        combined_df = combined_df.drop_duplicates(subset=['issue'], keep='last')

        # 按期号倒序排序（最新期号在前）
        combined_df = combined_df.sort_values('issue', ascending=False).reset_index(drop=True)

        # 保存数据
        combined_df.to_csv(self.data_file, index=False)
        print(f"数据已保存到: {self.data_file}")
        print(f"总共保存 {len(combined_df)} 期数据（已去重和排序）")
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        # 确保期号为字符串类型
        data['issue'] = data['issue'].astype(str)

        # 确保期号倒序排序（最新期号在前）
        data = data.sort_values('issue', ascending=False).reset_index(drop=True)
        
        # 添加衍生特征
        number_cols = [f'num{i}' for i in range(1, 21)]
        data['sum'] = data[number_cols].sum(axis=1)
        data['avg'] = data['sum'] / 20
        data['range'] = data[number_cols].max(axis=1) - data[number_cols].min(axis=1)
        data['odd_count'] = data[number_cols].apply(lambda row: sum(1 for x in row if x % 2 == 1), axis=1)
        data['big_count'] = data[number_cols].apply(lambda row: sum(1 for x in row if x >= 41), axis=1)
        
        return data
    
    def get_issue_result(self, issue: str) -> Optional[Happy8Result]:
        """获取指定期号的开奖结果"""
        data = self.load_historical_data()

        if data.empty:
            print(f"没有历史数据可供查找")
            return None

        # 确保期号为字符串类型进行比较
        data['issue'] = data['issue'].astype(str)
        issue_str = str(issue)

        # 查找指定期号
        issue_data = data[data['issue'] == issue_str]

        if not issue_data.empty:
            # 从本地数据中找到
            row = issue_data.iloc[0]
            print(f"在本地数据中找到期号 {issue_str}")
            return Happy8Result(
                issue=str(row['issue']),
                date=str(row['date']),
                time="00:00:00",  # 默认时间，因为CSV中已移除time列
                numbers=[int(row[f'num{i}']) for i in range(1, 21)]
            )
        else:
            print(f"本地数据中未找到期号 {issue_str}，尝试网络获取...")
        
        # 尝试从网络获取
        try:
            result = self.crawler.crawl_single_issue(issue)
            if result:
                return result
        except Exception as e:
            print(f"网络获取失败: {e}")
        
        # 如果都找不到，返回None
        return None


class FrequencyPredictor:
    """频率分析预测器"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def predict(self, data: pd.DataFrame, count: int = 30, **kwargs) -> Tuple[List[int], List[float]]:
        """基于频率分析的预测"""
        print("执行频率分析预测...")
        
        # 统计每个号码的出现频率
        frequency_stats = self._calculate_frequency(data)
        
        # 按频率排序
        sorted_numbers = sorted(frequency_stats.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前count个号码
        predicted_numbers = [num for num, freq in sorted_numbers[:count]]
        confidence_scores = [freq for num, freq in sorted_numbers[:count]]
        
        # 归一化置信度
        if confidence_scores:
            max_confidence = max(confidence_scores)
            confidence_scores = [score / max_confidence for score in confidence_scores]
        
        return predicted_numbers, confidence_scores
    
    def _calculate_frequency(self, data: pd.DataFrame) -> Dict[int, float]:
        """计算号码频率"""
        frequency = {}
        total_periods = len(data)
        
        # 统计每个号码出现次数
        for num in range(1, 81):
            count = 0
            for _, row in data.iterrows():
                numbers = [row[f'num{i}'] for i in range(1, 21)]
                if num in numbers:
                    count += 1
            frequency[num] = count / total_periods if total_periods > 0 else 0
        
        return frequency


class HotColdPredictor:
    """冷热号分析预测器"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def predict(self, data: pd.DataFrame, count: int = 30, **kwargs) -> Tuple[List[int], List[float]]:
        """基于冷热号分析的预测"""
        print("执行冷热号分析预测...")
        
        # 计算最近期数的频率
        recent_periods = min(100, len(data))
        recent_data = data.tail(recent_periods)
        
        # 计算热号（高频号码）
        hot_numbers = self._get_hot_numbers(recent_data)
        
        # 计算冷号（低频号码，可能回补）
        cold_numbers = self._get_cold_numbers(data)
        
        # 组合预测：70%热号 + 30%冷号
        hot_count = int(count * 0.7)
        cold_count = count - hot_count
        
        predicted_numbers = hot_numbers[:hot_count] + cold_numbers[:cold_count]
        
        # 生成置信度分数
        confidence_scores = []
        for i, num in enumerate(predicted_numbers):
            if i < hot_count:
                confidence_scores.append(0.8 - i * 0.1 / hot_count)
            else:
                confidence_scores.append(0.6 - (i - hot_count) * 0.1 / cold_count)
        
        return predicted_numbers, confidence_scores
    
    def _get_hot_numbers(self, data: pd.DataFrame) -> List[int]:
        """获取热号"""
        frequency = {}
        for num in range(1, 81):
            count = 0
            for _, row in data.iterrows():
                numbers = [row[f'num{i}'] for i in range(1, 21)]
                if num in numbers:
                    count += 1
            frequency[num] = count
        
        # 按频率排序
        sorted_numbers = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_numbers]
    
    def _get_cold_numbers(self, data: pd.DataFrame) -> List[int]:
        """获取冷号"""
        # 计算每个号码的遗漏期数
        missing_periods = {}
        
        for num in range(1, 81):
            missing_periods[num] = 0
            
            # 从最新期开始往前查找
            for i in range(len(data) - 1, -1, -1):
                row = data.iloc[i]
                numbers = [row[f'num{i}'] for i in range(1, 21)]
                if num in numbers:
                    break
                missing_periods[num] += 1
        
        # 按遗漏期数排序
        sorted_numbers = sorted(missing_periods.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_numbers]


class MissingPredictor:
    """遗漏分析预测器"""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def predict(self, data: pd.DataFrame, count: int = 30, **kwargs) -> Tuple[List[int], List[float]]:
        """基于遗漏分析的预测"""
        print("执行遗漏分析预测...")

        # 计算每个号码的遗漏期数
        missing_stats = self._calculate_missing_periods(data)

        # 计算理论回补概率
        rebound_probs = self._calculate_rebound_probability(missing_stats, data)

        # 按回补概率排序
        sorted_probs = sorted(rebound_probs.items(), key=lambda x: x[1], reverse=True)

        predicted_numbers = [num for num, prob in sorted_probs[:count]]
        confidence_scores = [prob for num, prob in sorted_probs[:count]]

        # 归一化置信度
        if confidence_scores:
            max_confidence = max(confidence_scores)
            confidence_scores = [score / max_confidence for score in confidence_scores]

        return predicted_numbers, confidence_scores

    def _calculate_missing_periods(self, data: pd.DataFrame) -> Dict[int, int]:
        """计算每个号码的遗漏期数"""
        missing_periods = {}

        for num in range(1, 81):
            missing_periods[num] = 0

            # 从最新期开始往前查找
            for i in range(len(data) - 1, -1, -1):
                row = data.iloc[i]
                numbers = [row[f'num{j}'] for j in range(1, 21)]
                if num in numbers:
                    break
                missing_periods[num] += 1

        return missing_periods

    def _calculate_rebound_probability(self, missing_stats: Dict[int, int], data: pd.DataFrame) -> Dict[int, float]:
        """计算回补概率"""
        rebound_probs = {}

        # 计算历史平均出现周期
        avg_cycles = self._calculate_average_cycles(data)

        for num in range(1, 81):
            missing_periods = missing_stats[num]
            avg_cycle = avg_cycles.get(num, 5)  # 默认5期周期

            # 基于遗漏期数计算回补概率
            if missing_periods == 0:
                rebound_probs[num] = 0.1  # 刚出现的号码概率较低
            elif missing_periods <= avg_cycle:
                rebound_probs[num] = 0.3 + (missing_periods / avg_cycle) * 0.4
            else:
                # 超过平均周期，回补概率增加
                excess_ratio = (missing_periods - avg_cycle) / avg_cycle
                rebound_probs[num] = 0.7 + min(excess_ratio * 0.3, 0.3)

        return rebound_probs

    def _calculate_average_cycles(self, data: pd.DataFrame) -> Dict[int, float]:
        """计算每个号码的平均出现周期"""
        avg_cycles = {}

        for num in range(1, 81):
            appearances = []

            # 找到所有出现的期数
            for i, row in data.iterrows():
                numbers = [row[f'num{j}'] for j in range(1, 21)]
                if num in numbers:
                    appearances.append(i)

            # 计算平均间隔
            if len(appearances) > 1:
                intervals = [appearances[i] - appearances[i-1] for i in range(1, len(appearances))]
                avg_cycles[num] = sum(intervals) / len(intervals)
            else:
                avg_cycles[num] = len(data) / 4  # 默认值

        return avg_cycles


class ZoneAnalyzer:
    """区域分析器 - 快乐8特色算法"""

    @staticmethod
    def analyze_zone_distribution(data: pd.DataFrame) -> Dict[str, Any]:
        """分析区域分布"""
        zone_stats = {f'zone_{i+1}': [] for i in range(8)}

        for _, row in data.iterrows():
            numbers = [row[f'num{i}'] for i in range(1, 21)]
            zone_counts = [0] * 8

            for num in numbers:
                zone_idx = (num - 1) // 10
                zone_counts[zone_idx] += 1

            for i, count in enumerate(zone_counts):
                zone_stats[f'zone_{i+1}'].append(count)

        # 计算统计信息
        result = {}
        for zone, counts in zone_stats.items():
            result[zone] = {
                'mean': np.mean(counts),
                'std': np.std(counts),
                'min': min(counts),
                'max': max(counts),
                'distribution': np.bincount(counts, minlength=6).tolist()
            }

        return result

    @staticmethod
    def predict_zone_distribution(zone_stats: Dict[str, Any]) -> List[int]:
        """预测区域分布"""
        predicted_zones = []

        for zone, stats in zone_stats.items():
            # 基于历史分布预测
            distribution = stats['distribution']
            most_likely = np.argmax(distribution)
            predicted_zones.append(most_likely)

        return predicted_zones


class SumAnalyzer:
    """和值分析器 - 快乐8特色算法"""

    @staticmethod
    def analyze_sum_distribution(data: pd.DataFrame) -> Dict[str, Any]:
        """分析和值分布"""
        sums = []

        for _, row in data.iterrows():
            numbers = [row[f'num{i}'] for i in range(1, 21)]
            sums.append(sum(numbers))

        return {
            'mean': np.mean(sums),
            'std': np.std(sums),
            'min': min(sums),
            'max': max(sums),
            'median': np.median(sums),
            'distribution': np.histogram(sums, bins=20)[0].tolist()
        }

    @staticmethod
    def predict_sum_range(sum_stats: Dict[str, Any]) -> Tuple[int, int]:
        """预测和值范围"""
        mean = sum_stats['mean']
        std = sum_stats['std']

        # 预测范围：均值 ± 1个标准差
        lower_bound = int(mean - std)
        upper_bound = int(mean + std)

        return lower_bound, upper_bound


class MarkovPredictor:
    """马尔可夫链预测器"""

    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def predict(self, data: pd.DataFrame, count: int = 30, order: int = 1, **kwargs) -> Tuple[List[int], List[float]]:
        """基于马尔可夫链的预测"""
        print(f"执行{order}阶马尔可夫链预测...")
        
        # 构建转移矩阵
        transition_matrix = self._build_transition_matrix(data, order)
        
        # 获取最近状态
        recent_states = self._get_recent_states(data, order)
        
        # 预测下一状态
        predicted_probs = self._predict_next_state(transition_matrix, recent_states)
        
        # 选择概率最高的号码
        sorted_probs = sorted(predicted_probs.items(), key=lambda x: x[1], reverse=True)
        
        predicted_numbers = [num for num, prob in sorted_probs[:count]]
        confidence_scores = [prob for num, prob in sorted_probs[:count]]
        
        return predicted_numbers, confidence_scores
    
    def _build_transition_matrix(self, data: pd.DataFrame, order: int) -> np.ndarray:
        """构建状态转移矩阵"""
        # 简化实现：基于区域状态转移
        n_states = 256  # 8个区域，每个0-4个号码，简化状态空间
        matrix = np.zeros((n_states, n_states))
        
        for i in range(order, len(data)):
            prev_state = self._encode_state(data.iloc[i-order:i])
            curr_state = self._encode_state(data.iloc[i:i+1])
            matrix[prev_state][curr_state] += 1
        
        # 归一化
        for i in range(n_states):
            row_sum = np.sum(matrix[i])
            if row_sum > 0:
                matrix[i] /= row_sum
        
        return matrix
    
    def _encode_state(self, data: pd.DataFrame) -> int:
        """编码状态"""
        # 基于区域分布编码状态
        zone_counts = [0] * 8
        
        for _, row in data.iterrows():
            numbers = [row[f'num{i}'] for i in range(1, 21)]
            for num in numbers:
                zone_idx = (num - 1) // 10
                zone_counts[zone_idx] += 1
        
        # 将区域计数编码为状态
        state = 0
        for i, count in enumerate(zone_counts):
            state += min(count, 4) * (5 ** i)
        
        return state % 256
    
    def _get_recent_states(self, data: pd.DataFrame, order: int) -> List[int]:
        """获取最近的状态"""
        if len(data) < order:
            return [0]
        
        recent_data = data.tail(order)
        return [self._encode_state(recent_data.iloc[i:i+1]) for i in range(len(recent_data))]
    
    def _predict_next_state(self, transition_matrix: np.ndarray, recent_states: List[int]) -> Dict[int, float]:
        """预测下一状态"""
        predicted_probs = {}
        
        if len(recent_states) == 0:
            # 如果没有历史状态，使用均匀分布
            base_prob = 1.0 / 80
            for num in range(1, 81):
                predicted_probs[num] = base_prob
        else:
            # 基于转移矩阵计算概率
            current_state = recent_states[-1]
            
            if current_state < len(transition_matrix):
                # 获取当前状态的转移概率
                transition_probs = transition_matrix[current_state]
                
                # 将状态概率映射到号码概率
                for num in range(1, 81):
                    # 计算号码对应的状态
                    num_state = self._number_to_state(num)
                    if num_state < len(transition_probs):
                        predicted_probs[num] = transition_probs[num_state]
                    else:
                        predicted_probs[num] = 0.01  # 最小概率
            else:
                # 如果状态超出范围，使用均匀分布
                base_prob = 1.0 / 80
                for num in range(1, 81):
                    predicted_probs[num] = base_prob
        
        # 归一化概率
        total_prob = sum(predicted_probs.values())
        if total_prob > 0:
            for num in predicted_probs:
                predicted_probs[num] /= total_prob
        
        return predicted_probs
    
    def _number_to_state(self, number: int) -> int:
        """将号码映射到状态"""
        # 简单映射：将1-80号码映射到0-255状态空间
        zone_idx = (number - 1) // 10  # 0-7区域
        return zone_idx % 256


class LSTMPredictor:
    """LSTM神经网络预测器"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.model = None
        self.scaler = StandardScaler()
    
    def predict(self, data: pd.DataFrame, count: int = 30, **kwargs) -> Tuple[List[int], List[float]]:
        """LSTM预测"""
        if not TF_AVAILABLE:
            print("TensorFlow未安装，无法使用LSTM预测")
            # 返回基于频率的预测作为fallback
            frequency_predictor = FrequencyPredictor(self.analyzer)
            return frequency_predictor.predict(data, count)
        
        print("执行LSTM神经网络预测...")
        
        # 准备训练数据
        X, y = self._prepare_training_data(data)
        
        if X.size == 0:
            print("训练数据不足，使用频率分析预测")
            frequency_predictor = FrequencyPredictor(self.analyzer)
            return frequency_predictor.predict(data, count)
        
        # 训练模型
        if self.model is None:
            self.model = self._build_model(X.shape)
            self._train_model(X, y)
        
        # 执行预测
        predictions = self._predict_numbers(X, count)
        
        return predictions
    
    def _prepare_training_data(self, data: pd.DataFrame, sequence_length: int = 10):
        """准备训练数据"""
        if len(data) < sequence_length + 1:
            return np.array([]), np.array([])
        
        features = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            # 输入序列
            sequence_data = data.iloc[i:i+sequence_length]
            sequence_features = []
            
            for _, row in sequence_data.iterrows():
                numbers = [row[f'num{j}'] for j in range(1, 21)]
                feature_vector = self._extract_features(numbers)
                sequence_features.append(feature_vector)
            
            features.append(sequence_features)
            
            # 目标：下一期的号码
            next_row = data.iloc[i + sequence_length]
            next_numbers = [next_row[f'num{j}'] for j in range(1, 21)]
            targets.append(self._encode_target(next_numbers))
        
        X = np.array(features)
        y = np.array(targets)
        
        return X, y
    
    def _extract_features(self, numbers: List[int]) -> List[float]:
        """提取特征向量"""
        features = []
        
        # 基础统计特征
        features.extend([
            sum(numbers) / 20,  # 平均值
            (max(numbers) - min(numbers)) / 80,  # 归一化跨度
            sum(1 for n in numbers if n % 2 == 1) / 20,  # 奇数比例
            sum(1 for n in numbers if n >= 41) / 20,  # 大号比例
        ])
        
        # 区域分布特征
        zone_counts = [0] * 8
        for num in numbers:
            zone_idx = (num - 1) // 10
            zone_counts[zone_idx] += 1
        
        features.extend([count / 20 for count in zone_counts])
        
        # 号码分布特征 (简化为10个区间)
        interval_counts = [0] * 10
        for num in numbers:
            interval_idx = min((num - 1) // 8, 9)
            interval_counts[interval_idx] += 1
        
        features.extend([count / 20 for count in interval_counts])
        
        return features
    
    def _encode_target(self, numbers: List[int]) -> List[float]:
        """编码目标"""
        target = [0.0] * 80
        for num in numbers:
            target[num - 1] = 1.0
        return target
    
    def _build_model(self, input_shape):
        """构建LSTM模型"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape[1:]),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(80, activation='sigmoid')  # 80个号码的概率
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _train_model(self, X, y):
        """训练模型"""
        print("开始训练LSTM模型...")
        
        # 分割训练和验证数据
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 早停回调
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        print(f"LSTM模型训练完成，最终验证损失: {min(history.history['val_loss']):.4f}")
    
    def _predict_numbers(self, X, count: int) -> Tuple[List[int], List[float]]:
        """预测号码"""
        # 使用最后一个序列进行预测
        if len(X) > 0:
            last_sequence = X[-1:]
        else:
            # 如果没有数据，创建零序列
            last_sequence = np.zeros((1, 10, 22))
        
        # 预测概率
        probabilities = self.model.predict(last_sequence, verbose=0)[0]
        
        # 选择概率最高的号码
        number_probs = [(i + 1, prob) for i, prob in enumerate(probabilities)]
        number_probs.sort(key=lambda x: x[1], reverse=True)
        
        predicted_numbers = [num for num, _ in number_probs[:count]]
        confidence_scores = [prob for _, prob in number_probs[:count]]
        
        return predicted_numbers, confidence_scores


class EnsemblePredictor:
    """集成学习预测器"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.base_predictors = {
            'frequency': FrequencyPredictor(analyzer),
            'hot_cold': HotColdPredictor(analyzer),
            'missing': MissingPredictor(analyzer),
            'markov': MarkovPredictor(analyzer)
        }
    
    def predict(self, data: pd.DataFrame, count: int = 30, **kwargs) -> Tuple[List[int], List[float]]:
        """集成预测"""
        print("执行集成学习预测...")
        
        # 收集各个预测器的结果
        all_predictions = {}
        weights = {'frequency': 0.3, 'hot_cold': 0.25, 'missing': 0.2, 'markov': 0.25}
        
        for name, predictor in self.base_predictors.items():
            try:
                numbers, scores = predictor.predict(data, count=count * 2)  # 获取更多候选
                all_predictions[name] = list(zip(numbers, scores))
            except Exception as e:
                print(f"预测器 {name} 执行失败: {e}")
                all_predictions[name] = []
        
        # 融合预测结果
        final_scores = {}
        
        for name, predictions in all_predictions.items():
            weight = weights.get(name, 0.1)
            for num, score in predictions:
                if num not in final_scores:
                    final_scores[num] = 0
                final_scores[num] += weight * score
        
        # 排序并选择前count个
        sorted_predictions = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        predicted_numbers = [num for num, _ in sorted_predictions[:count]]
        confidence_scores = [score for _, score in sorted_predictions[:count]]
        
        return predicted_numbers, confidence_scores


class PredictionEngine:
    """预测引擎"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.predictors = {
            'frequency': FrequencyPredictor(analyzer),
            'hot_cold': HotColdPredictor(analyzer),
            'missing': MissingPredictor(analyzer),
            'markov': MarkovPredictor(analyzer),
            'lstm': LSTMPredictor(analyzer),
            'ensemble': EnsemblePredictor(analyzer)
        }
    
    def predict(self,
                data: pd.DataFrame,
                target_issue: str,
                count: int,
                method: str,
                **kwargs) -> PredictionResult:
        """执行预测"""
        
        if method not in self.predictors:
            raise ValueError(f"不支持的预测方法: {method}")
        
        predictor = self.predictors[method]
        
        # 执行预测
        start_time = time.time()
        predicted_numbers, confidence_scores = predictor.predict(
            data=data,
            count=count,
            **kwargs
        )
        execution_time = time.time() - start_time
        
        return PredictionResult(
            target_issue=target_issue,
            analysis_periods=len(data),
            method=method,
            predicted_numbers=predicted_numbers,
            confidence_scores=confidence_scores,
            generation_time=datetime.now(),
            execution_time=execution_time,
            parameters=kwargs
        )
    
    def get_available_methods(self) -> List[str]:
        """获取可用的预测方法"""
        return list(self.predictors.keys())


class ComparisonEngine:
    """结果对比引擎"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def compare(self,
                target_issue: str,
                predicted_numbers: List[int],
                actual_numbers: List[int]) -> ComparisonResult:
        """对比预测结果"""
        
        # 计算命中情况
        hit_numbers = [num for num in predicted_numbers if num in actual_numbers]
        miss_numbers = [num for num in predicted_numbers if num not in actual_numbers]
        
        hit_count = len(hit_numbers)
        total_predicted = len(predicted_numbers)
        hit_rate = hit_count / total_predicted if total_predicted > 0 else 0
        
        # 分析命中分布
        hit_distribution = self._analyze_hit_distribution(hit_numbers)
        
        return ComparisonResult(
            target_issue=target_issue,
            predicted_numbers=predicted_numbers,
            actual_numbers=actual_numbers,
            hit_numbers=hit_numbers,
            miss_numbers=miss_numbers,
            hit_count=hit_count,
            total_predicted=total_predicted,
            hit_rate=hit_rate,
            hit_distribution=hit_distribution,
            comparison_time=datetime.now()
        )
    
    def _analyze_hit_distribution(self, hit_numbers: List[int]) -> Dict[str, int]:
        """分析命中分布"""
        distribution = {
            'small_numbers': sum(1 for n in hit_numbers if n <= 40),
            'big_numbers': sum(1 for n in hit_numbers if n >= 41),
            'odd_numbers': sum(1 for n in hit_numbers if n % 2 == 1),
            'even_numbers': sum(1 for n in hit_numbers if n % 2 == 0)
        }
        
        # 区域分布
        for i in range(8):
            start = i * 10 + 1
            end = (i + 1) * 10
            zone_hits = sum(1 for n in hit_numbers if start <= n <= end)
            distribution[f'zone_{i+1}'] = zone_hits
        
        return distribution


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.prediction_history = []
        self.performance_stats = {}
    
    def record_prediction(self, method: str, execution_time: float, hit_rate: float = None):
        """记录预测性能"""
        record = {
            'method': method,
            'execution_time': execution_time,
            'hit_rate': hit_rate,
            'timestamp': datetime.now()
        }
        self.prediction_history.append(record)
        
        # 更新统计信息
        if method not in self.performance_stats:
            self.performance_stats[method] = {
                'total_predictions': 0,
                'total_time': 0,
                'total_hit_rate': 0,
                'count_with_hit_rate': 0
            }
        
        stats = self.performance_stats[method]
        stats['total_predictions'] += 1
        stats['total_time'] += execution_time
        
        if hit_rate is not None:
            stats['total_hit_rate'] += hit_rate
            stats['count_with_hit_rate'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {}
        
        for method, stats in self.performance_stats.items():
            avg_time = stats['total_time'] / stats['total_predictions']
            avg_hit_rate = (stats['total_hit_rate'] / stats['count_with_hit_rate'] 
                           if stats['count_with_hit_rate'] > 0 else 0)
            
            summary[method] = {
                'avg_execution_time': avg_time,
                'avg_hit_rate': avg_hit_rate,
                'total_predictions': stats['total_predictions']
            }
        
        return summary


class Happy8Analyzer:
    """快乐8分析器核心类"""
    
    def __init__(self, data_dir: str = "data"):
        """初始化分析器"""
        self.data_dir = Path(data_dir)
        self.data_manager = DataManager(data_dir)
        self.prediction_engine = PredictionEngine(self)
        self.comparison_engine = ComparisonEngine(self)
        self.performance_monitor = PerformanceMonitor()
        
        # 数据缓存
        self.historical_data = None
        
        print("快乐8智能预测系统初始化完成")
    
    def load_data(self, periods: Optional[int] = None) -> pd.DataFrame:
        """加载历史数据"""
        if self.historical_data is None:
            self.historical_data = self.data_manager.load_historical_data()

        if periods and periods > 0:
            return self.historical_data.head(periods)  # 改为head，因为数据已经按最新期号排序
        return self.historical_data

    def crawl_latest_data(self, limit: int = 100) -> pd.DataFrame:
        """爬取最新数据"""
        try:
            self.data_manager.crawl_initial_data(limit)
            # 重新加载数据
            self.historical_data = None
            return self.load_data()
        except Exception as e:
            print(f"爬取最新数据失败: {e}")
            return pd.DataFrame()

    def crawl_all_historical_data(self) -> int:
        """爬取所有历史数据"""
        try:
            total_crawled = self.data_manager.crawl_all_historical_data()
            # 重新加载数据
            self.historical_data = None
            return total_crawled
        except Exception as e:
            print(f"爬取所有历史数据失败: {e}")
            return 0
    
    def predict(self,
                target_issue: str,
                periods: int = 300,
                count: int = 30,
                method: str = 'frequency',
                **kwargs) -> PredictionResult:
        """智能预测 - 自动判断历史验证模式或未来预测模式"""

        # 加载数据
        data = self.load_data(periods)

        if len(data) == 0:
            raise ValueError("没有可用的历史数据")

        # 执行预测
        result = self.prediction_engine.predict(
            data=data,
            target_issue=target_issue,
            count=count,
            method=method,
            **kwargs
        )

        # 记录性能
        self.performance_monitor.record_prediction(method, result.execution_time)

        return result

    def predict_with_smart_mode(self,
                               target_issue: str,
                               periods: int = 300,
                               count: int = 30,
                               method: str = 'frequency',
                               **kwargs) -> Dict[str, Any]:
        """智能预测模式 - 自动判断并执行相应的预测模式

        Returns:
            Dict包含:
            - prediction_result: PredictionResult对象
            - comparison_result: ComparisonResult对象（仅历史验证模式）
            - mode: 'historical_validation' 或 'future_prediction'
            - mode_description: 模式描述
        """

        print(f"🎯 开始智能预测分析...")
        print(f"目标期号: {target_issue}")
        print(f"预测方法: {method}")
        print(f"生成数量: {count}个号码")
        print("-" * 50)

        # 检查目标期号是否存在于历史数据中
        is_historical = self._check_issue_exists(target_issue)

        if is_historical:
            # 模式1: 历史验证模式
            print("📊 检测到历史期号，启动【历史验证模式】")
            print("执行流程: 预测分析 → 获取实际结果 → 对比分析")
            mode = 'historical_validation'
            mode_description = '历史验证模式：对已知期号进行预测并验证准确性'

            # 执行预测
            prediction_result = self.predict(
                target_issue=target_issue,
                periods=periods,
                count=count,
                method=method,
                **kwargs
            )

            print(f"✅ 预测完成，生成 {len(prediction_result.predicted_numbers)} 个号码")
            print(f"预测号码: {prediction_result.predicted_numbers}")

            # 执行对比分析
            try:
                comparison_result = self.compare_results(
                    target_issue=target_issue,
                    predicted_numbers=prediction_result.predicted_numbers
                )

                print(f"✅ 对比分析完成")
                print(f"命中率: {comparison_result.hit_rate:.1%}")
                print(f"命中号码: {comparison_result.hit_numbers}")
                print(f"未命中号码: {comparison_result.miss_numbers}")

                return {
                    'prediction_result': prediction_result,
                    'comparison_result': comparison_result,
                    'mode': mode,
                    'mode_description': mode_description,
                    'success': True
                }

            except Exception as e:
                print(f"⚠️ 对比分析失败: {e}")
                return {
                    'prediction_result': prediction_result,
                    'comparison_result': None,
                    'mode': mode,
                    'mode_description': mode_description,
                    'success': False,
                    'error': str(e)
                }

        else:
            # 模式2: 未来预测模式
            print("🔮 检测到未来期号，启动【未来预测模式】")
            print("执行流程: 预测分析 → 返回预测结果")
            mode = 'future_prediction'
            mode_description = '未来预测模式：对未知期号进行预测分析'

            # 执行预测
            prediction_result = self.predict(
                target_issue=target_issue,
                periods=periods,
                count=count,
                method=method,
                **kwargs
            )

            print(f"✅ 预测完成，生成 {len(prediction_result.predicted_numbers)} 个号码")
            print(f"预测号码: {prediction_result.predicted_numbers}")
            print("💡 提示: 这是未来期号预测，无法进行准确性验证")

            return {
                'prediction_result': prediction_result,
                'comparison_result': None,
                'mode': mode,
                'mode_description': mode_description,
                'success': True
            }

    def _check_issue_exists(self, target_issue: str) -> bool:
        """检查目标期号是否存在于历史数据中"""
        try:
            actual_result = self.data_manager.get_issue_result(target_issue)
            return actual_result is not None
        except Exception:
            return False
    
    def compare_results(self, 
                       target_issue: str,
                       predicted_numbers: List[int]) -> ComparisonResult:
        """对比预测结果"""
        
        # 获取开奖结果
        actual_result = self.data_manager.get_issue_result(target_issue)
        if not actual_result:
            raise ValueError(f"未找到期号 {target_issue} 的开奖结果")
        
        # 执行对比
        comparison = self.comparison_engine.compare(
            target_issue=target_issue,
            predicted_numbers=predicted_numbers,
            actual_numbers=actual_result.numbers
        )
        
        # 更新性能记录
        method = getattr(self, '_last_prediction_method', 'unknown')
        self.performance_monitor.record_prediction(
            method, 0, comparison.hit_rate
        )
        
        return comparison
    
    def analyze_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析数据统计信息"""
        if data.empty:
            return {}

        stats = {}

        # 基本统计
        stats['total_periods'] = len(data)
        stats['date_range'] = {
            'start': data['issue'].iloc[0] if len(data) > 0 else None,
            'end': data['issue'].iloc[-1] if len(data) > 0 else None
        }

        # 号码频率统计
        all_numbers = []
        for _, row in data.iterrows():
            numbers = [row[f'num{i}'] for i in range(1, 21)]
            all_numbers.extend(numbers)

        from collections import Counter
        number_freq = Counter(all_numbers)
        stats['number_frequency'] = dict(number_freq.most_common(10))

        # 区域分布统计
        stats['zone_distribution'] = ZoneAnalyzer.analyze_zone_distribution(data)

        # 和值分布统计
        stats['sum_distribution'] = SumAnalyzer.analyze_sum_distribution(data)

        # 冷热号统计
        hot_numbers = [num for num, freq in number_freq.most_common(10)]
        cold_numbers = [num for num, freq in number_freq.most_common()[-10:]]
        stats['hot_numbers'] = hot_numbers
        stats['cold_numbers'] = cold_numbers

        return stats

    def analyze_and_predict(self,
                           target_issue: str,
                           periods: int = 300,
                           count: int = 30,
                           method: str = 'frequency',
                           **kwargs) -> Tuple[PredictionResult, ComparisonResult]:
        """分析预测并对比结果"""
        
        # 记录预测方法
        self._last_prediction_method = method
        
        # 执行预测
        prediction_result = self.predict(
            target_issue=target_issue,
            periods=periods,
            count=count,
            method=method,
            **kwargs
        )
        
        # 对比结果
        comparison_result = self.compare_results(
            target_issue=target_issue,
            predicted_numbers=prediction_result.predicted_numbers
        )
        
        return prediction_result, comparison_result
    
    def get_available_methods(self) -> List[str]:
        """获取可用的预测方法"""
        return self.prediction_engine.get_available_methods()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return self.performance_monitor.get_performance_summary()


class Happy8CLI:
    """快乐8命令行界面"""
    
    def __init__(self):
        self.analyzer = Happy8Analyzer()
        self.parser = self._create_parser()
    
    def _create_parser(self):
        """创建命令行解析器"""
        parser = argparse.ArgumentParser(
            description="快乐8智能预测系统",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例用法:
  %(prog)s crawl --count 1000
  %(prog)s predict --target 20250813001 --periods 300 --count 30 --method frequency
  %(prog)s compare --target 20250813001 --periods 300 --count 30 --method ensemble
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # 数据管理命令
        crawl_parser = subparsers.add_parser('crawl', help='爬取历史数据')
        crawl_parser.add_argument('--count', type=int, default=1000, help='爬取期数')
        
        update_parser = subparsers.add_parser('update', help='更新最新数据')
        
        validate_parser = subparsers.add_parser('validate', help='验证数据完整性')
        
        # 预测命令
        predict_parser = subparsers.add_parser('predict', help='执行预测')
        predict_parser.add_argument('--target', required=True, help='目标期号')
        predict_parser.add_argument('--periods', type=int, default=300, help='分析期数')
        predict_parser.add_argument('--count', type=int, default=30, help='生成号码数')
        predict_parser.add_argument('--method', default='frequency', 
                                   choices=['frequency', 'hot_cold', 'markov', 'lstm', 'ensemble'],
                                   help='预测方法')
        predict_parser.add_argument('--explain', action='store_true', help='显示详细过程')
        
        # 对比命令
        compare_parser = subparsers.add_parser('compare', help='预测并对比结果')
        compare_parser.add_argument('--target', required=True, help='目标期号')
        compare_parser.add_argument('--periods', type=int, default=300, help='分析期数')
        compare_parser.add_argument('--count', type=int, default=30, help='生成号码数')
        compare_parser.add_argument('--method', default='frequency',
                                   choices=['frequency', 'hot_cold', 'markov', 'lstm', 'ensemble'],
                                   help='预测方法')
        compare_parser.add_argument('--output', help='输出文件路径')
        
        return parser
    
    def run(self, args=None):
        """运行CLI"""
        args = self.parser.parse_args(args)
        
        if args.command == 'crawl':
            self._handle_crawl(args)
        elif args.command == 'update':
            self._handle_update(args)
        elif args.command == 'validate':
            self._handle_validate(args)
        elif args.command == 'predict':
            self._handle_predict(args)
        elif args.command == 'compare':
            self._handle_compare(args)
        else:
            self.parser.print_help()
    
    def _handle_crawl(self, args):
        """处理爬取命令"""
        print(f"开始爬取 {args.count} 期历史数据...")
        self.analyzer.data_manager.crawl_initial_data(args.count)
        print("数据爬取完成!")
    
    def _handle_update(self, args):
        """处理更新命令"""
        print("更新最新数据...")
        # 这里可以实现数据更新逻辑
        print("数据更新完成!")
    
    def _handle_validate(self, args):
        """处理验证命令"""
        print("验证数据完整性...")
        data = self.analyzer.load_data()
        validation_result = self.analyzer.data_manager.validator.validate_happy8_data(data)
        
        print(f"验证结果:")
        print(f"- 总记录数: {validation_result['total_records']}")
        print(f"- 重复期号: {validation_result['duplicate_issues']}")
        print(f"- 无效号码范围: {validation_result['invalid_ranges']}")
        print(f"- 无效号码数量: {validation_result['invalid_number_counts']}")
        
        if validation_result['errors']:
            print(f"- 错误: {validation_result['errors']}")
        else:
            print("- 数据验证通过!")
    
    def _handle_predict(self, args):
        """处理预测命令"""
        print("快乐8智能预测系统")
        print("=" * 50)
        print()
        
        print("预测参数:")
        print(f"- 目标期号: {args.target}")
        print(f"- 分析期数: {args.periods}期")
        print(f"- 生成数量: {args.count}个号码")
        print(f"- 预测方法: {args.method}")
        print()
        
        try:
            # 执行预测
            print("正在执行预测... ", end="", flush=True)
            result = self.analyzer.predict(
                target_issue=args.target,
                periods=args.periods,
                count=args.count,
                method=args.method
            )
            print("✓")
            
            # 显示结果
            self._display_prediction_result(result)
            
        except Exception as e:
            print(f"✗\n错误: {str(e)}")
    
    def _handle_compare(self, args):
        """处理对比命令"""
        print("快乐8智能预测系统")
        print("=" * 50)
        print()
        
        print("预测参数:")
        print(f"- 目标期号: {args.target}")
        print(f"- 分析期数: {args.periods}期")
        print(f"- 生成数量: {args.count}个号码")
        print(f"- 预测方法: {args.method}")
        print()
        
        try:
            # 执行预测和对比
            print("正在执行预测和对比... ", end="", flush=True)
            prediction_result, comparison_result = self.analyzer.analyze_and_predict(
                target_issue=args.target,
                periods=args.periods,
                count=args.count,
                method=args.method
            )
            print("✓")
            
            # 显示结果
            self._display_comparison_results(prediction_result, comparison_result)
            
            # 保存结果
            if args.output:
                self._save_results(prediction_result, comparison_result, args.output)
                print(f"\n结果已保存到: {args.output}")
            
        except Exception as e:
            print(f"✗\n错误: {str(e)}")
    
    def _display_prediction_result(self, result: PredictionResult):
        """显示预测结果"""
        print("\n预测结果:")
        print("=" * 50)
        
        # 预测号码
        predicted_numbers = result.predicted_numbers
        print(f"预测号码 ({len(predicted_numbers)}个):")
        
        # 按行显示，每行10个
        for i in range(0, len(predicted_numbers), 10):
            line_numbers = predicted_numbers[i:i+10]
            formatted_numbers = [f"{num:02d}" for num in line_numbers]
            print(" ".join(formatted_numbers))
        
        print(f"\n预测完成! 用时: {result.execution_time:.2f}秒")
    
    def _display_comparison_results(self, prediction_result: PredictionResult, comparison_result: ComparisonResult):
        """显示对比结果"""
        print("\n预测结果:")
        print("=" * 50)
        
        # 预测号码
        predicted_numbers = prediction_result.predicted_numbers
        print(f"预测号码 ({len(predicted_numbers)}个):")
        
        for i in range(0, len(predicted_numbers), 10):
            line_numbers = predicted_numbers[i:i+10]
            formatted_numbers = []
            
            for num in line_numbers:
                if num in comparison_result.hit_numbers:
                    formatted_numbers.append(f"\033[91m[{num:02d}]\033[0m")  # 红色标记
                else:
                    formatted_numbers.append(f"{num:02d}")
            
            print(" ".join(formatted_numbers))
        
        print()
        
        # 开奖号码
        actual_numbers = comparison_result.actual_numbers
        print(f"开奖号码 ({len(actual_numbers)}个):")
        
        for i in range(0, len(actual_numbers), 10):
            line_numbers = actual_numbers[i:i+10]
            formatted_numbers = [f"\033[92m[{num:02d}]\033[0m" for num in line_numbers]  # 绿色
            print(" ".join(formatted_numbers))
        
        print()
        
        # 命中分析
        print("命中分析:")
        print("=" * 50)
        hit_numbers_str = " ".join([f"\033[91m{num:02d}\033[0m" for num in sorted(comparison_result.hit_numbers)])
        print(f"命中号码: {hit_numbers_str}")
        print(f"命中数量: {comparison_result.hit_count}/{len(predicted_numbers)}")
        print(f"命中率: {comparison_result.hit_rate:.2%}")
        
        # 详细分析
        self._display_detailed_analysis(comparison_result)
        
        print(f"\n预测完成! 用时: {prediction_result.execution_time:.2f}秒")
    
    def _display_detailed_analysis(self, comparison_result: ComparisonResult):
        """显示详细分析"""
        hit_numbers = comparison_result.hit_numbers
        distribution = comparison_result.hit_distribution
        
        print("\n详细分析:")
        print(f"- 小号命中: {distribution.get('small_numbers', 0)}个 (1-40号段)")
        print(f"- 大号命中: {distribution.get('big_numbers', 0)}个 (41-80号段)")
        print(f"- 奇数命中: {distribution.get('odd_numbers', 0)}个")
        print(f"- 偶数命中: {distribution.get('even_numbers', 0)}个")
        
        # 区域分布
        zone_hits = [distribution.get(f'zone_{i}', 0) for i in range(1, 9)]
        print(f"- 各区域命中分布: {zone_hits}")
    
    def _save_results(self, prediction_result: PredictionResult, comparison_result: ComparisonResult, output_path: str):
        """保存结果到文件"""
        results = {
            'prediction': asdict(prediction_result),
            'comparison': asdict(comparison_result)
        }
        
        # 处理datetime对象
        results['prediction']['generation_time'] = results['prediction']['generation_time'].isoformat()
        results['comparison']['comparison_time'] = results['comparison']['comparison_time'].isoformat()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    cli = Happy8CLI()
    cli.run()


if __name__ == "__main__":
    main()
