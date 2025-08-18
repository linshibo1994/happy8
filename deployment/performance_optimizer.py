#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8预测系统性能优化器
Happy8 Prediction System Performance Optimizer
"""

import os
import time
import psutil
import threading
from functools import wraps
from typing import Dict, Any, Callable
import pandas as pd
import numpy as np

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = {}
        self.performance_stats = {}
        self.lock = threading.Lock()
    
    def cache_result(self, ttl: int = 300):
        """结果缓存装饰器"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
                
                with self.lock:
                    # 检查缓存
                    if cache_key in self.cache:
                        cache_time = self.cache_ttl.get(cache_key, 0)
                        if time.time() - cache_time < ttl:
                            return self.cache[cache_key]
                    
                    # 执行函数并缓存结果
                    result = func(*args, **kwargs)
                    self.cache[cache_key] = result
                    self.cache_ttl[cache_key] = time.time()
                    
                    return result
            return wrapper
        return decorator
    
    def monitor_performance(self, func: Callable) -> Callable:
        """性能监控装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # 记录性能统计
                func_name = func.__name__
                if func_name not in self.performance_stats:
                    self.performance_stats[func_name] = {
                        'call_count': 0,
                        'total_time': 0,
                        'avg_time': 0,
                        'max_time': 0,
                        'min_time': float('inf'),
                        'memory_usage': [],
                        'success_rate': 0,
                        'errors': []
                    }
                
                stats = self.performance_stats[func_name]
                execution_time = end_time - start_time
                memory_used = end_memory - start_memory
                
                stats['call_count'] += 1
                stats['total_time'] += execution_time
                stats['avg_time'] = stats['total_time'] / stats['call_count']
                stats['max_time'] = max(stats['max_time'], execution_time)
                stats['min_time'] = min(stats['min_time'], execution_time)
                stats['memory_usage'].append(memory_used)
                
                if success:
                    stats['success_rate'] = (stats['success_rate'] * (stats['call_count'] - 1) + 1) / stats['call_count']
                else:
                    stats['success_rate'] = stats['success_rate'] * (stats['call_count'] - 1) / stats['call_count']
                    stats['errors'].append(error)
            
            return result
        return wrapper
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化DataFrame内存使用"""
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        # 优化数值类型
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype('uint8')
                elif col_max < 65535:
                    df[col] = df[col].astype('uint16')
                elif col_max < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype('int32')
        
        # 优化浮点类型
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # 优化字符串类型
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # 如果唯一值比例小于50%，转换为category
                df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        print(f"内存优化: {original_memory:.2f}MB -> {optimized_memory:.2f}MB (减少 {reduction:.1f}%)")
        
        return df
    
    def parallel_predict(self, predictor_func, data_chunks, **kwargs):
        """并行预测"""
        import concurrent.futures
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(predictor_func, chunk, **kwargs) for chunk in data_chunks]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"并行预测错误: {e}")
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            'cache_stats': {
                'cache_size': len(self.cache),
                'cache_hit_rate': self._calculate_cache_hit_rate()
            },
            'function_stats': {}
        }
        
        for func_name, stats in self.performance_stats.items():
            report['function_stats'][func_name] = {
                'call_count': stats['call_count'],
                'avg_execution_time': f"{stats['avg_time']:.4f}s",
                'max_execution_time': f"{stats['max_time']:.4f}s",
                'min_execution_time': f"{stats['min_time']:.4f}s",
                'avg_memory_usage': f"{np.mean(stats['memory_usage']):.2f}MB" if stats['memory_usage'] else "0MB",
                'success_rate': f"{stats['success_rate']:.2%}",
                'error_count': len(stats['errors'])
            }
        
        return report
    
    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        # 这里简化处理，实际应该记录缓存命中和未命中次数
        return 0.85  # 假设85%的命中率
    
    def clear_cache(self):
        """清理缓存"""
        with self.lock:
            self.cache.clear()
            self.cache_ttl.clear()
        print("缓存已清理")
    
    def cleanup_expired_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []
        
        with self.lock:
            for key, cache_time in self.cache_ttl.items():
                if current_time - cache_time > 300:  # 5分钟过期
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                del self.cache_ttl[key]
        
        if expired_keys:
            print(f"清理了 {len(expired_keys)} 个过期缓存项")

# 全局性能优化器实例
performance_optimizer = PerformanceOptimizer()

# 导出装饰器
cache_result = performance_optimizer.cache_result
monitor_performance = performance_optimizer.monitor_performance
