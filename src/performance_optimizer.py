#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化模块 - GPU加速和并行处理优化
"""

import os
import time
import psutil
import multiprocessing as mp
from typing import Dict, List, Any, Optional
import numpy as np


class PerformanceOptimizer:
    """性能优化器 - GPU加速和并行处理"""
    
    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.memory_info = psutil.virtual_memory()
        self.gpu_available = self._check_gpu_availability()
        
        print(f"🔧 性能优化器初始化")
        print(f"   CPU核心数: {self.cpu_count}")
        print(f"   内存总量: {self.memory_info.total / (1024**3):.1f}GB")
        print(f"   GPU可用: {'是' if self.gpu_available else '否'}")
    
    def _check_gpu_availability(self) -> bool:
        """检查GPU可用性"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                print(f"   GPU设备: {gpu_name} (数量: {gpu_count})")
                return True
            else:
                print("   GPU设备: 未检测到CUDA设备")
                return False
        except ImportError:
            print("   GPU设备: PyTorch未安装")
            return False
    
    def optimize_for_algorithm(self, algorithm_name: str, data_size: int) -> Dict[str, Any]:
        """为特定算法优化性能配置"""
        config = {
            'use_gpu': False,
            'batch_size': 32,
            'num_workers': 1,
            'memory_limit': None,
            'parallel_processes': 1
        }
        
        # 基于算法类型优化
        if algorithm_name in ['transformer', 'gnn', 'lstm']:
            # 深度学习算法
            config['use_gpu'] = self.gpu_available
            config['batch_size'] = min(64, max(16, data_size // 10))
            config['num_workers'] = min(4, self.cpu_count)
            
        elif algorithm_name in ['monte_carlo', 'advanced_ensemble']:
            # 计算密集型算法
            config['parallel_processes'] = min(self.cpu_count, 8)
            config['memory_limit'] = self.memory_info.available * 0.7
            
        elif algorithm_name in ['clustering', 'bayesian']:
            # 中等计算量算法
            config['parallel_processes'] = min(self.cpu_count // 2, 4)
            config['batch_size'] = min(128, data_size)
            
        elif algorithm_name == 'super_predictor':
            # 超级预测器 - 综合优化
            config['use_gpu'] = self.gpu_available
            config['parallel_processes'] = min(self.cpu_count, 6)
            config['memory_limit'] = self.memory_info.available * 0.8
            config['batch_size'] = 32
        
        print(f"🎯 {algorithm_name} 性能配置: {config}")
        return config
    
    def monitor_performance(self, func, *args, **kwargs):
        """性能监控装饰器"""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        # 执行函数
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        # 计算性能指标
        execution_time = end_time - start_time
        memory_used = (end_memory - start_memory) / (1024**2)  # MB
        
        performance_info = {
            'execution_time': execution_time,
            'memory_used': memory_used,
            'cpu_usage': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }
        
        print(f"📊 性能监控结果:")
        print(f"   执行时间: {execution_time:.2f}秒")
        print(f"   内存使用: {memory_used:.1f}MB")
        print(f"   CPU使用率: {performance_info['cpu_usage']:.1f}%")
        print(f"   内存使用率: {performance_info['memory_percent']:.1f}%")
        
        return result, performance_info
    
    def optimize_memory_usage(self, data_size: int) -> Dict[str, Any]:
        """内存使用优化"""
        available_memory = self.memory_info.available
        
        # 计算推荐的批处理大小
        if data_size * 1000 < available_memory:  # 假设每个数据点1KB
            batch_size = data_size
            use_chunking = False
        else:
            batch_size = max(10, available_memory // (1000 * 10))
            use_chunking = True
        
        config = {
            'batch_size': batch_size,
            'use_chunking': use_chunking,
            'memory_limit': available_memory * 0.8,
            'gc_frequency': 100 if use_chunking else 1000
        }
        
        return config
    
    def parallel_process_optimization(self, task_count: int, cpu_intensive: bool = True) -> int:
        """并行处理优化"""
        if cpu_intensive:
            # CPU密集型任务
            optimal_processes = min(self.cpu_count, task_count)
        else:
            # I/O密集型任务
            optimal_processes = min(self.cpu_count * 2, task_count)
        
        # 考虑内存限制
        memory_per_process = 500  # MB
        max_processes_by_memory = self.memory_info.available // (memory_per_process * 1024 * 1024)
        
        return min(optimal_processes, max_processes_by_memory, 16)  # 最多16个进程


class SystemBenchmark:
    """系统性能基准测试"""
    
    def __init__(self):
        self.optimizer = PerformanceOptimizer()
    
    def run_benchmark(self) -> Dict[str, float]:
        """运行系统性能基准测试"""
        print("🏃 开始系统性能基准测试...")
        
        results = {}
        
        # CPU计算性能测试
        results['cpu_performance'] = self._test_cpu_performance()
        
        # 内存访问性能测试
        results['memory_performance'] = self._test_memory_performance()
        
        # 并行处理性能测试
        results['parallel_performance'] = self._test_parallel_performance()
        
        # GPU性能测试（如果可用）
        if self.optimizer.gpu_available:
            results['gpu_performance'] = self._test_gpu_performance()
        
        print("✅ 系统性能基准测试完成")
        return results
    
    def _test_cpu_performance(self) -> float:
        """CPU性能测试"""
        print("测试CPU计算性能...")
        
        start_time = time.time()
        
        # 矩阵运算测试
        size = 1000
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        c = np.dot(a, b)
        
        end_time = time.time()
        cpu_score = 1000 / (end_time - start_time)  # 分数越高越好
        
        print(f"   CPU性能分数: {cpu_score:.1f}")
        return cpu_score
    
    def _test_memory_performance(self) -> float:
        """内存性能测试"""
        print("测试内存访问性能...")
        
        start_time = time.time()
        
        # 大数组创建和访问测试
        size = 10000000  # 10M elements
        arr = np.random.rand(size)
        result = np.sum(arr)
        
        end_time = time.time()
        memory_score = size / (end_time - start_time) / 1000000  # MB/s
        
        print(f"   内存性能分数: {memory_score:.1f} MB/s")
        return memory_score
    
    def _test_parallel_performance(self) -> float:
        """并行处理性能测试"""
        print("测试并行处理性能...")

        # 串行测试
        start_time = time.time()
        serial_results = [self._cpu_task(10000) for _ in range(100)]
        serial_time = time.time() - start_time

        # 并行测试
        start_time = time.time()
        with mp.Pool() as pool:
            parallel_results = pool.map(self._cpu_task, [10000] * 100)
        parallel_time = time.time() - start_time

        speedup = serial_time / parallel_time if parallel_time > 0 else 1

        print(f"   并行加速比: {speedup:.2f}x")
        return speedup

    def _cpu_task(self, n):
        """CPU密集型任务"""
        return sum(i * i for i in range(n))
    
    def _test_gpu_performance(self) -> float:
        """GPU性能测试"""
        print("测试GPU计算性能...")
        
        try:
            import torch
            
            device = torch.device('cuda')
            
            start_time = time.time()
            
            # GPU矩阵运算测试
            size = 2000
            a = torch.rand(size, size, device=device)
            b = torch.rand(size, size, device=device)
            c = torch.mm(a, b)
            torch.cuda.synchronize()
            
            end_time = time.time()
            gpu_score = 2000 / (end_time - start_time)
            
            print(f"   GPU性能分数: {gpu_score:.1f}")
            return gpu_score
            
        except Exception as e:
            print(f"   GPU测试失败: {e}")
            return 0.0


def optimize_system_performance():
    """系统性能优化主函数"""
    print("🚀 启动系统性能优化...")
    
    optimizer = PerformanceOptimizer()
    benchmark = SystemBenchmark()
    
    # 运行基准测试
    benchmark_results = benchmark.run_benchmark()
    
    # 生成优化建议
    recommendations = []
    
    if benchmark_results.get('cpu_performance', 0) < 100:
        recommendations.append("建议升级CPU或优化算法复杂度")
    
    if benchmark_results.get('memory_performance', 0) < 1000:
        recommendations.append("建议增加内存或使用内存映射")
    
    if benchmark_results.get('parallel_performance', 1) < 2:
        recommendations.append("建议优化并行算法或检查CPU核心数")
    
    if optimizer.gpu_available and benchmark_results.get('gpu_performance', 0) < 500:
        recommendations.append("建议优化GPU算法或升级显卡")
    
    print("\n💡 性能优化建议:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    if not recommendations:
        print("   系统性能良好，无需特别优化")
    
    return optimizer, benchmark_results


if __name__ == "__main__":
    optimize_system_performance()
