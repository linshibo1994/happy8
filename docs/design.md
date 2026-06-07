# 批量预测对比功能 - Design Document

## Overview

### 设计理念
批量预测对比功能采用模块化设计，在现有Happy8预测系统基础上扩展，充分利用已有的17种预测算法和数据管理功能。设计遵循单一职责原则，将批量预测逻辑、结果统计、数据可视化和文件导出分离为独立模块。

### 设计目标
1. **复用性**: 最大化复用现有的预测算法和数据处理逻辑
2. **可扩展性**: 支持未来增加新的统计指标和导出格式
3. **性能优化**: 通过并行处理提高批量预测效率
4. **用户体验**: 提供直观的界面和实时的进度反馈
5. **数据完整性**: 确保所有预测结果和统计数据的准确性

### 核心价值
- 为用户提供算法稳定性和准确性的量化分析工具
- 支持多轮预测的统计分析，帮助用户理解算法性能
- 通过可视化和数据导出，满足深度分析需求

## Architecture

### 整体架构
批量预测对比功能采用三层架构设计：

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  Parameter UI   │  │  Progress UI     │  │  Results UI │ │
│  │  参数配置界面     │  │  进度显示界面      │  │  结果展示界面 │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                     Business Layer                          │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │ BatchPredictor  │  │  StatisticEngine │  │ ExportEngine│ │
│  │   批量预测器     │  │    统计分析引擎    │  │  导出引擎    │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                             │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │ Happy8Analyzer  │  │   Data Models    │  │ File System │ │
│  │  现有预测系统     │  │    数据模型       │  │  文件系统    │ │
│  └─────────────────┘  └──────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 核心模块设计

#### 1. BatchPredictor (批量预测器)
- **职责**: 执行多轮预测任务，管理预测流程
- **接口**: 提供批量预测启动、进度查询、结果获取接口
- **特性**: 支持并发预测、异常恢复、进度回调

#### 2. StatisticEngine (统计分析引擎)
- **职责**: 对批量预测结果进行统计分析
- **功能**: 计算命中率分布、平均值、方差、置信区间等
- **输出**: 生成结构化的统计报告

#### 3. ExportEngine (导出引擎)
- **职责**: 将预测结果和统计数据导出为各种格式
- **支持格式**: Excel、CSV、JSON
- **特性**: 支持自定义导出模板和字段选择

### 数据流设计
1. **用户输入** → 参数验证 → **批量预测器**
2. **批量预测器** → 调用现有预测算法 → **收集预测结果**
3. **预测结果** → 统计分析引擎 → **生成统计报告**
4. **统计报告** → 前端展示 + **导出引擎** → 文件下载

## Components and Interfaces

### 1. BatchPredictor 类设计

```python
class BatchPredictor:
    def __init__(self, analyzer: Happy8Analyzer):
        """初始化批量预测器"""
        pass

    def execute_batch_prediction(self, config: BatchConfig,
                                progress_callback: callable = None) -> BatchResult:
        """执行批量预测"""
        pass

    def get_progress(self) -> dict:
        """获取当前进度"""
        pass

    def cancel_prediction(self):
        """取消预测任务"""
        pass
```

### 2. StatisticEngine 类设计

```python
class StatisticEngine:
    def analyze_batch_results(self, results: List[PredictionResult]) -> StatisticReport:
        """分析批量预测结果"""
        pass

    def calculate_hit_rate_distribution(self, results: List[PredictionResult]) -> dict:
        """计算命中率分布"""
        pass

    def generate_summary_stats(self, results: List[PredictionResult]) -> dict:
        """生成汇总统计"""
        pass
```

### 3. ExportEngine 类设计

```python
class ExportEngine:
    def export_to_excel(self, batch_result: BatchResult, filepath: str):
        """导出到Excel文件"""
        pass

    def export_to_csv(self, batch_result: BatchResult, filepath: str):
        """导出到CSV文件"""
        pass

    def generate_download_link(self, batch_result: BatchResult, format: str) -> str:
        """生成下载链接"""
        pass
```

### 4. 前端UI组件设计

#### BatchPredictionPage (Streamlit页面)
```python
def batch_prediction_page():
    """批量预测对比页面"""
    # 参数配置区域
    render_parameter_config()

    # 预测执行区域
    render_prediction_execution()

    # 结果展示区域
    render_results_display()

    # 导出下载区域
    render_export_section()
```

### 5. 核心接口定义

#### BatchConfig (配置参数)
```python
@dataclass
class BatchConfig:
    target_issue: str        # 目标期号
    analysis_periods: int    # 分析期数
    prediction_method: str   # 预测方法
    number_count: int        # 生成号码数量
    comparison_times: int    # 对比次数
    max_parallel: int = 4    # 最大并发数
```

#### BatchResult (批量预测结果)
```python
@dataclass
class BatchResult:
    config: BatchConfig                    # 配置参数
    predictions: List[PredictionResult]    # 每轮预测结果
    statistics: StatisticReport           # 统计分析结果
    execution_time: float                 # 总执行时间
    created_at: datetime                  # 创建时间
```

#### StatisticReport (统计报告)
```python
@dataclass
class StatisticReport:
    avg_hit_rate: float              # 平均命中率
    max_hit_rate: float              # 最高命中率
    min_hit_rate: float              # 最低命中率
    std_deviation: float             # 标准差
    hit_rate_distribution: dict      # 命中率分布
    confidence_interval: tuple       # 置信区间
    success_count: int               # 成功预测次数
    total_count: int                 # 总预测次数
```

## Data Models

### 数据模型设计

#### 1. 扩展现有的PredictionResult
```python
@dataclass
class EnhancedPredictionResult:
    """增强的预测结果，包含批量预测所需的额外信息"""
    # 继承原有字段
    predicted_numbers: List[int]
    confidence_scores: List[float]
    execution_time: float

    # 新增字段
    round_number: int           # 预测轮次
    hit_numbers: List[int]      # 命中的号码
    hit_count: int             # 命中数量
    hit_rate: float            # 命中率
    actual_numbers: List[int]   # 实际开奖号码
    timestamp: datetime         # 预测时间戳
```

#### 2. 批量预测会话管理
```python
@dataclass
class BatchSession:
    """批量预测会话信息"""
    session_id: str
    config: BatchConfig
    status: str  # 'running', 'completed', 'failed', 'cancelled'
    current_round: int
    results: List[EnhancedPredictionResult]
    start_time: datetime
    end_time: Optional[datetime]
    error_message: Optional[str]
```

#### 3. 统计分析数据模型
```python
@dataclass
class HitRateStats:
    """命中率统计"""
    rate: float
    count: int
    percentage: float

@dataclass
class DetailedStats:
    """详细统计信息"""
    hit_rate_ranges: Dict[str, HitRateStats]  # 如：'0-10%', '10-20%'等
    quartiles: List[float]                    # 四分位数
    percentiles: Dict[int, float]             # 百分位数
    outliers: List[int]                       # 异常值的轮次
```

### 数据库设计（可选）

如果需要持久化存储批量预测历史，可以设计以下表结构：

#### batch_sessions 表
```sql
CREATE TABLE batch_sessions (
    session_id VARCHAR(50) PRIMARY KEY,
    target_issue VARCHAR(20),
    analysis_periods INT,
    prediction_method VARCHAR(50),
    number_count INT,
    comparison_times INT,
    status VARCHAR(20),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### batch_predictions 表
```sql
CREATE TABLE batch_predictions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES batch_sessions(session_id),
    round_number INT,
    predicted_numbers TEXT,  -- JSON格式
    hit_numbers TEXT,        -- JSON格式
    hit_count INT,
    hit_rate DECIMAL(5,2),
    execution_time DECIMAL(10,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Error Handling

### 异常处理策略

#### 1. 分层异常处理
```python
class BatchPredictionError(Exception):
    """批量预测基础异常类"""
    pass

class InvalidConfigError(BatchPredictionError):
    """无效配置参数异常"""
    pass

class PredictionExecutionError(BatchPredictionError):
    """预测执行异常"""
    pass

class DataAccessError(BatchPredictionError):
    """数据访问异常"""
    pass

class ExportError(BatchPredictionError):
    """导出文件异常"""
    pass
```

#### 2. 错误处理机制

##### 参数验证错误
- **触发条件**: 用户输入无效的期号、期数、次数等参数
- **处理方式**: 前端实时验证 + 后端二次验证
- **用户反馈**: 清晰的错误提示信息，指导用户正确输入

##### 预测执行错误
- **触发条件**: 算法执行失败、数据不足、网络异常等
- **处理方式**:
  - 单轮失败：记录错误，继续后续预测
  - 连续失败（3次）：暂停批量预测，提示用户
  - 网络异常：自动重试机制（最多3次）
- **用户反馈**: 显示具体失败轮次和错误原因

##### 资源限制错误
- **触发条件**: 内存不足、CPU过载、超时等
- **处理方式**:
  - 动态调整并发数
  - 设置合理的超时时间
  - 内存使用监控和垃圾回收
- **用户反馈**: 建议减少对比次数或选择较快的算法

#### 3. 恢复机制

##### 断点续传
- 保存每轮预测的中间结果
- 异常中断后可从上次成功的轮次继续
- 提供"继续未完成的预测"功能

##### 数据一致性保证
- 使用事务机制确保数据完整性
- 异常回滚机制，避免部分数据丢失
- 定期检查点保存

#### 4. 用户体验优化

##### 优雅降级
- 算法不可用时自动切换到备选算法
- 部分功能失败时不影响其他功能使用
- 提供基础版本的功能作为备选

##### 友好的错误提示
```python
ERROR_MESSAGES = {
    "invalid_issue": "期号格式不正确，请输入7位数字期号（如：2025091）",
    "invalid_periods": "分析期数应在10-500之间",
    "invalid_times": "对比次数应在1-100之间",
    "prediction_failed": "第{round}轮预测失败：{error_detail}",
    "network_error": "网络连接异常，正在重试...",
    "timeout_error": "预测超时，建议减少分析期数或选择更快的算法"
}
```

## Testing Strategy

### 测试策略设计

#### 1. 单元测试 (Unit Tests)

##### BatchPredictor 测试
```python
class TestBatchPredictor(unittest.TestCase):
    def test_execute_single_prediction(self):
        """测试单轮预测功能"""
        pass

    def test_execute_batch_prediction(self):
        """测试批量预测功能"""
        pass

    def test_progress_tracking(self):
        """测试进度追踪功能"""
        pass

    def test_cancellation(self):
        """测试取消预测功能"""
        pass
```

##### StatisticEngine 测试
```python
class TestStatisticEngine(unittest.TestCase):
    def test_hit_rate_calculation(self):
        """测试命中率计算准确性"""
        pass

    def test_statistics_generation(self):
        """测试统计指标生成"""
        pass

    def test_edge_cases(self):
        """测试边界情况（如全部命中、全部不命中）"""
        pass
```

##### ExportEngine 测试
```python
class TestExportEngine(unittest.TestCase):
    def test_excel_export(self):
        """测试Excel导出功能"""
        pass

    def test_csv_export(self):
        """测试CSV导出功能"""
        pass

    def test_file_format_validation(self):
        """测试导出文件格式验证"""
        pass
```

#### 2. 集成测试 (Integration Tests)

##### 完整流程测试
- 参数配置 → 批量预测 → 统计分析 → 结果展示 → 文件导出
- 测试数据流的完整性和正确性
- 验证各模块间的接口调用

##### 异常场景测试
- 网络异常时的处理
- 数据不完整时的处理
- 算法执行失败时的处理

#### 3. 性能测试 (Performance Tests)

##### 负载测试
```python
def test_batch_prediction_performance():
    """测试批量预测性能"""
    # 测试不同对比次数下的执行时间
    for times in [10, 20, 50, 100]:
        start_time = time.time()
        # 执行批量预测
        end_time = time.time()
        assert (end_time - start_time) < expected_time(times)
```

##### 内存使用测试
```python
def test_memory_usage():
    """测试内存使用情况"""
    # 监控批量预测过程中的内存使用
    # 确保内存使用在合理范围内
    pass
```

#### 4. 用户界面测试 (UI Tests)

##### 自动化UI测试（使用Selenium）
```python
class TestBatchPredictionUI:
    def test_parameter_input(self):
        """测试参数输入界面"""
        pass

    def test_prediction_execution(self):
        """测试预测执行流程"""
        pass

    def test_results_display(self):
        """测试结果展示界面"""
        pass

    def test_file_download(self):
        """测试文件下载功能"""
        pass
```

#### 5. 数据验证测试

##### 算法一致性测试
```python
def test_algorithm_consistency():
    """测试同一算法多次执行的一致性"""
    # 虽然结果可能不同，但应该在合理范围内
    pass

def test_hit_rate_accuracy():
    """测试命中率计算的准确性"""
    # 使用已知的测试数据验证命中率计算
    pass
```

#### 6. 端到端测试 (E2E Tests)

##### 典型用户场景测试
1. **场景1**: 新用户首次使用批量预测功能
2. **场景2**: 高级用户进行复杂的多算法对比
3. **场景3**: 用户中断预测任务后继续执行
4. **场景4**: 用户导出结果进行后续分析

#### 7. 测试数据准备

##### 测试数据集
```python
TEST_DATA = {
    "historical_issues": ["2025001", "2025002", "2025003", ...],
    "known_results": {
        "2025001": [1, 5, 12, 15, 23, 28, 34, 41, 46, 52, 58, 61, 65, 69, 72, 74, 76, 77, 79, 80],
        # 更多测试数据...
    },
    "algorithm_expected_ranges": {
        "frequency": {"min_hit_rate": 0.15, "max_hit_rate": 0.35},
        "markov_chain": {"min_hit_rate": 0.20, "max_hit_rate": 0.40},
        # 其他算法的期望范围...
    }
}
```

#### 8. 测试执行计划

##### 开发阶段测试
- 编写代码时进行单元测试
- 完成模块后进行集成测试
- 每日构建时运行自动化测试

##### 发布前测试
- 完整的端到端测试
- 性能和负载测试
- 用户验收测试

##### 生产环境监控
- 关键指标监控（响应时间、成功率等）
- 异常日志监控
- 用户反馈收集
