"""数字对频率分析专项测试。"""

import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.happy8_analyzer import (
    TOTAL_PAIR_TYPES,
    PairFrequencyAnalyzer,
    analyze_pair_frequency_core,
    extract_number_pairs,
)


def _row(issue, numbers):
    data = {"issue": str(issue)}
    for index, number in enumerate(numbers, 1):
        data[f"num{index}"] = number
    return data


def _draw(start, end):
    return list(range(start, end + 1))


class MutableDataManager:
    """测试用可变数据管理器。"""

    def __init__(self, data):
        self.data = data

    def load_historical_data(self):
        return self.data


def test_extract_number_pairs_example_uses_valid_draw():
    """示例中的20个号码应生成190个数字对。"""
    pairs = extract_number_pairs(list(range(1, 21)))

    assert len(pairs) == 190
    assert pairs[0] == (1, 2)
    assert pairs[-1] == (19, 20)


def test_pair_frequency_uses_valid_periods_and_skips_invalid_rows():
    """分母应使用有效行数，无效行只计数跳过。"""
    data = pd.DataFrame(
        [
            _row("2026001", _draw(1, 20)),
            _row("2026002", _draw(2, 21)),
            _row("2026003", [1] * 20),
            _row("2026004", _draw(1, 19) + [81]),
        ]
    )

    result = analyze_pair_frequency_core(data, "2026004", 4)

    assert result.actual_periods == 2
    assert result.valid_periods == 2
    assert result.skipped_periods == 2
    assert result.skipped_invalid_rows == 2
    assert result.total_pairs == TOTAL_PAIR_TYPES
    assert len(result.frequency_items) == TOTAL_PAIR_TYPES

    pair_1_2 = result.find_pair(1, 2)
    pair_20_21 = result.find_pair(20, 21)
    pair_79_80 = result.find_pair(79, 80)

    assert pair_1_2 is not None
    assert pair_1_2.count == 1
    assert pair_1_2.percentage == 50.0
    assert pair_20_21 is not None
    assert pair_20_21.count == 1
    assert pair_20_21.percentage == 50.0
    assert pair_79_80 is not None
    assert pair_79_80.count == 0
    assert pair_79_80.percentage == 0.0


def test_pair_frequency_cache_key_changes_when_range_data_changes():
    """历史数据内容变化后，相同参数不应复用旧缓存。"""
    first_data = pd.DataFrame(
        [
            _row("2026001", _draw(1, 20)),
            _row("2026002", _draw(2, 21)),
        ]
    )
    second_data = pd.DataFrame(
        [
            _row("2026001", _draw(1, 20)),
            _row("2026002", _draw(41, 60)),
        ]
    )
    data_manager = MutableDataManager(first_data)
    analyzer = PairFrequencyAnalyzer(
        data_manager=data_manager,
        cache_size=10,
        enable_parallel=False,
    )

    first_result = analyzer.analyze_pair_frequency("2026002", 2, use_cache=True)
    data_manager.data = second_data
    second_result = analyzer.analyze_pair_frequency("2026002", 2, use_cache=True)

    assert first_result.find_pair(2, 21).count == 1
    assert second_result.find_pair(2, 21).count == 0
    assert second_result.find_pair(41, 60).count == 1

    stats = analyzer.get_cache_info()
    assert stats["cache_size"] == 2
    assert stats["hit_count"] == 0


def test_pair_frequency_cached_result_is_used_for_identical_data():
    """数据未变化时，相同参数应命中缓存。"""
    data = pd.DataFrame(
        [
            _row("2026001", _draw(1, 20)),
            _row("2026002", _draw(2, 21)),
        ]
    )
    analyzer = PairFrequencyAnalyzer(
        data_manager=SimpleNamespace(load_historical_data=lambda: data),
        cache_size=10,
        enable_parallel=False,
    )

    first_result = analyzer.analyze_pair_frequency("2026002", 2, use_cache=True)
    second_result = analyzer.analyze_pair_frequency("2026002", 2, use_cache=True)

    assert second_result is first_result
    assert analyzer.get_cache_info()["hit_count"] == 1
