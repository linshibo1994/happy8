"""基础统计预测器确定性行为测试。"""

from datetime import date

import pandas as pd
import pytest

from app.services.happy8_algorithm_adapter import (
    Happy8AlgorithmAdapter,
    FrequencyPredictor,
    HotColdPredictor,
    MissingPredictor,
)


def make_engine_frame(draws):
    """构造原始引擎格式数据；调用方按最新期在前传入。"""
    rows = []
    for index, numbers in enumerate(draws):
        row = {
            "issue": f"2026{len(draws) - index:03d}",
            "date": "2026-01-01",
        }
        for position, number in enumerate(numbers, 1):
            row[f"num{position}"] = number
        rows.append(row)
    return pd.DataFrame(rows)


def make_draw(*numbers, filler_start=61):
    """生成20个不重复号码，便于测试只关心少量目标号码。"""
    selected = list(numbers)
    for number in range(filler_start, 81):
        if len(selected) >= 20:
            break
        if number not in selected:
            selected.append(number)
    return selected


def test_frequency_predictor_returns_empty_for_empty_data():
    """频率预测器在空数据下不应返回全量零置信候选。"""
    predictor = FrequencyPredictor(analyzer=None)

    numbers, scores = predictor.predict(pd.DataFrame(), count=5)

    assert numbers == []
    assert scores == []


def test_frequency_predictor_confidence_uses_absolute_frequency():
    """频率置信度应是实际出现率，而不是按最高频率二次归一化。"""
    data = make_engine_frame(
        [
            [1, *range(4, 23)],
            [1, *range(23, 42)],
            [2, *range(42, 61)],
            [3, *range(61, 80)],
        ]
    )
    predictor = FrequencyPredictor(analyzer=None)

    numbers, scores = predictor.predict(data, count=3)

    assert numbers[:3] == [1, 2, 3]
    assert scores[:3] == [0.5, 0.25, 0.25]


def test_hot_cold_predictor_uses_head_as_recent_window_and_deduplicates():
    """冷热预测应使用最新head窗口，且冷热组合结果不能重复。"""
    recent_draws = [
        list(range(1, 21))
        for _ in range(100)
    ]
    older_draws = [
        list(range(21, 41))
        for _ in range(100)
    ]
    data = make_engine_frame([*recent_draws, *older_draws])
    predictor = HotColdPredictor(analyzer=None)

    hot_numbers = predictor._get_hot_numbers(data.head(100))
    cold_numbers = predictor._get_cold_numbers(data)
    numbers, scores = predictor.predict(data, count=30)

    assert hot_numbers[:20] == list(range(1, 21))
    assert cold_numbers[:20] == list(range(41, 61))
    assert len(numbers) == 30
    assert len(numbers) == len(set(numbers))
    assert len(scores) == len(numbers)


def test_missing_predictor_current_missing_starts_from_newest_row():
    """当前遗漏应从第0行最新期开奖向历史方向累计。"""
    data = make_engine_frame(
        [
            make_draw(1, 2),
            make_draw(2, 3),
            make_draw(3, 4),
        ]
    )
    predictor = MissingPredictor(analyzer=None)

    missing = predictor._calculate_missing_periods(data)
    avg_cycles = predictor._calculate_average_cycles(data)

    assert missing[1] == 0
    assert missing[2] == 0
    assert missing[3] == 1
    assert missing[4] == 2
    assert avg_cycles[2] == 1
    assert avg_cycles[3] == 1
    assert avg_cycles[80] == 4


@pytest.mark.asyncio
async def test_adapter_missing_fallback_works_without_original_analyzer():
    """后端missing回退不应依赖原始分析器实例，也要按当前遗漏排序。"""
    adapter = Happy8AlgorithmAdapter.__new__(Happy8AlgorithmAdapter)
    adapter.original_analyzer = None
    adapter.data_manager = None
    historical_data = [
        {
            "issue": f"2026{index:03d}",
            "date": date(2026, 1, index).isoformat(),
            "numbers": draw,
        }
        for index, draw in enumerate(
            [
                make_draw(1, 2),
                make_draw(2, 3),
                make_draw(3, 4),
            ],
            1,
        )
    ]

    result = await adapter._create_missing_predictor(historical_data, count=5, params={})
    df = adapter.convert_db_to_happy8_format(historical_data)
    missing = adapter._calculate_current_missing_periods(df)

    assert missing[2] == 1
    assert missing[5] == 3
    assert result["predicted_numbers"][:2] == [2, 5]
    assert len(result["predicted_numbers"]) == 5
    assert len(result["predicted_numbers"]) == len(set(result["predicted_numbers"]))
    assert result["analysis_data"]["algorithm"] == "missing"


@pytest.mark.asyncio
async def test_adapter_reports_missing_available_without_original_analyzer():
    """missing有内置回退时，算法信息也应报告可用。"""
    adapter = Happy8AlgorithmAdapter.__new__(Happy8AlgorithmAdapter)
    adapter.original_analyzer = None
    adapter.data_manager = None

    result = await adapter.get_algorithm_info("missing")

    assert result["available"] is True
    assert result["predictor_class"] == "MissingPredictor(Fallback)"
