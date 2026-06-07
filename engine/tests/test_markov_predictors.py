"""Markov预测器跨期状态转移测试。"""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from happy8_analyzer import (
    AdaptiveMarkovPredictor,
    Markov2ndPredictor,
    Markov3rdPredictor,
    MarkovPredictor,
    calculate_markov_state_scores,
)


def make_draw(numbers, filler_numbers):
    """构造20个不重复号码。"""
    selected = list(numbers)
    for number in filler_numbers:
        if len(selected) >= 20:
            break
        if number not in selected:
            selected.append(number)
    return selected


def make_markov_frame(state_rows):
    """根据每期特别放入的号码构造引擎格式数据。"""
    rows = []
    for index, numbers in enumerate(state_rows, start=1):
        row = {"issue": f"2026{index:03d}", "date": "2026-01-01"}
        filler_numbers = range(31, 50) if index % 2 == 1 else range(50, 69)
        for position, number in enumerate(make_draw(numbers, filler_numbers), start=1):
            row[f"num{position}"] = number
        rows.append(row)
    return pd.DataFrame(rows)


def assert_valid_prediction(numbers, scores, expected_count):
    """验证预测结果数量、去重和置信度范围。"""
    assert len(numbers) == expected_count
    assert len(scores) == expected_count
    assert len(set(numbers)) == expected_count
    assert all(1 <= num <= 80 for num in numbers)
    assert all(0 <= score <= 1 for score in scores)


def test_markov_predictors_are_order_invariant_and_count_safe():
    """Markov预测应显式按期号正序建模，输入升序或降序结果一致。"""
    data = make_markov_frame([
        [1],
        [21],
        [1],
        [21],
        [1],
    ])
    descending_data = data.sort_values("issue", ascending=False).reset_index(drop=True)

    for predictor_cls in [MarkovPredictor, Markov2ndPredictor, Markov3rdPredictor]:
        predictor = predictor_cls(analyzer=None)
        ascending_numbers, ascending_scores = predictor.predict(data, count=90)
        descending_numbers, descending_scores = predictor.predict(descending_data, count=90)

        assert ascending_numbers == descending_numbers
        assert ascending_scores == descending_scores
        assert_valid_prediction(ascending_numbers, ascending_scores, 80)


def test_markov_uses_recent_cross_period_state_transition():
    """1阶Markov应使用最近一期状态预测下一期，而不是只看同期开奖共现。"""
    data = make_markov_frame([
        [1],
        [21],
        [1],
        [21],
        [1],
    ])

    scores = calculate_markov_state_scores(data, order=1)
    numbers, _ = MarkovPredictor(analyzer=None).predict(data, count=5)

    assert scores[21] > scores[1]
    assert numbers[0] == 21


def test_high_order_markov_falls_back_to_lower_order_transition():
    """高阶当前状态未出现时，应回退到低阶条件转移而不是直接用全局频率。"""
    data = make_markov_frame([
        [1],
        [21],
        [1],
        [21],
        [1],
    ])

    first_order_scores = calculate_markov_state_scores(data, order=1)
    third_order_scores = calculate_markov_state_scores(data, order=3)

    assert first_order_scores[21] > first_order_scores[1]
    assert third_order_scores[21] > third_order_scores[1]


def test_high_order_markov_falls_back_when_history_too_short_for_requested_order():
    """请求阶数超过历史长度时，应使用可训练低阶Markov而不是频率基线。"""
    data = make_markov_frame([
        [1],
        [21],
        [1],
    ])

    first_order_scores = calculate_markov_state_scores(data, order=1)
    third_order_scores = calculate_markov_state_scores(data, order=3)

    assert first_order_scores[21] > first_order_scores[1]
    assert third_order_scores[21] > third_order_scores[1]


def test_adaptive_markov_fuses_full_per_number_scores():
    """自适应Markov应融合完整80个单号评分并返回唯一号码。"""
    data = make_markov_frame([
        [1],
        [21],
        [1],
        [21],
        [1],
        [21],
        [1],
    ])
    predictor = AdaptiveMarkovPredictor(analyzer=None)

    numbers, scores = predictor.predict(data, count=30)

    assert_valid_prediction(numbers, scores, 30)
    assert 21 in numbers[:5]
