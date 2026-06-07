"""Monte Carlo和Bayesian预测器的算法语义测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from happy8_analyzer import BayesianPredictor, MonteCarloPredictor


def make_history(periods=6):
    """构造稳定偏向1-20号的历史数据。"""
    rows = []
    for issue_offset in range(periods):
        row = {
            "issue": f"20260{issue_offset:02d}",
            "date": "2026-01-01",
        }
        for index, num in enumerate(range(1, 21), start=1):
            row[f"num{index}"] = num
        rows.append(row)

    return pd.DataFrame(rows)


def assert_valid_prediction(numbers, scores, expected_count):
    assert len(numbers) == expected_count
    assert len(scores) == expected_count
    assert len(set(numbers)) == expected_count
    assert all(1 <= num <= 80 for num in numbers)
    assert all(0 <= score <= 1 for score in scores)


def test_monte_carlo_uses_reproducible_without_replacement_sampling():
    """相同随机种子应产生相同的无放回模拟结果。"""
    predictor = MonteCarloPredictor(analyzer=None)
    data = make_history()

    first_numbers, first_scores = predictor.predict(
        data,
        count=12,
        num_simulations=300,
        random_seed=2026,
    )
    second_numbers, second_scores = predictor.predict(
        data,
        count=12,
        num_simulations=300,
        random_seed=2026,
    )

    assert first_numbers == second_numbers
    assert first_scores == second_scores
    assert_valid_prediction(first_numbers, first_scores, 12)


def test_monte_carlo_official_uniform_model_matches_happy8_draw_shape():
    """官方均匀模型应模拟1-80中每期20个号码的无放回抽样。"""
    predictor = MonteCarloPredictor(analyzer=None)

    numbers, scores = predictor.predict(
        make_history(),
        count=80,
        sampling_model="official_uniform",
        num_simulations=500,
        random_seed=7,
    )

    assert sorted(numbers) == list(range(1, 81))
    assert_valid_prediction(numbers, scores, 80)
    assert sum(scores) == pytest.approx(20.0)


def test_bayesian_uses_dirichlet_posterior_mean_scores():
    """Bayesian预测器应按Dirichlet后验均值给高频号码排序。"""
    predictor = BayesianPredictor(analyzer=None)
    data = make_history(periods=5)

    posterior_alpha = predictor._build_dirichlet_posterior(data, prior_strength=1.0)
    posterior_mean = predictor._calculate_posterior_mean(posterior_alpha)
    numbers, scores = predictor.predict(data, count=10, prior_strength=1.0)

    assert predictor.prior_strength == 1.0
    assert np.all(posterior_alpha[:20] == 6.0)
    assert np.all(posterior_alpha[20:] == 1.0)
    assert posterior_mean.sum() == pytest.approx(1.0)
    assert numbers == list(range(1, 11))
    assert scores == sorted(scores, reverse=True)
    assert_valid_prediction(numbers, scores, 10)


@pytest.mark.parametrize("predictor_cls", [MonteCarloPredictor, BayesianPredictor])
@pytest.mark.parametrize("prior_strength", [0, -1, np.nan, np.inf, -np.inf])
def test_predictors_reject_invalid_prior_strength(predictor_cls, prior_strength):
    """Dirichlet平滑参数必须为正数。"""
    predictor = predictor_cls(analyzer=None)

    with pytest.raises(ValueError):
        predictor.predict(make_history(), prior_strength=prior_strength)


def test_bayesian_empty_data_uses_finite_symmetric_prior():
    """空数据下Bayesian只能返回有限的对称先验排序分。"""
    predictor = BayesianPredictor(analyzer=None)

    numbers, scores = predictor.predict(pd.DataFrame(), count=5, prior_strength=1.0)

    assert numbers == [1, 2, 3, 4, 5]
    assert scores == [0.25] * 5


def test_bayesian_none_data_uses_finite_symmetric_prior():
    """None数据下Bayesian也应使用有限对称先验。"""
    predictor = BayesianPredictor(analyzer=None)

    numbers, scores = predictor.predict(None, count=5, prior_strength=1.0)

    assert numbers == [1, 2, 3, 4, 5]
    assert scores == [0.25] * 5


def test_monte_carlo_none_data_is_reproducible():
    """None数据下蒙特卡洛应使用对称先验并保持可复现。"""
    predictor = MonteCarloPredictor(analyzer=None)

    first = predictor.predict(None, count=5, num_simulations=100, random_seed=11)
    second = predictor.predict(None, count=5, num_simulations=100, random_seed=11)

    assert first == second
    assert_valid_prediction(first[0], first[1], 5)


def test_bayesian_ignores_invalid_history_numbers():
    """非法历史号码不能污染Dirichlet后验计数。"""
    predictor = BayesianPredictor(analyzer=None)
    data = make_history(periods=1)
    data["num3"] = data["num3"].astype(object)
    data.loc[0, "num1"] = 0
    data.loc[0, "num2"] = 81
    data.loc[0, "num3"] = "bad"

    posterior_alpha = predictor._build_dirichlet_posterior(data, prior_strength=1.0)

    assert posterior_alpha[0] == 1.0
    assert posterior_alpha[1] == 1.0
    assert posterior_alpha[2] == 1.0
    assert posterior_alpha[3] == 2.0
