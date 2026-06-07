"""Happy8算法引擎回归测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ENGINE_ROOT = Path(__file__).resolve().parents[2] / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from happy8_analyzer import (  # noqa: E402
    BayesianPredictor,
    ClusteringPredictor,
    FrequencyPredictor,
    Happy8Analyzer,
    HotColdPredictor,
    MarkovPredictor,
    MissingPredictor,
    MonteCarloPredictor,
    analyze_pair_frequency_core,
    calculate_markov_state_scores,
    count_pair_frequencies,
)


def make_draw(issue: str, numbers: list[int]) -> dict:
    row = {"issue": issue, "date": "2026-01-01"}
    for idx, number in enumerate(numbers, 1):
        row[f"num{idx}"] = number
    return row


def make_dataframe(rows: list[tuple[str, list[int]]]) -> pd.DataFrame:
    return pd.DataFrame([make_draw(issue, numbers) for issue, numbers in rows])


def test_missing_and_hot_cold_use_newest_first_direction():
    """遗漏和冷热应按最新期在前计算当前状态。"""
    data = make_dataframe(
        [
            ("2026004", list(range(61, 81))),
            ("2026003", list(range(41, 61))),
            ("2026002", list(range(21, 41))),
            ("2026001", list(range(1, 21))),
        ]
    )

    hot_cold = HotColdPredictor(None)
    missing = MissingPredictor(None)

    assert hot_cold._get_hot_numbers(data.head(2))[:5] == [41, 42, 43, 44, 45]
    assert hot_cold._get_cold_numbers(data)[:5] == [1, 2, 3, 4, 5]

    missing_periods = missing._calculate_missing_periods(data)
    assert missing_periods[61] == 0
    assert missing_periods[1] == 3


def test_frequency_predictor_does_not_forge_empty_data_scores():
    """频率分析在空数据上不能伪造1-80全量候选。"""
    assert FrequencyPredictor(None).predict(pd.DataFrame(), 5) == ([], [])


def test_pair_frequency_uses_valid_periods_and_reports_skips():
    """数字对统计分母应使用有效期数，并报告无效行。"""
    valid_numbers = list(range(1, 21))
    invalid_numbers = list(range(1, 20)) + [19]
    data = make_dataframe(
        [
            ("2026002", valid_numbers),
            ("2026001", invalid_numbers),
        ]
    )

    pair_counts, valid_periods, skipped_periods = count_pair_frequencies(
        data, "2026001", "2026002", return_metadata=True
    )
    assert len(pair_counts) == 190
    assert valid_periods == 1
    assert skipped_periods == 1

    result = analyze_pair_frequency_core(data, "2026002", 2)
    assert result.valid_periods == 1
    assert result.skipped_periods == 1
    assert result.frequency_items[0].percentage == 100.0


def test_markov_scores_use_cross_period_state():
    """Markov评分应学习跨期状态，而不是同期开奖共现。"""
    data = make_dataframe(
        [
            ("2026005", [1, *range(22, 41)]),
            ("2026004", [1, *range(42, 61)]),
            ("2026003", [1, *range(22, 41)]),
            ("2026002", [1, *range(42, 61)]),
            ("2026001", [1, *range(22, 41)]),
        ]
    )

    scores = calculate_markov_state_scores(data, order=1)
    numbers, _ = MarkovPredictor(None).predict(data, 5)

    assert scores[1] > scores[80]
    assert numbers[0] == 1


def test_monte_carlo_is_reproducible_with_seed():
    """蒙特卡洛模拟应可通过随机种子复现。"""
    data = make_dataframe(
        [
            ("2026003", list(range(1, 21))),
            ("2026002", list(range(21, 41))),
            ("2026001", list(range(41, 61))),
        ]
    )
    predictor = MonteCarloPredictor(None)

    first = predictor.predict(data, 10, num_simulations=200, random_seed=7)
    second = predictor.predict(data, 10, num_simulations=200, random_seed=7)

    assert first == second


def test_clustering_uses_latest_scaled_feature_for_prediction():
    """聚类预测应使用最新期的标准化特征，不应把最旧期当最新。"""
    rows = []
    for idx in range(12):
        start = 1 + (idx % 4) * 20
        rows.append((f"2026{idx + 1:03d}", list(range(start, start + 20))))
    data = make_dataframe(list(reversed(rows)))

    predictor = ClusteringPredictor(None)
    features = predictor._extract_clustering_features(data)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    clustering = predictor._kmeans_clustering(features_scaled)
    numbers, scores = predictor._predict_from_clusters(clustering, features_scaled, data, 10)

    assert len(numbers) == 10
    assert len(scores) == 10
    assert len(set(numbers)) == 10


def test_clustering_handles_none_and_count_boundaries():
    """聚类预测应兼容None输入和count边界。"""
    predictor = ClusteringPredictor(None)
    data = make_dataframe(
        [(f"2026{i:03d}", list(range(1, 21))) for i in range(1, 13)]
    )

    assert predictor.predict(None, 5) == ([], [])
    assert predictor._generate_cluster_prediction([0], data, -1) == ([], [])

    numbers, scores = predictor._generate_cluster_prediction([0], data, 81)
    assert len(numbers) == 80
    assert len(scores) == 80
    assert len(set(numbers)) == 80
    assert all(1 <= num <= 80 for num in numbers)


def test_clustering_skips_invalid_draw_rows():
    """聚类特征和簇内频次不能被非法开奖记录污染。"""
    predictor = ClusteringPredictor(None)
    valid = make_draw("2026002", list(range(1, 21)))
    invalid = make_draw("2026001", [0, *range(2, 21)])
    duplicate = make_draw("2026000", [1, *range(1, 20)])
    data = pd.DataFrame([valid, invalid, duplicate])

    filtered = predictor._extract_clustering_features(
        pd.DataFrame([valid])
    )
    assert filtered.shape[0] == 1

    numbers, scores = predictor._generate_cluster_prediction([0, 1, 2], data, 5)
    assert numbers == [1, 2, 3, 4, 5]
    assert scores == [1.0] * 5


def test_bayesian_posterior_prefers_observed_numbers():
    """Dirichlet后验评分应反映历史观测次数。"""
    data = make_dataframe(
        [
            ("2026003", list(range(1, 21))),
            ("2026002", list(range(1, 21))),
            ("2026001", list(range(21, 41))),
        ]
    )

    numbers, scores = BayesianPredictor(None).predict(data, 5)

    assert numbers[:5] == [1, 2, 3, 4, 5]
    assert all(0 <= score <= 1 for score in scores)


def test_analyzer_prediction_window_excludes_target_issue(tmp_path):
    """历史目标期预测只能使用目标期之前的数据。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data = make_dataframe(
        [
            ("2026003", list(range(41, 61))),
            ("2026002", list(range(21, 41))),
            ("2026001", list(range(1, 21))),
        ]
    )
    data.to_csv(data_dir / "happy8_results.csv", index=False)

    analyzer = Happy8Analyzer(str(data_dir))
    window = analyzer._select_prediction_training_data("2026002", 10)

    assert window["issue"].tolist() == ["2026001"]
