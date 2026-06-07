"""Happy8引擎预测器关键逻辑测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from happy8_analyzer import ClusteringPredictor, GraphNeuralNetworkPredictor


def make_draw_dataframe(issue_numbers):
    """构造符合引擎格式的开奖DataFrame。"""
    rows = []
    for issue, numbers in issue_numbers:
        row = {"issue": issue, "date": "2026-01-01"}
        for index, number in enumerate(numbers, start=1):
            row[f"num{index}"] = number
        rows.append(row)
    return pd.DataFrame(rows)


def make_rotating_draws(period_count):
    """生成每期20个唯一号码的可预测测试数据。"""
    issue_numbers = []
    for period in range(period_count):
        start = period * 3
        numbers = [((start + offset) % 80) + 1 for offset in range(20)]
        issue_numbers.append((f"2026{period + 1:03d}", numbers))
    return make_draw_dataframe(issue_numbers)


def test_clustering_predict_sorts_latest_issue_before_feature_extraction():
    """聚类预测应先按最新期排序，再提取特征。"""

    class CapturingClusteringPredictor(ClusteringPredictor):
        def __init__(self):
            super().__init__(analyzer=None)
            self.captured_issues = []

        def _extract_clustering_features(self, data):
            self.captured_issues = data["issue"].tolist()
            return np.arange(220, dtype=float).reshape(10, 22)

        def _kmeans_clustering(self, features_scaled):
            return {
                "kmeans": {
                    "labels": np.zeros(len(features_scaled), dtype=int),
                    "centers": np.array([features_scaled[0]]),
                    "score": 0.5,
                }
            }

        def _predict_from_clusters(self, clustering_results, features_scaled, data, count):
            return [1, 2, 3], [1.0, 0.8, 0.6]

    data = make_rotating_draws(10).sample(frac=1, random_state=7).reset_index(drop=True)
    predictor = CapturingClusteringPredictor()

    predictor.predict(data, count=3)

    assert predictor.captured_issues[0] == "2026010"
    assert predictor.captured_issues[-1] == "2026001"


def test_clustering_predicts_target_cluster_in_scaled_space():
    """聚类目标簇选择必须使用标准化后的特征空间。"""
    predictor = ClusteringPredictor(analyzer=None)
    data = make_draw_dataframe(
        [
            ("2026003", list(range(41, 61))),
            ("2026002", list(range(21, 41))),
            ("2026001", list(range(1, 21))),
        ]
    )
    features_scaled = np.array(
        [
            [10.0, 0.0],
            [0.0, 0.0],
            [0.1, 0.0],
        ]
    )
    clustering_results = {
        "kmeans": {
            "labels": np.array([1, 0, 0]),
            "centers": np.array([[0.0, 0.0], [10.0, 0.0]]),
            "score": 0.9,
        }
    }

    predicted_numbers, confidence_scores = predictor._predict_from_clusters(
        clustering_results, features_scaled, data, count=5
    )

    assert predicted_numbers == [41, 42, 43, 44, 45]
    assert confidence_scores == [1.0] * 5


def test_gnn_training_samples_use_history_window_to_next_period_target():
    """图传播训练样本应是历史窗口预测下一期的80维多标签目标。"""
    predictor = GraphNeuralNetworkPredictor(analyzer=None)
    data = make_rotating_draws(8).sort_values("issue", ascending=False).reset_index(drop=True)

    samples = predictor._prepare_graph_training_samples(data, window_size=3)

    assert len(samples) == 5
    first_scores, first_target = samples[0]
    assert first_scores.shape == (80,)
    assert first_target.shape == (80,)
    assert first_target.sum() == 20

    expected_first_target_numbers = make_rotating_draws(8).iloc[3][
        [f"num{i}" for i in range(1, 21)]
    ].tolist()
    for number in expected_first_target_numbers:
        assert first_target[number - 1] == 1.0


def test_gnn_graph_propagation_prediction_is_deterministic():
    """图传播预测不应包含随机扰动，同一输入应得到完全相同结果。"""
    predictor = GraphNeuralNetworkPredictor(analyzer=None)
    data = make_rotating_draws(24).sort_values("issue", ascending=False).reset_index(drop=True)

    samples = predictor._prepare_graph_training_samples(data, window_size=6)
    score_weights = predictor._train_gnn_model(samples)
    adjacency_matrix, node_features = predictor._build_number_graph(data)

    first_prediction = predictor._predict_with_gnn(score_weights, adjacency_matrix, node_features, 10)
    second_prediction = predictor._predict_with_gnn(score_weights, adjacency_matrix, node_features, 10)

    assert first_prediction == second_prediction
