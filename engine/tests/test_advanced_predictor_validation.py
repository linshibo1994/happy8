"""高级集成和深度学习预测器的时间验证语义测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from happy8_analyzer import (  # noqa: E402
    AdvancedEnsemblePredictor,
    GraphNeuralNetworkPredictor,
    LSTMPredictor,
    SuperPredictor,
    HighConfidencePredictor,
    TransformerPredictor,
    calculate_multilabel_jaccard_score,
    split_time_series_tail_validation,
)


def make_draw(issue: str, start: int) -> dict:
    """构造一期开奖结果。"""
    row = {"issue": issue, "date": "2026-01-01"}
    numbers = list(range(start, start + 20))
    for index, number in enumerate(numbers, start=1):
        row[f"num{index}"] = number
    return row


def make_history(periods: int = 16) -> pd.DataFrame:
    """构造号码区间随期号递增轮换的历史数据。"""
    starts = [1, 21, 41, 61]
    rows = []
    for offset in range(periods):
        issue = f"2026{offset + 1:03d}"
        rows.append(make_draw(issue, starts[offset % len(starts)]))
    return pd.DataFrame(rows)


def test_tail_validation_split_keeps_latest_samples_for_validation():
    """时间序列验证应固定使用后段样本，不能随机抽验证集。"""
    X = np.arange(10).reshape(10, 1)
    y = np.arange(10)

    X_train, X_val, y_train, y_val = split_time_series_tail_validation(X, y, validation_fraction=0.3)

    assert X_train.ravel().tolist() == list(range(7))
    assert X_val.ravel().tolist() == [7, 8, 9]
    assert y_train.tolist() == list(range(7))
    assert y_val.tolist() == [7, 8, 9]


def test_multilabel_jaccard_uses_top_k_prediction_sets():
    """多标签Jaccard应按每期top-k集合评分，而不是训练集准确率。"""
    y_true = np.zeros((1, 80))
    y_true[0, [0, 1, 2]] = 1
    scores = np.zeros((1, 80))
    scores[0, [0, 1, 9]] = [0.9, 0.8, 0.7]

    assert calculate_multilabel_jaccard_score(y_true, scores, top_k=3) == pytest.approx(0.5)


def test_lstm_training_and_prediction_windows_are_time_ordered():
    """LSTM样本和预测窗口应按期号正序，不包含未来目标样本。"""
    data = make_history(periods=14).sample(frac=1.0, random_state=7).reset_index(drop=True)
    predictor = LSTMPredictor(None)

    X, y = predictor._prepare_training_data(data, sequence_length=3)
    latest_sequence = predictor._build_latest_sequence(data, sequence_length=3)

    assert len(X) == 11
    assert y[0][:20].sum() == 0
    assert y[0][60:80].sum() == 20
    assert latest_sequence.shape == (1, 3, 22)
    assert latest_sequence[0, -1, 0] == pytest.approx(sum(range(21, 41)) / 20)


def test_transformer_uses_latest_observed_window_without_random_fallback():
    """Transformer预测窗口应来自最新可见历史，不使用随机序列兜底。"""
    data = make_history(periods=14).sample(frac=1.0, random_state=3).reset_index(drop=True)
    predictor = TransformerPredictor(None)

    sequences, targets = predictor._prepare_sequences(data, seq_length=3)
    latest_sequence = predictor._build_latest_sequence(data, seq_length=3)
    X_array, y_array = predictor._encode_training_arrays(sequences, targets)

    assert len(sequences) == 11
    assert targets[0] == list(range(61, 81))
    assert latest_sequence == list(range(61, 81)) + list(range(1, 21)) + list(range(21, 41))
    assert X_array.shape == (11, 60)
    assert y_array[0, 60:80].sum() == 20
    with pytest.raises(ValueError):
        predictor._build_latest_sequence(make_history(periods=2), seq_length=3)


def test_graph_training_weights_are_fit_without_validation_tail():
    """GNN权重训练应保留尾段样本做Jaccard验证。"""
    data = make_history(periods=18)
    predictor = GraphNeuralNetworkPredictor(None)
    samples = predictor._prepare_graph_training_samples(data, window_size=4)

    weights, validation_jaccard = predictor._train_gnn_model(samples)

    assert weights.shape == (80,)
    assert 0 <= validation_jaccard <= 1
    assert len(samples) == 14


def test_advanced_ensemble_prediction_features_use_latest_window():
    """高级集成预测下一期时应使用最新窗口，而不是最后一个训练样本。"""
    data = make_history(periods=8).sample(frac=1.0, random_state=11).reset_index(drop=True)
    predictor = AdvancedEnsemblePredictor(None)

    X, y = predictor._prepare_ensemble_data(data)
    prediction_features = predictor._build_prediction_features(data)

    assert X.shape == (3, 15)
    assert y.shape == (3, 80)
    assert prediction_features.shape == (1, 15)
    assert not np.array_equal(prediction_features, X[-1:].reshape(1, -1))
    assert prediction_features[0, -3] == pytest.approx(sum(range(61, 81)) / 20)


def test_advanced_ensemble_handles_none_and_invalid_draw_rows():
    """高级集成应兼容None输入，并跳过无效开奖记录。"""
    predictor = AdvancedEnsemblePredictor(None)
    valid_data = make_history(periods=12)
    invalid_data = valid_data.copy()
    invalid_data.loc[0, "num1"] = 0
    invalid_data.loc[1, "num2"] = 81
    invalid_data.loc[2, "num3"] = invalid_data.loc[2, "num4"]

    assert predictor.predict(None, count=5) == ([], [])

    X_valid, y_valid = predictor._prepare_ensemble_data(valid_data)
    X_invalid, y_invalid = predictor._prepare_ensemble_data(invalid_data)

    assert X_invalid.shape[0] < X_valid.shape[0]
    assert y_invalid.shape[1] == 80
    assert np.all((y_invalid == 0) | (y_invalid == 1))


def test_lstm_fallback_metadata_when_tensorflow_unavailable(monkeypatch):
    """LSTM降级到频率分析时应记录实际执行算法。"""
    import happy8_analyzer

    predictor = LSTMPredictor(None)
    data = make_history(periods=6)
    monkeypatch.setattr(happy8_analyzer, "TF_AVAILABLE", False)

    numbers, scores = predictor.predict(data, count=5)

    assert len(numbers) == 5
    assert len(scores) == 5
    assert predictor.last_fallback_algorithm == "frequency"
    assert predictor.last_fallback_reason == "tensorflow_unavailable"


def test_super_predictor_empty_data_and_duplicate_fallback_sources(monkeypatch):
    """综合融合器不能在空数据伪造候选，也不能重复融合同一降级信号。"""
    predictor = SuperPredictor(None)
    data = make_history(periods=6)

    assert predictor.predict(pd.DataFrame(), count=5) == ([], [])

    class FixedPredictor:
        def __init__(self, fallback_algorithm=None):
            self.last_fallback_algorithm = fallback_algorithm

        def predict(self, _data, count):
            return [1, 2, 3], [1.0, 0.5, 0.25]

    predictor.predictors = {
        "frequency": FixedPredictor(),
        "transformer": FixedPredictor("frequency"),
        "bayesian": FixedPredictor(),
    }
    predictor.predictor_groups = {
        "frequency": "statistical",
        "transformer": "deep_sequence",
        "bayesian": "bayesian",
    }

    numbers, scores = predictor.predict(data, count=3)

    assert numbers == [1, 2, 3]
    assert scores == [1.0, 0.5, 0.25]


def test_super_predictor_returns_empty_when_all_fused_candidates_invalid():
    """综合融合器在所有候选被过滤后应返回空结果。"""
    predictor = SuperPredictor(None)
    data = make_history(periods=6)

    class InvalidPredictor:
        last_fallback_algorithm = None

        def predict(self, _data, _count):
            return [0, 81, "bad", 1], [1.0, 1.0, 1.0, float("nan")]

    predictor.predictors = {"invalid": InvalidPredictor()}
    predictor.predictor_groups = {"invalid": "statistical"}

    assert predictor.predict(data, count=5) == ([], [])


def test_high_confidence_hard_rejects_invalid_or_empty_outputs():
    """质量门控应硬拒绝空、重复、越界和非数字候选。"""
    predictor = HighConfidencePredictor(None)
    data = make_history(periods=30)

    assert predictor.predict(None, count=5) == ([], [])
    assert predictor._validate_model_output([]) == 0.0
    assert predictor._validate_model_output([1, 1, 2, 3, 4]) == 0.0
    assert predictor._validate_model_output([81, 2, 3, 4, 5]) == 0.0
    assert predictor._validate_model_output(["bad", 2, 3, 4, 5]) == 0.0
    assert predictor._calculate_prediction_stability(data, [81, 2, 3, 4, 5]) == 0.0
    assert predictor._validate_statistical_consistency(data, [1, 1, 2, 3, 4]) == 0.0


def test_high_confidence_skips_invalid_history_rows():
    """质量评分内部统计只应使用有效历史行。"""
    predictor = HighConfidencePredictor(None)
    data = make_history(periods=12)
    invalid = data.copy()
    invalid.loc[0, "num1"] = 0
    invalid.loc[1, "num2"] = 81
    invalid.loc[2, "num3"] = invalid.loc[2, "num4"]

    assert predictor._calculate_pattern_strength(invalid) >= 0
    dimensions = predictor._evaluate_confidence_dimensions(invalid, list(range(1, 6)), [1.0] * 5)
    assert 0 <= dimensions["data_quality"] <= 1
    assert 0 <= dimensions["statistical_significance"] <= 1
