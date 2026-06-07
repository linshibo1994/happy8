"""深度学习与高级集成的时间窗口语义测试。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from happy8_analyzer import (  # noqa: E402
    AdvancedEnsemblePredictor,
    LSTMPredictor,
    TransformerPredictor,
)


def _row(issue, start):
    row = {"issue": issue, "date": "2026-01-01"}
    for index, number in enumerate(range(start, start + 20), start=1):
        row[f"num{index}"] = number
    return row


def make_frame(starts):
    """构造可按特征均值识别时间顺序的历史数据。"""
    rows = [
        _row(f"2026{index:03d}", start)
        for index, start in enumerate(starts, start=1)
    ]
    return pd.DataFrame(rows)


def test_lstm_training_and_prediction_windows_use_oldest_to_newest_order():
    """LSTM训练样本和预测窗口都应按时间升序组织。"""
    data = make_frame([1, 11, 21, 31, 41, 51, 61, 1, 11, 21, 31, 41])
    shuffled = data.sort_values("issue", ascending=False).reset_index(drop=True)
    predictor = LSTMPredictor(analyzer=None)

    X, y = predictor._prepare_training_data(shuffled, sequence_length=3)
    latest_sequence = predictor._build_latest_sequence(shuffled, sequence_length=3)

    assert X[0, :, 0].tolist() == [10.5, 20.5, 30.5]
    assert np.flatnonzero(y[0])[:3].tolist() == [30, 31, 32]
    assert latest_sequence[0, :, 0].tolist() == [30.5, 40.5, 50.5]


def test_transformer_training_and_prediction_windows_use_oldest_to_newest_order():
    """Transformer训练和预测输入不能把最新期当作窗口开头。"""
    data = make_frame([1, 11, 21, 31, 41, 51, 61, 1, 11, 21, 31, 41])
    shuffled = data.sort_values("issue", ascending=False).reset_index(drop=True)
    predictor = TransformerPredictor(analyzer=None)

    sequences, targets = predictor._prepare_sequences(shuffled)
    latest_sequence = predictor._build_latest_sequence(shuffled)

    assert sequences[0][:5] == [1, 2, 3, 4, 5]
    assert targets[0][:5] == [31, 32, 33, 34, 35]
    assert latest_sequence[:5] == [21, 22, 23, 24, 25]
    assert latest_sequence[-5:] == [56, 57, 58, 59, 60]


def test_advanced_ensemble_predicts_from_latest_history_window():
    """高级集成预测特征应来自最新5期历史，而不是最后一个训练样本。"""
    data = make_frame([1, 11, 21, 31, 41, 51, 61])
    shuffled = data.sort_values("issue", ascending=False).reset_index(drop=True)
    predictor = AdvancedEnsemblePredictor(analyzer=None)

    X, y = predictor._prepare_ensemble_data(shuffled)
    prediction_features = predictor._build_prediction_features(shuffled)

    assert X[-1].reshape(1, -1).tolist() != prediction_features.tolist()
    assert prediction_features[0, 0] == 30.5
    assert prediction_features[0, -3] == 70.5
    assert np.flatnonzero(y[-1])[:3].tolist() == [60, 61, 62]


def test_advanced_ensemble_extracts_positive_class_probabilities():
    """多输出模型单类概率不能被误读成正类概率。"""

    class FakeModel:
        classes_ = [
            np.array([0]),
            np.array([0, 1]),
            *[np.array([0]) for _ in range(78)],
        ]

        def predict_proba(self, _sample):
            return [
                np.array([[1.0]]),
                np.array([[0.25, 0.75]]),
                *[np.array([[1.0]]) for _ in range(78)],
            ]

    predictor = AdvancedEnsemblePredictor(analyzer=None)
    scores = predictor._extract_positive_probabilities(
        FakeModel(),
        np.zeros((1, 15)),
    )

    assert scores[0] == 0.0
    assert scores[1] == 0.75
    assert scores[2] == 0.0
