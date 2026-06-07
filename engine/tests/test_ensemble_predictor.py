"""基础集成预测器边界和融合语义测试。"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from happy8_analyzer import EnsemblePredictor, rank_number_scores  # noqa: E402


class FixedPredictor:
    """固定返回结果的假预测器。"""

    def __init__(self, numbers=None, scores=None, error=None):
        self.numbers = numbers or []
        self.scores = scores or []
        self.error = error

    def predict(self, data, count):
        if self.error:
            raise self.error
        return self.numbers, self.scores


def make_predictor(base_predictors):
    predictor = EnsemblePredictor(analyzer=None)
    predictor.base_predictors = base_predictors
    return predictor


def test_ensemble_count_boundaries_and_all_failed():
    """count边界和全失败场景应返回稳定结果。"""
    predictor = make_predictor({
        "frequency": FixedPredictor([1], [1.0]),
        "hot_cold": FixedPredictor([2], [1.0]),
    })

    assert predictor.predict([], count=-1) == ([], [])
    assert predictor.predict([], count=0) == ([], [])

    numbers, scores = predictor.predict([], count=90)
    assert numbers == [1, 2]
    assert scores == [1.0, pytest.approx(0.25 / 0.3)]

    failed = make_predictor({
        "frequency": FixedPredictor(error=RuntimeError("boom")),
        "hot_cold": FixedPredictor(error=RuntimeError("boom")),
    })
    assert failed.predict([], count=5) == ([], [])


def test_ensemble_weighted_fusion_and_stable_sorting():
    """基础集成应按固定权重加权并在同分时按号码升序。"""
    predictor = make_predictor({
        "frequency": FixedPredictor([1, 3], [1.0, 0.5]),
        "hot_cold": FixedPredictor([2], [1.0]),
        "missing": FixedPredictor([3], [0.5]),
        "markov": FixedPredictor([2], [0.2]),
    })

    numbers, scores = predictor.predict([], count=3)

    assert numbers == [1, 2, 3]
    assert scores == [
        1.0,
        pytest.approx(0.3 / 0.3),
        pytest.approx(0.25 / 0.3),
    ]


def test_ensemble_filters_invalid_candidates_and_scores():
    """非法号码和非有限分数不能进入基础集成输出。"""
    predictor = make_predictor({
        "frequency": FixedPredictor([0, 81, "bad", 1, 2], [1.0, 1.0, 1.0, float("nan"), 0.5]),
        "hot_cold": FixedPredictor([3], [float("inf")]),
        "missing": FixedPredictor([4], [1.0]),
    })

    numbers, scores = predictor.predict([], count=5)

    assert numbers == [4, 2]
    assert scores == [1.0, pytest.approx(0.75)]


def test_ensemble_returns_empty_when_all_candidates_are_invalid():
    """全部候选过滤后为空时，基础集成应返回空结果。"""
    predictor = make_predictor({
        "frequency": FixedPredictor([0, 81, "bad"], [1.0, 1.0, 1.0]),
        "hot_cold": FixedPredictor([1, 2], [float("nan"), float("inf")]),
    })

    assert predictor.predict([], count=5) == ([], [])


def test_rank_number_scores_filters_invalid_number_keys_and_scores():
    """公共排序函数应过滤非法号码键和非有限分数。"""
    numbers, scores = rank_number_scores(
        {
            0: 1.0,
            81: 1.0,
            "bad": 1.0,
            None: 1.0,
            float("nan"): 1.0,
            float("inf"): 1.0,
            float("-inf"): 1.0,
            7: 0.7,
            8: float("nan"),
            9: float("inf"),
            10: float("-inf"),
        },
        count=5,
    )

    assert numbers == [7]
    assert scores == [1.0]


def test_rank_number_scores_returns_empty_when_all_candidates_invalid():
    """公共排序函数在全部候选非法时应返回空结果。"""
    assert rank_number_scores(
        {
            0: 1.0,
            81: 1.0,
            "bad": 1.0,
            float("inf"): 1.0,
            1: float("nan"),
            2: float("inf"),
            3: float("-inf"),
        },
        count=5,
    ) == ([], [])
