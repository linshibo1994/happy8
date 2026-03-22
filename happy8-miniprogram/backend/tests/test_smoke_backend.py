"""后端关键功能冒烟测试。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.schemas.prediction_schemas import (
    SUPPORTED_ALGORITHMS,
    BatchPredictionRequest,
    PredictionRequest,
)
from app.api.v1.lottery import serialize_lottery_result
from app.models.membership import MembershipLevel
from app.services.lottery_service import LotteryService
from app.services.membership_service import MembershipService
from app.services.prediction_service import PredictionService
from app.utils.algorithm_config_updater import update_algorithm_configs


def test_prediction_request_supports_all_algorithms():
    """单次预测请求应接受所有支持算法。"""
    for algorithm in SUPPORTED_ALGORITHMS:
        request = PredictionRequest(
            algorithm=algorithm,
            target_issue="2026001",
            periods=30,
            count=5,
        )
        assert request.algorithm == algorithm


def test_batch_prediction_request_supports_all_algorithms():
    """批量预测请求应接受所有支持算法。"""
    request = BatchPredictionRequest(
        algorithms=SUPPORTED_ALGORITHMS,
        target_issue="2026001",
        periods=30,
        count=5,
    )
    assert request.algorithms == SUPPORTED_ALGORITHMS


def test_prediction_request_rejects_unsupported_algorithm():
    """不支持算法应在模型校验阶段失败。"""
    with pytest.raises(ValueError):
        PredictionRequest(
            algorithm="not_exists",
            target_issue="2026001",
            periods=30,
            count=5,
        )


def test_serialize_lottery_result_with_dict_passthrough():
    """开奖结果序列化应兼容字典输入。"""
    payload = {"issue": "2026001", "numbers": [1, 2, 3]}
    assert serialize_lottery_result(payload) == payload


def test_serialize_lottery_result_with_orm_like_object():
    """开奖结果序列化应兼容ORM对象输入。"""
    row = SimpleNamespace(
        id=1,
        issue="2026001",
        draw_date=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:00"),
        numbers=[1, 2, 3],
        sum_value=6,
        odd_count=2,
        even_count=1,
        big_count=0,
        small_count=3,
        zone_distribution={"zone_1": 3},
    )
    data = serialize_lottery_result(row)
    assert data["issue"] == "2026001"
    assert data["numbers"] == [1, 2, 3]


def test_parse_draw_datetime_supports_date_and_datetime():
    """开奖时间解析应兼容仅日期与日期时间格式。"""
    dt1 = LotteryService._parse_draw_datetime("2026-03-22", "09:05:00")
    dt2 = LotteryService._parse_draw_datetime("2026/03/22", "")
    assert dt1.year == 2026 and dt1.hour == 9
    assert dt2.year == 2026 and dt2.month == 3 and dt2.day == 22


def test_parse_default_params_is_backward_compatible():
    """默认参数解析应兼容字符串、字典和非法数据。"""
    assert PredictionService._parse_default_params({"a": 1}) == {"a": 1}
    assert PredictionService._parse_default_params('{"a": 1}') == {"a": 1}
    assert PredictionService._parse_default_params("not-json") == {}
    assert PredictionService._parse_default_params(None) == {}


@pytest.mark.asyncio
async def test_check_permission_accepts_case_insensitive_level():
    """会员权限判断应支持大小写混用的等级字符串。"""
    service = MembershipService(db=None, cache=None)
    membership = SimpleNamespace(level=MembershipLevel.VIP)

    service.get_user_membership = AsyncMock(return_value=membership)
    service.check_membership_validity = AsyncMock(return_value=True)

    assert await service.check_permission(1, "vip")
    assert await service.check_permission(1, "VIP")
    assert await service.check_permission(1, "free")
    assert not await service.check_permission(1, "premium")


def test_readme_core_algorithms_are_fully_configured():
    """README声明的17种核心算法应全部出现在后端配置中。"""
    expected_core_algorithms = {
        "frequency",
        "hot_cold",
        "missing",
        "markov",
        "markov_2nd",
        "markov_3rd",
        "adaptive_markov",
        "transformer",
        "gnn",
        "monte_carlo",
        "clustering",
        "advanced_ensemble",
        "bayesian",
        "super_predictor",
        "high_confidence",
        "lstm",
        "ensemble",
    }
    configured = {item["algorithm_name"] for item in update_algorithm_configs()}
    assert expected_core_algorithms.issubset(configured)
    assert len(configured) >= 17


@pytest.mark.asyncio
async def test_execute_original_prediction_dispatches_all_supported_algorithms():
    """所有支持算法都应能进入对应执行分支并返回结果。"""

    class DummyAdapter:
        def __init__(self):
            self.calls = []

        async def frequency_analysis(self, *args, **kwargs):
            self.calls.append(("frequency_analysis", None))
            return {"predicted_numbers": [1, 2, 3], "confidence_score": 0.5}

        async def hot_cold_analysis(self, *args, **kwargs):
            self.calls.append(("hot_cold_analysis", None))
            return {"predicted_numbers": [1, 2, 3], "confidence_score": 0.5}

        async def missing_analysis(self, *args, **kwargs):
            self.calls.append(("missing_analysis", None))
            return {"predicted_numbers": [1, 2, 3], "confidence_score": 0.5}

        async def markov_analysis(self, *args, **kwargs):
            self.calls.append(("markov_analysis", None))
            return {"predicted_numbers": [1, 2, 3], "confidence_score": 0.5}

        async def ml_ensemble_analysis(self, *args, **kwargs):
            self.calls.append(("ml_ensemble_analysis", None))
            return {"predicted_numbers": [1, 2, 3], "confidence_score": 0.5}

        async def deep_learning_analysis(self, *args, **kwargs):
            self.calls.append(("deep_learning_analysis", None))
            return {"predicted_numbers": [1, 2, 3], "confidence_score": 0.5}

        async def super_predictor_analysis(self, *args, **kwargs):
            self.calls.append(("super_predictor_analysis", None))
            return {"predicted_numbers": [1, 2, 3], "confidence_score": 0.5}

        async def execute_original_algorithm(self, algorithm, *args, **kwargs):
            self.calls.append(("execute_original_algorithm", algorithm))
            return {"predicted_numbers": [1, 2, 3], "confidence_score": 0.5}

    service = PredictionService.__new__(PredictionService)
    service.algorithm_mapping = {
        "frequency": "frequency",
        "hot_cold": "hot_cold",
        "missing": "missing",
        "markov": "adaptive_markov",
        "ml_ensemble": "advanced_ensemble",
        "deep_learning": "transformer",
        "super_predictor": "super_predictor",
        "markov_basic": "markov",
        "markov_2nd": "markov_2nd",
        "markov_3rd": "markov_3rd",
        "adaptive_markov": "adaptive_markov",
        "lstm": "lstm",
        "transformer": "transformer",
        "gnn": "gnn",
        "bayesian": "bayesian",
        "monte_carlo": "monte_carlo",
        "clustering": "clustering",
        "ensemble_basic": "ensemble",
        "ensemble": "ensemble",
        "advanced_ensemble": "advanced_ensemble",
        "high_confidence": "high_confidence",
    }
    service.algorithm_adapter = DummyAdapter()
    service._get_historical_data = AsyncMock(
        return_value=[
            {"issue": f"20260{i:02d}", "date": "2026-01-01", "numbers": list(range(1, 21))}
            for i in range(1, 15)
        ]
    )

    expected_call_type = {
        "frequency": "frequency_analysis",
        "hot_cold": "hot_cold_analysis",
        "missing": "missing_analysis",
        "markov": "markov_analysis",
        "markov_basic": "markov_analysis",
        "ml_ensemble": "ml_ensemble_analysis",
        "ensemble_basic": "ml_ensemble_analysis",
        "deep_learning": "deep_learning_analysis",
        "super_predictor": "super_predictor_analysis",
    }

    for algorithm in service.algorithm_mapping:
        before = len(service.algorithm_adapter.calls)
        result = await service._execute_original_prediction(
            algorithm=algorithm,
            target_issue="2026999",
            periods=30,
            count=5,
            params={},
        )
        assert result["predicted_numbers"] == [1, 2, 3]
        call_type, call_algo = service.algorithm_adapter.calls[before]
        assert call_type == expected_call_type.get(algorithm, "execute_original_algorithm")
        if call_type == "execute_original_algorithm":
            assert call_algo == service.algorithm_mapping[algorithm]
