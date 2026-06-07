"""预测训练窗口回归测试。"""

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
ENGINE_ROOT = PROJECT_ROOT / "engine"
for path in (BACKEND_ROOT, ENGINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.models.base import Base
from app.models.prediction import LotteryResult
from app.services.prediction_service import PredictionService
from happy8_analyzer import Happy8Analyzer, PredictionResult


def _make_engine_frame(issues):
    rows = []
    for issue in issues:
        row = {"issue": issue, "date": "2026-01-01"}
        for index in range(1, 21):
            row[f"num{index}"] = index
        rows.append(row)
    return pd.DataFrame(rows)


def _make_engine_analyzer_with_data(data):
    captured = {}

    class FakeDataManager:
        def load_historical_data(self):
            return data

    class FakePredictionEngine:
        def predict(self, data, target_issue, count, method, **kwargs):
            captured["issues"] = data["issue"].tolist()
            return PredictionResult(
                target_issue=target_issue,
                analysis_periods=len(data),
                method=method,
                predicted_numbers=[1, 2, 3],
                confidence_scores=[1.0, 0.8, 0.6],
                generation_time=datetime.now(),
                execution_time=0.01,
                parameters=kwargs,
            )

    analyzer = Happy8Analyzer.__new__(Happy8Analyzer)
    analyzer.data_manager = FakeDataManager()
    analyzer.prediction_engine = FakePredictionEngine()
    analyzer.performance_monitor = SimpleNamespace(record_prediction=lambda *args: None)
    analyzer.historical_data = None
    return analyzer, captured


def test_engine_predict_excludes_target_and_later_issues_for_historical_target():
    """历史目标期预测只允许使用目标期之前的数据。"""
    data = _make_engine_frame(["2026001", "2026005", "2026003", "2026004", "2026002"])
    analyzer, captured = _make_engine_analyzer_with_data(data)

    result = analyzer.predict(
        target_issue="2026004",
        periods=2,
        count=3,
        method="frequency",
    )

    assert captured["issues"] == ["2026003", "2026002"]
    assert result.analysis_periods == 2


def test_engine_load_data_returns_newest_issue_first():
    """算法入口加载数据后应稳定保持最新期在前。"""
    data = _make_engine_frame(["2026001", "2026005", "2026003", "2026004", "2026002"])
    analyzer, _ = _make_engine_analyzer_with_data(data)

    loaded = analyzer.load_data(periods=3)

    assert loaded["issue"].tolist() == ["2026005", "2026004", "2026003"]


def test_engine_predict_uses_latest_history_for_future_target():
    """未来目标期预测使用最新历史窗口。"""
    data = _make_engine_frame(["2026001", "2026005", "2026003", "2026004", "2026002"])
    analyzer, captured = _make_engine_analyzer_with_data(data)

    analyzer.predict(
        target_issue="2026999",
        periods=3,
        count=3,
        method="frequency",
    )

    assert captured["issues"] == ["2026005", "2026004", "2026003"]


@pytest.fixture()
def lottery_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    draw_dates = {
        1: datetime(2026, 1, 5),
        2: datetime(2026, 1, 1),
        3: datetime(2026, 1, 4),
        4: datetime(2026, 1, 2),
        5: datetime(2026, 1, 3),
    }
    for index in range(1, 6):
        numbers = list(range(1, 21))
        session.add(
            LotteryResult(
                issue=f"2026{index:03d}",
                draw_date=draw_dates[index],
                numbers=numbers,
                sum_value=sum(numbers),
                odd_count=sum(1 for number in numbers if number % 2 == 1),
                even_count=sum(1 for number in numbers if number % 2 == 0),
                big_count=sum(1 for number in numbers if number >= 41),
                small_count=sum(1 for number in numbers if number <= 40),
            )
        )
    session.commit()

    try:
        yield session
    finally:
        session.close()


@pytest.mark.asyncio
async def test_backend_historical_data_excludes_target_and_later_issues(lottery_session):
    """后端历史目标期训练数据不能包含目标期及之后期号。"""
    service = PredictionService.__new__(PredictionService)
    service.db = lottery_session

    historical_data = await service._get_historical_data(2, "2026004")

    assert [item["issue"] for item in historical_data] == ["2026002", "2026003"]


@pytest.mark.asyncio
async def test_backend_historical_data_uses_latest_history_for_future_target(lottery_session):
    """后端未来目标期训练数据使用最新历史数据。"""
    service = PredictionService.__new__(PredictionService)
    service.db = lottery_session

    historical_data = await service._get_historical_data(2, "2026999")

    assert [item["issue"] for item in historical_data] == ["2026004", "2026005"]


@pytest.mark.asyncio
async def test_execute_original_prediction_passes_target_issue_to_history_query():
    """执行原始预测时必须将目标期传给历史查询，防止默认取最新期泄露。"""

    class FakeAdapter:
        async def frequency_analysis(self, historical_data, count, params):
            return {"predicted_numbers": [1, 2, 3], "confidence_score": 0.8}

    service = PredictionService.__new__(PredictionService)
    service.algorithm_mapping = {"frequency": "frequency"}
    service.algorithm_adapter = FakeAdapter()
    service._get_historical_data = AsyncMock(
        return_value=[
            {"issue": f"2026{index:03d}", "date": "2026-01-01", "numbers": list(range(1, 21))}
            for index in range(1, 12)
        ]
    )

    await service._execute_original_prediction(
        algorithm="frequency",
        target_issue="2026004",
        periods=10,
        count=3,
        params={},
    )

    service._get_historical_data.assert_awaited_once_with(10, "2026004")
