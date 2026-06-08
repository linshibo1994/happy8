"""快乐8数据沙盘服务测试。"""

import asyncio
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.base import Base
from app.models.prediction import LotteryResult
from app.services.lottery_sandbox_service import LotterySandboxService
from app.services.lottery_service import LotteryService


@pytest.fixture
def lottery_session():
    """创建隔离的内存数据库会话。"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


def add_result(session, issue, day_offset, numbers):
    """写入一期开奖样本。"""
    numbers = sorted(numbers)
    draw_date = datetime(2026, 1, 1) + timedelta(days=day_offset)
    payload = LotteryService._build_result_payload(issue, draw_date, numbers)
    session.add(LotteryResult(**payload))
    session.commit()


def complete_numbers(seed_numbers):
    """补足快乐8每期20个号码，避免干扰指定规则。"""
    numbers = list(seed_numbers)
    candidate = 1
    while len(numbers) < 20:
        if candidate not in numbers:
            numbers.append(candidate)
        candidate += 5
    return sorted(set(numbers))[:20]


def test_filter_supports_two_three_four_consecutive(lottery_session):
    """沙盘过滤应识别两连、三连、四连。"""
    add_result(lottery_session, "2026001", 1, complete_numbers([11, 12]))
    add_result(lottery_session, "2026002", 2, complete_numbers([21, 22, 23]))
    add_result(lottery_session, "2026003", 3, complete_numbers([31, 32, 33, 34]))

    service = LotterySandboxService(lottery_session)

    two = asyncio.run(service.filter_results(rules=["two_consecutive"]))
    three = asyncio.run(service.filter_results(rules=["three_consecutive"]))
    four = asyncio.run(service.filter_results(rules=["four_consecutive"]))

    assert two["total"] == 3
    assert three["total"] == 2
    assert four["total"] == 1
    assert four["results"][0]["issue"] == "2026003"


def test_filter_supports_eight_zone_cross(lottery_session):
    """八区跨界应识别10/11、20/21等区界相邻号码。"""
    add_result(lottery_session, "2026001", 1, complete_numbers([10, 11]))
    add_result(lottery_session, "2026002", 2, complete_numbers([12, 13]))

    service = LotterySandboxService(lottery_session)
    data = asyncio.run(service.filter_results(rules=["eight_zone_cross"]))

    assert data["total"] == 1
    assert data["results"][0]["issue"] == "2026001"
    assert data["results"][0]["features"]["eight_zone_cross"] is True


def test_filter_supports_gap_and_consecutive_gap(lottery_session):
    """沙盘过滤应识别隔号，以及同期开奖同时存在连号和隔号。"""
    add_result(lottery_session, "2026001", 1, complete_numbers([41, 43]))
    add_result(lottery_session, "2026002", 2, complete_numbers([51, 52, 54]))

    service = LotterySandboxService(lottery_session)

    gap = asyncio.run(service.filter_results(rules=["gap_number"]))
    consecutive_gap = asyncio.run(service.filter_results(rules=["consecutive_gap_number"]))

    assert gap["total"] == 2
    assert consecutive_gap["total"] == 1
    assert consecutive_gap["results"][0]["issue"] == "2026002"


def test_interval_reports_insufficient_sample(lottery_session):
    """规则命中样本不足时应返回可用结果和明确提示。"""
    add_result(lottery_session, "2026001", 1, complete_numbers([61, 62, 63, 64]))

    service = LotterySandboxService(lottery_session)
    data = asyncio.run(service.analyze_intervals(rules=["four_consecutive"]))

    assert data["matched_count"] == 1
    assert data["sample_size"] == 0
    assert data["is_sample_sufficient"] is False
    assert data["message"] == "间隔样本不足，结论仅供参考"


def test_history_total_uses_real_count(lottery_session):
    """历史列表 total 应返回匹配条件真实总数，而不是当前页数量。"""
    for index in range(5):
        add_result(
            lottery_session,
            f"202600{index + 1}",
            index,
            complete_numbers([index + 11, index + 12]),
        )

    service = LotteryService(lottery_session)
    page = asyncio.run(service.get_historical_results(limit=2, offset=0))
    total = asyncio.run(service.count_historical_results())

    assert len(page) == 2
    assert total == 5


def test_external_fetch_falls_back_when_public_source_fails(monkeypatch, lottery_session):
    """外部公开来源失败时，应降级到既有同步来源而不是抛错。"""
    service = LotteryService(lottery_session)

    async def failed_public_source(latest_issue):
        return []

    async def fallback_source(latest_result):
        return [
            LotteryService._build_result_payload(
                "2026999",
                datetime(2026, 6, 8),
                list(range(1, 21)),
            )
        ]

    monkeypatch.setattr(service, "_fetch_from_public_sources", failed_public_source)
    monkeypatch.setattr(service, "_fetch_from_engine_crawler", fallback_source)

    data = asyncio.run(service._fetch_latest_from_source(None))

    assert len(data) == 1
    assert data[0]["issue"] == "2026999"
