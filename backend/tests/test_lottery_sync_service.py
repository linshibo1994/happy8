"""开奖数据同步协调服务测试。"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from app.services.lottery_sync_service import LotterySyncService


@pytest.mark.asyncio
async def test_page_sync_uses_cooldown(monkeypatch):
    """页面连续刷新时应复用最近一次同步结果，避免频繁抓取外部数据源。"""
    service = LotterySyncService()
    now = datetime(2026, 6, 7, 10, 0, tzinfo=timezone(timedelta(hours=8)))
    monkeypatch.setattr(service, "_now", lambda: now)
    service._run_sync = AsyncMock(
        return_value={
            "updated_count": 1,
            "latest_result": {"issue": "2026001", "numbers": list(range(1, 21))},
            "synced_at": now.isoformat(),
            "skipped": False,
            "reason": "同步完成",
            "trigger": "page",
        }
    )

    first_summary = await service.sync_latest(trigger="page")
    second_summary = await service.sync_latest(trigger="page")

    assert first_summary["updated_count"] == 1
    assert second_summary["skipped"] is True
    assert service._run_sync.await_count == 1


@pytest.mark.asyncio
async def test_force_sync_ignores_page_cooldown(monkeypatch):
    """管理员或定时任务强制同步不应受页面冷却时间影响。"""
    service = LotterySyncService()
    now = datetime(2026, 6, 7, 10, 0, tzinfo=timezone(timedelta(hours=8)))
    monkeypatch.setattr(service, "_now", lambda: now)
    service._last_page_sync_at = now
    service._run_sync = AsyncMock(
        return_value={
            "updated_count": 0,
            "latest_result": None,
            "synced_at": now.isoformat(),
            "skipped": False,
            "reason": "同步完成",
            "trigger": "admin",
        }
    )

    summary = await service.sync_latest(force=True, trigger="admin")

    assert summary["skipped"] is False
    service._run_sync.assert_awaited_once()


def test_next_daily_run_at_rolls_to_tomorrow(monkeypatch):
    """当天 00:01 已过时，下次执行时间应顺延到明天 00:01。"""
    service = LotterySyncService()
    now = datetime(2026, 6, 7, 10, 0, tzinfo=timezone(timedelta(hours=8)))
    monkeypatch.setattr(service, "_now", lambda: now)

    next_run_at = service._next_daily_run_at()

    assert next_run_at.date().isoformat() == "2026-06-08"
    assert next_run_at.hour == 0
    assert next_run_at.minute == 1
