"""开奖数据同步协调服务。"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.core.config import settings
from app.core.database import db_manager
from app.services.lottery_service import LotteryService

logger = logging.getLogger(__name__)


class LotterySyncService:
    """统一管理手动、页面触发和定时开奖同步。"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._scheduler_task: Optional[asyncio.Task] = None
        self._last_page_sync_at: Optional[datetime] = None
        self._last_summary: Dict[str, Any] = {
            "updated_count": 0,
            "latest_result": None,
            "synced_at": None,
            "skipped": True,
            "reason": "尚未同步",
        }

    async def sync_latest(self, *, force: bool = False, trigger: str = "manual") -> Dict[str, Any]:
        """同步最新开奖数据，页面触发时会按冷却时间跳过频繁请求。"""
        now = self._now()
        if not force and self._should_skip_page_sync(trigger, now):
            return {
                **self._last_summary,
                "updated_count": 0,
                "skipped": True,
                "reason": "页面刷新触发过于频繁，已复用最近一次同步结果",
                "trigger": trigger,
            }

        if trigger == "page":
            self._last_page_sync_at = now

        if self._lock.locked():
            return {
                **self._last_summary,
                "updated_count": 0,
                "skipped": True,
                "reason": "已有同步任务正在执行",
                "trigger": trigger,
            }

        async with self._lock:
            try:
                summary = await self._run_sync(trigger)
                self._last_summary = summary
                return summary
            except Exception as exc:
                logger.exception("开奖数据同步失败，触发来源: %s", trigger)
                self._last_summary = {
                    **self._last_summary,
                    "updated_count": 0,
                    "skipped": False,
                    "reason": f"同步失败: {exc}",
                    "trigger": trigger,
                    "synced_at": self._now().isoformat(),
                }
                raise

    async def start_scheduler(self) -> None:
        """启动每日定时同步任务。"""
        if not settings.LOTTERY_AUTO_SYNC_ENABLED:
            logger.info("开奖数据自动同步已关闭")
            return

        if self._scheduler_task and not self._scheduler_task.done():
            return

        self._scheduler_task = asyncio.create_task(self._daily_sync_loop())
        logger.info(
            "开奖数据自动同步已启动，每天 %02d:%02d 执行",
            settings.LOTTERY_DAILY_SYNC_HOUR,
            settings.LOTTERY_DAILY_SYNC_MINUTE,
        )

    async def stop_scheduler(self) -> None:
        """停止每日定时同步任务。"""
        if not self._scheduler_task:
            return

        self._scheduler_task.cancel()
        try:
            await self._scheduler_task
        except asyncio.CancelledError:
            logger.info("开奖数据自动同步任务已停止")
        finally:
            self._scheduler_task = None

    async def _run_sync(self, trigger: str) -> Dict[str, Any]:
        session_factory = db_manager.get_session_factory()
        db = session_factory()
        try:
            service = LotteryService(db)
            summary = await service.sync_latest_data_with_summary()
            return {
                **summary,
                "skipped": False,
                "reason": "同步完成",
                "trigger": trigger,
            }
        finally:
            db.close()

    async def _daily_sync_loop(self) -> None:
        while True:
            next_run_at = self._next_daily_run_at()
            wait_seconds = max((next_run_at - self._now()).total_seconds(), 1)
            logger.info("下次开奖数据自动同步时间: %s", next_run_at.isoformat())
            await asyncio.sleep(wait_seconds)

            try:
                await self.sync_latest(force=True, trigger="scheduler")
            except Exception:
                logger.exception("每日开奖数据自动同步执行失败")

    def _should_skip_page_sync(self, trigger: str, now: datetime) -> bool:
        if trigger != "page" or self._last_page_sync_at is None:
            return False

        cooldown = timedelta(seconds=settings.LOTTERY_PAGE_SYNC_COOLDOWN_SECONDS)
        return now - self._last_page_sync_at < cooldown

    def _next_daily_run_at(self) -> datetime:
        now = self._now()
        run_at = now.replace(
            hour=settings.LOTTERY_DAILY_SYNC_HOUR,
            minute=settings.LOTTERY_DAILY_SYNC_MINUTE,
            second=0,
            microsecond=0,
        )
        if run_at <= now:
            run_at += timedelta(days=1)
        return run_at

    @staticmethod
    def _now() -> datetime:
        try:
            tz = ZoneInfo(settings.APP_TIMEZONE)
        except ZoneInfoNotFoundError:
            tz = timezone(timedelta(hours=8))
        return datetime.now(tz)


lottery_sync_service = LotterySyncService()
