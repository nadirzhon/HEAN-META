"""Clock and scheduler for time-based operations."""

import asyncio
import time
from collections.abc import Callable
from datetime import datetime, timedelta

from hean.logging import get_logger

logger = get_logger(__name__)


class Clock:
    """Monotonic clock and scheduler."""

    def __init__(self) -> None:
        """Initialize the clock."""
        self._start_time = time.monotonic()
        self._start_datetime = datetime.utcnow()
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []

    def now(self) -> datetime:
        """Get current UTC datetime."""
        return datetime.utcnow()

    def monotonic_time(self) -> float:
        """Get monotonic time since clock start."""
        return time.monotonic() - self._start_time

    def schedule_periodic(self, callback: Callable[[], None], interval: timedelta) -> None:
        """Schedule a periodic callback."""
        if not self._running:
            raise RuntimeError("Clock must be started before scheduling")

        async def periodic_task() -> None:
            while self._running:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    logger.error(f"Periodic task error: {e}", exc_info=True)
                await asyncio.sleep(interval.total_seconds())

        task = asyncio.create_task(periodic_task())
        self._tasks.append(task)

    def schedule_at(self, callback: Callable[[], None], when: datetime) -> None:
        """Schedule a callback at a specific time."""
        if not self._running:
            raise RuntimeError("Clock must be started before scheduling")

        async def scheduled_task() -> None:
            now = self.now()
            if when > now:
                delay = (when - now).total_seconds()
                await asyncio.sleep(delay)
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Scheduled task error: {e}", exc_info=True)

        task = asyncio.create_task(scheduled_task())
        self._tasks.append(task)

    async def start(self) -> None:
        """Start the clock."""
        self._running = True
        logger.info("Clock started")

    async def stop(self) -> None:
        """Stop the clock and cancel all tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Clock stopped")
