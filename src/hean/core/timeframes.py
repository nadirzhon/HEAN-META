"""Timeframe utilities and candle aggregation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Tick
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Candle:
    """OHLCV candle."""

    symbol: str
    timeframe: str  # e.g. "1m", "5m", "1h", "1d"
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def _floor_time(ts: datetime, delta: timedelta) -> datetime:
    """Floor a datetime to the nearest lower multiple of `delta` since epoch."""
    seconds = ts.timestamp()
    step = delta.total_seconds()
    floored = seconds - (seconds % step)
    return datetime.utcfromtimestamp(floored)


class CandleAggregator:
    """Aggregate ticks into candles on multiple timeframes.

    Listens to TICK events on the EventBus and emits CANDLE events with payload:
        {
            "timeframe": str,
            "candle": Candle,
        }
    """

    _TF_DELTAS: dict[str, timedelta] = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1),
    }

    def __init__(
        self,
        bus: EventBus,
        timeframes: Iterable[str] | None = None,
    ) -> None:
        self._bus = bus
        self._timeframes: list[str] = (
            list(timeframes) if timeframes else list(self._TF_DELTAS.keys())
        )
        self._candles: dict[tuple[str, str], Candle] = {}
        self._running = False

    async def start(self) -> None:
        """Start aggregation by subscribing to TICK events."""
        if self._running:
            return
        self._running = True
        self._bus.subscribe(EventType.TICK, self._on_tick)
        logger.info("[CandleAggregator] Started with timeframes=%s", self._timeframes)

    async def stop(self) -> None:
        """Stop aggregation and unsubscribe."""
        if not self._running:
            return
        self._running = False
        self._bus.unsubscribe(EventType.TICK, self._on_tick)
        logger.info("[CandleAggregator] Stopped")

    async def _on_tick(self, event: Event) -> None:
        """Handle incoming tick and update all configured timeframes."""
        if not self._running:
            return

        tick: Tick = event.data["tick"]
        ts = tick.timestamp

        for tf in self._timeframes:
            delta = self._TF_DELTAS[tf]
            bucket_start = _floor_time(ts, delta)
            bucket_end = bucket_start + delta
            key = (tick.symbol, tf)

            candle = self._candles.get(key)
            if candle is None or ts >= candle.close_time:
                # Close previous candle (if any) and emit
                if candle is not None:
                    await self._emit_candle(tf, candle)

                # Start new candle
                candle = Candle(
                    symbol=tick.symbol,
                    timeframe=tf,
                    open_time=bucket_start,
                    close_time=bucket_end,
                    open=tick.price,
                    high=tick.price,
                    low=tick.price,
                    close=tick.price,
                    volume=tick.volume,
                )
                self._candles[key] = candle
            else:
                # Update existing candle
                candle.high = max(candle.high, tick.price)
                candle.low = min(candle.low, tick.price)
                candle.close = tick.price
                candle.volume += tick.volume

    async def _emit_candle(self, timeframe: str, candle: Candle) -> None:
        """Publish a CANDLE event for the given candle."""
        await self._bus.publish(
            Event(
                event_type=EventType.CANDLE,
                data={
                    "timeframe": timeframe,
                    "candle": candle,
                },
            )
        )
