"""In-memory market data cache for ticks and klines."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
from typing import Any, Deque

from hean.core.timeframes import Candle
from hean.core.types import Tick
from hean.logging import get_logger

logger = get_logger(__name__)


def _ts_ms(value: datetime | None) -> int | None:
    """Convert datetime to unix ms."""
    if value is None:
        return None
    return int(value.timestamp() * 1000)


class MarketDataStore:
    """Lightweight ring-buffer store for market data snapshots."""

    def __init__(self, max_candles: int = 1000) -> None:
        self._candles: dict[tuple[str, str], Deque[dict[str, Any]]] = {}
        self._last_tick: dict[str, dict[str, Any]] = {}
        self._max_candles = max_candles
        self._lock = asyncio.Lock()

    async def record_tick(self, tick: Tick) -> dict[str, Any]:
        """Cache the most recent tick per symbol."""
        payload = {
            "symbol": tick.symbol,
            "price": tick.price,
            "volume": getattr(tick, "volume", 0.0),
            "bid": getattr(tick, "bid", None),
            "ask": getattr(tick, "ask", None),
            "ts": tick.timestamp.isoformat() if hasattr(tick, "timestamp") else datetime.utcnow().isoformat(),
            "ts_ms": _ts_ms(getattr(tick, "timestamp", None)) or int(datetime.utcnow().timestamp() * 1000),
        }
        async with self._lock:
            self._last_tick[tick.symbol] = payload
        return payload

    async def record_candle(self, timeframe: str, candle: Candle) -> dict[str, Any]:
        """Append a closed candle to the ring buffer."""
        data = {
            "symbol": candle.symbol,
            "timeframe": timeframe,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "open_time": candle.open_time.isoformat(),
            "close_time": candle.close_time.isoformat(),
            "open_time_ms": _ts_ms(candle.open_time),
            "close_time_ms": _ts_ms(candle.close_time),
        }
        key = (candle.symbol, timeframe)
        async with self._lock:
            buffer = self._candles.get(key)
            if buffer is None:
                buffer = deque(maxlen=self._max_candles)
                self._candles[key] = buffer
            buffer.append(data)
        return data

    async def latest_tick(self, symbol: str | None = None) -> dict[str, Any] | None:
        """Return last tick for symbol (or any symbol if not provided)."""
        async with self._lock:
            if symbol:
                return self._last_tick.get(symbol)
            if self._last_tick:
                # Return last inserted tick (iteration order stable in Py3.7+)
                last_symbol = next(reversed(self._last_tick))
                return self._last_tick.get(last_symbol)
            return None

    async def get_klines(self, symbol: str, timeframe: str = "1m", limit: int = 200) -> list[dict[str, Any]]:
        """Return up to `limit` klines for symbol/timeframe (oldest first)."""
        async with self._lock:
            candles = list(self._candles.get((symbol, timeframe), []))
        if not candles:
            return []
        if limit <= 0:
            return []
        return candles[-limit:]

    async def snapshot(self, symbol: str | None = None, timeframe: str = "1m", limit: int = 200) -> dict[str, Any]:
        """Return consolidated market snapshot for UI priming."""
        async with self._lock:
            target_symbol = symbol
            if not target_symbol:
                if self._last_tick:
                    target_symbol = next(reversed(self._last_tick))
                elif self._candles:
                    target_symbol = next(iter(self._candles.keys()))[0]
            target_symbol = target_symbol or "BTCUSDT"

        klines = await self.get_klines(target_symbol, timeframe, limit)
        last_tick = await self.latest_tick(target_symbol)
        return {
            "symbol": target_symbol,
            "timeframe": timeframe,
            "klines": klines,
            "last_tick": last_tick,
            "count": len(klines),
        }


# Singleton store shared across API modules
market_data_store = MarketDataStore()
