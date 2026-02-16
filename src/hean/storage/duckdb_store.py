"""DuckDB Storage - Persistent storage for ticks, physics, brain analyses.

Subscribes to EventBus events and persists them to DuckDB.
Background flush loop ensures writes are batched for performance.
"""

import asyncio
import time
from collections import deque
from pathlib import Path
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class DuckDBStore:
    """Persistent storage using DuckDB."""

    def __init__(
        self,
        bus: EventBus,
        db_path: str = "data/hean.duckdb",
        flush_interval: float = 5.0,
        batch_size: int = 500,
    ) -> None:
        self._bus = bus
        self._db_path = db_path
        self._flush_interval = flush_interval
        self._batch_size = batch_size

        # Write buffers
        self._tick_buffer: deque[dict[str, Any]] = deque(maxlen=10000)
        self._physics_buffer: deque[dict[str, Any]] = deque(maxlen=5000)
        self._brain_buffer: deque[dict[str, Any]] = deque(maxlen=1000)

        self._conn = None
        self._running = False
        self._flush_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the DuckDB store."""
        try:
            import duckdb
        except ImportError:
            logger.warning("DuckDB not installed, storage disabled. Install: pip install duckdb")
            return

        # Ensure data directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = duckdb.connect(self._db_path)
        self._create_tables()

        self._running = True
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        self._bus.subscribe(EventType.CONTEXT_UPDATE, self._handle_context_update)

        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info(f"DuckDB store started: {self._db_path}")

    async def stop(self) -> None:
        """Stop and flush remaining data."""
        self._running = False
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        self._bus.unsubscribe(EventType.CONTEXT_UPDATE, self._handle_context_update)

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        if self._conn:
            self._flush_ticks()
            self._flush_physics()
            self._flush_brain()
            self._conn.close()

        logger.info("DuckDB store stopped")

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                symbol VARCHAR,
                price DOUBLE,
                volume DOUBLE,
                timestamp DOUBLE,
                inserted_at DOUBLE DEFAULT epoch_ms(now()) / 1000
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS physics_snapshots (
                symbol VARCHAR,
                temperature DOUBLE,
                temperature_regime VARCHAR,
                entropy DOUBLE,
                entropy_state VARCHAR,
                phase VARCHAR,
                phase_confidence DOUBLE,
                szilard_profit DOUBLE,
                should_trade BOOLEAN,
                trade_reason VARCHAR,
                size_multiplier DOUBLE,
                timestamp DOUBLE,
                inserted_at DOUBLE DEFAULT epoch_ms(now()) / 1000
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS brain_analyses (
                timestamp VARCHAR,
                summary VARCHAR,
                market_regime VARCHAR,
                signal_action VARCHAR,
                signal_confidence DOUBLE,
                thoughts_json VARCHAR,
                inserted_at DOUBLE DEFAULT epoch_ms(now()) / 1000
            )
        """)

    async def _handle_tick(self, event: Event) -> None:
        """Buffer tick events."""
        tick = event.data.get("tick")
        if tick is None:
            return

        self._tick_buffer.append({
            "symbol": tick.symbol,
            "price": tick.price,
            "volume": tick.volume,
            "timestamp": tick.timestamp.timestamp() if hasattr(tick.timestamp, "timestamp") else time.time(),
        })

    async def _handle_context_update(self, event: Event) -> None:
        """Buffer physics/brain events."""
        data = event.data
        context_type = data.get("context_type", "")

        if context_type == "physics":
            physics = data.get("physics", {})
            self._physics_buffer.append(physics)

    def store_brain_analysis(self, analysis: dict[str, Any]) -> None:
        """Store a brain analysis (called by brain client)."""
        self._brain_buffer.append(analysis)

    async def _flush_loop(self) -> None:
        """Periodic flush loop."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                if not self._running or not self._conn:
                    break

                self._flush_ticks()
                self._flush_physics()
                self._flush_brain()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"DuckDB flush error: {e}")

    def _flush_ticks(self) -> None:
        """Flush tick buffer to DuckDB."""
        if not self._tick_buffer or not self._conn:
            return

        batch = []
        while self._tick_buffer and len(batch) < self._batch_size:
            batch.append(self._tick_buffer.popleft())

        if batch:
            try:
                self._conn.executemany(
                    "INSERT INTO ticks (symbol, price, volume, timestamp) VALUES (?, ?, ?, ?)",
                    [(t["symbol"], t["price"], t["volume"], t["timestamp"]) for t in batch],
                )
            except Exception as e:
                logger.warning(f"DuckDB tick flush error: {e}")

    def _flush_physics(self) -> None:
        """Flush physics buffer to DuckDB."""
        if not self._physics_buffer or not self._conn:
            return

        batch = []
        while self._physics_buffer and len(batch) < self._batch_size:
            batch.append(self._physics_buffer.popleft())

        if batch:
            try:
                self._conn.executemany(
                    """INSERT INTO physics_snapshots
                    (symbol, temperature, temperature_regime, entropy, entropy_state,
                     phase, phase_confidence, szilard_profit, should_trade, trade_reason,
                     size_multiplier, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            p.get("symbol", ""),
                            p.get("temperature", 0),
                            p.get("temperature_regime", ""),
                            p.get("entropy", 0),
                            p.get("entropy_state", ""),
                            p.get("phase", ""),
                            p.get("phase_confidence", 0),
                            p.get("szilard_profit", 0),
                            p.get("should_trade", False),
                            p.get("trade_reason", ""),
                            p.get("size_multiplier", 0.5),
                            p.get("timestamp", 0),
                        )
                        for p in batch
                    ],
                )
            except Exception as e:
                logger.warning(f"DuckDB physics flush error: {e}")

    def _flush_brain(self) -> None:
        """Flush brain buffer to DuckDB."""
        if not self._brain_buffer or not self._conn:
            return

        batch = []
        while self._brain_buffer and len(batch) < 100:
            batch.append(self._brain_buffer.popleft())

        if batch:
            try:
                import json

                self._conn.executemany(
                    """INSERT INTO brain_analyses
                    (timestamp, summary, market_regime, signal_action, signal_confidence, thoughts_json)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            a.get("timestamp", ""),
                            a.get("summary", ""),
                            a.get("market_regime", ""),
                            a.get("signal", {}).get("action", "") if a.get("signal") else "",
                            a.get("signal", {}).get("confidence", 0) if a.get("signal") else 0,
                            json.dumps(a.get("thoughts", [])),
                        )
                        for a in batch
                    ],
                )
            except Exception as e:
                logger.warning(f"DuckDB brain flush error: {e}")

    # Query methods

    def query_ticks(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 100,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        """Query stored ticks."""
        if not self._conn:
            return []

        try:
            if since:
                result = self._conn.execute(
                    "SELECT symbol, price, volume, timestamp FROM ticks WHERE symbol = ? AND timestamp > ? ORDER BY timestamp DESC LIMIT ?",
                    [symbol, since, limit],
                ).fetchall()
            else:
                result = self._conn.execute(
                    "SELECT symbol, price, volume, timestamp FROM ticks WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                    [symbol, limit],
                ).fetchall()

            return [
                {"symbol": r[0], "price": r[1], "volume": r[2], "timestamp": r[3]}
                for r in result
            ]
        except Exception as e:
            logger.warning(f"DuckDB tick query error: {e}")
            return []

    def query_physics(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query stored physics snapshots."""
        if not self._conn:
            return []

        try:
            result = self._conn.execute(
                """SELECT symbol, temperature, temperature_regime, entropy, entropy_state,
                   phase, phase_confidence, szilard_profit, should_trade, trade_reason,
                   size_multiplier, timestamp
                   FROM physics_snapshots WHERE symbol = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                [symbol, limit],
            ).fetchall()

            return [
                {
                    "symbol": r[0],
                    "temperature": r[1],
                    "temperature_regime": r[2],
                    "entropy": r[3],
                    "entropy_state": r[4],
                    "phase": r[5],
                    "phase_confidence": r[6],
                    "szilard_profit": r[7],
                    "should_trade": r[8],
                    "trade_reason": r[9],
                    "size_multiplier": r[10],
                    "timestamp": r[11],
                }
                for r in result
            ]
        except Exception as e:
            logger.warning(f"DuckDB physics query error: {e}")
            return []

    def query_brain(self, limit: int = 20) -> list[dict[str, Any]]:
        """Query stored brain analyses."""
        if not self._conn:
            return []

        try:
            import json

            result = self._conn.execute(
                """SELECT timestamp, summary, market_regime, signal_action,
                   signal_confidence, thoughts_json
                   FROM brain_analyses ORDER BY inserted_at DESC LIMIT ?""",
                [limit],
            ).fetchall()

            return [
                {
                    "timestamp": r[0],
                    "summary": r[1],
                    "market_regime": r[2],
                    "signal": {"action": r[3], "confidence": r[4]} if r[3] else None,
                    "thoughts": json.loads(r[5]) if r[5] else [],
                }
                for r in result
            ]
        except Exception as e:
            logger.warning(f"DuckDB brain query error: {e}")
            return []

    def get_ohlcv_candles(
        self,
        symbol: str,
        timeframe: str,  # e.g., '1m', '5m', '1h'
        start_ts: float | None = None,
        end_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Aggregates ticks into OHLCV candles.

        Args:
            symbol: The trading symbol (e.g., 'BTCUSDT').
            timeframe: The candle timeframe ('1m', '5m', '1h', etc.).
            start_ts: Optional start unix timestamp.
            end_ts: Optional end unix timestamp.

        Returns:
            A list of OHLCV dictionaries.
        """
        if not self._conn:
            logger.warning("DuckDB connection not available.")
            return []

        timeframe_map = {
            '1m': '1 minute', '5m': '5 minutes', '15m': '15 minutes',
            '1h': '1 hour', '4h': '4 hours', '1d': '1 day'
        }
        if timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(timeframe_map.keys())}")

        interval = timeframe_map[timeframe]

        # Build query
        base_query = f"""
            SELECT
                epoch(time_bucket(INTERVAL '{interval}', to_timestamp(timestamp))) AS candle_ts,
                first(price) AS open,
                max(price) AS high,
                min(price) AS low,
                last(price) AS close,
                sum(volume) AS volume
            FROM ticks
            WHERE symbol = ?
        """
        params = [symbol]

        if start_ts:
            base_query += " AND timestamp >= ?"
            params.append(start_ts)
        if end_ts:
            base_query += " AND timestamp < ?"
            params.append(end_ts)

        base_query += " GROUP BY candle_ts ORDER BY candle_ts"

        try:
            result = self._conn.execute(base_query, params).fetchall()
            return [
                {
                    "timestamp": int(r[0] * 1000),  # to ms for backtester
                    "open": r[1],
                    "high": r[2],
                    "low": r[3],
                    "close": r[4],
                    "volume": r[5],
                }
                for r in result
            ]
        except Exception as e:
            logger.error(f"Failed to get OHLCV candles: {e}")
            return []
