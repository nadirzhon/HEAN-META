"""Trading metrics aggregation service for funnel observability."""

from __future__ import annotations

import asyncio
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)


class TradingMetrics:
    """In-memory metrics aggregator with ring buffers for trading funnel observability."""

    def __init__(self, window_seconds: int = 300):
        """Initialize metrics service.
        
        Args:
            window_seconds: Time window for per-minute aggregations (default 5 minutes)
        """
        self._window_seconds = window_seconds
        self._lock = asyncio.Lock()
        
        # Session counters (since start)
        self._session_counters = {
            "signals_total": 0,
            "decisions_create": 0,
            "decisions_skip": 0,
            "decisions_block": 0,
            "orders_created": 0,
            "orders_filled": 0,
            "orders_canceled": 0,
            "orders_rejected": 0,
            "orders_open": 0,
            "positions_open": 0,
            "positions_closed": 0,
            "pnl_unrealized": 0.0,
            "pnl_realized": 0.0,
            "equity": 0.0,
        }
        
        # Per-minute ring buffers (timestamp -> count)
        self._minute_buffers: dict[str, deque[tuple[float, int | float]]] = {
            "signals_total": deque(maxlen=window_seconds),
            "decisions_create": deque(maxlen=window_seconds),
            "decisions_skip": deque(maxlen=window_seconds),
            "decisions_block": deque(maxlen=window_seconds),
            "orders_created": deque(maxlen=window_seconds),
            "orders_filled": deque(maxlen=window_seconds),
            "orders_canceled": deque(maxlen=window_seconds),
            "orders_rejected": deque(maxlen=window_seconds),
            "positions_open": deque(maxlen=window_seconds),
            "positions_closed": deque(maxlen=window_seconds),
        }
        
        # Breakdown by symbol and strategy
        self._symbol_breakdown: dict[str, Counter] = defaultdict(Counter)
        self._strategy_breakdown: dict[str, Counter] = defaultdict(Counter)
        
        # Reason codes tracking
        self._reason_codes: deque[tuple[float, str]] = deque(maxlen=1000)
        
        # Timestamps
        self._last_signal_ts: float | None = None
        self._last_order_ts: float | None = None
        self._last_fill_ts: float | None = None
        
        # Engine state
        self._engine_state: str = "STOPPED"
        self._mode: str = "paper"
        
        # Current state snapshots
        self._active_orders_count: int = 0
        self._active_positions_count: int = 0
        
        self._start_time = time.time()

    async def record_signal(self, symbol: str, strategy_id: str) -> None:
        """Record a signal detection."""
        async with self._lock:
            now = time.time()
            self._session_counters["signals_total"] += 1
            self._minute_buffers["signals_total"].append((now, 1))
            self._last_signal_ts = now
            self._symbol_breakdown["signals"][symbol] += 1
            self._strategy_breakdown["signals"][strategy_id] += 1

    async def record_decision(
        self, decision: str, reason_code: str | None = None, symbol: str | None = None, strategy_id: str | None = None
    ) -> None:
        """Record an order decision."""
        async with self._lock:
            now = time.time()
            decision_upper = decision.upper()
            if decision_upper in ("CREATE", "ACCEPTED", "ENTRY"):
                self._session_counters["decisions_create"] += 1
                self._minute_buffers["decisions_create"].append((now, 1))
            elif decision_upper in ("SKIP", "HOLD"):
                self._session_counters["decisions_skip"] += 1
                self._minute_buffers["decisions_skip"].append((now, 1))
            elif decision_upper in ("BLOCK", "REJECT", "SUPPRESSED"):
                self._session_counters["decisions_block"] += 1
                self._minute_buffers["decisions_block"].append((now, 1))
            
            if reason_code:
                self._reason_codes.append((now, reason_code))
            
            if symbol:
                self._symbol_breakdown["decisions"][symbol] += 1
            if strategy_id:
                self._strategy_breakdown["decisions"][strategy_id] += 1

    async def record_order(
        self, event_type: str, symbol: str | None = None, strategy_id: str | None = None
    ) -> None:
        """Record an order event."""
        async with self._lock:
            now = time.time()
            event_upper = event_type.upper()
            if event_upper in ("ORDER_CREATED", "ORDER_PLACED", "ORDER_SUBMITTED"):
                self._session_counters["orders_created"] += 1
                self._minute_buffers["orders_created"].append((now, 1))
                self._last_order_ts = now
            elif event_upper in ("ORDER_FILLED", "FILLED"):
                self._session_counters["orders_filled"] += 1
                self._minute_buffers["orders_filled"].append((now, 1))
                self._last_fill_ts = now
            elif event_upper in ("ORDER_CANCELLED", "CANCELED"):
                self._session_counters["orders_canceled"] += 1
                self._minute_buffers["orders_canceled"].append((now, 1))
            elif event_upper in ("ORDER_REJECTED", "REJECTED"):
                self._session_counters["orders_rejected"] += 1
                self._minute_buffers["orders_rejected"].append((now, 1))
            
            if symbol:
                self._symbol_breakdown["orders"][symbol] += 1
            if strategy_id:
                self._strategy_breakdown["orders"][strategy_id] += 1

    async def record_position(
        self, event_type: str, symbol: str | None = None, strategy_id: str | None = None
    ) -> None:
        """Record a position event."""
        async with self._lock:
            now = time.time()
            event_upper = event_type.upper()
            if event_upper in ("POSITION_OPENED", "OPEN"):
                self._session_counters["positions_open"] += 1
                self._minute_buffers["positions_open"].append((now, 1))
            elif event_upper in ("POSITION_CLOSED", "CLOSE"):
                self._session_counters["positions_closed"] += 1
                self._minute_buffers["positions_closed"].append((now, 1))
            
            if symbol:
                self._symbol_breakdown["positions"][symbol] += 1
            if strategy_id:
                self._strategy_breakdown["positions"][strategy_id] += 1

    async def update_pnl(self, unrealized: float, realized: float, equity: float) -> None:
        """Update PnL metrics."""
        async with self._lock:
            self._session_counters["pnl_unrealized"] = unrealized
            self._session_counters["pnl_realized"] = realized
            self._session_counters["equity"] = equity

    async def update_state(
        self, engine_state: str, mode: str, active_orders: int, active_positions: int
    ) -> None:
        """Update engine state and active counts."""
        async with self._lock:
            self._engine_state = engine_state
            self._mode = mode
            self._active_orders_count = active_orders
            self._active_positions_count = active_positions
            self._session_counters["orders_open"] = active_orders
            self._session_counters["positions_open"] = active_positions

    def _count_in_window(self, buffer: deque[tuple[float, int | float]], window_seconds: int) -> int | float:
        """Count events in time window."""
        now = time.time()
        cutoff = now - window_seconds
        total = 0
        for ts, value in buffer:
            if ts >= cutoff:
                total += value
        return total

    async def get_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics for last 1m, 5m, and session."""
        async with self._lock:
            now = time.time()
            
            # Calculate per-minute counts
            last_1m = {}
            last_5m = {}
            for key, buffer in self._minute_buffers.items():
                last_1m[key] = self._count_in_window(buffer, 60)
                last_5m[key] = self._count_in_window(buffer, 300)
            
            # Top reason codes (last 5 minutes)
            cutoff = now - 300
            recent_reasons = [code for ts, code in self._reason_codes if ts >= cutoff]
            reason_counts = Counter(recent_reasons)
            top_reasons = [
                {"code": code, "count": count, "pct": round(count / max(len(recent_reasons), 1) * 100, 1)}
                for code, count in reason_counts.most_common(10)
            ]
            
            # Top symbols and strategies
            top_symbols = [
                {"symbol": sym, "count": count}
                for sym, count in self._symbol_breakdown["signals"].most_common(5)
            ]
            top_strategies = [
                {"strategy_id": sid, "count": count}
                for sid, count in self._strategy_breakdown["signals"].most_common(5)
            ]
            
            return {
                "counters": {
                    "last_1m": last_1m,
                    "last_5m": last_5m,
                    "session": dict(self._session_counters),
                },
                "top_reasons_for_skip_block": top_reasons,
                "active_orders_count": self._active_orders_count,
                "active_positions_count": self._active_positions_count,
                "last_signal_ts": datetime.fromtimestamp(self._last_signal_ts, tz=timezone.utc).isoformat()
                if self._last_signal_ts
                else None,
                "last_order_ts": datetime.fromtimestamp(self._last_order_ts, tz=timezone.utc).isoformat()
                if self._last_order_ts
                else None,
                "last_fill_ts": datetime.fromtimestamp(self._last_fill_ts, tz=timezone.utc).isoformat()
                if self._last_fill_ts
                else None,
                "engine_state": self._engine_state,
                "mode": self._mode,
                "top_symbols": top_symbols,
                "top_strategies": top_strategies,
                "uptime_sec": round(time.time() - self._start_time, 2),
            }


# Global instance
trading_metrics = TradingMetrics()
