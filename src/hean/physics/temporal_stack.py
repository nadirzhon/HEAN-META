"""Temporal Stack - 5-level market analysis engine.

Analyzes markets across 5 time levels simultaneously:
- LEVEL 5 MACRO (days-weeks): Overall trend and cycle phase
- LEVEL 4 SESSION (hours): Trading session and mode
- LEVEL 3 TACTICS (minutes): Current situation and order flow
- LEVEL 2 EXECUTION (seconds): Entry points and R:R
- LEVEL 1 MICRO (milliseconds): Orderbook and slippage
"""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from hean.logging import get_logger

logger = get_logger(__name__)


class TimeLevel(Enum):
    MACRO = 5
    SESSION = 4
    TACTICS = 3
    EXECUTION = 2
    MICRO = 1


class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class TemporalState:
    level: TimeLevel
    name: str
    timeframe: str
    trend: TrendDirection
    phase: str
    dominant_force: str
    summary: str
    confidence: float
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "name": self.name,
            "timeframe": self.timeframe,
            "trend": self.trend.value,
            "phase": self.phase,
            "dominant_force": self.dominant_force,
            "summary": self.summary,
            "confidence": self.confidence,
            "details": self.details,
        }


class TemporalStack:
    """5-level temporal analysis engine."""

    def __init__(self, symbols: list[str] | None = None):
        self.symbols = symbols or ["BTCUSDT"]
        self._price_history: dict[str, deque] = {
            s: deque(maxlen=10000) for s in self.symbols
        }
        self._volume_history: dict[str, deque] = {
            s: deque(maxlen=1000) for s in self.symbols
        }
        self._order_flow_delta: dict[str, float] = {s: 0.0 for s in self.symbols}
        self._states: dict[TimeLevel, TemporalState] = {}
        self._last_update = datetime.utcnow()

    def update(self, symbol: str, price: float, volume: float = 0.0) -> None:
        """Update with new tick data."""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=10000)
            self._volume_history[symbol] = deque(maxlen=1000)

        now = datetime.utcnow()
        self._price_history[symbol].append((now, price))
        self._volume_history[symbol].append((now, volume))
        self._analyze_levels(symbol)
        self._last_update = now

    def _analyze_levels(self, symbol: str) -> None:
        prices = self._price_history[symbol]
        if len(prices) < 10:
            return

        price_values = np.array([p[1] for p in prices])
        current_price = price_values[-1]

        self._states[TimeLevel.MACRO] = self._analyze_macro(symbol, price_values)
        self._states[TimeLevel.SESSION] = self._analyze_session(symbol, price_values, current_price)
        self._states[TimeLevel.TACTICS] = self._analyze_tactics(symbol, price_values, current_price)
        self._states[TimeLevel.EXECUTION] = self._analyze_execution(symbol, price_values, current_price)
        self._states[TimeLevel.MICRO] = self._analyze_micro(symbol, price_values, current_price)

    def _analyze_macro(self, symbol: str, prices: np.ndarray) -> TemporalState:
        if len(prices) >= 200:
            ema = self._ema(prices, 200)
            current = prices[-1]
            if current > ema * 1.02:
                trend, force, phase = TrendDirection.BULLISH, "Institutional accumulation", "trending"
            elif current < ema * 0.98:
                trend, force, phase = TrendDirection.BEARISH, "Distribution pressure", "trending"
            else:
                trend, force, phase = TrendDirection.NEUTRAL, "Range-bound", "accumulation"
        elif len(prices) >= 20:
            ema = self._ema(prices, 20)
            current = prices[-1]
            if current > ema * 1.02:
                trend, force, phase = TrendDirection.BULLISH, "Buying pressure", "trending"
            elif current < ema * 0.98:
                trend, force, phase = TrendDirection.BEARISH, "Selling pressure", "trending"
            else:
                trend, force, phase = TrendDirection.NEUTRAL, "Consolidation", "accumulation"
        else:
            trend, force, phase = TrendDirection.NEUTRAL, "Insufficient data", "unknown"

        return TemporalState(
            level=TimeLevel.MACRO, name="MACRO", timeframe="days-weeks",
            trend=trend, phase=phase, dominant_force=force,
            summary=f"{symbol} {trend.value} trend, {phase} phase",
            confidence=0.8,
        )

    def _analyze_session(self, symbol: str, prices: np.ndarray, current: float) -> TemporalState:
        hour = datetime.utcnow().hour
        session = "Asia" if hour < 8 else ("London" if hour < 16 else "New York")

        if len(prices) >= 60:
            recent = prices[-60:]
            range_pct = (np.max(recent) - np.min(recent)) / np.min(recent) * 100
            if range_pct > 2.0:
                mode, strategy = "Breakout", "Momentum"
            elif range_pct < 0.5:
                mode, strategy = "Range", "Mean-reversion"
            else:
                mode, strategy = "Trending", "Trend-following"
        else:
            mode, strategy = "Unknown", "Adaptive"

        return TemporalState(
            level=TimeLevel.SESSION, name="SESSION", timeframe="hours",
            trend=TrendDirection.NEUTRAL, phase=mode.lower(),
            dominant_force=session,
            summary=f"{session} session: {mode} mode, use {strategy}",
            confidence=0.7,
            details={"session": session, "mode": mode, "strategy": strategy},
        )

    def _analyze_tactics(self, symbol: str, prices: np.ndarray, current: float) -> TemporalState:
        delta = self._order_flow_delta.get(symbol, 0.0)

        if len(prices) >= 20:
            recent = prices[-20:]
            resistance = float(np.max(recent))
            support = float(np.min(recent))
            if abs(current - resistance) / current < 0.002:
                situation = f"Testing resistance at {resistance:.2f}"
            elif abs(current - support) / current < 0.002:
                situation = f"Testing support at {support:.2f}"
            else:
                situation = f"Mid-range {support:.2f}-{resistance:.2f}"
        else:
            situation = "Establishing range"

        return TemporalState(
            level=TimeLevel.TACTICS, name="TACTICS", timeframe="minutes",
            trend=TrendDirection.NEUTRAL, phase="analysis",
            dominant_force="order_flow",
            summary=f"{situation}. Flow: ${delta/1e6:.1f}M",
            confidence=0.6,
            details={"situation": situation, "order_flow_delta": delta},
        )

    def _analyze_execution(self, symbol: str, prices: np.ndarray, current: float) -> TemporalState:
        if len(prices) >= 20:
            low = float(np.min(prices[-20:]))
            stop = low * 0.998
            risk = abs(current - stop)
            tp = current + risk * 4.2
            rr = (tp - current) / risk if risk > 0 else 0
            summary = f"Entry: {current:.2f}, Stop: {stop:.2f}, R:R={rr:.1f}"
        else:
            stop, tp, rr = current * 0.99, current * 1.02, 2.0
            summary = f"Entry: {current:.2f} (insufficient data)"

        return TemporalState(
            level=TimeLevel.EXECUTION, name="EXECUTION", timeframe="seconds",
            trend=TrendDirection.NEUTRAL, phase="ready",
            dominant_force="technical", summary=summary, confidence=0.75,
            details={"entry_price": current, "stop_loss": stop, "take_profit": tp, "rr_ratio": rr},
        )

    def _analyze_micro(self, symbol: str, prices: np.ndarray, current: float) -> TemporalState:
        if len(prices) >= 10:
            vol = float(np.std(prices[-10:]))
            slippage = (vol / current) * 100
        else:
            slippage = 0.02

        order_type = "Limit" if slippage < 0.05 else "Market"
        state = "Tight spread" if slippage < 0.05 else "Wide spread"

        return TemporalState(
            level=TimeLevel.MICRO, name="MICRO", timeframe="milliseconds",
            trend=TrendDirection.NEUTRAL, phase="execution",
            dominant_force="orderbook",
            summary=f"{state}. {order_type} at {current:.2f}, slippage ~{slippage:.3f}%",
            confidence=0.9,
            details={"order_type": order_type, "slippage_pct": slippage},
        )

    def _ema(self, prices: np.ndarray, period: int) -> float:
        if len(prices) < period:
            return float(np.mean(prices))
        recent = prices[-period:]
        alpha = 2.0 / (period + 1.0)
        ema = recent[0]
        for p in recent[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return float(ema)

    def get_state(self, level: TimeLevel) -> TemporalState | None:
        return self._states.get(level)

    def get_all_states(self) -> dict[TimeLevel, TemporalState]:
        return self._states.copy()

    def get_stack_dict(self) -> dict[str, Any]:
        return {
            "levels": {str(l.value): s.to_dict() for l, s in self._states.items()},
            "last_update": self._last_update.isoformat(),
        }
