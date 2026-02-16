"""MarketGenomeDetector — Unified multi-dimensional market state synthesis.

Subscribes to TICK, PHYSICS_UPDATE, ORDER_BOOK_UPDATE, FUNDING_UPDATE, REGIME_UPDATE
and publishes a consolidated MARKET_GENOME_UPDATE event every N seconds per symbol.

This is the foundational layer for strategies to consume instead of manually
synthesizing regime state from fragmented sources.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)

GENOME_VERSION = 1


@dataclass
class MarketGenome:
    """Comprehensive market state for a single symbol."""

    symbol: str
    timestamp: float

    # Regime (synthesized from physics + trend + volatility)
    regime: str = "range"  # bull_trend, bear_trend, range, breakout, crisis
    regime_confidence: float = 0.5

    # Volatility
    volatility_state: str = "normal"  # low, normal, high, extreme
    realized_vol_1m: float = 0.0
    realized_vol_5m: float = 0.0
    vol_regime_percentile: float = 50.0  # 0-100

    # Thermodynamics (from PHYSICS_UPDATE)
    temperature: float = 0.0
    entropy: float = 0.0
    phase: str = "unknown"
    ssd_mode: str = "normal"

    # Liquidity (from ORDER_BOOK_UPDATE)
    liquidity_profile: str = "normal"  # deep, normal, thinning, desert
    bid_ask_spread_bps: float = 10.0
    orderbook_imbalance: float = 0.0  # -1.0 to 1.0

    # Momentum
    trend_strength: float = 0.0  # -1.0 to 1.0
    mean_reversion_score: float = 0.0  # 0.0-1.0

    # Funding
    funding_rate: float = 0.0
    funding_pressure: str = "neutral"  # long_heavy, short_heavy, neutral

    # Meta
    genome_version: int = GENOME_VERSION
    data_freshness: float = 0.0  # seconds since last tick

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "regime": self.regime,
            "regime_confidence": self.regime_confidence,
            "volatility_state": self.volatility_state,
            "realized_vol_1m": self.realized_vol_1m,
            "realized_vol_5m": self.realized_vol_5m,
            "vol_regime_percentile": self.vol_regime_percentile,
            "temperature": self.temperature,
            "entropy": self.entropy,
            "phase": self.phase,
            "ssd_mode": self.ssd_mode,
            "liquidity_profile": self.liquidity_profile,
            "bid_ask_spread_bps": self.bid_ask_spread_bps,
            "orderbook_imbalance": self.orderbook_imbalance,
            "trend_strength": self.trend_strength,
            "mean_reversion_score": self.mean_reversion_score,
            "funding_rate": self.funding_rate,
            "funding_pressure": self.funding_pressure,
            "genome_version": self.genome_version,
            "data_freshness": self.data_freshness,
        }


@dataclass
class _SymbolState:
    """Internal rolling window state per symbol."""

    prices: deque = field(default_factory=lambda: deque(maxlen=300))  # 5 min at 1/s
    timestamps: deque = field(default_factory=lambda: deque(maxlen=300))
    returns_1m: deque = field(default_factory=lambda: deque(maxlen=60))
    returns_5m: deque = field(default_factory=lambda: deque(maxlen=300))
    vol_history: deque = field(default_factory=lambda: deque(maxlen=8640))  # 24h at 10s

    # Latest from each event source
    last_physics: dict[str, Any] = field(default_factory=dict)
    last_orderbook: dict[str, Any] = field(default_factory=dict)
    last_funding: dict[str, Any] = field(default_factory=dict)
    last_regime: str = "NORMAL"
    last_tick_time: float = 0.0


class MarketGenomeDetector:
    """Synthesizes multiple event streams into a unified MarketGenome per symbol."""

    def __init__(self, bus: EventBus, interval: float = 10.0) -> None:
        self._bus = bus
        self._interval = interval
        self._symbols: dict[str, _SymbolState] = {}
        self._genomes: dict[str, MarketGenome] = {}
        self._running = False
        self._publish_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the genome detector."""
        self._running = True
        self._bus.subscribe(EventType.TICK, self._on_tick)
        self._bus.subscribe(EventType.PHYSICS_UPDATE, self._on_physics)
        self._bus.subscribe(EventType.ORDER_BOOK_UPDATE, self._on_orderbook)
        self._bus.subscribe(EventType.FUNDING_UPDATE, self._on_funding)
        self._bus.subscribe(EventType.REGIME_UPDATE, self._on_regime)
        self._publish_task = asyncio.create_task(self._publish_loop())
        logger.info("MarketGenomeDetector started (interval=%.1fs)", self._interval)

    async def stop(self) -> None:
        """Stop the genome detector."""
        self._running = False
        if self._publish_task:
            self._publish_task.cancel()
            try:
                await self._publish_task
            except asyncio.CancelledError:
                pass
        self._bus.unsubscribe(EventType.TICK, self._on_tick)
        self._bus.unsubscribe(EventType.PHYSICS_UPDATE, self._on_physics)
        self._bus.unsubscribe(EventType.ORDER_BOOK_UPDATE, self._on_orderbook)
        self._bus.unsubscribe(EventType.FUNDING_UPDATE, self._on_funding)
        self._bus.unsubscribe(EventType.REGIME_UPDATE, self._on_regime)
        logger.info("MarketGenomeDetector stopped")

    def get_genome(self, symbol: str) -> MarketGenome | None:
        """Get latest genome for a symbol."""
        return self._genomes.get(symbol)

    # --- Event handlers ---

    async def _on_tick(self, event: Event) -> None:
        tick = event.data.get("tick")
        if tick is None:
            return
        symbol = tick.symbol
        state = self._get_or_create(symbol)
        now = time.time()

        # Record price
        if state.prices:
            prev = state.prices[-1]
            if prev > 0:
                ret = (tick.price - prev) / prev
                state.returns_1m.append(ret)
                state.returns_5m.append(ret)
        state.prices.append(tick.price)
        state.timestamps.append(now)
        state.last_tick_time = now

    async def _on_physics(self, event: Event) -> None:
        symbol = event.data.get("symbol")
        physics = event.data.get("physics")
        if symbol and physics:
            self._get_or_create(symbol).last_physics = physics

    async def _on_orderbook(self, event: Event) -> None:
        symbol = event.data.get("symbol")
        if symbol:
            self._get_or_create(symbol).last_orderbook = event.data

    async def _on_funding(self, event: Event) -> None:
        symbol = event.data.get("symbol")
        if symbol:
            self._get_or_create(symbol).last_funding = event.data

    async def _on_regime(self, event: Event) -> None:
        symbol = event.data.get("symbol")
        regime = event.data.get("regime")
        if symbol and regime:
            r = regime.value if hasattr(regime, "value") else str(regime)
            self._get_or_create(symbol).last_regime = r

    # --- Computation ---

    async def _publish_loop(self) -> None:
        """Periodically compute and publish genomes."""
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                if not self._running:
                    break
                for symbol in list(self._symbols.keys()):
                    genome = self._compute_genome(symbol)
                    self._genomes[symbol] = genome
                    await self._bus.publish(
                        Event(
                            event_type=EventType.MARKET_GENOME_UPDATE,
                            data={"symbol": symbol, "genome": genome.to_dict()},
                        )
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("MarketGenomeDetector publish error")

    def _compute_genome(self, symbol: str) -> MarketGenome:
        """Compute a MarketGenome from current state."""
        state = self._symbols[symbol]
        now = time.time()

        # Volatility
        vol_1m = self._realized_vol(state.returns_1m)
        vol_5m = self._realized_vol(state.returns_5m)
        vol_state = self._classify_volatility(vol_1m)

        # Store vol for percentile ranking
        if vol_1m > 0:
            state.vol_history.append(vol_1m)
        vol_pct = self._vol_percentile(vol_1m, state.vol_history)

        # Trend strength
        trend = self._compute_trend_strength(state.prices)

        # Mean reversion score (high when price is far from mean)
        mr_score = self._compute_mean_reversion(state.prices)

        # Physics
        physics = state.last_physics
        temperature = physics.get("temperature", 0.0)
        entropy = physics.get("entropy", 0.0)
        phase = physics.get("phase", "unknown")
        ssd_mode = physics.get("ssd_mode", "normal")

        # Liquidity from orderbook
        spread_bps, imbalance, liq_profile = self._compute_liquidity(state.last_orderbook)

        # Funding
        funding_rate, funding_pressure = self._compute_funding(state.last_funding)

        # Synthesize regime
        regime, confidence = self._synthesize_regime(
            phase=phase,
            trend_strength=trend,
            vol_state=vol_state,
            ssd_mode=ssd_mode,
            base_regime=state.last_regime,
        )

        # Data freshness
        freshness = now - state.last_tick_time if state.last_tick_time > 0 else 999.0

        return MarketGenome(
            symbol=symbol,
            timestamp=now,
            regime=regime,
            regime_confidence=confidence,
            volatility_state=vol_state,
            realized_vol_1m=vol_1m,
            realized_vol_5m=vol_5m,
            vol_regime_percentile=vol_pct,
            temperature=temperature,
            entropy=entropy,
            phase=phase,
            ssd_mode=ssd_mode,
            liquidity_profile=liq_profile,
            bid_ask_spread_bps=spread_bps,
            orderbook_imbalance=imbalance,
            trend_strength=trend,
            mean_reversion_score=mr_score,
            funding_rate=funding_rate,
            funding_pressure=funding_pressure,
            data_freshness=freshness,
        )

    # --- Calculation helpers ---

    @staticmethod
    def _realized_vol(returns: deque) -> float:
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns, dtype=np.float64)
        return float(np.std(arr))

    @staticmethod
    def _classify_volatility(vol_1m: float) -> str:
        if vol_1m < 0.0005:
            return "low"
        if vol_1m < 0.002:
            return "normal"
        if vol_1m < 0.005:
            return "high"
        return "extreme"

    @staticmethod
    def _vol_percentile(current_vol: float, history: deque) -> float:
        if len(history) < 10 or current_vol <= 0:
            return 50.0
        arr = np.array(history, dtype=np.float64)
        return float(np.searchsorted(np.sort(arr), current_vol) / len(arr) * 100)

    @staticmethod
    def _compute_trend_strength(prices: deque) -> float:
        """Trend strength from -1.0 (strong bear) to 1.0 (strong bull)."""
        if len(prices) < 20:
            return 0.0
        arr = np.array(prices, dtype=np.float64)
        # Use linear regression slope normalized by price
        n = len(arr)
        x = np.arange(n, dtype=np.float64)
        x_mean = x.mean()
        y_mean = arr.mean()
        if y_mean == 0:
            return 0.0
        slope = np.sum((x - x_mean) * (arr - y_mean)) / np.sum((x - x_mean) ** 2)
        # Normalize: slope per tick as fraction of mean price
        normalized = (slope / y_mean) * n
        return float(np.clip(normalized * 10.0, -1.0, 1.0))

    @staticmethod
    def _compute_mean_reversion(prices: deque) -> float:
        """Mean reversion score: 0.0 (at mean) to 1.0 (far from mean)."""
        if len(prices) < 20:
            return 0.0
        arr = np.array(prices, dtype=np.float64)
        mean = arr.mean()
        std = arr.std()
        if std == 0 or mean == 0:
            return 0.0
        z_score = abs(arr[-1] - mean) / std
        return float(np.clip(z_score / 3.0, 0.0, 1.0))

    @staticmethod
    def _compute_liquidity(ob_data: dict[str, Any]) -> tuple[float, float, str]:
        """Compute liquidity metrics from orderbook data."""
        if not ob_data:
            return 10.0, 0.0, "normal"

        bid = ob_data.get("best_bid", 0.0) or ob_data.get("bid", 0.0)
        ask = ob_data.get("best_ask", 0.0) or ob_data.get("ask", 0.0)

        if not bid or not ask or bid <= 0:
            return 10.0, 0.0, "normal"

        spread_bps = (ask - bid) / bid * 10000
        bid_depth = ob_data.get("bid_depth", 0.0) or 0.0
        ask_depth = ob_data.get("ask_depth", 0.0) or 0.0
        total_depth = bid_depth + ask_depth

        imbalance = 0.0
        if total_depth > 0:
            imbalance = (bid_depth - ask_depth) / total_depth

        if spread_bps < 5 and abs(imbalance) < 0.3:
            profile = "deep"
        elif spread_bps < 15 and abs(imbalance) < 0.5:
            profile = "normal"
        elif spread_bps < 50:
            profile = "thinning"
        else:
            profile = "desert"

        return spread_bps, imbalance, profile

    @staticmethod
    def _compute_funding(funding_data: dict[str, Any]) -> tuple[float, str]:
        """Compute funding rate and pressure."""
        if not funding_data:
            return 0.0, "neutral"

        rate = funding_data.get("rate", 0.0) or 0.0
        if rate > 0.0005:
            pressure = "long_heavy"
        elif rate < -0.0005:
            pressure = "short_heavy"
        else:
            pressure = "neutral"

        return rate, pressure

    @staticmethod
    def _synthesize_regime(
        phase: str,
        trend_strength: float,
        vol_state: str,
        ssd_mode: str,
        base_regime: str,
    ) -> tuple[str, float]:
        """Synthesize regime from multiple signals.

        Returns (regime_name, confidence).
        """
        # Crisis: extreme volatility or SSD silent mode
        if vol_state == "extreme" or ssd_mode == "silent":
            return "crisis", 0.9

        # Breakout: phase transition + high vol + strong trend
        phase_lower = phase.lower() if phase else ""
        if vol_state == "high" and abs(trend_strength) > 0.5:
            if phase_lower in ("vapor", "markup", "markdown"):
                return "breakout", 0.8

        # Bull trend
        if trend_strength > 0.5 and vol_state in ("normal", "low"):
            if phase_lower in ("water", "markup", "accumulation"):
                return "bull_trend", min(0.9, 0.5 + abs(trend_strength))
            return "bull_trend", min(0.8, 0.4 + abs(trend_strength))

        # Bear trend
        if trend_strength < -0.5 and vol_state in ("normal", "low"):
            if phase_lower in ("water", "markdown", "distribution"):
                return "bear_trend", min(0.9, 0.5 + abs(trend_strength))
            return "bear_trend", min(0.8, 0.4 + abs(trend_strength))

        # Range
        if abs(trend_strength) < 0.3 and vol_state in ("low", "normal"):
            if phase_lower in ("ice", "accumulation"):
                return "range", 0.8
            return "range", 0.6

        # Default: use base regime mapping
        regime_map = {"IMPULSE": "breakout", "RANGE": "range", "NORMAL": "range"}
        return regime_map.get(base_regime.upper(), "range"), 0.5

    def _get_or_create(self, symbol: str) -> _SymbolState:
        if symbol not in self._symbols:
            self._symbols[symbol] = _SymbolState()
        return self._symbols[symbol]
