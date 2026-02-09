"""
Causal Mesh: Dynamic Bayesian Network for Top 50 Bybit Perpetual Pairs
Monitors real-time influence between assets and triggers satellite trades
when central influence (BTC/ETH) moves.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Signal, Tick
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CausalLink:
    """Represents a causal link in the Dynamic Bayesian Network."""
    source: str  # Source symbol (e.g., "BTCUSDT")
    target: str  # Target symbol (e.g., "ETHUSDT")
    influence_strength: float  # 0-1, strength of causal influence
    lag_ms: int  # Time lag in milliseconds
    correlation: float  # Real-time correlation
    last_update: datetime = field(default_factory=datetime.utcnow)
    trade_trigger_count: int = 0  # Number of trades triggered by this link


@dataclass
class CentralInfluence:
    """Represents central influence nodes (BTC, ETH, etc.)."""
    symbol: str
    influence_score: float  # How much this asset influences others
    current_price: float
    price_change_pct: float  # Recent price change percentage
    timestamp: datetime = field(default_factory=datetime.utcnow)
    satellite_count: int = 0  # Number of satellite assets


@dataclass
class SatelliteTrade:
    """Satellite trade triggered by central influence movement."""
    symbol: str  # Satellite asset to trade
    central_symbol: str  # Central influence that triggered this
    direction: str  # "buy" or "sell"
    expected_lag_ms: int  # Expected lag before price movement
    confidence: float  # 0-1, confidence in the trade
    timestamp: datetime = field(default_factory=datetime.utcnow)


class DynamicBayesianNetwork:
    """
    Dynamic Bayesian Network for modeling causal relationships.
    Updates conditional probabilities in real-time.
    """

    def __init__(self, symbols: list[str]):
        """Initialize DBN with symbols."""
        self.symbols = symbols
        self.n = len(symbols)

        # Conditional probability tables (CPT)
        # cpt[target][source] = P(target | source)
        self.cpt: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Parent sets (which symbols influence each symbol)
        self.parents: dict[str, set[str]] = defaultdict(set)

        # Price histories for each symbol (last 1000 ticks)
        self.price_history: dict[str, deque] = {s: deque(maxlen=1000) for s in symbols}
        self.timestamp_history: dict[str, deque] = {s: deque(maxlen=1000) for s in symbols}

        # Initialize uniform CPT
        for target in symbols:
            for source in symbols:
                if source != target:
                    self.cpt[target][source] = 0.5  # Start with neutral probability

    def update_price(self, symbol: str, price: float, timestamp: datetime):
        """Update price history for a symbol."""
        if symbol not in self.price_history:
            return

        self.price_history[symbol].append(price)
        self.timestamp_history[symbol].append(timestamp)

    def calculate_influence(self, source: str, target: str) -> tuple[float, int]:
        """
        Calculate causal influence strength and lag.
        Returns: (influence_strength, lag_ms)
        """
        if source not in self.price_history or target not in self.price_history:
            return 0.0, 0

        source_prices = np.array(list(self.price_history[source]))
        target_prices = np.array(list(self.price_history[target]))
        source_timestamps = list(self.timestamp_history[source])
        target_timestamps = list(self.timestamp_history[target])

        if len(source_prices) < 50 or len(target_prices) < 50:
            return 0.0, 0

        # Calculate correlation
        min_len = min(len(source_prices), len(target_prices))
        source_returns = np.diff(source_prices[-min_len:]) / source_prices[-min_len:-1]
        target_returns = np.diff(target_prices[-min_len:]) / target_prices[-min_len:-1]

        if len(source_returns) < 10:
            return 0.0, 0

        # Find optimal lag using cross-correlation
        max_lag = min(20, len(source_returns) // 4)
        best_correlation = 0.0
        best_lag = 0

        for lag in range(max_lag):
            if lag >= len(source_returns):
                break
            source_shifted = source_returns[lag:]
            target_aligned = target_returns[:len(source_shifted)]

            if len(source_shifted) < 10:
                continue

            corr = np.corrcoef(source_shifted, target_aligned)[0, 1]
            if not np.isnan(corr) and abs(corr) > abs(best_correlation):
                best_correlation = corr
                best_lag = lag

        # Calculate average time difference for lag
        lag_ms = 0
        if len(source_timestamps) > best_lag + 1 and len(target_timestamps) > 1:
            try:
                source_time = source_timestamps[-best_lag-1] if best_lag < len(source_timestamps) else source_timestamps[0]
                target_time = target_timestamps[-1]
                lag_delta = target_time - source_time
                lag_ms = int(lag_delta.total_seconds() * 1000)
            except (IndexError, AttributeError) as e:
                logger.warning(f"Failed to calculate lag time delta: {e}")
                lag_ms = best_lag * 100  # Estimate: 100ms per lag step

        # Influence strength = absolute correlation
        influence_strength = abs(best_correlation)

        # Update CPT
        self.cpt[target][source] = 0.9 * self.cpt[target][source] + 0.1 * influence_strength

        # Update parent set if influence is strong
        if influence_strength > 0.3:
            self.parents[target].add(source)
        elif influence_strength < 0.1:
            self.parents[target].discard(source)

        return influence_strength, lag_ms

    def predict_target_change(self, source: str, source_change_pct: float, target: str) -> tuple[float, float]:
        """
        Predict target price change given source change.
        Returns: (predicted_change_pct, confidence)
        """
        if source not in self.cpt[target]:
            return 0.0, 0.0

        # Use CPT to predict change
        influence = self.cpt[target][source]
        predicted_change = source_change_pct * influence

        # Confidence based on CPT strength
        confidence = min(1.0, influence * 1.5)

        return predicted_change, confidence


class CausalMesh:
    """
    Causal Mesh: Monitors Top 50 Bybit Perpetual pairs using Dynamic Bayesian Network.
    Triggers satellite trades when central influence moves.
    """

    # Top 50 Bybit Perpetual pairs (prioritized)
    TOP_50_PERPETUALS = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "SHIBUSDT", "TONUSDT",
        "DOTUSDT", "MATICUSDT", "LINKUSDT", "UNIUSDT", "BCHUSDT",
        "LTCUSDT", "NEARUSDT", "ATOMUSDT", "ETCUSDT", "APTUSDT",
        "XLMUSDT", "FILUSDT", "ICPUSDT", "ALGOUSDT", "VETUSDT",
        "ARBUSDT", "OPUSDT", "AAVEUSDT", "GRTUSDT", "INJUSDT",
        "SANDUSDT", "MANAUSDT", "THETAUSDT", "AXSUSDT", "EOSUSDT",
        "FLOWUSDT", "AUDIOUSDT", "GALAUSDT", "IMXUSDT", "APEUSDT",
        "CHZUSDT", "ENJUSDT", "HBARUSDT", "EGLDUSDT", "XTZUSDT",
        "ZILUSDT", "ROSEUSDT", "IOTAUSDT", "ZECUSDT", "DASHUSDT",
    ]

    # Central influence symbols (usually BTC and ETH)
    CENTRAL_INFLUENCE = ["BTCUSDT", "ETHUSDT"]

    def __init__(self, bus: EventBus):
        """Initialize Causal Mesh."""
        self._bus = bus
        self._dbn = DynamicBayesianNetwork(self.TOP_50_PERPETUALS)

        # Causal links map: source -> List[CausalLink]
        self._causal_links: dict[str, list[CausalLink]] = defaultdict(list)

        # Central influence tracking
        self._central_influences: dict[str, CentralInfluence] = {}
        for symbol in self.CENTRAL_INFLUENCE:
            self._central_influences[symbol] = CentralInfluence(
                symbol=symbol,
                influence_score=1.0,  # High initial influence
                current_price=0.0,
                price_change_pct=0.0
            )

        # Recent price changes (for detecting central movements)
        self._previous_prices: dict[str, float] = {}
        self._price_change_threshold = 0.002  # 0.2% change triggers satellite trades

        # Satellite trades queue
        self._satellite_trades: deque = deque(maxlen=1000)

        # Running state
        self._running = False
        self._update_task: asyncio.Task | None = None

        # Statistics
        self._total_links_discovered = 0
        self._total_satellite_trades_triggered = 0

    async def start(self) -> None:
        """Start the Causal Mesh."""
        if self._running:
            return

        self._running = True
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        self._update_task = asyncio.create_task(self._update_mesh_loop())

        logger.info("Causal Mesh started: Monitoring Top 50 Bybit Perpetual pairs")

    async def stop(self) -> None:
        """Stop the Causal Mesh."""
        self._running = False
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        logger.info("Causal Mesh stopped")

    async def _handle_tick(self, event: Event) -> None:
        """Handle tick events to update price history."""
        if not self._running:
            return

        tick: Tick = event.data["tick"]
        symbol = tick.symbol

        if symbol not in self.TOP_50_PERPETUALS:
            return

        price = tick.price
        timestamp = datetime.utcnow()

        # Update DBN price history
        self._dbn.update_price(symbol, price, timestamp)

        # Track price changes for central influence
        if symbol in self._previous_prices:
            price_change = (price - self._previous_prices[symbol]) / self._previous_prices[symbol]

            # Update central influence if this is a central symbol
            if symbol in self._central_influences:
                central = self._central_influences[symbol]
                central.current_price = price
                central.price_change_pct = price_change
                central.timestamp = timestamp

                # Check if central influence moved significantly
                if abs(price_change) > self._price_change_threshold:
                    await self._trigger_satellite_trades(symbol, price_change)

        self._previous_prices[symbol] = price

    async def _update_mesh_loop(self) -> None:
        """Background loop to update causal links in the mesh."""
        while self._running:
            try:
                await self._recalculate_causal_links()
                await asyncio.sleep(5.0)  # Recalculate every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in causal mesh update loop: {e}")
                await asyncio.sleep(1.0)

    async def _recalculate_causal_links(self) -> None:
        """Recalculate all causal links in the mesh."""
        # Calculate influence from central symbols to all others
        for central_symbol in self.CENTRAL_INFLUENCE:
            for target_symbol in self.TOP_50_PERPETUALS:
                if central_symbol == target_symbol:
                    continue

                influence, lag_ms = self._dbn.calculate_influence(central_symbol, target_symbol)

                # Create or update causal link
                existing_link = None
                for link in self._causal_links[central_symbol]:
                    if link.target == target_symbol:
                        existing_link = link
                        break

                if existing_link:
                    existing_link.influence_strength = influence
                    existing_link.lag_ms = lag_ms
                    existing_link.last_update = datetime.utcnow()
                else:
                    new_link = CausalLink(
                        source=central_symbol,
                        target=target_symbol,
                        influence_strength=influence,
                        lag_ms=lag_ms,
                        correlation=influence
                    )
                    self._causal_links[central_symbol].append(new_link)
                    self._total_links_discovered += 1

        # Also calculate influence between major pairs (not just from central)
        # This helps identify secondary causal chains
        major_pairs = ["SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"]
        for source in major_pairs:
            for target in self.TOP_50_PERPETUALS:
                if source == target or source in self.CENTRAL_INFLUENCE:
                    continue

                influence, lag_ms = self._dbn.calculate_influence(source, target)
                if influence > 0.4:  # Only track strong links
                    # Similar update logic as above
                    pass

    async def _trigger_satellite_trades(self, central_symbol: str, price_change_pct: float) -> None:
        """
        Trigger satellite trades when central influence moves.
        Central influence has moved, so predict and trigger trades on laggard assets.
        """
        direction = "buy" if price_change_pct > 0 else "sell"

        # Get all causal links from this central symbol
        links = self._causal_links.get(central_symbol, [])

        # Sort by influence strength (strongest first)
        sorted_links = sorted(links, key=lambda x: x.influence_strength, reverse=True)

        # Trigger trades on top satellite assets
        triggered_count = 0
        for link in sorted_links[:10]:  # Top 10 satellite assets
            if link.influence_strength < 0.3:  # Skip weak links
                continue

            # Predict target price change
            predicted_change, confidence = self._dbn.predict_target_change(
                central_symbol, price_change_pct, link.target
            )

            if confidence < 0.4:  # Skip low confidence predictions
                continue

            # Create satellite trade signal
            satellite_trade = SatelliteTrade(
                symbol=link.target,
                central_symbol=central_symbol,
                direction=direction,
                expected_lag_ms=link.lag_ms,
                confidence=confidence
            )

            self._satellite_trades.append(satellite_trade)
            link.trade_trigger_count += 1
            triggered_count += 1
            self._total_satellite_trades_triggered += 1

            # Publish signal to event bus
            signal = Signal(
                symbol=link.target,
                side=direction,
                confidence=confidence,
                strategy_id="causal_mesh",
                metadata={
                    "central_symbol": central_symbol,
                    "central_change_pct": price_change_pct,
                    "predicted_change_pct": predicted_change,
                    "expected_lag_ms": link.lag_ms,
                    "influence_strength": link.influence_strength
                }
            )

            await self._bus.publish(Event(
                event_type=EventType.SIGNAL,
                data={"signal": signal}
            ))

            logger.info(
                f"Causal Mesh: Triggered {direction} signal on {link.target} "
                f"(central: {central_symbol}, change: {price_change_pct:.4f}, "
                f"confidence: {confidence:.2f}, lag: {link.lag_ms}ms)"
            )

        if triggered_count > 0:
            logger.info(f"Causal Mesh: Triggered {triggered_count} satellite trades from {central_symbol} movement")

    def get_causal_links(self, source: str | None = None) -> dict[str, list[CausalLink]]:
        """Get causal links, optionally filtered by source."""
        if source:
            return {source: self._causal_links.get(source, [])}
        return dict(self._causal_links)

    def get_central_influences(self) -> dict[str, CentralInfluence]:
        """Get current central influence status."""
        return dict(self._central_influences)

    def get_satellite_trades(self, limit: int = 100) -> list[SatelliteTrade]:
        """Get recent satellite trades."""
        return list(self._satellite_trades)[-limit:]

    def get_statistics(self) -> dict:
        """Get causal mesh statistics."""
        total_links = sum(len(links) for links in self._causal_links.values())
        strong_links = sum(
            1 for links in self._causal_links.values()
            for link in links
            if link.influence_strength > 0.5
        )

        return {
            "total_links": total_links,
            "strong_links": strong_links,
            "total_links_discovered": self._total_links_discovered,
            "total_satellite_trades_triggered": self._total_satellite_trades_triggered,
            "central_influences": {
                symbol: {
                    "influence_score": inf.influence_score,
                    "price_change_pct": inf.price_change_pct,
                    "satellite_count": inf.satellite_count
                }
                for symbol, inf in self._central_influences.items()
            }
        }
