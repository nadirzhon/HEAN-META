"""
Swarm Intelligence System with Topological Data Analysis (TDA) Integration
All agents receive topology_score as a mandatory input feature.
Topology is now the primary sense through which the system perceives reality.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Signal, Tick
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Agent:
    """Swarm agent with topology-aware signal generation."""

    agent_id: str
    strategy_id: str
    confidence_threshold: float = 0.6
    momentum_score: float = 0.0
    topology_score: float = 1.0  # Mandatory: structural stability (0-1)
    signal_confidence: float = 0.0
    last_signal_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SwarmIntelligence:
    """Swarm-based agent system with TDA topology integration.

    All agents must receive topology_score as a mandatory input.
    If topology_score indicates structural collapse, decrease signal confidence by 50%
    regardless of momentum.
    """

    def __init__(self, bus: EventBus) -> None:
        """Initialize the swarm intelligence system."""
        self._bus = bus
        self._agents: dict[str, Agent] = {}
        self._topology_score_cache: dict[str, float] = {}  # symbol -> topology_score
        self._market_topology_score: float = 1.0
        self._running = False

        # Correlation Matrix Module for Delta-Neutral Hedging
        self._correlation_matrix: dict[tuple[str, str], float] = {}  # (symbol_a, symbol_b) -> correlation
        self._returns_history: dict[str, deque[float]] = {}  # symbol -> rolling returns
        self._price_history: dict[str, deque[float]] = {}  # symbol -> rolling prices
        self._correlation_window: int = 100  # Window size for correlation calculation
        self._min_correlation_window: int = 50  # Minimum data points required
        self._active_positions: dict[str, float] = {}  # symbol -> position_size (for delta tracking)
        self._delta_neutral_target: float = 0.0  # Target delta (0 = fully neutral)

        # Try to import FastWarden (C++ TDA engine)
        self._fast_warden: Any = None
        try:
            import graph_engine_py  # type: ignore
            self._fast_warden = graph_engine_py.FastWarden()
            logger.info("FastWarden (TDA Engine) initialized successfully")
        except ImportError:
            logger.warning("FastWarden not available. Using fallback topology scoring.")

    def register_agent(self, agent_id: str, strategy_id: str, **kwargs: Any) -> Agent:
        """Register a new agent in the swarm.

        Args:
            agent_id: Unique agent identifier
            strategy_id: Strategy this agent belongs to
            **kwargs: Additional agent configuration

        Returns:
            Registered Agent instance
        """
        agent = Agent(
            agent_id=agent_id,
            strategy_id=strategy_id,
            **kwargs
        )
        self._agents[agent_id] = agent
        logger.info(f"Agent {agent_id} registered for strategy {strategy_id}")
        return agent

    def update_topology_score(self, symbol: str, topology_score: float | None = None) -> None:
        """Update topology score for a symbol.

        Args:
            symbol: Trading symbol
            topology_score: Topology score (0-1). If None, fetch from FastWarden.
        """
        if topology_score is not None:
            self._topology_score_cache[symbol] = topology_score
        elif self._fast_warden is not None:
            try:
                self._market_topology_score = self._fast_warden.get_market_topology_score()
                # Use market-level score as proxy for individual symbol
                # In production, would have per-symbol scores
                self._topology_score_cache[symbol] = self._market_topology_score
            except Exception as e:
                logger.warning(f"Failed to get topology score from FastWarden: {e}")
                self._topology_score_cache[symbol] = 1.0  # Default: stable
        else:
            # Fallback: assume stable market
            self._topology_score_cache[symbol] = 1.0

    def get_topology_score(self, symbol: str) -> float:
        """Get topology score for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Topology score (0-1, 1=stable, 0=collapsing)
        """
        if symbol in self._topology_score_cache:
            return self._topology_score_cache[symbol]

        # Fetch if not cached
        self.update_topology_score(symbol)
        return self._topology_score_cache.get(symbol, 1.0)

    def evaluate_signal(
        self,
        agent_id: str,
        symbol: str,
        side: str,
        momentum_score: float,
        entry_price: float,
        **kwargs: Any
    ) -> Signal | None:
        """Evaluate and generate signal with topology-aware confidence.

        CRITICAL: If topology_score indicates structural collapse (score < 0.3),
        decrease signal confidence by 50% regardless of momentum.

        Args:
            agent_id: Agent identifier
            symbol: Trading symbol
            side: "buy" or "sell"
            momentum_score: Momentum-based signal strength (0-1)
            entry_price: Entry price
            **kwargs: Additional signal parameters

        Returns:
            Signal if confidence threshold met, None otherwise
        """
        if agent_id not in self._agents:
            logger.warning(f"Agent {agent_id} not found in swarm")
            return None

        agent = self._agents[agent_id]

        # MANDATORY: Get topology score
        topology_score = self.get_topology_score(symbol)
        agent.topology_score = topology_score

        # Update agent's momentum score
        agent.momentum_score = momentum_score

        # Calculate base signal confidence from momentum
        base_confidence = momentum_score

        # TOPOLOGY-DRIVEN LOGIC: Decrease confidence by 50% if structural collapse
        # Topology score < 0.3 indicates structural collapse
        structural_collapse_threshold = 0.3

        if topology_score < structural_collapse_threshold:
            # Structural collapse detected: decrease confidence by 50% regardless of momentum
            logger.warning(
                f"Structural collapse detected for {symbol}: "
                f"topology_score={topology_score:.3f} < {structural_collapse_threshold}, "
                f"reducing confidence by 50%"
            )
            base_confidence = base_confidence * 0.5
        else:
            # Topology is stable: use topology score to modulate confidence
            # Higher topology score = higher confidence
            topology_modulator = 0.5 + (topology_score * 0.5)  # Map [0,1] -> [0.5, 1.0]
            base_confidence = base_confidence * topology_modulator

        agent.signal_confidence = base_confidence

        # Check if confidence meets threshold
        if base_confidence < agent.confidence_threshold:
            logger.debug(
                f"Agent {agent_id} signal rejected: "
                f"confidence={base_confidence:.3f} < threshold={agent.confidence_threshold:.3f}, "
                f"topology={topology_score:.3f}, momentum={momentum_score:.3f}"
            )
            return None

        # Generate signal
        signal = Signal(
            strategy_id=agent.strategy_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            metadata={
                "agent_id": agent_id,
                "momentum_score": momentum_score,
                "topology_score": topology_score,
                "signal_confidence": base_confidence,
                "topology_driven": topology_score < structural_collapse_threshold,
                **kwargs
            }
        )

        agent.last_signal_time = datetime.utcnow()

        logger.info(
            f"Agent {agent_id} signal generated: {symbol} {side} @ {entry_price:.2f}, "
            f"confidence={base_confidence:.3f} (momentum={momentum_score:.3f}, "
            f"topology={topology_score:.3f})"
        )

        return signal

    async def start(self) -> None:
        """Start the swarm intelligence system."""
        self._running = True
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        self._bus.subscribe(EventType.ORDER_BOOK_UPDATE, self._handle_orderbook_update)
        self._bus.subscribe(EventType.POSITION_OPENED, self._handle_position_opened)
        self._bus.subscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        logger.info("Swarm Intelligence system started with TDA topology integration and Correlation Matrix")

    async def stop(self) -> None:
        """Stop the swarm intelligence system."""
        self._running = False
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        self._bus.unsubscribe(EventType.ORDER_BOOK_UPDATE, self._handle_orderbook_update)
        self._bus.unsubscribe(EventType.POSITION_OPENED, self._handle_position_opened)
        self._bus.unsubscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        logger.info("Swarm Intelligence system stopped")

    async def _handle_tick(self, event: Event) -> None:
        """Handle tick events to update topology scores and correlation matrix."""
        tick: Tick = event.data["tick"]

        # Update topology score cache periodically
        # In production, this would trigger FastWarden updates
        if tick.symbol not in self._topology_score_cache:
            self.update_topology_score(tick.symbol)

        # Update correlation matrix (for delta-neutral hedging)
        self._update_correlation_data(tick)

        # Check for delta-neutral hedge opportunities during high-frequency scalping
        await self._check_delta_neutral_hedge(tick.symbol)

    async def _handle_orderbook_update(self, event: Event) -> None:
        """Handle orderbook updates to feed FastWarden."""
        if self._fast_warden is None:
            return

        orderbook_data = event.data.get("orderbook", {})
        symbol = orderbook_data.get("symbol")

        if not symbol:
            return

        # Extract bid/ask levels
        bids = orderbook_data.get("bids", [])
        asks = orderbook_data.get("asks", [])

        if not bids or not asks:
            return

        try:
            # Convert to FastWarden format: (price, size) tuples
            bid_prices = [float(bid[0]) for bid in bids]
            bid_sizes = [float(bid[1]) for bid in bids]
            ask_prices = [float(ask[0]) for ask in asks]
            ask_sizes = [float(ask[1]) for ask in asks]

            # Update FastWarden with L2 orderbook snapshot
            self._fast_warden.update_orderbook(
                symbol,
                bid_prices,
                bid_sizes,
                ask_prices,
                ask_sizes
            )

            # Update topology score after orderbook update
            self.update_topology_score(symbol)

        except Exception as e:
            logger.warning(f"Failed to update FastWarden orderbook for {symbol}: {e}")

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def get_all_agents(self) -> list[Agent]:
        """Get all registered agents."""
        return list(self._agents.values())

    def get_market_topology_score(self) -> float:
        """Get market-wide topology score."""
        return self._market_topology_score

    # ==================== Correlation Matrix Module ====================

    def _update_correlation_data(self, tick: Tick) -> None:
        """Update price and return history for correlation calculation."""
        symbol = tick.symbol

        # Initialize data structures if needed
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self._correlation_window)
            self._returns_history[symbol] = deque(maxlen=self._correlation_window)

        # Update price history
        self._price_history[symbol].append(tick.price)

        # Calculate and store normalized return
        if len(self._price_history[symbol]) >= 2:
            prices = list(self._price_history[symbol])
            prev_price = prices[-2]
            if prev_price > 0:
                ret = (prices[-1] - prev_price) / prev_price
                self._returns_history[symbol].append(ret)

        # Recalculate correlation matrix periodically (every 10 ticks to reduce computation)
        if len(self._returns_history[symbol]) >= self._min_correlation_window and \
           len(self._returns_history[symbol]) % 10 == 0:
            self._recalculate_correlation_matrix()

    def _calculate_pearson_correlation(
        self, returns_a: list[float], returns_b: list[float]
    ) -> float:
        """Calculate Pearson correlation coefficient between two return series.

        Formula: r = Σ((Xi - X̄)(Yi - Ȳ)) / √(Σ(Xi - X̄)² * Σ(Yi - Ȳ)²)
        """
        if len(returns_a) != len(returns_b) or len(returns_a) < 2:
            return 0.0

        n = len(returns_a)

        # Calculate means
        mean_a = sum(returns_a) / n
        mean_b = sum(returns_b) / n

        # Calculate covariance and variances
        covariance = sum((returns_a[i] - mean_a) * (returns_b[i] - mean_b) for i in range(n))
        variance_a = sum((r - mean_a) ** 2 for r in returns_a)
        variance_b = sum((r - mean_b) ** 2 for r in returns_b)

        # Avoid division by zero
        denominator = math.sqrt(variance_a * variance_b)
        if denominator == 0:
            return 0.0

        correlation = covariance / denominator

        # Clamp to [-1, 1] (floating point safety)
        return max(-1.0, min(1.0, correlation))

    def _recalculate_correlation_matrix(self) -> None:
        """Recalculate correlation matrix for all symbol pairs."""
        symbols = list(self._returns_history.keys())

        # Clear old correlations
        self._correlation_matrix.clear()

        # Calculate pairwise correlations
        for i, symbol_a in enumerate(symbols):
            returns_a = list(self._returns_history[symbol_a])
            if len(returns_a) < self._min_correlation_window:
                continue

            for j, symbol_b in enumerate(symbols):
                if i >= j:  # Only calculate upper triangle (symmetric)
                    continue

                returns_b = list(self._returns_history[symbol_b])
                if len(returns_b) < self._min_correlation_window:
                    continue

                # Align lengths (use minimum length)
                min_len = min(len(returns_a), len(returns_b))
                if min_len < self._min_correlation_window:
                    continue

                aligned_a = returns_a[-min_len:]
                aligned_b = returns_b[-min_len:]

                # Calculate correlation
                correlation = self._calculate_pearson_correlation(aligned_a, aligned_b)

                # Store both directions (symmetric)
                self._correlation_matrix[(symbol_a, symbol_b)] = correlation
                self._correlation_matrix[(symbol_b, symbol_a)] = correlation

        logger.debug(f"Correlation matrix updated: {len(self._correlation_matrix)} pairs")

    async def _check_delta_neutral_hedge(self, symbol: str) -> None:
        """Check if delta-neutral hedge is needed for high-frequency scalping positions.

        During high-frequency scalping, automatically open hedge positions to maintain
        a delta-neutral portfolio when correlations are high.
        """
        # Calculate current portfolio delta
        current_delta = sum(self._active_positions.values())

        # Check if delta deviates significantly from target (delta-neutral = 0)
        delta_deviation = abs(current_delta - self._delta_neutral_target)

        if delta_deviation < 0.01:  # Already neutral (within 1% tolerance)
            return

        # Find highly correlated symbols for hedging
        hedge_candidates = self._find_hedge_candidates(symbol)

        if not hedge_candidates:
            return

        # Open hedge positions to neutralize delta
        hedge_size = -current_delta  # Opposite direction to neutralize

        for hedge_symbol, correlation in hedge_candidates:
            # Only hedge if correlation is strong (|correlation| > 0.7)
            if abs(correlation) < 0.7:
                continue

            # Adjust hedge size based on correlation
            adjusted_hedge_size = hedge_size * correlation

            # Emit hedge signal
            hedge_signal = Signal(
                strategy_id="swarm_delta_neutral",
                symbol=hedge_symbol,
                side="sell" if adjusted_hedge_size > 0 else "buy",
                size=abs(adjusted_hedge_size),
                entry_price=0.0,  # Will be filled by executor
                metadata={
                    "delta_neutral_hedge": True,
                    "original_symbol": symbol,
                    "correlation": correlation,
                    "delta_deviation": delta_deviation,
                    "target_delta": self._delta_neutral_target,
                }
            )

            await self._bus.publish(
                Event(
                    event_type=EventType.SIGNAL,
                    data={"signal": hedge_signal}
                )
            )

            logger.info(
                f"Delta-neutral hedge signal: {hedge_symbol} {hedge_signal.side} "
                f"{abs(adjusted_hedge_size):.4f} (correlation={correlation:.3f}, "
                f"delta_deviation={delta_deviation:.4f})"
            )

            # Limit to one hedge position per check to avoid over-hedging
            break

    def _find_hedge_candidates(self, symbol: str) -> list[tuple[str, float]]:
        """Find symbols highly correlated with the given symbol for hedging.

        Returns:
            List of (hedge_symbol, correlation) tuples, sorted by |correlation|
        """
        candidates: list[tuple[str, float]] = []

        for (symbol_a, symbol_b), correlation in self._correlation_matrix.items():
            if symbol_a == symbol:
                candidates.append((symbol_b, correlation))
            elif symbol_b == symbol:
                candidates.append((symbol_a, correlation))

        # Sort by absolute correlation (highest first)
        candidates.sort(key=lambda x: abs(x[1]), reverse=True)

        return candidates[:5]  # Top 5 candidates

    def _handle_position_opened(self, event: Event) -> None:
        """Handle position opened event to update delta tracking."""
        position = event.data.get("position")
        if not position:
            return

        symbol = position.get("symbol")
        size = position.get("size", 0.0)
        side = position.get("side", "buy")

        # Update active positions (long = positive, short = negative)
        if symbol:
            current_size = self._active_positions.get(symbol, 0.0)
            delta_change = size if side == "long" else -size
            self._active_positions[symbol] = current_size + delta_change

    def _handle_position_closed(self, event: Event) -> None:
        """Handle position closed event to update delta tracking."""
        position = event.data.get("position")
        if not position:
            return

        symbol = position.get("symbol")

        # Remove from active positions
        if symbol and symbol in self._active_positions:
            del self._active_positions[symbol]

    def get_correlation_matrix(self) -> dict[tuple[str, str], float]:
        """Get current correlation matrix.

        Returns:
            Dictionary mapping (symbol_a, symbol_b) tuples to correlation values
        """
        return self._correlation_matrix.copy()

    def get_correlation(self, symbol_a: str, symbol_b: str) -> float:
        """Get correlation between two symbols.

        Returns 0.0 if not available.
        """
        return self._correlation_matrix.get((symbol_a, symbol_b), 0.0)

    def get_portfolio_delta(self) -> float:
        """Get current portfolio delta (sum of all position sizes)."""
        return sum(self._active_positions.values())

    def set_delta_neutral_target(self, target: float) -> None:
        """Set target delta for delta-neutral portfolio (default: 0.0)."""
        self._delta_neutral_target = target
        logger.info(f"Delta-neutral target set to {target:.4f}")
