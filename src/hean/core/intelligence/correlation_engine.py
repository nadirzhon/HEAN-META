"""Multi-pair correlation engine for statistical arbitrage.

Calculates real-time Pearson correlation between assets and identifies
price gaps for pair trading opportunities.
"""

import math
from collections import deque
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Signal, Tick
from hean.logging import get_logger

logger = get_logger(__name__)


class CorrelationEngine:
    """Calculates real-time Pearson correlation between assets and triggers pair trades.

    Identifies "Price Gaps" where highly correlated assets diverge.
    Strategy: Long the laggard, Short the leader when correlation is high but prices diverge.
    """

    def __init__(self, bus: EventBus, symbols: list[str] | None = None) -> None:
        """Initialize the correlation engine.

        Args:
            bus: Event bus for publishing signals
            symbols: List of symbols to track (defaults to top 20 crypto assets)
        """
        self._bus = bus
        self._symbols = symbols or self._get_top_assets()
        
        # Price history for each symbol (rolling window)
        self._price_history: dict[str, deque[float]] = {
            symbol: deque(maxlen=100) for symbol in self._symbols
        }
        
        # Normalized returns (for correlation calculation)
        self._returns_history: dict[str, deque[float]] = {
            symbol: deque(maxlen=100) for symbol in self._symbols
        }
        
        # Correlation matrix (calculated periodically)
        self._correlation_matrix: dict[tuple[str, str], float] = {}
        
        # Minimum correlation threshold for pair trading
        self._min_correlation = 0.7  # 70% correlation minimum
        
        # Price gap threshold (standard deviations)
        self._gap_threshold = 2.0  # 2 standard deviations
        
        # Active pair trades
        self._active_pairs: dict[tuple[str, str], dict[str, Any]] = {}
        
        logger.info(f"Correlation engine initialized with {len(self._symbols)} symbols")

    def _get_top_assets(self) -> list[str]:
        """Get top 20 crypto assets by market cap.
        
        Returns list of trading symbols.
        """
        # Top 20 by market cap (2024)
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "TRXUSDT", "LINKUSDT", "AVAXUSDT",
            "DOTUSDT", "MATICUSDT", "SHIBUSDT", "LTCUSDT", "UNIUSDT",
            "ATOMUSDT", "ETCUSDT", "XLMUSDT", "BCHUSDT", "FILUSDT"
        ]

    async def start(self) -> None:
        """Start the correlation engine."""
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        logger.info("Correlation engine started")

    async def stop(self) -> None:
        """Stop the correlation engine."""
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        logger.info("Correlation engine stopped")

    async def _handle_tick(self, event: Event) -> None:
        """Handle tick events to update price history."""
        tick: Tick = event.data["tick"]
        
        if tick.symbol not in self._symbols:
            return
        
        # Update price history
        self._price_history[tick.symbol].append(tick.price)
        
        # Calculate and store normalized return
        if len(self._price_history[tick.symbol]) >= 2:
            prices = list(self._price_history[tick.symbol])
            prev_price = prices[-2]
            if prev_price > 0:
                ret = (prices[-1] - prev_price) / prev_price
                self._returns_history[tick.symbol].append(ret)
        
        # Update correlations and check for pair trading opportunities
        if len(self._returns_history[tick.symbol]) >= 50:  # Minimum window for correlation
            await self._update_correlations()
            await self._check_pair_opportunities()

    def _calculate_pearson_correlation(
        self, returns_a: list[float], returns_b: list[float]
    ) -> float:
        """Calculate Pearson correlation coefficient between two return series.
        
        Formula: r = Σ((Xi - X̄)(Yi - Ȳ)) / √(Σ(Xi - X̄)² * Σ(Yi - Ȳ)²)
        
        Args:
            returns_a: Return series for asset A
            returns_b: Return series for asset B
            
        Returns:
            Pearson correlation coefficient (-1.0 to 1.0)
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

    async def _update_correlations(self) -> None:
        """Update correlation matrix for all symbol pairs."""
        symbols = list(self._symbols)
        
        # Clear old correlations
        self._correlation_matrix.clear()
        
        # Calculate pairwise correlations
        for i, symbol_a in enumerate(symbols):
            if len(self._returns_history[symbol_a]) < 50:
                continue
                
            returns_a = list(self._returns_history[symbol_a])
            
            for j, symbol_b in enumerate(symbols):
                if i >= j:  # Only calculate upper triangle (symmetric)
                    continue
                    
                if len(self._returns_history[symbol_b]) < 50:
                    continue
                
                returns_b = list(self._returns_history[symbol_b])
                
                # Align lengths (use minimum length)
                min_len = min(len(returns_a), len(returns_b))
                if min_len < 50:
                    continue
                
                aligned_a = returns_a[-min_len:]
                aligned_b = returns_b[-min_len:]
                
                # Calculate correlation
                correlation = self._calculate_pearson_correlation(aligned_a, aligned_b)
                
                # Store both directions (symmetric)
                self._correlation_matrix[(symbol_a, symbol_b)] = correlation
                self._correlation_matrix[(symbol_b, symbol_a)] = correlation
        
        logger.debug(f"Updated correlation matrix with {len(self._correlation_matrix)} pairs")

    def _calculate_z_score(self, value: float, mean: float, std: float) -> float:
        """Calculate z-score (standard deviations from mean)."""
        if std == 0:
            return 0.0
        return (value - mean) / std

    def _identify_price_gap(
        self, symbol_a: str, symbol_b: str
    ) -> tuple[float, str, str] | None:
        """Identify price gaps between correlated assets.
        
        Returns:
            Tuple of (gap_z_score, laggard_symbol, leader_symbol) or None
        """
        if (symbol_a, symbol_b) not in self._correlation_matrix:
            return None
        
        correlation = self._correlation_matrix[(symbol_a, symbol_b)]
        
        # Only consider highly correlated pairs
        if abs(correlation) < self._min_correlation:
            return None
        
        # Need price history for both
        if (
            len(self._price_history[symbol_a]) < 50
            or len(self._price_history[symbol_b]) < 50
        ):
            return None
        
        prices_a = list(self._price_history[symbol_a])
        prices_b = list(self._price_history[symbol_b])
        
        # Calculate price ratio (A/B)
        min_len = min(len(prices_a), len(prices_b))
        ratios = [
            prices_a[-min_len + i] / prices_b[-min_len + i]
            if prices_b[-min_len + i] > 0
            else 1.0
            for i in range(min_len)
        ]
        
        if len(ratios) < 50:
            return None
        
        # Calculate mean and std of ratio
        mean_ratio = sum(ratios) / len(ratios)
        variance = sum((r - mean_ratio) ** 2 for r in ratios) / len(ratios)
        std_ratio = math.sqrt(variance) if variance > 0 else 0.0
        
        if std_ratio == 0:
            return None
        
        # Current ratio
        current_ratio = ratios[-1]
        
        # Calculate z-score
        z_score = self._calculate_z_score(current_ratio, mean_ratio, std_ratio)
        
        # Check if gap exceeds threshold
        if abs(z_score) < self._gap_threshold:
            return None
        
        # Determine laggard and leader
        if z_score > 0:
            # A/B is high → A is expensive relative to B → B is laggard, A is leader
            return (z_score, symbol_b, symbol_a)
        else:
            # A/B is low → A is cheap relative to B → A is laggard, B is leader
            return (abs(z_score), symbol_a, symbol_b)

    async def _check_pair_opportunities(self) -> None:
        """Check for pair trading opportunities and emit signals."""
        symbols = list(self._symbols)
        
        for i, symbol_a in enumerate(symbols):
            for j, symbol_b in enumerate(symbols):
                if i >= j:
                    continue
                
                # Skip if already in a pair trade
                if (symbol_a, symbol_b) in self._active_pairs:
                    continue
                
                gap_info = self._identify_price_gap(symbol_a, symbol_b)
                if gap_info is None:
                    continue
                
                z_score, laggard, leader = gap_info
                
                # Create pair trade signal: Long laggard, Short leader
                await self._emit_pair_signal(laggard, leader, z_score)

    async def _emit_pair_signal(self, laggard: str, leader: str, z_score: float) -> None:
        """Emit pair trading signal.
        
        Strategy: Long the laggard (expect it to catch up), Short the leader (expect it to revert).
        """
        # Get current prices
        if (
            len(self._price_history[laggard]) == 0
            or len(self._price_history[leader]) == 0
        ):
            return
        
        laggard_price = self._price_history[laggard][-1]
        leader_price = self._price_history[leader][-1]
        
        # Calculate position sizes (equal notional)
        # For pair trading, we want equal dollar exposure
        notional = 100.0  # Base notional (would be configurable)
        
        laggard_size = notional / laggard_price
        leader_size = notional / leader_price
        
        # Long laggard signal
        long_signal = Signal(
            strategy_id="correlation_pair",
            symbol=laggard,
            side="buy",
            size=laggard_size,
            entry_price=laggard_price,
            stop_loss=laggard_price * 0.98,  # 2% stop
            take_profit=laggard_price * 1.04,  # 4% target
            metadata={
                "pair_trade": True,
                "pair_symbol": leader,
                "z_score": z_score,
                "trade_type": "long_laggard"
            }
        )
        
        # Short leader signal
        short_signal = Signal(
            strategy_id="correlation_pair",
            symbol=leader,
            side="sell",
            size=leader_size,
            entry_price=leader_price,
            stop_loss=leader_price * 1.02,  # 2% stop
            take_profit=leader_price * 0.96,  # 4% target
            metadata={
                "pair_trade": True,
                "pair_symbol": laggard,
                "z_score": z_score,
                "trade_type": "short_leader"
            }
        )
        
        # Publish signals
        await self._bus.publish(
            Event(
                event_type=EventType.SIGNAL,
                data={"signal": long_signal}
            )
        )
        
        await self._bus.publish(
            Event(
                event_type=EventType.SIGNAL,
                data={"signal": short_signal}
            )
        )
        
        # Track active pair
        self._active_pairs[(laggard, leader)] = {
            "z_score": z_score,
            "entry_time": self._price_history[laggard][-1],  # Using price as timestamp proxy
            "laggard_price": laggard_price,
            "leader_price": leader_price
        }
        
        logger.info(
            f"Pair trade signal: Long {laggard} @ {laggard_price:.2f}, "
            f"Short {leader} @ {leader_price:.2f}, z-score={z_score:.2f}"
        )

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
