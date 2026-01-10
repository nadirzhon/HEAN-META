"""Execution edge estimator for signal filtering."""

from collections import deque

from hean.config import settings
from hean.core.regime import Regime
from hean.core.types import Signal, Tick
from hean.logging import get_logger
from hean.paper_trade_assist import (
    get_edge_threshold_reduction_pct,
    is_paper_assist_enabled,
    log_allow_reason,
    log_block_reason,
)

logger = get_logger(__name__)


class ExecutionEdgeEstimator:
    """Estimates execution edge for trading signals.

    Edge calculation includes:
    - Expected move toward take_profit (in bps)
    - Spread cost (ask-bid)
    - Maker fill probability proxy (based on spread size and regime)
    - Volatility penalty (higher volatility reduces certainty)
    - Regime adjustment (IMPULSE: higher threshold, RANGE: stricter threshold)
    """

    def __init__(self) -> None:
        """Initialize the edge estimator."""
        self._volatility_history: dict[str, deque[float]] = {}
        self._window_size = 20  # Lookback window for volatility
        self._signals_blocked_by_edge = 0
        self._edge_sum = 0.0
        self._edge_count = 0

    def estimate_edge(self, signal: Signal, tick: Tick, regime: Regime) -> float:
        """Estimate execution edge in basis points.

        Args:
            signal: Trading signal
            tick: Current market tick
            regime: Current market regime

        Returns:
            Estimated edge in basis points (bps)
        """
        if not tick.bid or not tick.ask:
            return -1000.0  # No edge if no bid/ask

        # Calculate spread cost in bps
        spread = tick.ask - tick.bid
        spread_bps = (spread / tick.price) * 10000 if tick.price > 0 else 0

        # Calculate expected move toward take_profit in bps
        expected_move_bps = 0.0
        if signal.take_profit:
            if signal.side == "buy":
                move = (signal.take_profit - signal.entry_price) / signal.entry_price if signal.entry_price != 0 else 0.0
            else:  # sell
                move = (signal.entry_price - signal.take_profit) / signal.entry_price if signal.entry_price != 0 else 0.0
            expected_move_bps = move * 10000

        # Calculate maker fill probability proxy
        # Improved model: more realistic fill probability based on spread and offset
        # With maker_price_offset_bps=2, orders are placed 2 bps away from best bid/ask
        # This gives reasonable fill probability (typically 60-80% depending on spread)
        base_fill_prob = 0.75  # Increased from 0.8 to 0.75 (more realistic)

        # Spread penalty: larger spreads reduce fill probability
        # Normalize spread penalty: spread of 8 bps = 50% penalty, spread of 4 bps = 25% penalty
        spread_penalty = min(spread_bps / 16.0, 0.6)  # Max 60% penalty for very large spreads

        # Regime adjustments: RANGE has tighter spreads (higher fill prob), IMPULSE has wider spreads
        if regime == Regime.RANGE:
            fill_prob = base_fill_prob * (1.0 - spread_penalty * 0.4)  # Less penalty in RANGE
        elif regime == Regime.IMPULSE:
            fill_prob = base_fill_prob * (1.0 - spread_penalty * 1.2)  # More penalty in IMPULSE
        else:  # NORMAL
            fill_prob = base_fill_prob * (1.0 - spread_penalty * 0.8)  # Moderate penalty

        # Clamp between 0.3 and 0.95 (more realistic range)
        fill_prob = max(0.3, min(0.95, fill_prob))

        # Calculate volatility penalty
        volatility_penalty = self._get_volatility_penalty(tick.symbol, tick.price)

        # Calculate raw edge (expected move adjusted for fill probability)
        raw_edge_bps = expected_move_bps * fill_prob - spread_bps

        # Apply volatility penalty
        edge_bps = raw_edge_bps * (1.0 - volatility_penalty)

        # Regime adjustment (already factored into fill_prob, but add small adjustment)
        if regime == Regime.IMPULSE:
            # Allow slightly higher edge threshold (reduce penalty by 5%)
            edge_bps *= 1.05
        elif regime == Regime.RANGE:
            # Stricter threshold (add 5% penalty)
            edge_bps *= 0.95

        return edge_bps

    def _get_volatility_penalty(self, symbol: str, current_price: float) -> float:
        """Calculate volatility penalty based on recent price history.

        Higher volatility reduces certainty, thus reducing edge.

        Returns:
            Volatility penalty (0.0 to 1.0)
        """
        if symbol not in self._volatility_history:
            self._volatility_history[symbol] = deque(maxlen=self._window_size)
            return 0.1  # Default low penalty if no history

        history = list(self._volatility_history[symbol])
        if len(history) < 5:
            return 0.1  # Low penalty if insufficient data

        # Calculate rolling volatility
        returns = []
        for i in range(1, len(history)):
            if history[i - 1] > 0:
                ret = abs((history[i] - history[i - 1]) / history[i - 1])
                returns.append(ret)

        if not returns:
            return 0.1

        volatility = sum(returns) / len(returns)

        # Convert volatility to penalty (0.0 to 0.5 max)
        # High volatility (e.g., 0.01 = 1%) -> higher penalty
        penalty = min(volatility * 50, 0.5)  # Cap at 50% penalty

        return penalty

    def update_price_history(self, symbol: str, price: float) -> None:
        """Update price history for volatility calculation."""
        if symbol not in self._volatility_history:
            self._volatility_history[symbol] = deque(maxlen=self._window_size)
        self._volatility_history[symbol].append(price)

    def get_min_edge_threshold(self, regime: Regime) -> float:
        """Get minimum edge threshold for a regime.

        Args:
            regime: Current market regime

        Returns:
            Minimum edge threshold in bps
        """
        # REDUCED BY 50% FOR DEBUG
        if regime == Regime.IMPULSE:
            base_threshold = settings.min_edge_bps_impulse * 0.5
        elif regime == Regime.RANGE:
            base_threshold = settings.min_edge_bps_range * 0.5
        else:  # NORMAL
            base_threshold = settings.min_edge_bps_normal * 0.5
        
        # Apply paper assist reduction
        if is_paper_assist_enabled():
            reduction_pct = get_edge_threshold_reduction_pct()
            base_threshold = base_threshold * (1.0 - reduction_pct / 100.0)
        
        return base_threshold

    def should_emit_signal(self, signal: Signal, tick: Tick, regime: Regime) -> bool:
        """Check if signal should be emitted based on edge estimation.

        Args:
            signal: Trading signal
            tick: Current market tick
            regime: Current market regime

        Returns:
            True if signal should be emitted, False otherwise
        """
        edge_bps = self.estimate_edge(signal, tick, regime)
        min_threshold = self.get_min_edge_threshold(regime)

        # Track metrics
        self._edge_sum += edge_bps
        self._edge_count += 1

        if edge_bps < min_threshold:
            self._signals_blocked_by_edge += 1
            log_block_reason(
                "edge_reject",
                measured_value=edge_bps,
                threshold=min_threshold,
                symbol=tick.symbol,
                strategy_id=signal.strategy_id,
                agent_name=signal.strategy_id,
            )
            logger.debug(
                f"Signal blocked by edge: edge={edge_bps:.1f} bps < "
                f"threshold={min_threshold} bps (regime={regime.value})"
            )
            return False

        log_allow_reason("edge_ok", symbol=tick.symbol, strategy_id=signal.strategy_id)
        return True

    def get_metrics(self) -> dict[str, float]:
        """Get edge estimator metrics.

        Returns:
            Dictionary with metrics:
            - signals_blocked_by_edge: Number of signals blocked
            - avg_edge_bps: Average edge in bps
        """
        avg_edge = self._edge_sum / self._edge_count if self._edge_count > 0 else 0.0

        return {
            "signals_blocked_by_edge": float(self._signals_blocked_by_edge),
            "avg_edge_bps": avg_edge,
            "total_signals_evaluated": float(self._edge_count),
        }

    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        self._signals_blocked_by_edge = 0
        self._edge_sum = 0.0
        self._edge_count = 0
