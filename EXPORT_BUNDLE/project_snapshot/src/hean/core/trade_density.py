"""Trade density control - prevents strategies from starving by relaxing filters.

Tracks per-strategy trade activity and progressively relaxes SECONDARY filters
(never risk, never KillSwitch) when no trades occur for extended periods.
"""

from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)


class TradeDensityTracker:
    """Tracks trade density per strategy and provides relaxation factors.

    Relaxation rules:
    - After 5 days idle: reduce volatility filter strictness by 10%
    - After 7 days: widen time window
    - After trade occurs: reset density relaxation
    """

    def __init__(self) -> None:
        """Initialize trade density tracker."""
        # Per-strategy tracking
        self._last_trade_timestamp: dict[str, datetime] = {}
        self._trade_history: dict[str, deque[datetime]] = defaultdict(
            lambda: deque(maxlen=30)  # Track last 30 trades
        )
        self._lookback_days = 7  # Track trades in last N days

        # Relaxation thresholds (in days)
        self._relaxation_threshold_1 = 5  # First relaxation at 5 days
        self._relaxation_threshold_2 = 7  # Second relaxation at 7 days

        # Relaxation factors
        self._volatility_relaxation_pct = 0.10  # 10% reduction
        self._time_window_expansion_hours = 2  # Expand time window by 2 hours

    def record_trade(self, strategy_id: str, timestamp: datetime | None = None) -> None:
        """Record a trade for a strategy.

        Args:
            strategy_id: Strategy identifier
            timestamp: Trade timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        self._last_trade_timestamp[strategy_id] = timestamp
        self._trade_history[strategy_id].append(timestamp)

        logger.debug(f"Trade density: recorded trade for {strategy_id} at {timestamp}")

    def get_idle_days(self, strategy_id: str, current_time: datetime | None = None) -> float:
        """Get number of days since last trade.

        Args:
            strategy_id: Strategy identifier
            current_time: Current time (defaults to now)

        Returns:
            Number of days since last trade, or 0.0 if no trades recorded
        """
        if current_time is None:
            current_time = datetime.utcnow()

        if strategy_id not in self._last_trade_timestamp:
            # No trades recorded - return a large number to trigger relaxation
            # Use a reasonable default (e.g., 30 days) for new strategies
            return 30.0

        time_since_trade = current_time - self._last_trade_timestamp[strategy_id]
        return time_since_trade.total_seconds() / (24 * 3600)

    def get_trades_last_N_days(
        self, strategy_id: str, days: int | None = None, current_time: datetime | None = None
    ) -> int:
        """Get number of trades in last N days.

        Args:
            strategy_id: Strategy identifier
            days: Number of days to look back (defaults to self._lookback_days)
            current_time: Current time (defaults to now)

        Returns:
            Number of trades in the specified period
        """
        if days is None:
            days = self._lookback_days

        if current_time is None:
            current_time = datetime.utcnow()

        cutoff = current_time - timedelta(days=days)
        trades = self._trade_history[strategy_id]

        return sum(1 for trade_time in trades if trade_time >= cutoff)

    def get_relaxation_level(self, strategy_id: str, current_time: datetime | None = None) -> int:
        """Get current relaxation level (0 = none, 1 = first, 2 = second).

        Args:
            strategy_id: Strategy identifier
            current_time: Current time (defaults to now)

        Returns:
            Relaxation level: 0 (none), 1 (first threshold), 2 (second threshold)
        """
        idle_days = self.get_idle_days(strategy_id, current_time)

        if idle_days >= self._relaxation_threshold_2:
            return 2
        elif idle_days >= self._relaxation_threshold_1:
            return 1
        else:
            return 0

    def get_volatility_relaxation_factor(
        self, strategy_id: str, current_time: datetime | None = None
    ) -> float:
        """Get volatility filter relaxation factor.

        Returns a multiplier to apply to the required volatility expansion ratio.
        Example: 0.9 means reduce required ratio by 10%.

        Args:
            strategy_id: Strategy identifier
            current_time: Current time (defaults to now)

        Returns:
            Relaxation factor (1.0 = no relaxation, < 1.0 = relaxed)
        """
        level = self.get_relaxation_level(strategy_id, current_time)

        if level >= 1:
            # Apply relaxation: reduce required ratio by relaxation_pct
            # If required ratio is 1.15, with 10% relaxation we want 1.15 * 0.9 = 1.035
            return 1.0 - self._volatility_relaxation_pct
        else:
            return 1.0

    def get_time_window_expansion_hours(
        self, strategy_id: str, current_time: datetime | None = None
    ) -> int:
        """Get time window expansion in hours.

        Args:
            strategy_id: Strategy identifier
            current_time: Current time (defaults to now)

        Returns:
            Number of hours to expand time window (0 = no expansion)
        """
        level = self.get_relaxation_level(strategy_id, current_time)

        if level >= 2:
            return self._time_window_expansion_hours
        else:
            return 0

    def get_density_state(
        self, strategy_id: str, current_time: datetime | None = None
    ) -> dict[str, Any]:
        """Get complete density state for a strategy.

        Args:
            strategy_id: Strategy identifier
            current_time: Current time (defaults to now)

        Returns:
            Dictionary with density state metrics
        """
        if current_time is None:
            current_time = datetime.utcnow()

        idle_days = self.get_idle_days(strategy_id, current_time)
        trades_last_7_days = self.get_trades_last_N_days(strategy_id, 7, current_time)
        relaxation_level = self.get_relaxation_level(strategy_id, current_time)

        return {
            "idle_days": idle_days,
            "trades_last_7_days": trades_last_7_days,
            "density_relaxation_level": relaxation_level,
            "volatility_relaxation_factor": self.get_volatility_relaxation_factor(
                strategy_id, current_time
            ),
            "time_window_expansion_hours": self.get_time_window_expansion_hours(
                strategy_id, current_time
            ),
        }


# Global trade density tracker instance
trade_density = TradeDensityTracker()
