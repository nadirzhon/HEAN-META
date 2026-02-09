"""Capital allocation to strategies."""

from collections import deque
from datetime import date

from hean.config import settings
from hean.core.regime import Regime
from hean.logging import get_logger
from hean.portfolio.capital_pressure import CapitalPressure
from hean.portfolio.strategy_memory import StrategyMemory

logger = get_logger(__name__)


class CapitalAllocator:
    """Allocates capital to different strategies with adaptive weight adjustment."""

    def __init__(self) -> None:
        """Initialize the capital allocator."""
        self._allocations: dict[str, float] = {}
        self._weights: dict[str, float] = {}  # Strategy weights (0.0 to 1.0)
        self._last_rebalance_date: date | None = None
        self._weight_history: list[dict[str, float]] = []  # Daily weight history
        self._rolling_pnl: dict[str, deque[float]] = {}  # Rolling PnL for PF calculation
        self._rolling_window = 30  # Days for rolling profit factor
        self._strategy_memory = StrategyMemory()
        # Active capital pressure layer (transient, short‑term)
        self._capital_pressure = CapitalPressure()

    def _get_enabled_strategies(self) -> list[str]:
        """Get list of enabled strategies."""
        enabled = []
        if settings.funding_harvester_enabled:
            enabled.append("funding_harvester")
        if settings.basis_arbitrage_enabled:
            enabled.append("basis_arbitrage")
        if settings.impulse_engine_enabled:
            enabled.append("impulse_engine")
        return enabled

    def _initialize_weights(self, strategies: list[str]) -> None:
        """Initialize equal weights for all strategies."""
        if not strategies:
            return

        equal_weight = 1.0 / len(strategies)
        for strategy_id in strategies:
            self._weights[strategy_id] = equal_weight
            if strategy_id not in self._rolling_pnl:
                self._rolling_pnl[strategy_id] = deque(maxlen=self._rolling_window)

    def _calculate_profit_factor(
        self, strategy_id: str, strategy_metrics: dict[str, dict[str, float]] | None = None
    ) -> float:
        """Calculate rolling profit factor for a strategy.

        Profit Factor = Sum of wins / Sum of losses

        If strategy_metrics is provided, uses that PF directly.
        Otherwise calculates from rolling PnL.
        """
        # If metrics provided, use the PF from there (more accurate)
        if strategy_metrics and strategy_id in strategy_metrics:
            pf = strategy_metrics[strategy_id].get("profit_factor", 1.0)
            if pf > 0:
                return pf

        # Fallback to rolling PnL calculation
        pnl_history = list(self._rolling_pnl.get(strategy_id, deque()))
        if not pnl_history:
            return 1.0  # Default neutral PF

        wins = sum(p for p in pnl_history if p > 0)
        losses = abs(sum(p for p in pnl_history if p < 0))

        if losses == 0:
            return wins if wins > 0 else 1.0

        return wins / losses

    def _calculate_recent_drawdown(
        self, strategy_metrics: dict[str, dict[str, float]], strategy_id: str
    ) -> float:
        """Calculate recent drawdown for a strategy."""
        if strategy_id not in strategy_metrics:
            return 0.0

        return strategy_metrics[strategy_id].get("max_drawdown_pct", 0.0)

    def _adjust_weights(
        self, strategy_metrics: dict[str, dict[str, float]], regime: Regime | None = None
    ) -> dict[str, float]:
        """Adjust strategy weights based on performance.

        Rules:
        - PF < 1.0 → decrease weight
        - PF > 1.3 → increase weight
        - max daily change ±20%
        - Apply StrategyMemory penalties before normalization
        """
        strategies = list(self._weights.keys())
        if not strategies:
            return {}

        # Use default regime if not provided
        if regime is None:
            regime = Regime.NORMAL

        new_weights: dict[str, float] = {}

        for strategy_id in strategies:
            current_weight = self._weights.get(strategy_id, 0.0)

            # Calculate profit factor (use metrics if available)
            pf = self._calculate_profit_factor(strategy_id, strategy_metrics=strategy_metrics)

            # Calculate recent drawdown
            recent_dd = self._calculate_recent_drawdown(strategy_metrics, strategy_id)

            # Update strategy memory equity for drawdown tracking
            if strategy_id in strategy_metrics:
                # Estimate equity from metrics (simplified - would use actual equity)
                pnl = strategy_metrics[strategy_id].get("pnl", 0.0)
                # Use a base equity estimate (in real system, would track actual equity)
                estimated_equity = 10000.0 + pnl  # Simplified
                self._strategy_memory.update_equity(strategy_id, estimated_equity)

            # Determine adjustment direction
            adjustment = 0.0

            # PF-based adjustment
            if pf < 1.0:
                # Decrease weight for poor performance
                adjustment = -0.15  # -15% base adjustment
            elif pf > 1.3:
                # Increase weight for strong performance
                adjustment = 0.15  # +15% base adjustment
            elif pf > 1.0:
                # Slight increase for positive PF (but not exactly 1.0)
                adjustment = 0.05  # +5% for PF between 1.0 and 1.3
            else:
                # PF == 1.0 exactly - no adjustment
                adjustment = 0.0

            # Drawdown penalty (reduce weight if high drawdown)
            if recent_dd > 10.0:  # > 10% drawdown
                adjustment -= 0.05  # Additional -5% penalty

            # Apply max daily change limit (±20%)
            adjustment = max(-0.2, min(0.2, adjustment))

            # Calculate new weight
            new_weight = current_weight * (1.0 + adjustment)

            # Apply StrategyMemory penalties BEFORE normalization
            penalty_multiplier = self._strategy_memory.get_penalty_multiplier(strategy_id, regime)
            new_weight *= penalty_multiplier

            # Apply active capital pressure BEFORE normalization
            pressure_multiplier = self._capital_pressure.get_multiplier(strategy_id)
            new_weight *= pressure_multiplier

            # Ensure minimum weight (after penalty)
            new_weight = max(0.05, new_weight)  # Minimum 5% weight
            new_weights[strategy_id] = new_weight

        # Normalize weights to sum to 1.0 (after penalties applied)
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for strategy_id in new_weights:
                new_weights[strategy_id] /= total_weight
        else:
            # Fallback to equal weights if all are zero
            equal_weight = 1.0 / len(strategies)
            for strategy_id in strategies:
                new_weights[strategy_id] = equal_weight

        return new_weights

    def _update_rolling_pnl(self, strategy_metrics: dict[str, dict[str, float]]) -> None:
        """Update rolling PnL history for profit factor calculation."""
        for strategy_id in self._weights.keys():
            if strategy_id in strategy_metrics:
                # Get daily PnL change (simplified - would track actual daily PnL)
                strategy_metrics[strategy_id].get("pnl", 0.0)
                # For rolling window, we'd track daily changes
                # For now, use current PnL as proxy
                if strategy_id not in self._rolling_pnl:
                    self._rolling_pnl[strategy_id] = deque(maxlen=self._rolling_window)

                # In real system, would track daily PnL deltas
                # For now, we'll use the strategy's win/loss data
                wins = strategy_metrics[strategy_id].get("wins", 0)
                losses = strategy_metrics[strategy_id].get("losses", 0)

                # Simulate rolling PnL by adding recent wins/losses
                # This is simplified - real system would track actual daily PnL
                if wins > 0 or losses > 0:
                    # Add a sample PnL entry (positive for wins, negative for losses)
                    # In practice, this would be actual daily PnL changes
                    pass

    def update_weights(self, strategy_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
        """Update strategy weights based on performance metrics.

        Should be called daily to adjust capital allocation.

        Returns:
            Updated weights dictionary
        """
        today = date.today()  # Use date.today() for consistency with tests

        # Initialize weights if first time
        if not self._weights:
            strategies = self._get_enabled_strategies()
            self._initialize_weights(strategies)

        # Force equal allocation mode - skip adaptive logic
        from hean.config import settings
        if settings.force_equal_allocation:
            strategies = self._get_enabled_strategies()
            if strategies:
                equal_weight = 1.0 / len(strategies)
                self._weights = dict.fromkeys(strategies, equal_weight)
                logger.info(f"Force equal allocation: {self._weights}")
            return self._weights

        # Only adjust once per day
        if self._last_rebalance_date == today:
            return self._weights

        # Update rolling PnL
        self._update_rolling_pnl(strategy_metrics)

        # Update active capital pressure state (decay + drawdown observations)
        self._capital_pressure.update_from_metrics(strategy_metrics)

        # Adjust weights (returns normalized weights)
        new_weights = self._adjust_weights(strategy_metrics)

        # Verify daily change limits (compare normalized weights)
        # Store pre-enforcement weights for comparison
        new_weights.copy()

        for strategy_id in new_weights:
            old_weight = self._weights.get(strategy_id, 0.0)
            new_weight = new_weights[strategy_id]
            change_pct = (
                abs((new_weight - old_weight) / old_weight * 100) if old_weight > 0 else 0.0
            )

            if change_pct > 20.0:
                # Enforce max change
                if new_weight > old_weight:
                    new_weight = old_weight * 1.2
                else:
                    new_weight = old_weight * 0.8
                new_weights[strategy_id] = new_weight

        # Re-normalize after enforcing limits
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for strategy_id in new_weights:
                new_weights[strategy_id] /= total_weight

        # Update weights
        self._weights = new_weights
        self._last_rebalance_date = today

        # Record weight history
        self._weight_history.append({**new_weights, "_date": today.isoformat()})

        logger.info(f"Updated strategy weights: {new_weights}")

        return new_weights

    def allocate(
        self, equity: float, strategy_metrics: dict[str, dict[str, float]] | None = None
    ) -> dict[str, float]:
        """Allocate capital to strategies based on adaptive weights.

        Args:
            equity: Total equity available
            strategy_metrics: Optional strategy performance metrics for weight adjustment

        Returns:
            Dictionary mapping strategy_id to allocated capital
        """
        # Reserve cash
        reserved = equity * settings.cash_reserve_rate
        available = equity - reserved

        # Get enabled strategies
        enabled_strategies = self._get_enabled_strategies()

        if not enabled_strategies:
            logger.warning("No strategies enabled")
            return {}

        # Initialize weights if first time
        if not self._weights:
            self._initialize_weights(enabled_strategies)

        # Force equal allocation if configured
        if settings.force_equal_allocation and enabled_strategies:
            equal_weight = 1.0 / len(enabled_strategies)
            self._weights = dict.fromkeys(enabled_strategies, equal_weight)
        # Update weights if metrics provided (adaptive routing)
        elif strategy_metrics:
            self.update_weights(strategy_metrics)

        # Allocate based on weights
        allocations: dict[str, float] = {}
        for strategy_id in enabled_strategies:
            weight = self._weights.get(strategy_id, 0.0)
            allocations[strategy_id] = available * weight

        self._allocations = allocations
        logger.debug(f"Capital allocation: {allocations}")

        return allocations

    def get_allocation(self, strategy_id: str) -> float:
        """Get allocated capital for a strategy."""
        return self._allocations.get(strategy_id, 0.0)

    def get_weights(self) -> dict[str, float]:
        """Get current strategy weights."""
        return self._weights.copy()

    def get_weight_history(self) -> list[dict[str, float]]:
        """Get weight history over time."""
        return self._weight_history.copy()

    def get_strategy_memory(self) -> StrategyMemory:
        """Get the strategy memory instance for external updates."""
        return self._strategy_memory

    def get_capital_pressure(self) -> CapitalPressure:
        """Get the capital pressure instance for external updates/tests."""
        return self._capital_pressure
