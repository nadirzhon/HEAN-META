"""Recursive Kelly Criterion for dynamic position sizing.

Calculates optimal position sizes based on win rate and average win/loss ratios
for each active agent in the Swarm (strategies).

Enhanced with:
- Confidence-based scaling (higher confidence = closer to full Kelly)
- Adaptive fractional Kelly (adjusts based on recent performance)
- Streak tracking (reduce size after losing streaks)
- Bayesian win rate estimation (smoothed with prior)
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from hean.logging import get_logger
from hean.portfolio.accounting import PortfolioAccounting

logger = get_logger(__name__)


@dataclass
class StrategyPerformanceTracker:
    """Tracks performance metrics for a strategy."""
    strategy_id: str
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=50))
    current_streak: int = 0  # Positive = wins, negative = losses
    max_streak: int = 0
    min_streak: int = 0
    last_update: datetime | None = None

    # Adaptive Kelly state
    adaptive_fraction: float = 0.25  # Starts at default
    fraction_adjustment_rate: float = 0.05  # How fast to adjust


class KellyCriterion:
    """Recursive Kelly Criterion position sizing.

    The Kelly Criterion calculates the optimal fraction of capital to bet:
    f* = (p * b - q) / b

    Where:
    - f* = optimal fraction of capital
    - p = win probability (win rate)
    - q = loss probability (1 - p)
    - b = odds ratio (average win / average loss)

    For trading, we use fractional Kelly (typically 0.25x to 0.5x) for safety.
    """

    # Adaptive Kelly parameters
    MIN_FRACTIONAL_KELLY = 0.15  # Never go below 15%
    MAX_FRACTIONAL_KELLY = 0.50  # Never exceed 50%
    STREAK_PENALTY_THRESHOLD = 3  # Reduce after 3 consecutive losses
    STREAK_BOOST_THRESHOLD = 5  # Increase after 5 consecutive wins

    # Confidence scaling parameters
    MIN_CONFIDENCE_MULT = 0.5  # Low confidence = 50% of Kelly size
    MAX_CONFIDENCE_MULT = 1.5  # High confidence = 150% of Kelly size

    # Bayesian prior for win rate smoothing
    PRIOR_WINS = 5  # Assume 5 wins initially
    PRIOR_LOSSES = 5  # Assume 5 losses initially (50% win rate prior)

    def __init__(self, accounting: PortfolioAccounting, fractional_kelly: float = 0.25) -> None:
        """Initialize the Kelly Criterion calculator.

        Args:
            accounting: Portfolio accounting for strategy metrics
            fractional_kelly: Fraction of full Kelly to use (default 0.25 = quarter Kelly)
                - 1.0 = full Kelly (aggressive, high variance)
                - 0.5 = half Kelly (moderate)
                - 0.25 = quarter Kelly (conservative, recommended)
        """
        self._accounting = accounting
        self._base_fractional_kelly = max(0.1, min(1.0, fractional_kelly))  # Clamp to [0.1, 1.0]
        self._fractional_kelly = self._base_fractional_kelly

        # Performance tracking per strategy
        self._strategy_trackers: dict[str, StrategyPerformanceTracker] = {}

        # Global performance tracking
        self._total_trades = 0
        self._total_wins = 0
        self._global_streak = 0

        logger.info(f"Kelly Criterion initialized with fractional_kelly={self._fractional_kelly}")

    def calculate_kelly_fraction(
        self, strategy_id: str, strategy_metrics: dict[str, dict[str, float]] | None = None
    ) -> float:
        """Calculate optimal Kelly fraction for a strategy.

        Args:
            strategy_id: Strategy identifier
            strategy_metrics: Optional strategy metrics dictionary

        Returns:
            Optimal Kelly fraction (0.0 to 1.0), or 0.0 if insufficient data
        """
        # Get strategy metrics
        if strategy_metrics is None:
            strategy_metrics = self._accounting.get_strategy_metrics()

        if strategy_id not in strategy_metrics:
            return 0.0

        metrics = strategy_metrics[strategy_id]

        # Get win/loss statistics
        wins = metrics.get("wins", 0)
        losses = metrics.get("losses", 0)
        total_trades = wins + losses

        # Need minimum trades for statistical significance
        if total_trades < 10:
            logger.debug(f"Insufficient trades for Kelly: {total_trades} < 10")
            return 0.0

        # Calculate win probability (p)
        win_rate = wins / total_trades

        # Calculate average win and average loss
        avg_win = metrics.get("avg_win", 0.0)
        avg_loss = abs(metrics.get("avg_loss", 0.0))  # Use absolute value

        # Handle edge cases
        if avg_loss == 0:
            if avg_win > 0:
                # Only wins - use conservative fraction
                return min(0.1, win_rate * self._fractional_kelly)
            else:
                # No data
                return 0.0

        # Calculate odds ratio (b = average win / average loss)
        odds_ratio = avg_win / avg_loss

        # Kelly formula: f* = (p * b - q) / b
        # Where q = 1 - p
        loss_probability = 1.0 - win_rate

        # Kelly fraction
        kelly_fraction = (win_rate * odds_ratio - loss_probability) / odds_ratio

        # CRITICAL: Reject trades with negative edge
        # Negative Kelly means expected value is negative - NEVER trade
        if kelly_fraction <= 0:
            logger.warning(
                f"REJECTING trade for {strategy_id}: negative Kelly edge "
                f"(kelly={kelly_fraction:.4f}, win_rate={win_rate:.2%}, odds={odds_ratio:.2f})"
            )
            return 0.0

        # Apply fractional Kelly
        fractional_kelly = kelly_fraction * self._fractional_kelly

        # Clamp to safe bounds [0, 0.5]
        # Never risk more than 50% of capital per strategy
        fractional_kelly = max(0.0, min(0.5, fractional_kelly))

        logger.debug(
            f"Kelly for {strategy_id}: win_rate={win_rate:.2%}, "
            f"odds={odds_ratio:.2f}, kelly={kelly_fraction:.4f}, "
            f"fractional={fractional_kelly:.4f}"
        )

        return fractional_kelly

    def calculate_strategy_allocation(
        self,
        equity: float,
        strategy_metrics: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Calculate capital allocation across strategies using Kelly Criterion.

        Each strategy gets capital proportional to its Kelly edge.
        This creates a "swarm" where better-performing strategies get more capital.

        Args:
            equity: Total equity available
            strategy_metrics: Dictionary of strategy metrics

        Returns:
            Dictionary mapping strategy_id to allocated capital
        """
        # Calculate Kelly fraction for each strategy
        kelly_fractions: dict[str, float] = {}

        for strategy_id in strategy_metrics:
            kelly_frac = self.calculate_kelly_fraction(strategy_id, strategy_metrics)
            kelly_fractions[strategy_id] = kelly_frac

        # Normalize Kelly fractions (they become weights)
        total_kelly = sum(kelly_fractions.values())

        if total_kelly == 0:
            # No valid Kelly fractions - use equal allocation
            logger.warning("No valid Kelly fractions, using equal allocation")
            equal_allocation = equity / len(strategy_metrics) if strategy_metrics else 0.0
            return dict.fromkeys(strategy_metrics.keys(), equal_allocation)

        # Normalize and allocate capital
        allocations: dict[str, float] = {}

        for strategy_id, kelly_frac in kelly_fractions.items():
            # Weight = normalized Kelly fraction
            weight = kelly_frac / total_kelly
            allocations[strategy_id] = equity * weight

        logger.info(f"Kelly-based allocation: {allocations}")

        return allocations

    def calculate_position_size_kelly(
        self,
        strategy_id: str,
        equity: float,
        current_price: float,
        stop_loss: float,
        strategy_metrics: dict[str, dict[str, float]] | None = None,
    ) -> float:
        """Calculate position size using Kelly Criterion.

        This integrates Kelly with risk-based position sizing:
        - Calculate Kelly fraction for the strategy
        - Apply to risk-based size calculation

        Args:
            strategy_id: Strategy identifier
            equity: Available equity
            current_price: Current asset price
            stop_loss: Stop loss price
            strategy_metrics: Optional strategy metrics

        Returns:
            Position size in asset units
        """
        # Get Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(strategy_id, strategy_metrics)

        if kelly_fraction <= 0:
            return 0.0

        # Calculate stop loss distance
        stop_distance_pct = abs((current_price - stop_loss) / current_price) * 100

        if stop_distance_pct <= 0:
            return 0.0

        # Calculate position size
        # Risk amount = equity * kelly_fraction
        risk_amount = equity * kelly_fraction

        # Position size = risk_amount / stop_distance_per_unit
        stop_distance_per_unit = current_price * (stop_distance_pct / 100.0)
        position_size = risk_amount / stop_distance_per_unit

        # Ensure minimum size
        min_size = (equity * 0.001) / current_price  # 0.1% minimum
        position_size = max(position_size, min_size)

        logger.debug(
            f"Kelly position size for {strategy_id}: "
            f"kelly_frac={kelly_fraction:.4f}, risk=${risk_amount:.2f}, "
            f"stop_distance={stop_distance_pct:.2f}%, size={position_size:.6f}"
        )

        return position_size

    def get_strategy_win_rate(self, strategy_id: str) -> float:
        """Get win rate for a strategy.

        Returns:
            Win rate (0.0 to 1.0), or 0.5 if no data
        """
        strategy_metrics = self._accounting.get_strategy_metrics()

        if strategy_id not in strategy_metrics:
            return 0.5  # Neutral

        metrics = strategy_metrics[strategy_id]
        wins = metrics.get("wins", 0)
        losses = metrics.get("losses", 0)
        total = wins + losses

        if total == 0:
            return 0.5

        return wins / total

    def get_strategy_edge(self, strategy_id: str) -> float:
        """Get mathematical edge for a strategy (Kelly-based).

        Returns:
            Edge value (can be negative), or 0.0 if insufficient data
        """
        strategy_metrics = self._accounting.get_strategy_metrics()

        if strategy_id not in strategy_metrics:
            return 0.0

        metrics = strategy_metrics[strategy_id]
        wins = metrics.get("wins", 0)
        losses = metrics.get("losses", 0)
        total = wins + losses

        if total < 10:
            return 0.0

        win_rate = wins / total
        avg_win = metrics.get("avg_win", 0.0)
        avg_loss = abs(metrics.get("avg_loss", 0.0))

        if avg_loss == 0:
            return 0.0

        odds_ratio = avg_win / avg_loss
        edge = (win_rate * odds_ratio - (1.0 - win_rate)) / odds_ratio

        return edge * self._fractional_kelly

    def calculate_kelly_with_confidence(
        self,
        strategy_id: str,
        signal_confidence: float,
        strategy_metrics: dict[str, dict[str, float]] | None = None,
    ) -> float:
        """Calculate Kelly fraction scaled by signal confidence.

        Higher confidence signals get sizes closer to full Kelly.
        Lower confidence signals are scaled down.

        Args:
            strategy_id: Strategy identifier
            signal_confidence: Signal confidence (0.0 to 1.0)
            strategy_metrics: Optional strategy metrics

        Returns:
            Confidence-scaled Kelly fraction
        """
        # Get base Kelly fraction
        base_kelly = self.calculate_kelly_fraction(strategy_id, strategy_metrics)

        if base_kelly <= 0:
            return 0.0

        # Scale by confidence
        # confidence 0.5 = baseline, 1.0 = max boost, 0.0 = max reduction
        confidence_multiplier = self.MIN_CONFIDENCE_MULT + (
            (self.MAX_CONFIDENCE_MULT - self.MIN_CONFIDENCE_MULT) * signal_confidence
        )

        # Get streak adjustment
        tracker = self._get_or_create_tracker(strategy_id)
        streak_multiplier = self._calculate_streak_multiplier(tracker)

        # Apply adaptive fractional Kelly for this strategy
        adaptive_kelly = tracker.adaptive_fraction

        # Combine all factors
        final_kelly = base_kelly * confidence_multiplier * streak_multiplier * (adaptive_kelly / self._base_fractional_kelly)

        # Clamp to safe bounds
        final_kelly = max(0.0, min(0.5, final_kelly))

        logger.debug(
            f"Kelly with confidence for {strategy_id}: base={base_kelly:.4f}, "
            f"confidence={signal_confidence:.2f}, conf_mult={confidence_multiplier:.2f}, "
            f"streak_mult={streak_multiplier:.2f}, final={final_kelly:.4f}"
        )

        return final_kelly

    def record_trade_result(self, strategy_id: str, is_win: bool, pnl_pct: float) -> None:
        """Record a trade result for adaptive Kelly calculation.

        Args:
            strategy_id: Strategy identifier
            is_win: Whether the trade was profitable
            pnl_pct: Profit/loss percentage
        """
        tracker = self._get_or_create_tracker(strategy_id)

        # Record trade
        tracker.recent_trades.append({
            "is_win": is_win,
            "pnl_pct": pnl_pct,
            "timestamp": datetime.utcnow(),
        })
        tracker.last_update = datetime.utcnow()

        # Update streak
        if is_win:
            if tracker.current_streak >= 0:
                tracker.current_streak += 1
            else:
                tracker.current_streak = 1
            tracker.max_streak = max(tracker.max_streak, tracker.current_streak)
        else:
            if tracker.current_streak <= 0:
                tracker.current_streak -= 1
            else:
                tracker.current_streak = -1
            tracker.min_streak = min(tracker.min_streak, tracker.current_streak)

        # Update global stats
        self._total_trades += 1
        if is_win:
            self._total_wins += 1
            self._global_streak = self._global_streak + 1 if self._global_streak >= 0 else 1
        else:
            self._global_streak = self._global_streak - 1 if self._global_streak <= 0 else -1

        # Adapt fractional Kelly based on recent performance
        self._adapt_kelly_fraction(tracker)

        logger.debug(
            f"Trade recorded for {strategy_id}: win={is_win}, pnl={pnl_pct:.2f}%, "
            f"streak={tracker.current_streak}, adaptive_kelly={tracker.adaptive_fraction:.3f}"
        )

    def _get_or_create_tracker(self, strategy_id: str) -> StrategyPerformanceTracker:
        """Get or create a performance tracker for a strategy."""
        if strategy_id not in self._strategy_trackers:
            self._strategy_trackers[strategy_id] = StrategyPerformanceTracker(
                strategy_id=strategy_id,
                adaptive_fraction=self._base_fractional_kelly,
            )
        return self._strategy_trackers[strategy_id]

    def _calculate_streak_multiplier(self, tracker: StrategyPerformanceTracker) -> float:
        """Calculate position size multiplier based on current streak.

        Losing streaks reduce size, winning streaks can slightly increase.
        """
        streak = tracker.current_streak

        if streak <= -self.STREAK_PENALTY_THRESHOLD:
            # Losing streak - reduce size progressively
            # 3 losses = 0.9x, 5 losses = 0.7x, 7 losses = 0.5x
            penalty_factor = abs(streak) - self.STREAK_PENALTY_THRESHOLD
            multiplier = max(0.5, 1.0 - (penalty_factor * 0.1))
            logger.debug(
                f"Losing streak {streak}: applying {multiplier:.2f}x multiplier"
            )
            return multiplier

        elif streak >= self.STREAK_BOOST_THRESHOLD:
            # Winning streak - slight boost (but capped to avoid overconfidence)
            # 5 wins = 1.05x, 10 wins = 1.1x (capped)
            bonus_factor = streak - self.STREAK_BOOST_THRESHOLD
            multiplier = min(1.1, 1.0 + (bonus_factor * 0.01))
            return multiplier

        return 1.0

    def _adapt_kelly_fraction(self, tracker: StrategyPerformanceTracker) -> None:
        """Adapt the fractional Kelly based on recent performance.

        Strategy: Increase Kelly when doing well, decrease when doing poorly.
        """
        if len(tracker.recent_trades) < 10:
            return  # Not enough data

        # Calculate recent win rate
        recent_wins = sum(1 for t in tracker.recent_trades if t["is_win"])
        recent_win_rate = recent_wins / len(tracker.recent_trades)

        # Target adjustment based on win rate
        # > 60% win rate: increase Kelly
        # < 40% win rate: decrease Kelly
        # 40-60%: maintain
        if recent_win_rate > 0.60:
            target_adjustment = tracker.fraction_adjustment_rate
        elif recent_win_rate < 0.40:
            target_adjustment = -tracker.fraction_adjustment_rate
        else:
            target_adjustment = 0.0

        # Apply adjustment
        new_fraction = tracker.adaptive_fraction + target_adjustment
        new_fraction = max(self.MIN_FRACTIONAL_KELLY, min(self.MAX_FRACTIONAL_KELLY, new_fraction))
        tracker.adaptive_fraction = new_fraction

    def calculate_bayesian_win_rate(
        self, strategy_id: str, strategy_metrics: dict[str, dict[str, float]] | None = None
    ) -> float:
        """Calculate Bayesian-smoothed win rate.

        Uses a prior to avoid extreme values with limited data.
        With 0 trades, returns 0.5 (prior).
        With many trades, converges to actual win rate.

        Args:
            strategy_id: Strategy identifier
            strategy_metrics: Optional strategy metrics

        Returns:
            Bayesian-smoothed win rate (0.0 to 1.0)
        """
        if strategy_metrics is None:
            strategy_metrics = self._accounting.get_strategy_metrics()

        if strategy_id not in strategy_metrics:
            return 0.5  # Prior

        metrics = strategy_metrics[strategy_id]
        wins = metrics.get("wins", 0)
        losses = metrics.get("losses", 0)

        # Bayesian smoothing with conjugate prior
        smoothed_wins = wins + self.PRIOR_WINS
        smoothed_losses = losses + self.PRIOR_LOSSES
        smoothed_total = smoothed_wins + smoothed_losses

        return smoothed_wins / smoothed_total

    def get_strategy_performance_summary(self, strategy_id: str) -> dict:
        """Get performance summary for a strategy.

        Returns:
            Dictionary with performance metrics
        """
        tracker = self._strategy_trackers.get(strategy_id)
        if tracker is None:
            return {
                "strategy_id": strategy_id,
                "trades": 0,
                "current_streak": 0,
                "adaptive_kelly": self._base_fractional_kelly,
            }

        recent_wins = sum(1 for t in tracker.recent_trades if t["is_win"]) if tracker.recent_trades else 0
        recent_total = len(tracker.recent_trades)

        return {
            "strategy_id": strategy_id,
            "trades": recent_total,
            "recent_wins": recent_wins,
            "recent_win_rate": recent_wins / recent_total if recent_total > 0 else 0.5,
            "current_streak": tracker.current_streak,
            "max_win_streak": tracker.max_streak,
            "max_loss_streak": abs(tracker.min_streak),
            "adaptive_kelly": tracker.adaptive_fraction,
            "last_update": tracker.last_update.isoformat() if tracker.last_update else None,
        }

    def get_global_performance(self) -> dict:
        """Get global performance across all strategies."""
        return {
            "total_trades": self._total_trades,
            "total_wins": self._total_wins,
            "global_win_rate": self._total_wins / self._total_trades if self._total_trades > 0 else 0.5,
            "global_streak": self._global_streak,
            "strategies_tracked": len(self._strategy_trackers),
            "base_fractional_kelly": self._base_fractional_kelly,
        }
