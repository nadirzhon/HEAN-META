"""Recursive Kelly Criterion for dynamic position sizing.

Calculates optimal position sizes based on win rate and average win/loss ratios
for each active agent in the Swarm (strategies).
"""

from typing import Any

from hean.logging import get_logger
from hean.portfolio.accounting import PortfolioAccounting

logger = get_logger(__name__)


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
        self._fractional_kelly = max(0.1, min(1.0, fractional_kelly))  # Clamp to [0.1, 1.0]
        
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
            return {sid: equal_allocation for sid in strategy_metrics.keys()}
        
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
