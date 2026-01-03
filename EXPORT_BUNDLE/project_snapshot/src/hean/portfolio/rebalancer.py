"""Profit reinvestment and rebalancing with smart reinvestment."""

from hean.config import settings
from hean.logging import get_logger
from hean.portfolio.smart_reinvestor import SmartReinvestor

logger = get_logger(__name__)


class Rebalancer:
    """Handles profit reinvestment with smart adaptive reinvestment.

    Now uses SmartReinvestor for intelligent, adaptive reinvestment
    that protects capital while maximizing growth.
    """

    def __init__(self) -> None:
        """Initialize the rebalancer with smart reinvestor."""
        self._last_rebalance_equity: float | None = None
        self._smart_reinvestor = SmartReinvestor(
            base_reinvest_rate=settings.smart_reinvest_base_rate,
            min_reserve_pct=settings.smart_reinvest_min_reserve_pct,
            drawdown_threshold=settings.smart_reinvest_drawdown_threshold,
        )

    def should_reinvest(self, current_equity: float) -> bool:
        """Check if profits should be reinvested."""
        if self._last_rebalance_equity is None:
            self._last_rebalance_equity = current_equity
            return False

        # Reinvest if equity has grown
        if current_equity > self._last_rebalance_equity:
            return True

        return False

    def calculate_reinvestment(
        self,
        current_equity: float,
        initial_capital: float,
        drawdown_pct: float = 0.0,
        rolling_pf: float = 1.0,
    ) -> float:
        """Calculate reinvestment amount using smart reinvestor.

        Args:
            current_equity: Current portfolio equity
            initial_capital: Initial capital
            drawdown_pct: Current drawdown percentage (for smart reinvestment)
            rolling_pf: Rolling profit factor (for smart reinvestment)

        Returns:
            Reinvestment amount
        """
        if self._last_rebalance_equity is None:
            self._last_rebalance_equity = current_equity
            return 0.0

        profit = current_equity - self._last_rebalance_equity
        if profit <= 0:
            return 0.0

        # Use smart reinvestor for adaptive reinvestment
        reinvest_amount = self._smart_reinvestor.calculate_smart_reinvestment(
            profit=profit,
            current_equity=current_equity,
            initial_capital=initial_capital,
            drawdown_pct=drawdown_pct,
            rolling_pf=rolling_pf,
        )

        # NO HARD LIMIT on growth - smart reinvestor handles protection
        # Removed: max_equity = initial_capital * 2.0 limit

        if reinvest_amount > 0:
            logger.info(
                f"Smart reinvestment: profit=${profit:.2f}, reinvest=${reinvest_amount:.2f}, "
                f"DD={drawdown_pct:.1f}%, PF={rolling_pf:.2f}"
            )

        return reinvest_amount

    def should_reinvest_after_trade(
        self,
        profit: float,
        current_equity: float,
        drawdown_pct: float,
        rolling_pf: float,
    ) -> bool:
        """Check if should reinvest after a profitable trade.

        Args:
            profit: Profit from the trade
            current_equity: Current portfolio equity
            drawdown_pct: Current drawdown percentage
            rolling_pf: Rolling profit factor

        Returns:
            True if should reinvest, False otherwise
        """
        return self._smart_reinvestor.should_reinvest_after_trade(
            profit, current_equity, drawdown_pct, rolling_pf
        )

    def record_rebalance(self, equity: float) -> None:
        """Record that a rebalance has occurred."""
        self._last_rebalance_equity = equity
