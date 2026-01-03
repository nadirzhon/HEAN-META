"""Smart reinvestment logic for maximizing capital growth."""

from hean.logging import get_logger

logger = get_logger(__name__)


class SmartReinvestor:
    """Smart reinvestment calculator that adapts reinvestment rate based on performance.

    Logic:
    - High drawdown (>12%): Conservative (30-40% reinvestment)
    - Low drawdown (<5%) + Good PF (>1.2): Aggressive (80-90% reinvestment)
    - Medium conditions: Moderate (50-70% reinvestment)
    - Always keeps some profit as buffer
    """

    def __init__(
        self,
        min_reinvest_pct: float = 30.0,
        max_reinvest_pct: float = 90.0,
        conservative_dd_threshold: float = 12.0,
        aggressive_dd_threshold: float = 5.0,
        good_pf_threshold: float = 1.2,
    ) -> None:
        """Initialize smart reinvestor.

        Args:
            min_reinvest_pct: Minimum reinvestment percentage (conservative mode)
            max_reinvest_pct: Maximum reinvestment percentage (aggressive mode)
            conservative_dd_threshold: Drawdown threshold for conservative mode
            aggressive_dd_threshold: Drawdown threshold for aggressive mode
            good_pf_threshold: Profit factor threshold for aggressive mode
        """
        self._min_reinvest_pct = min_reinvest_pct
        self._max_reinvest_pct = max_reinvest_pct
        self._conservative_dd_threshold = conservative_dd_threshold
        self._aggressive_dd_threshold = aggressive_dd_threshold
        self._good_pf_threshold = good_pf_threshold

    def calculate_smart_reinvestment(
        self,
        profit: float,
        current_equity: float,
        initial_capital: float,
        drawdown_pct: float,
        rolling_pf: float = 1.0,
    ) -> float:
        """Calculate smart reinvestment amount.

        Args:
            profit: Profit from the closed position
            current_equity: Current portfolio equity
            initial_capital: Initial capital
            drawdown_pct: Current drawdown percentage
            rolling_pf: Rolling profit factor for the strategy

        Returns:
            Amount to reinvest (can be 0 if conditions are poor)
        """
        if profit <= 0:
            return 0.0

        # Determine reinvestment rate based on conditions
        if drawdown_pct >= self._conservative_dd_threshold:
            # High drawdown: Conservative reinvestment
            reinvest_pct = self._min_reinvest_pct
            logger.debug(
                f"Conservative reinvestment: drawdown={drawdown_pct:.1f}% >= {self._conservative_dd_threshold}%"
            )
        elif (
            drawdown_pct <= self._aggressive_dd_threshold and rolling_pf >= self._good_pf_threshold
        ):
            # Low drawdown + good PF: Aggressive reinvestment
            reinvest_pct = self._max_reinvest_pct
            logger.debug(
                f"Aggressive reinvestment: drawdown={drawdown_pct:.1f}% <= {self._aggressive_dd_threshold}%, "
                f"PF={rolling_pf:.2f} >= {self._good_pf_threshold}"
            )
        else:
            # Medium conditions: Moderate reinvestment (interpolate)
            # Linear interpolation between min and max based on conditions
            dd_factor = (self._conservative_dd_threshold - drawdown_pct) / (
                self._conservative_dd_threshold - self._aggressive_dd_threshold
            )
            pf_factor = min(rolling_pf / self._good_pf_threshold, 1.0)
            combined_factor = (dd_factor + pf_factor) / 2.0
            reinvest_pct = self._min_reinvest_pct + (
                (self._max_reinvest_pct - self._min_reinvest_pct) * combined_factor
            )
            logger.debug(
                f"Moderate reinvestment: drawdown={drawdown_pct:.1f}%, PF={rolling_pf:.2f}, "
                f"rate={reinvest_pct:.1f}%"
            )

        # Calculate reinvestment amount
        reinvest_amount = profit * (reinvest_pct / 100.0)

        # Safety: Never reinvest more than profit
        reinvest_amount = min(reinvest_amount, profit)

        # Safety: Ensure we don't reinvest if equity is too low
        if current_equity < initial_capital * 0.5:  # Less than 50% of initial
            reinvest_amount = 0.0
            logger.warning(
                f"Reinvestment blocked: equity ${current_equity:.2f} < 50% of initial ${initial_capital:.2f}"
            )

        return reinvest_amount
