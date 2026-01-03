"""Capital preservation mode for automatic protection.

This module implements automatic capital preservation mode that:
- Activates when problems are detected (drawdown > 12%, PF < 0.9, 3+ consecutive losses)
- Reduces risk per trade to 0.5% (from 1%)
- Disables leverage (1x only)
- Only allows conservative strategies
- Focuses on recovery, not growth
"""

from hean.logging import get_logger

logger = get_logger(__name__)


class CapitalPreservationMode:
    """Capital preservation mode for automatic protection.

    Automatically activates when problems are detected and implements
    conservative trading rules to protect capital.
    """

    def __init__(
        self,
        drawdown_threshold: float = 12.0,
        pf_threshold: float = 0.9,
        consecutive_losses_threshold: int = 3,
    ) -> None:
        """Initialize capital preservation mode.

        Args:
            drawdown_threshold: Drawdown % to activate (default 12%)
            pf_threshold: PF threshold to activate (default 0.9)
            consecutive_losses_threshold: Consecutive losses to activate (default 3)
        """
        self._drawdown_threshold = drawdown_threshold
        self._pf_threshold = pf_threshold
        self._consecutive_losses_threshold = consecutive_losses_threshold
        self._active = False
        self._activation_reason = ""

    def should_activate(
        self, drawdown_pct: float, rolling_pf: float, consecutive_losses: int
    ) -> bool:
        """Check if capital preservation mode should be activated.

        Args:
            drawdown_pct: Current drawdown percentage
            rolling_pf: Rolling profit factor
            consecutive_losses: Number of consecutive losses

        Returns:
            True if should activate, False otherwise
        """
        if drawdown_pct > self._drawdown_threshold:
            return True
        if rolling_pf < self._pf_threshold:
            return True
        if consecutive_losses >= self._consecutive_losses_threshold:
            return True
        return False

    def activate(self, reason: str) -> None:
        """Activate capital preservation mode.

        Args:
            reason: Reason for activation
        """
        if not self._active:
            self._active = True
            self._activation_reason = reason
            logger.critical(
                f"CapitalPreservationMode: ACTIVATED - {reason}. "
                f"Risk reduced to 0.5%, leverage disabled, only conservative strategies allowed."
            )

    def deactivate(self) -> None:
        """Deactivate capital preservation mode."""
        if self._active:
            logger.info("CapitalPreservationMode: DEACTIVATED. Resuming normal trading parameters.")
            self._active = False
            self._activation_reason = ""

    def update(self, drawdown_pct: float, rolling_pf: float, consecutive_losses: int) -> None:
        """Update capital preservation mode state.

        Automatically activates/deactivates based on current conditions.

        Args:
            drawdown_pct: Current drawdown percentage
            rolling_pf: Rolling profit factor
            consecutive_losses: Number of consecutive losses
        """
        if self._active:
            # Check if conditions improved enough to deactivate
            # Need drawdown < 8%, PF > 1.1, and no recent consecutive losses
            if drawdown_pct < 8.0 and rolling_pf > 1.1 and consecutive_losses == 0:
                self.deactivate()
        else:
            # Check if should activate
            if self.should_activate(drawdown_pct, rolling_pf, consecutive_losses):
                reason = ""
                if drawdown_pct > self._drawdown_threshold:
                    reason = f"Drawdown {drawdown_pct:.1f}% > {self._drawdown_threshold}%"
                elif rolling_pf < self._pf_threshold:
                    reason = f"PF {rolling_pf:.2f} < {self._pf_threshold}"
                elif consecutive_losses >= self._consecutive_losses_threshold:
                    reason = f"{consecutive_losses} consecutive losses >= {self._consecutive_losses_threshold}"
                self.activate(reason)

    def get_risk_pct(self, base_risk_pct: float) -> float:
        """Get adjusted risk percentage in preservation mode.

        Args:
            base_risk_pct: Base risk percentage (typically 1.0%)

        Returns:
            Adjusted risk percentage (0.5% in preservation mode)
        """
        if self._active:
            return base_risk_pct * 0.5  # Half the risk
        return base_risk_pct

    def get_max_leverage(self, base_max_leverage: float) -> float:
        """Get adjusted max leverage in preservation mode.

        Args:
            base_max_leverage: Base max leverage

        Returns:
            Adjusted max leverage (1.0 in preservation mode)
        """
        if self._active:
            return 1.0  # NO leverage in preservation mode
        return base_max_leverage

    def is_strategy_allowed(self, strategy_id: str) -> bool:
        """Check if a strategy is allowed in preservation mode.

        Only conservative strategies are allowed:
        - Funding Harvester
        - Grid Trading
        - Basis Arbitrage (conservative)

        Aggressive strategies are blocked:
        - Impulse Engine
        - Scalping

        Args:
            strategy_id: Strategy identifier

        Returns:
            True if strategy is allowed, False otherwise
        """
        if not self._active:
            return True  # All strategies allowed when not in preservation mode

        # Conservative strategies
        conservative_strategies = [
            "funding_harvester",
            "basis_arbitrage",
            "grid_trading",
        ]

        strategy_lower = strategy_id.lower()
        for conservative in conservative_strategies:
            if conservative in strategy_lower:
                return True

        # Block aggressive strategies
        aggressive_strategies = ["impulse_engine", "scalping"]
        for aggressive in aggressive_strategies:
            if aggressive in strategy_lower:
                logger.debug(f"CapitalPreservationMode: Blocking aggressive strategy {strategy_id}")
                return False

        # Default: allow if not explicitly blocked
        return True

    @property
    def is_active(self) -> bool:
        """Check if capital preservation mode is active."""
        return self._active

    @property
    def activation_reason(self) -> str:
        """Get reason for activation."""
        return self._activation_reason
