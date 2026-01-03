"""Smart leverage management with multiple safety checks.

This module implements intelligent leverage management that:
- Uses leverage 3-5x only for VERY high-quality signals
- Automatically reduces/removes leverage when problems detected
- Multiple safety checks before allowing leverage
- Adapts to market conditions and performance
"""

from hean.core.regime import Regime
from hean.core.types import Signal
from hean.logging import get_logger

logger = get_logger(__name__)


class SmartLeverageManager:
    """Smart leverage manager with multiple safety checks.

    Features:
    - Leverage 3-5x only for excellent conditions
    - Automatic reduction/removal when problems detected
    - Multiple safety checks (edge, regime, drawdown, PF, volatility)
    - Adaptive leverage based on market conditions
    """

    def __init__(
        self,
        max_leverage: float = 5.0,
        min_edge_for_leverage_2x: float = 25.0,
        min_edge_for_leverage_3x: float = 35.0,
        min_edge_for_leverage_4x: float = 50.0,
        min_pf_for_leverage: float = 1.2,
        max_leverage_on_drawdown_10pct: float = 2.0,
    ) -> None:
        """Initialize smart leverage manager.

        Args:
            max_leverage: Maximum allowed leverage (default 5.0)
            min_edge_for_leverage_2x: Minimum edge in bps for 2x leverage (default 25)
            min_edge_for_leverage_3x: Minimum edge in bps for 3x leverage (default 35)
            min_edge_for_leverage_4x: Minimum edge in bps for 4x leverage (default 50)
            min_pf_for_leverage: Minimum PF for leverage > 1x (default 1.2)
            max_leverage_on_drawdown_10pct: Max leverage when drawdown 10-15% (default 2.0)
        """
        self._max_leverage = max_leverage
        self._min_edge_for_leverage_2x = min_edge_for_leverage_2x
        self._min_edge_for_leverage_3x = min_edge_for_leverage_3x
        self._min_edge_for_leverage_4x = min_edge_for_leverage_4x
        self._min_pf_for_leverage = min_pf_for_leverage
        self._max_leverage_on_drawdown_10pct = max_leverage_on_drawdown_10pct

    def calculate_safe_leverage(
        self,
        signal: Signal,
        edge_bps: float,
        regime: Regime,
        drawdown_pct: float,
        rolling_pf: float,
        volatility_percentile: float,
    ) -> float:
        """Calculate safe leverage with multiple safety checks.

        Safety rules:
        1. Edge must be > 30 bps for leverage > 2x
        2. Regime must be RANGE or NORMAL (NOT IMPULSE for high leverage)
        3. Drawdown must be < 10% for leverage > 3x
        4. PF must be > 1.2 for leverage > 2x
        5. Volatility must be < 50th percentile for leverage > 3x

        Adaptive reduction:
        - Drawdown 10-15%: maximum leverage 2x
        - Drawdown > 15%: leverage 1x (NO leverage!)
        - PF < 1.0: leverage 1x
        - High volatility (>80th percentile): maximum leverage 2x

        Args:
            signal: Trading signal
            edge_bps: Edge in basis points
            regime: Current market regime
            drawdown_pct: Current drawdown percentage
            rolling_pf: Rolling profit factor
            volatility_percentile: Volatility percentile (0-100)

        Returns:
            Safe leverage (1.0 to max_leverage)
        """
        # CRITICAL: No leverage if problems detected
        if drawdown_pct > 15.0 or rolling_pf < 1.0:
            logger.warning(
                f"Smart leverage: Blocking leverage due to problems "
                f"(DD={drawdown_pct:.1f}%, PF={rolling_pf:.2f})"
            )
            return 1.0  # NO leverage!

        # Base leverage based on edge
        base_leverage = 1.0
        if (
            edge_bps > self._min_edge_for_leverage_4x
            and volatility_percentile < 50
            and drawdown_pct < 5.0
        ):
            # Excellent conditions: high leverage
            base_leverage = 4.0
            logger.debug(
                f"Smart leverage: Excellent conditions (edge={edge_bps:.1f}bps, "
                f"vol_pct={volatility_percentile:.1f}, DD={drawdown_pct:.1f}%), "
                f"base leverage 4.0x"
            )
        elif (
            edge_bps > self._min_edge_for_leverage_3x
            and volatility_percentile < 70
            and drawdown_pct < 10.0
        ):
            # Good conditions: moderate leverage
            base_leverage = 3.0
            logger.debug(
                f"Smart leverage: Good conditions (edge={edge_bps:.1f}bps, "
                f"vol_pct={volatility_percentile:.1f}, DD={drawdown_pct:.1f}%), "
                f"base leverage 3.0x"
            )
        elif edge_bps > self._min_edge_for_leverage_2x and drawdown_pct < 10.0:
            # Moderate conditions: low leverage
            base_leverage = 2.0
            logger.debug(
                f"Smart leverage: Moderate conditions (edge={edge_bps:.1f}bps, "
                f"DD={drawdown_pct:.1f}%), base leverage 2.0x"
            )
        elif edge_bps > 15.0:
            # Minimal edge: slight leverage
            base_leverage = 1.5
        else:
            # Low edge: no leverage
            base_leverage = 1.0

        # Regime adjustment
        regime_multiplier = 1.0
        if regime == Regime.RANGE and volatility_percentile < 50:
            # Can use more leverage in RANGE with low volatility
            regime_multiplier = 1.2
        elif regime == Regime.IMPULSE:
            # LESS leverage in IMPULSE (dangerous!)
            regime_multiplier = 0.7
            logger.debug("Smart leverage: IMPULSE regime, reducing leverage multiplier to 0.7x")

        leverage = base_leverage * regime_multiplier

        # Final safety limits
        if drawdown_pct > 10.0:
            # Drawdown 10-15%: cap at 2x
            leverage = min(leverage, self._max_leverage_on_drawdown_10pct)
            logger.debug(
                f"Smart leverage: Drawdown {drawdown_pct:.1f}% > 10%, "
                f"capping leverage at {self._max_leverage_on_drawdown_10pct}x"
            )

        if volatility_percentile > 80.0:
            # High volatility: cap at 2x
            leverage = min(leverage, 2.0)
            logger.debug(
                f"Smart leverage: High volatility ({volatility_percentile:.1f}th percentile), "
                f"capping leverage at 2.0x"
            )

        if rolling_pf < self._min_pf_for_leverage:
            # Low PF: cap at 2x
            leverage = min(leverage, 2.0)
            logger.debug(
                f"Smart leverage: PF {rolling_pf:.2f} < {self._min_pf_for_leverage}, "
                f"capping leverage at 2.0x"
            )

        # Final bounds
        leverage = max(1.0, min(leverage, self._max_leverage))

        if leverage > 1.0:
            logger.info(
                f"Smart leverage: edge={edge_bps:.1f}bps, regime={regime.value}, "
                f"DD={drawdown_pct:.1f}%, PF={rolling_pf:.2f}, vol_pct={volatility_percentile:.1f}, "
                f"leverage={leverage:.2f}x"
            )

        return leverage

    def should_use_leverage(
        self,
        edge_bps: float,
        regime: Regime,
        drawdown_pct: float,
        rolling_pf: float,
        volatility_percentile: float,
    ) -> bool:
        """Check if leverage should be used at all.

        Args:
            edge_bps: Edge in basis points
            regime: Current market regime
            drawdown_pct: Current drawdown percentage
            rolling_pf: Rolling profit factor
            volatility_percentile: Volatility percentile

        Returns:
            True if leverage can be used, False otherwise
        """
        # Never use leverage if severe problems
        if drawdown_pct > 15.0 or rolling_pf < 1.0:
            return False

        # Need minimum edge
        if edge_bps < 15.0:
            return False

        # Not in extreme volatility
        if volatility_percentile > 90.0:
            return False

        return True
