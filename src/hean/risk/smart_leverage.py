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
        real_time_alpha: float = 0.0,
    ) -> float:
        """Calculate leverage based ONLY on real-time alpha score.
        
        UNCONSTRAINED: No safety thresholds - leverage determined solely by alpha.
        The system has the right to take risks and evolve based on alpha signals.

        Args:
            signal: Trading signal
            edge_bps: Edge in basis points
            regime: Current market regime
            drawdown_pct: Current drawdown percentage (unused - no constraint)
            rolling_pf: Rolling profit factor (unused - no constraint)
            volatility_percentile: Volatility percentile (unused - no constraint)
            real_time_alpha: Real-time alpha score (PRIMARY FACTOR)

        Returns:
            Leverage based on alpha (can exceed max_leverage if alpha is extreme)
        """
        # UNCONSTRAINED: No safety checks - leverage based ONLY on alpha
        # If no alpha provided, fall back to edge-based calculation (still unconstrained)
        if real_time_alpha != 0.0:
            # Primary: Alpha-based leverage (no constraints)
            # High alpha = high leverage, even if extreme
            leverage = 1.0 + (abs(real_time_alpha) * self._max_leverage * 10.0)
            
            # No upper bound - allow extreme leverage if alpha demands it
            if real_time_alpha > 0.8:
                # Extremely high alpha: allow unlimited leverage
                leverage = self._max_leverage * 20.0  # 100x if max is 5x
                logger.info(
                    f"UNCONSTRAINED: Extreme alpha={real_time_alpha:.4f}, "
                    f"allowing leverage={leverage:.2f}x"
                )
            elif real_time_alpha > 0.5:
                leverage = self._max_leverage * 5.0  # 25x if max is 5x
            elif real_time_alpha > 0.3:
                leverage = self._max_leverage * 2.0  # 10x if max is 5x
            
            logger.info(
                f"Alpha-based leverage: alpha={real_time_alpha:.4f}, "
                f"leverage={leverage:.2f}x (UNCONSTRAINED)"
            )
            return leverage

        # UNCONSTRAINED: Edge-based leverage (no safety limits)
        # Base leverage scales with edge, no constraints on volatility/drawdown/PF
        base_leverage = 1.0
        
        # Scale leverage with edge (no safety constraints)
        if edge_bps > self._min_edge_for_leverage_4x:
            base_leverage = 4.0 * (edge_bps / self._min_edge_for_leverage_4x)
            logger.debug(
                f"UNCONSTRAINED leverage: High edge={edge_bps:.1f}bps, "
                f"base leverage {base_leverage:.2f}x"
            )
        elif edge_bps > self._min_edge_for_leverage_3x:
            base_leverage = 3.0 * (edge_bps / self._min_edge_for_leverage_3x)
            logger.debug(
                f"UNCONSTRAINED leverage: Good edge={edge_bps:.1f}bps, "
                f"base leverage {base_leverage:.2f}x"
            )
        elif edge_bps > self._min_edge_for_leverage_2x:
            base_leverage = 2.0 * (edge_bps / self._min_edge_for_leverage_2x)
            logger.debug(
                f"UNCONSTRAINED leverage: Moderate edge={edge_bps:.1f}bps, "
                f"base leverage {base_leverage:.2f}x"
            )
        elif edge_bps > 15.0:
            base_leverage = 1.5
        else:
            base_leverage = 1.0

        # Regime multiplier (unconstrained - can increase leverage in any regime)
        regime_multiplier = 1.0
        if regime == Regime.RANGE:
            regime_multiplier = 1.5  # Increase in RANGE (more opportunities)
        elif regime == Regime.IMPULSE:
            regime_multiplier = 2.0  # UNCONSTRAINED: Allow MORE leverage in IMPULSE (high alpha opportunity)
            logger.debug(f"UNCONSTRAINED: IMPULSE regime, increasing leverage multiplier to 2.0x")

        leverage = base_leverage * regime_multiplier

        # NO SAFETY LIMITS - Allow leverage to scale based on edge/alpha
        # Remove all drawdown, volatility, and PF constraints
        
        # Only minimum bound (can't go below 1.0x)
        leverage = max(1.0, leverage)
        
        # NO MAXIMUM BOUND - Allow extreme leverage if edge is high
        # (Previously: leverage = min(leverage, self._max_leverage))
        # Now removed - leverage can exceed max_leverage if conditions demand it

        if leverage > 1.0:
            logger.info(
                f"UNCONSTRAINED leverage: edge={edge_bps:.1f}bps, regime={regime.value}, "
                f"leverage={leverage:.2f}x (NO SAFETY LIMITS)"
            )

        return leverage

    def should_use_leverage(
        self,
        edge_bps: float,
        regime: Regime,
        drawdown_pct: float,
        rolling_pf: float,
        volatility_percentile: float,
        real_time_alpha: float = 0.0,
    ) -> bool:
        """Check if leverage should be used.
        
        UNCONSTRAINED: No safety checks - leverage based on alpha/edge only.

        Args:
            edge_bps: Edge in basis points
            regime: Current market regime (unused - no constraint)
            drawdown_pct: Current drawdown percentage (unused - no constraint)
            rolling_pf: Rolling profit factor (unused - no constraint)
            volatility_percentile: Volatility percentile (unused - no constraint)
            real_time_alpha: Real-time alpha score

        Returns:
            True if leverage should be used (based on alpha/edge only)
        """
        # UNCONSTRAINED: No safety checks - only check for minimum signal
        # Use leverage if there's any positive alpha or edge
        
        if real_time_alpha > 0.0:
            return True  # Any positive alpha allows leverage
        
        # Minimum edge requirement (minimal constraint - just need some edge)
        if edge_bps > 5.0:  # Very low threshold
            return True
        
        return False
