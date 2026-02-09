"""Smart leverage management with multiple safety checks.

This module implements intelligent leverage management that:
- Uses leverage 3-5x only for VERY high-quality signals
- Automatically reduces/removes leverage when problems detected
- Multiple safety checks before allowing leverage
- Adapts to market conditions and performance
- RESTORED SAFETY CONSTRAINTS for production use
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
    - Bayesian confidence adjustment
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

        # Performance tracking for Bayesian adjustment
        self._leverage_outcomes: list[tuple[float, bool]] = []  # (leverage, profitable)
        self._max_outcomes_history = 100

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
        """Calculate safe leverage with multiple safety constraints.

        PRODUCTION-SAFE: All safety thresholds are enforced.
        Leverage is determined by edge quality with safety bounds.

        Args:
            signal: Trading signal
            edge_bps: Edge in basis points
            regime: Current market regime
            drawdown_pct: Current drawdown percentage
            rolling_pf: Rolling profit factor
            volatility_percentile: Volatility percentile (0-100)
            real_time_alpha: Real-time alpha score (0.0 to 1.0)

        Returns:
            Safe leverage (bounded by max_leverage and safety constraints)
        """
        # Start with base leverage of 1.0
        base_leverage = 1.0

        # ============ EDGE-BASED LEVERAGE SCALING ============
        # Scale leverage based on edge quality
        if edge_bps > self._min_edge_for_leverage_4x:
            base_leverage = 4.0
            logger.debug(
                f"High edge={edge_bps:.1f}bps >= {self._min_edge_for_leverage_4x}, "
                f"base leverage 4.0x"
            )
        elif edge_bps > self._min_edge_for_leverage_3x:
            base_leverage = 3.0
            logger.debug(
                f"Good edge={edge_bps:.1f}bps >= {self._min_edge_for_leverage_3x}, "
                f"base leverage 3.0x"
            )
        elif edge_bps > self._min_edge_for_leverage_2x:
            base_leverage = 2.0
            logger.debug(
                f"Moderate edge={edge_bps:.1f}bps >= {self._min_edge_for_leverage_2x}, "
                f"base leverage 2.0x"
            )
        elif edge_bps > 15.0:
            base_leverage = 1.5
        else:
            base_leverage = 1.0

        # ============ ALPHA BOOST (BOUNDED) ============
        # Alpha can boost leverage, but within safe limits
        if real_time_alpha > 0.0:
            # Alpha boost: max 50% increase for very high alpha
            alpha_boost = min(real_time_alpha * 0.5, 0.5)
            base_leverage *= (1.0 + alpha_boost)
            logger.debug(
                f"Alpha boost: alpha={real_time_alpha:.3f}, "
                f"boost={alpha_boost:.2f}, leverage now {base_leverage:.2f}x"
            )

        # ============ REGIME MULTIPLIER ============
        regime_multiplier = 1.0
        if regime == Regime.RANGE:
            # RANGE: More predictable, allow slight increase
            regime_multiplier = 1.1
        elif regime == Regime.IMPULSE:
            # IMPULSE: Higher opportunity but also higher risk
            # Only allow increase if edge is very strong
            if edge_bps > self._min_edge_for_leverage_4x:
                regime_multiplier = 1.2
            else:
                regime_multiplier = 0.9  # Reduce for lower edge in IMPULSE
        elif regime == Regime.NORMAL:
            regime_multiplier = 1.0

        leverage = base_leverage * regime_multiplier

        # ============ SAFETY CONSTRAINT: DRAWDOWN ============
        # Reduce leverage based on drawdown severity
        if drawdown_pct >= 15.0:
            # Severe drawdown: max 1.5x leverage
            leverage = min(leverage, 1.5)
            logger.info(
                f"SAFETY: Drawdown {drawdown_pct:.1f}% >= 15%, "
                f"limiting leverage to 1.5x"
            )
        elif drawdown_pct >= 10.0:
            # Moderate drawdown: max 2x leverage
            leverage = min(leverage, self._max_leverage_on_drawdown_10pct)
            logger.info(
                f"SAFETY: Drawdown {drawdown_pct:.1f}% >= 10%, "
                f"limiting leverage to {self._max_leverage_on_drawdown_10pct}x"
            )
        elif drawdown_pct >= 5.0:
            # Light drawdown: reduce by 25%
            leverage *= 0.75
            logger.debug(
                f"Drawdown {drawdown_pct:.1f}% >= 5%, reducing leverage by 25%"
            )

        # ============ SAFETY CONSTRAINT: PROFIT FACTOR ============
        # Reduce leverage if PF is poor
        if rolling_pf < 1.0:
            # Losing money: max 1.0x (no leverage)
            leverage = min(leverage, 1.0)
            logger.info(
                f"SAFETY: PF {rolling_pf:.2f} < 1.0, disabling leverage"
            )
        elif rolling_pf < self._min_pf_for_leverage:
            # Marginal PF: reduce leverage
            pf_penalty = (self._min_pf_for_leverage - rolling_pf) / self._min_pf_for_leverage
            leverage *= (1.0 - pf_penalty * 0.5)  # Max 50% reduction
            logger.debug(
                f"PF {rolling_pf:.2f} < {self._min_pf_for_leverage}, "
                f"reducing leverage by {pf_penalty * 50:.0f}%"
            )

        # ============ SAFETY CONSTRAINT: VOLATILITY ============
        # Reduce leverage in high volatility environments
        if volatility_percentile >= 95:
            # Extreme volatility: max 1.5x
            leverage = min(leverage, 1.5)
            logger.info(
                f"SAFETY: Volatility P{volatility_percentile:.0f} >= 95, "
                f"limiting leverage to 1.5x"
            )
        elif volatility_percentile >= 85:
            # High volatility: reduce by 30%
            leverage *= 0.7
            logger.debug(
                f"Volatility P{volatility_percentile:.0f} >= 85, "
                f"reducing leverage by 30%"
            )
        elif volatility_percentile >= 75:
            # Elevated volatility: reduce by 15%
            leverage *= 0.85
            logger.debug(
                f"Volatility P{volatility_percentile:.0f} >= 75, "
                f"reducing leverage by 15%"
            )

        # ============ BAYESIAN ADJUSTMENT ============
        # Adjust based on historical leverage outcomes
        bayesian_factor = self._get_bayesian_adjustment(leverage)
        leverage *= bayesian_factor

        # ============ FINAL BOUNDS ============
        # Always enforce minimum 1.0x and maximum max_leverage
        leverage = max(1.0, min(leverage, self._max_leverage))

        if leverage > 1.0:
            logger.info(
                f"Safe leverage calculated: edge={edge_bps:.1f}bps, "
                f"regime={regime.value}, dd={drawdown_pct:.1f}%, "
                f"pf={rolling_pf:.2f}, vol_pct={volatility_percentile:.0f}, "
                f"leverage={leverage:.2f}x"
            )

        return leverage

    def _get_bayesian_adjustment(self, proposed_leverage: float) -> float:
        """Get Bayesian adjustment based on historical leverage outcomes.

        Reduces leverage if high-leverage trades have been losing.

        Args:
            proposed_leverage: The proposed leverage level

        Returns:
            Adjustment factor (0.5 to 1.0)
        """
        if len(self._leverage_outcomes) < 10:
            return 1.0  # Not enough data

        # Get outcomes for similar leverage levels
        similar_outcomes = [
            (lev, profitable)
            for lev, profitable in self._leverage_outcomes
            if abs(lev - proposed_leverage) < 1.0
        ]

        if len(similar_outcomes) < 5:
            return 1.0  # Not enough similar outcomes

        # Calculate win rate for similar leverage
        wins = sum(1 for _, profitable in similar_outcomes if profitable)
        win_rate = wins / len(similar_outcomes)

        # If win rate is below 50% for this leverage level, reduce
        if win_rate < 0.5:
            adjustment = 0.5 + win_rate  # Range: 0.5 to 1.0
            logger.debug(
                f"Bayesian adjustment: leverage~{proposed_leverage:.1f}x has "
                f"{win_rate:.0%} win rate, adjusting by {adjustment:.2f}"
            )
            return adjustment

        return 1.0

    def record_outcome(self, leverage_used: float, profitable: bool) -> None:
        """Record the outcome of a leveraged trade for Bayesian learning.

        Args:
            leverage_used: The leverage that was used
            profitable: Whether the trade was profitable
        """
        self._leverage_outcomes.append((leverage_used, profitable))

        # Keep only recent history
        if len(self._leverage_outcomes) > self._max_outcomes_history:
            self._leverage_outcomes = self._leverage_outcomes[-self._max_outcomes_history:]

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

        PRODUCTION-SAFE: All safety checks are enforced.

        Args:
            edge_bps: Edge in basis points
            regime: Current market regime
            drawdown_pct: Current drawdown percentage
            rolling_pf: Rolling profit factor
            volatility_percentile: Volatility percentile
            real_time_alpha: Real-time alpha score

        Returns:
            True if leverage should be used (based on safety criteria)
        """
        # ============ HARD BLOCKS ============

        # Block leverage if drawdown is too high
        if drawdown_pct >= 15.0:
            logger.info(f"Leverage blocked: drawdown {drawdown_pct:.1f}% >= 15%")
            return False

        # Block leverage if PF is below 1.0
        if rolling_pf < 1.0:
            logger.info(f"Leverage blocked: PF {rolling_pf:.2f} < 1.0")
            return False

        # Block leverage in extreme volatility without strong alpha
        if volatility_percentile >= 95 and real_time_alpha < 0.5:
            logger.info(
                f"Leverage blocked: extreme volatility P{volatility_percentile:.0f} "
                f"without strong alpha ({real_time_alpha:.2f})"
            )
            return False

        # ============ EDGE REQUIREMENTS ============

        # Require minimum edge for leverage
        if real_time_alpha > 0.3:
            # High alpha: lower edge requirement
            min_edge = 10.0
        else:
            # Standard edge requirement
            min_edge = 15.0

        if edge_bps < min_edge:
            logger.debug(
                f"Leverage not recommended: edge {edge_bps:.1f}bps < {min_edge}bps"
            )
            return False

        # ============ REGIME CHECK ============

        # In IMPULSE regime, require higher edge unless alpha is strong
        if regime == Regime.IMPULSE and edge_bps < 25.0 and real_time_alpha < 0.5:
            logger.debug(
                f"Leverage not recommended in IMPULSE: "
                f"edge {edge_bps:.1f}bps < 25bps and alpha {real_time_alpha:.2f} < 0.5"
            )
            return False

        return True

    def get_metrics(self) -> dict:
        """Get leverage manager metrics.

        Returns:
            Dictionary with leverage metrics
        """
        if not self._leverage_outcomes:
            return {
                "total_leveraged_trades": 0,
                "leverage_win_rate": 0.0,
                "avg_leverage_used": 0.0,
            }

        wins = sum(1 for _, profitable in self._leverage_outcomes if profitable)
        avg_leverage = sum(lev for lev, _ in self._leverage_outcomes) / len(self._leverage_outcomes)

        return {
            "total_leveraged_trades": len(self._leverage_outcomes),
            "leverage_win_rate": wins / len(self._leverage_outcomes),
            "avg_leverage_used": avg_leverage,
        }
