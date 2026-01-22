"""Position sizing based on risk parameters with smart protection."""

from hean.config import settings
from hean.core.regime import Regime
from hean.core.types import Signal
from hean.logging import get_logger
from hean.portfolio.accounting import PortfolioAccounting
from hean.risk.capital_preservation import CapitalPreservationMode
from hean.risk.dynamic_risk import DynamicRiskManager
from hean.risk.kelly_criterion import KellyCriterion
from hean.risk.smart_leverage import SmartLeverageManager

logger = get_logger(__name__)


class PositionSizer:
    """Calculates position size based on risk parameters and market regime.

    Enhanced with smart protection mechanisms:
    - Adaptive size reduction when problems detected
    - Size increase only in excellent conditions
    - Integration with capital preservation mode
    """

    def __init__(self) -> None:
        """Initialize position sizer with dynamic risk manager."""
        self._dynamic_risk = DynamicRiskManager()
        self._leverage_manager = SmartLeverageManager()
        self._capital_preservation: CapitalPreservationMode | None = None
        self._last_calculation: dict[str, float] | None = None

    def calculate_size(
        self,
        signal: Signal,
        equity: float,
        current_price: float,
        regime: Regime = Regime.NORMAL,
        rolling_pf: float | None = None,
        recent_drawdown: float | None = None,
        volatility_percentile: float | None = None,
        edge_bps: float | None = None,
    ) -> float:
        """Calculate position size based on risk parameters and regime with smart protection.

        Uses stop loss distance to determine position size such that
        max_trade_risk_pct of equity is at risk.

        Smart protection:
        - Drawdown > 10%: reduce size by 30%
        - PF < 1.0: reduce size by 50%
        - PF > 1.5 and drawdown < 5%: can increase size by 20%
        - Capital preservation mode: reduce risk to 0.5%

        Regime adjustments:
        - RANGE: 0.7x
        - NORMAL: 1.0x
        - IMPULSE: 1.2x
        """
        # Get base risk percentage (adjusted by capital preservation mode if active)
        base_risk_pct = settings.max_trade_risk_pct
        if self._capital_preservation and self._capital_preservation.is_active:
            base_risk_pct = self._capital_preservation.get_risk_pct(base_risk_pct)
            logger.debug(
                f"Position sizing: Capital preservation mode active, "
                f"risk reduced from {settings.max_trade_risk_pct}% to {base_risk_pct}%"
            )

        if signal.stop_loss is None:
            # No stop loss - use a conservative default size
            risk_pct = base_risk_pct / 2  # Half of max risk
            # Default: assume 2% stop distance
            stop_distance_pct = 2.0
        else:
            risk_pct = base_risk_pct
            # Calculate stop distance as percentage
            if signal.side == "buy":
                stop_distance_pct = abs((current_price - signal.stop_loss) / current_price) * 100
            else:  # sell
                stop_distance_pct = abs((signal.stop_loss - current_price) / current_price) * 100

        # CRITICAL FIX: Ensure stop_distance_pct is always valid
        if stop_distance_pct <= 0 or stop_distance_pct > 50:
            logger.warning(
                f"Invalid stop distance: {stop_distance_pct}%, using default 2%"
            )
            stop_distance_pct = 2.0

        # Calculate risk amount
        risk_amount = equity * (risk_pct / 100.0)

        # Position size = risk_amount / stop_distance_per_unit
        # stop_distance_per_unit = price * stop_distance_pct / 100
        stop_distance_per_unit = current_price * (stop_distance_pct / 100.0)
        position_size = risk_amount / stop_distance_per_unit

        # Apply regime multiplier (base multiplier)
        regime_multiplier = self._get_regime_multiplier(regime)
        position_size *= regime_multiplier

        # Smart leverage calculation
        leverage = 1.0
        if edge_bps is not None:
            # Get max leverage (adjusted by capital preservation mode if active)
            max_leverage = settings.max_leverage
            if self._capital_preservation and self._capital_preservation.is_active:
                max_leverage = self._capital_preservation.get_max_leverage(max_leverage)

            leverage = self._leverage_manager.calculate_safe_leverage(
                signal=signal,
                edge_bps=edge_bps,
                regime=regime,
                drawdown_pct=recent_drawdown or 0.0,
                rolling_pf=rolling_pf or 1.0,
                volatility_percentile=volatility_percentile or 50.0,
            )
            # Ensure leverage doesn't exceed max (after capital preservation adjustment)
            leverage = min(leverage, max_leverage)

        # Apply dynamic risk multiplier if metrics provided
        if (
            rolling_pf is not None
            and recent_drawdown is not None
            and volatility_percentile is not None
        ):
            dynamic_multiplier = self._dynamic_risk.calculate_risk_multiplier(
                current_regime=regime,
                rolling_pf=rolling_pf,
                recent_drawdown=recent_drawdown,
                volatility_percentile=volatility_percentile,
            )
            position_size *= dynamic_multiplier

            # Smart protection multipliers
            protection_multiplier = self._calculate_protection_multiplier(
                recent_drawdown, rolling_pf, volatility_percentile
            )
            position_size *= protection_multiplier

            logger.info(
                f"Position sizing: equity=${equity:.2f}, risk_pct={risk_pct}%, "
                f"stop_distance={stop_distance_pct:.2f}%, regime={regime.value}, "
                f"regime_mult={regime_multiplier}, dynamic_mult={dynamic_multiplier:.2f}, "
                f"protection_mult={protection_multiplier:.2f}, leverage={leverage:.2f}x, "
                f"size={position_size:.6f}"
            )
            
            # CRITICAL: Ensure size doesn't become 0 after multipliers
            if position_size <= 0:
                min_size_value = (equity * 0.001) / current_price
                absolute_min = 0.001
                position_size = max(min_size_value, absolute_min)
                logger.warning(
                    f"Position size became 0 after multipliers, using minimum {position_size:.6f}"
                )
        else:
            logger.debug(
                f"Position sizing: equity={equity:.2f}, risk_pct={risk_pct}%, "
                f"stop_distance={stop_distance_pct:.2f}%, regime={regime.value}, "
                f"multiplier={regime_multiplier}, leverage={leverage:.2f}x, "
                f"size={position_size:.4f}"
            )

        # Apply leverage to position size
        # Note: In practice, leverage affects margin requirements, not position size directly
        # This is a simplified model where we scale position size by leverage
        position_size *= leverage

        # Ensure minimum position size (critical: never return 0)
        # Minimum size based on equity: at least 0.1% of equity worth
        min_size_value = (equity * 0.001) / current_price  # 0.1% of equity in units
        # Also ensure absolute minimum (e.g., 0.001 BTC or equivalent)
        absolute_min = 0.001
        min_size = max(min_size_value, absolute_min)

        if position_size < min_size:
            logger.debug(
                f"Position size {position_size:.6f} below minimum {min_size:.6f}, "
                f"using minimum size"
            )
            position_size = min_size

        # Cache calculation context for downstream telemetry
        self._last_calculation = {
            "leverage": leverage,
            "risk_pct": risk_pct,
            "stop_distance_pct": stop_distance_pct,
            "regime_multiplier": regime_multiplier,
            "edge_bps": edge_bps or 0.0,
        }

        return position_size

    def get_last_calculation(self) -> dict[str, float] | None:
        """Return details from the most recent sizing calculation."""
        return self._last_calculation

    def set_capital_preservation(self, capital_preservation: CapitalPreservationMode) -> None:
        """Set capital preservation mode instance.

        Args:
            capital_preservation: Capital preservation mode instance
        """
        self._capital_preservation = capital_preservation

    def _calculate_protection_multiplier(
        self, drawdown_pct: float, rolling_pf: float, volatility_percentile: float
    ) -> float:
        """Calculate protection multiplier based on current conditions.

        Smart protection rules:
        - Drawdown > 10%: reduce size by 30%
        - PF < 1.0: reduce size by 50%
        - High volatility (>80th percentile): reduce size by 20%
        - PF > 1.5 and drawdown < 5%: can increase size by 20% (only in excellent conditions!)

        Args:
            drawdown_pct: Current drawdown percentage
            rolling_pf: Rolling profit factor
            volatility_percentile: Volatility percentile

        Returns:
            Protection multiplier (0.3x to 1.2x)
        """
        multiplier = 1.0

        # Protection: reduce size when problems detected
        if drawdown_pct > 10.0:
            multiplier *= 0.7  # Reduce by 30%
            logger.debug(
                f"Position sizing protection: Drawdown {drawdown_pct:.1f}% > 10%, "
                f"reducing size by 30%"
            )

        if rolling_pf < 1.0:
            multiplier *= 0.5  # Reduce by 50%
            logger.debug(
                f"Position sizing protection: PF {rolling_pf:.2f} < 1.0, reducing size by 50%"
            )

        if volatility_percentile > 80.0:
            multiplier *= 0.8  # Reduce by 20%
            logger.debug(
                f"Position sizing protection: High volatility ({volatility_percentile:.1f}th percentile), "
                f"reducing size by 20%"
            )

        # Growth: increase size only in excellent conditions
        if rolling_pf > 1.5 and drawdown_pct < 5.0:
            multiplier *= 1.2  # Increase by 20%
            logger.debug(
                f"Position sizing protection: Excellent conditions (PF={rolling_pf:.2f}, "
                f"DD={drawdown_pct:.1f}%), increasing size by 20%"
            )

        # Bounds: between 0.3x and 1.2x
        multiplier = max(0.3, min(multiplier, 1.2))

        return multiplier

    def _get_regime_multiplier(self, regime: Regime) -> float:
        """Get position size multiplier for regime."""
        multipliers = {
            Regime.RANGE: 0.7,
            Regime.NORMAL: 1.0,
            Regime.IMPULSE: 1.2,
        }
        return multipliers.get(regime, 1.0)

    def get_dynamic_risk_manager(self) -> DynamicRiskManager:
        """Get the dynamic risk manager instance."""
        return self._dynamic_risk

    def update_volatility(self, volatility: float) -> None:
        """Update volatility history for percentile calculation."""
        self._dynamic_risk.update_volatility(volatility)
