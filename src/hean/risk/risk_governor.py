"""Risk Governor - Multi-level risk management replacing binary killswitch.

Provides graduated risk levels:
- NORMAL: Normal operation
- SOFT_BRAKE: Reduced sizing (50%), increased cooldowns
- QUARANTINE: Per-symbol blocking
- HARD_STOP: Emergency halt

BUG FIX (2026-01):
- Fixed drawdown calculation to use high water mark (peak equity) instead of initial capital
- This prevents blocking trading when in profit but experiencing a pullback
- Added proper peak tracking for accurate drawdown measurement
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class RiskState(str, Enum):
    """Risk governor state levels."""
    NORMAL = "NORMAL"
    SOFT_BRAKE = "SOFT_BRAKE"
    QUARANTINE = "QUARANTINE"
    HARD_STOP = "HARD_STOP"


class RiskGovernor:
    """Multi-level risk governor replacing binary killswitch.

    Provides graduated risk management with clear thresholds,
    recommendations, and clear paths back to normal operation.
    """

    def __init__(self, bus: EventBus) -> None:
        """Initialize risk governor.

        Args:
            bus: Event bus for publishing state changes
        """
        self._bus = bus
        self._enabled = getattr(settings, "risk_governor_enabled", True)
        self._state = RiskState.NORMAL
        self._level = 0  # 0=NORMAL, 1=SOFT_BRAKE, 2=QUARANTINE, 3=HARD_STOP
        self._reason_codes: list[str] = []
        self._metric: str | None = None
        self._value: float | None = None
        self._threshold: float | None = None
        self._blocked_at: datetime | None = None
        self._quarantined_symbols: set[str] = set()

        # High Water Mark tracking for accurate drawdown calculation
        # BUG FIX: Previously calculated drawdown from initial_capital, which
        # caused issues when in profit - a pullback from profit could be
        # misinterpreted or the state wouldn't properly track recovery
        self._peak_equity: float = 0.0  # Track highest equity reached
        self._initial_capital: float = 0.0  # Store initial capital
        self._last_equity: float = 0.0  # Last known equity

        # Drawdown tracking
        self._current_drawdown_pct: float = 0.0
        self._max_drawdown_pct: float = 0.0

        logger.info(f"Risk Governor initialized: enabled={self._enabled}")

    def get_state(self) -> dict[str, Any]:
        """Get current risk governor state.

        Returns:
            Risk state dictionary
        """
        return {
            "risk_state": self._state.value,
            "level": self._level,
            "reason_codes": list(self._reason_codes),
            "metric": self._metric,
            "value": self._value,
            "threshold": self._threshold,
            "recommended_action": self._get_recommended_action(),
            "clear_rule": self._get_clear_rule(),
            "quarantined_symbols": list(self._quarantined_symbols),
            "blocked_at": self._blocked_at.isoformat() if self._blocked_at else None,
            "can_clear": self._can_clear(),
            # High water mark tracking (for debugging profit bug)
            "peak_equity": self._peak_equity,
            "initial_capital": self._initial_capital,
            "last_equity": self._last_equity,
            "current_drawdown_pct": self._current_drawdown_pct,
            "max_drawdown_pct": self._max_drawdown_pct,
        }

    async def check_and_update(
        self,
        equity: float,
        initial_capital: float,
        positions_count: int,
        orders_count: int,
    ) -> RiskState:
        """Check risk conditions and update state if needed.

        Uses HIGH WATER MARK (peak equity) for drawdown calculation to correctly
        handle profit scenarios. Drawdown is measured from the highest equity
        achieved, not from initial capital.

        Args:
            equity: Current equity
            initial_capital: Initial capital
            positions_count: Number of open positions
            orders_count: Number of open orders

        Returns:
            Current risk state
        """
        if not self._enabled:
            return RiskState.NORMAL

        # Initialize tracking on first call
        if self._initial_capital == 0.0:
            self._initial_capital = initial_capital
            self._peak_equity = max(initial_capital, equity)
            logger.info(
                f"[RiskGovernor] Initialized: initial_capital={initial_capital:.2f}, "
                f"peak_equity={self._peak_equity:.2f}"
            )

        # Update peak equity (high water mark)
        if equity > self._peak_equity:
            old_peak = self._peak_equity
            self._peak_equity = equity
            logger.debug(
                f"[RiskGovernor] New peak equity: {old_peak:.2f} → {equity:.2f}"
            )

        self._last_equity = equity

        # Calculate drawdown from HIGH WATER MARK (not initial capital!)
        # This is the correct way to measure drawdown:
        # - If we're at $110 from $100 initial and drop to $105, drawdown is 4.5% (from $110 peak)
        # - NOT 0% because we're still above initial capital
        if self._peak_equity > 0:
            drawdown_from_peak = ((self._peak_equity - equity) / self._peak_equity) * 100
        else:
            drawdown_from_peak = 0.0

        # Also calculate drawdown from initial (for reporting)
        if initial_capital > 0:
            drawdown_from_initial = ((initial_capital - equity) / initial_capital) * 100
        else:
            drawdown_from_initial = 0.0

        # Use the MORE CONSERVATIVE of the two for risk management
        # If drawdown_from_initial is negative (we're in profit), use peak-based
        # If both are positive (loss from both perspectives), use the larger one
        if drawdown_from_initial < 0:
            # We're in profit vs initial capital
            # Use drawdown from peak to catch profit pullbacks
            drawdown_pct = max(0.0, drawdown_from_peak)
            drawdown_source = "peak"
        else:
            # We're in loss vs initial capital
            # Use the larger drawdown for safety
            drawdown_pct = max(drawdown_from_initial, drawdown_from_peak)
            drawdown_source = "initial" if drawdown_from_initial >= drawdown_from_peak else "peak"

        # Track max drawdown
        if drawdown_pct > self._max_drawdown_pct:
            self._max_drawdown_pct = drawdown_pct

        self._current_drawdown_pct = drawdown_pct

        # Log state for debugging
        logger.debug(
            f"[RiskGovernor] equity={equity:.2f}, peak={self._peak_equity:.2f}, "
            f"drawdown={drawdown_pct:.2f}% (from {drawdown_source}), "
            f"state={self._state.value}"
        )

        # Check for HARD_STOP conditions
        if drawdown_pct >= 20.0:  # 20% drawdown
            await self._escalate_to(
                RiskState.HARD_STOP,
                reason_codes=["MAX_DRAWDOWN_HARD"],
                metric="drawdown_pct",
                value=drawdown_pct,
                threshold=20.0,
            )
        # Check for QUARANTINE conditions
        elif drawdown_pct >= 15.0:  # 15% drawdown
            await self._escalate_to(
                RiskState.QUARANTINE,
                reason_codes=["MAX_DRAWDOWN_QUARANTINE"],
                metric="drawdown_pct",
                value=drawdown_pct,
                threshold=15.0,
            )
        # Check for SOFT_BRAKE conditions
        elif drawdown_pct >= 10.0:  # 10% drawdown
            await self._escalate_to(
                RiskState.SOFT_BRAKE,
                reason_codes=["MAX_DRAWDOWN_SOFT"],
                metric="drawdown_pct",
                value=drawdown_pct,
                threshold=10.0,
            )
        # Check for de-escalation (recovery)
        elif self._state != RiskState.NORMAL and drawdown_pct < 5.0:
            # Can de-escalate if drawdown reduced below 5% and cooldown period passed
            if self._can_deescalate():
                await self._deescalate_to(RiskState.NORMAL)

        return self._state

    async def quarantine_symbol(self, symbol: str, reason: str) -> None:
        """Quarantine a specific symbol (block trading).

        Args:
            symbol: Symbol to quarantine
            reason: Reason code
        """
        if symbol not in self._quarantined_symbols:
            self._quarantined_symbols.add(symbol)
            logger.warning(f"Symbol quarantined: {symbol}, reason: {reason}")

            await self._bus.publish(Event(
                event_type=EventType.KILLSWITCH_TRIGGERED,  # Reuse existing event type
                data={
                    "type": "SYMBOL_QUARANTINED",
                    "symbol": symbol,
                    "reason": reason,
                    "quarantined_symbols": list(self._quarantined_symbols),
                }
            ))

    async def clear_quarantine(self, symbol: str | None = None) -> dict[str, Any]:
        """Clear symbol quarantine.

        Args:
            symbol: Specific symbol to clear, or None to clear all

        Returns:
            Status dictionary
        """
        if symbol:
            if symbol in self._quarantined_symbols:
                self._quarantined_symbols.remove(symbol)
                logger.info(f"Symbol quarantine cleared: {symbol}")
                return {"status": "cleared", "symbol": symbol}
            else:
                return {"status": "not_quarantined", "symbol": symbol}
        else:
            count = len(self._quarantined_symbols)
            self._quarantined_symbols.clear()
            logger.info(f"All symbol quarantines cleared: {count} symbols")
            return {"status": "cleared_all", "count": count}

    async def clear_all(self, force: bool = False, reset_peak: bool = False) -> dict[str, Any]:
        """Clear all risk governor blocks.

        Args:
            force: Force clear even if conditions not met
            reset_peak: Also reset peak equity to current equity (useful after manual intervention)

        Returns:
            Status dictionary
        """
        if not force and not self._can_clear():
            return {
                "status": "cannot_clear",
                "message": "Conditions not met for clearing",
                "clear_rule": self._get_clear_rule(),
            }

        # Clear state
        old_state = self._state
        self._state = RiskState.NORMAL
        self._level = 0
        self._reason_codes.clear()
        self._metric = None
        self._value = None
        self._threshold = None
        self._blocked_at = None
        self._quarantined_symbols.clear()

        # Optionally reset peak equity (useful when manually resetting after drawdown)
        old_peak = self._peak_equity
        if reset_peak and self._last_equity > 0:
            self._peak_equity = self._last_equity
            self._max_drawdown_pct = 0.0
            self._current_drawdown_pct = 0.0
            logger.info(
                f"[RiskGovernor] Peak equity reset: {old_peak:.2f} → {self._peak_equity:.2f}"
            )

        logger.info(f"Risk Governor cleared: {old_state.value} → NORMAL (force={force}, reset_peak={reset_peak})")

        await self._bus.publish(Event(
            event_type=EventType.KILLSWITCH_TRIGGERED,  # Reuse
            data={
                "type": "RISK_STATE_UPDATE",
                "state": self._state.value,
                "previous_state": old_state.value,
                "cleared": True,
                "forced": force,
                "peak_reset": reset_peak,
            }
        ))

        return {
            "status": "cleared",
            "risk_state": self._state.value,
            "message": f"Risk governor cleared from {old_state.value}",
            "peak_equity": self._peak_equity,
        }

    def reset_peak_equity(self, new_peak: float | None = None) -> dict[str, Any]:
        """Reset the peak equity (high water mark).

        Useful after:
        - Manual position closure
        - Strategy restart
        - Recovery from a drawdown event

        Args:
            new_peak: New peak value. If None, uses current equity.

        Returns:
            Status dictionary
        """
        old_peak = self._peak_equity
        if new_peak is not None:
            self._peak_equity = new_peak
        elif self._last_equity > 0:
            self._peak_equity = self._last_equity
        else:
            return {"status": "error", "message": "No equity data available"}

        self._max_drawdown_pct = 0.0
        self._current_drawdown_pct = 0.0

        logger.info(f"[RiskGovernor] Peak equity manually reset: {old_peak:.2f} → {self._peak_equity:.2f}")

        return {
            "status": "reset",
            "old_peak": old_peak,
            "new_peak": self._peak_equity,
        }

    def is_symbol_allowed(self, symbol: str) -> bool:
        """Check if trading allowed for symbol.

        Args:
            symbol: Symbol to check

        Returns:
            True if allowed, False if blocked
        """
        if self._state == RiskState.HARD_STOP:
            return False
        if self._state == RiskState.QUARANTINE and symbol in self._quarantined_symbols:
            return False
        return True

    async def check_signal_allowed(self, symbol: str, strategy_id: str, signal_metadata: dict | None = None) -> bool:
        """Check if signal is allowed and publish RISK_BLOCKED if not.

        This method should be called by TradingSystem when processing signals.
        It combines is_symbol_allowed check with event publishing for observability.

        Args:
            symbol: Trading symbol
            strategy_id: Strategy generating the signal
            signal_metadata: Optional signal metadata for diagnostic purposes

        Returns:
            True if signal allowed, False if blocked
        """
        if not self.is_symbol_allowed(symbol):
            # Determine block reason
            if self._state == RiskState.HARD_STOP:
                reason = "risk_hard_stop"
                details = "Trading halted - HARD_STOP state"
            elif self._state == RiskState.QUARANTINE:
                reason = "risk_quarantine"
                details = f"Symbol {symbol} quarantined"
            else:
                reason = "risk_unknown"
                details = "Unknown risk block"

            # Publish RISK_BLOCKED event
            await self._bus.publish(Event(
                event_type=EventType.RISK_BLOCKED,
                data={
                    "symbol": symbol,
                    "strategy_id": strategy_id,
                    "reason": reason,
                    "details": details,
                    "risk_state": self._state.value,
                    "risk_level": self._level,
                    "signal_metadata": signal_metadata or {},
                }
            ))

            logger.info(f"[RiskGovernor] Signal blocked: {strategy_id}/{symbol} - {details}")
            return False

        return True

    def get_size_multiplier(self, regime: str | None = None) -> float:
        """Get position size multiplier based on risk state and regime.

        Enhanced with regime-aware sizing:
        - IMPULSE regime in NORMAL state: can use 1.2x (slight boost)
        - RANGE regime: always reduce by 30% (mean-reversion territory)
        - Combines with risk state multiplier

        Args:
            regime: Optional market regime ("IMPULSE", "NORMAL", "RANGE")

        Returns:
            Size multiplier (0.0-1.2)
        """
        # Base multiplier from risk state
        if self._state == RiskState.HARD_STOP:
            return 0.0
        elif self._state == RiskState.SOFT_BRAKE:
            base_mult = 0.5  # 50% sizing
        elif self._state == RiskState.QUARANTINE:
            base_mult = 0.75  # 75% sizing for non-quarantined symbols
        else:
            base_mult = 1.0  # Normal

        # Regime-based adjustment
        regime_mult = 1.0
        if regime:
            regime_upper = regime.upper()
            if regime_upper == "IMPULSE":
                # Strong momentum - can size up slightly in NORMAL risk state
                if self._state == RiskState.NORMAL:
                    regime_mult = 1.15  # 15% boost for momentum trades
                # In SOFT_BRAKE, no boost
            elif regime_upper == "RANGE":
                # Range-bound market - reduce size (mean-reversion preferred)
                regime_mult = 0.7  # 30% reduction
            # NORMAL regime: no adjustment (regime_mult stays 1.0)

        # Drawdown-based dynamic adjustment
        drawdown_mult = 1.0
        if self._current_drawdown_pct > 0:
            if self._current_drawdown_pct > 15.0:
                drawdown_mult = 0.3  # Severe drawdown - minimal sizing
            elif self._current_drawdown_pct > 10.0:
                drawdown_mult = 0.5  # Significant drawdown
            elif self._current_drawdown_pct > 5.0:
                drawdown_mult = 0.7  # Moderate drawdown

        # Combine multipliers
        final_mult = base_mult * regime_mult * drawdown_mult

        # Clamp to reasonable bounds
        return max(0.0, min(1.5, final_mult))

    def get_regime_recommendation(self, regime: str) -> dict[str, Any]:
        """Get trading recommendations for current regime.

        Args:
            regime: Market regime ("IMPULSE", "NORMAL", "RANGE")

        Returns:
            Recommendations for the regime
        """
        regime_upper = regime.upper() if regime else "NORMAL"

        # Base recommendations
        recommendations = {
            "regime": regime_upper,
            "risk_state": self._state.value,
            "allow_new_positions": self.is_trading_allowed(),
            "size_multiplier": self.get_size_multiplier(regime),
        }

        # Regime-specific advice
        if regime_upper == "IMPULSE":
            recommendations.update({
                "preferred_strategies": ["momentum", "breakout", "trend_following"],
                "avoid_strategies": ["mean_reversion", "range_trading"],
                "stop_loss_adjustment": "tighter",  # Tight stops in momentum
                "take_profit_adjustment": "extended",  # Let winners run
                "maker_vs_taker": "prefer_taker",  # Speed over rebate
            })
        elif regime_upper == "RANGE":
            recommendations.update({
                "preferred_strategies": ["mean_reversion", "range_trading", "market_making"],
                "avoid_strategies": ["momentum", "breakout"],
                "stop_loss_adjustment": "wider",  # Allow for noise
                "take_profit_adjustment": "conservative",  # Take quick profits
                "maker_vs_taker": "prefer_maker",  # Collect rebates
            })
        else:  # NORMAL
            recommendations.update({
                "preferred_strategies": ["all"],
                "avoid_strategies": [],
                "stop_loss_adjustment": "standard",
                "take_profit_adjustment": "standard",
                "maker_vs_taker": "balanced",
            })

        return recommendations

    async def _escalate_to(
        self,
        new_state: RiskState,
        reason_codes: list[str],
        metric: str,
        value: float,
        threshold: float,
    ) -> None:
        """Escalate to higher risk level.

        Args:
            new_state: New risk state
            reason_codes: Reason codes
            metric: Metric that triggered escalation
            value: Current value
            threshold: Threshold crossed
        """
        if new_state.value == self._state.value:
            return  # Already at this level

        old_state = self._state
        self._state = new_state
        self._level = list(RiskState).index(new_state)
        self._reason_codes = reason_codes
        self._metric = metric
        self._value = value
        self._threshold = threshold
        self._blocked_at = datetime.utcnow()

        logger.warning(
            f"Risk Governor escalated: {old_state.value} → {new_state.value}, "
            f"reason={reason_codes}, {metric}={value:.2f} >= {threshold:.2f}"
        )

        # Publish RISK_ALERT for state transitions
        await self._bus.publish(Event(
            event_type=EventType.RISK_ALERT,
            data={
                "type": "RISK_STATE_UPDATE",
                "state": self._state.value,
                "previous_state": old_state.value,
                "reason_codes": reason_codes,
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "recommended_action": self._get_recommended_action(),
                "clear_rule": self._get_clear_rule(),
            }
        ))

    async def _deescalate_to(self, new_state: RiskState) -> None:
        """De-escalate to lower risk level.

        Args:
            new_state: New risk state
        """
        old_state = self._state
        self._state = new_state
        self._level = list(RiskState).index(new_state)

        if new_state == RiskState.NORMAL:
            self._reason_codes.clear()
            self._metric = None
            self._value = None
            self._threshold = None
            self._blocked_at = None

        logger.info(f"Risk Governor de-escalated: {old_state.value} → {new_state.value}")

        await self._bus.publish(Event(
            event_type=EventType.KILLSWITCH_TRIGGERED,  # Reuse
            data={
                "type": "RISK_STATE_UPDATE",
                "state": self._state.value,
                "previous_state": old_state.value,
                "deescalated": True,
            }
        ))

    def _can_clear(self) -> bool:
        """Check if risk governor can be cleared.

        Returns:
            True if can be cleared
        """
        if self._state == RiskState.NORMAL:
            return True

        # Require cooldown period (1 hour minimum)
        if self._blocked_at:
            cooldown = datetime.utcnow() - self._blocked_at
            if cooldown < timedelta(hours=1):
                return False

        return True

    def _can_deescalate(self) -> bool:
        """Check if can de-escalate from current state.

        Returns:
            True if can de-escalate
        """
        # Require cooldown period (30 minutes minimum)
        if self._blocked_at:
            cooldown = datetime.utcnow() - self._blocked_at
            if cooldown < timedelta(minutes=30):
                return False

        return True

    def _get_recommended_action(self) -> str:
        """Get recommended action for current state.

        Returns:
            Recommended action string
        """
        if self._state == RiskState.NORMAL:
            return "Continue normal operation"
        elif self._state == RiskState.SOFT_BRAKE:
            return "Reduce position sizing by 50%. Monitor drawdown closely."
        elif self._state == RiskState.QUARANTINE:
            return "Review quarantined symbols. Consider manual intervention."
        elif self._state == RiskState.HARD_STOP:
            return "EMERGENCY: Close all positions. Review strategy before resuming."
        return "Unknown state"

    def _get_clear_rule(self) -> str:
        """Get rule for clearing current state.

        Returns:
            Clear rule string
        """
        if self._state == RiskState.NORMAL:
            return "N/A - already normal"
        elif self._state == RiskState.SOFT_BRAKE:
            return "Drawdown must reduce to <5% and wait 30 minutes"
        elif self._state == RiskState.QUARANTINE:
            return "Drawdown must reduce to <5% and wait 30 minutes, or manually clear quarantined symbols"
        elif self._state == RiskState.HARD_STOP:
            return "Drawdown must reduce to <5% and wait 1 hour, or force clear with confirmation"
        return "Unknown state"
