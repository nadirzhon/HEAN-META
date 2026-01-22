"""Risk Governor - Multi-level risk management replacing binary killswitch.

Provides graduated risk levels:
- NORMAL: Normal operation
- SOFT_BRAKE: Reduced sizing (50%), increased cooldowns
- QUARANTINE: Per-symbol blocking
- HARD_STOP: Emergency halt
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Literal

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
        }

    async def check_and_update(
        self,
        equity: float,
        initial_capital: float,
        positions_count: int,
        orders_count: int,
    ) -> RiskState:
        """Check risk conditions and update state if needed.

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

        # Calculate drawdown
        drawdown_pct = ((initial_capital - equity) / initial_capital * 100) if initial_capital > 0 else 0.0

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

    async def clear_all(self, force: bool = False) -> dict[str, Any]:
        """Clear all risk governor blocks.

        Args:
            force: Force clear even if conditions not met

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

        logger.info(f"Risk Governor cleared: {old_state.value} → NORMAL (force={force})")

        await self._bus.publish(Event(
            event_type=EventType.KILLSWITCH_TRIGGERED,  # Reuse
            data={
                "type": "RISK_STATE_UPDATE",
                "state": self._state.value,
                "previous_state": old_state.value,
                "cleared": True,
                "forced": force,
            }
        ))

        return {
            "status": "cleared",
            "risk_state": self._state.value,
            "message": f"Risk governor cleared from {old_state.value}",
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

    def get_size_multiplier(self) -> float:
        """Get position size multiplier based on risk state.

        Returns:
            Size multiplier (0.0-1.0)
        """
        if self._state == RiskState.HARD_STOP:
            return 0.0
        elif self._state == RiskState.SOFT_BRAKE:
            return 0.5  # 50% sizing
        elif self._state == RiskState.QUARANTINE:
            return 0.75  # 75% sizing for non-quarantined symbols
        return 1.0  # Normal

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

        await self._bus.publish(Event(
            event_type=EventType.KILLSWITCH_TRIGGERED,  # Reuse
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
