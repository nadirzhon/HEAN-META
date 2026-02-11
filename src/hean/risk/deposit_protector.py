"""Deposit protection - critical system to prevent loss of initial capital."""

import asyncio

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class DepositProtector:
    """Protects deposit from falling below initial capital.

    Critical system that ensures equity NEVER falls below initial capital.
    This is the highest priority protection mechanism.
    """

    def __init__(self, bus: EventBus, initial_capital: float) -> None:
        """Initialize deposit protector.

        Args:
            bus: Event bus for publishing killswitch events
            initial_capital: Initial capital amount that must be protected
        """
        self._bus = bus
        self._initial_capital = initial_capital
        self._triggered = False
        logger.info(f"Deposit protector initialized for ${initial_capital:.2f}")

    def check_equity(self, equity: float) -> tuple[bool, str]:
        """Check if equity is safe.

        CRITICAL: Never allow equity below initial capital.

        Args:
            equity: Current portfolio equity

        Returns:
            (is_safe, reason) tuple.
            - is_safe=False means trading must stop immediately
            - is_safe=True means trading can continue
        """
        # Check drawdown from initial capital
        # Allow a 10% buffer before triggering (spreads and unrealized PnL fluctuations)
        drop_pct = ((self._initial_capital - equity) / self._initial_capital) * 100 if equity < self._initial_capital else 0.0

        # Killswitch threshold (default 30% drop from initial)
        if drop_pct >= settings.killswitch_drawdown_pct:
            if not self._triggered:
                self._triggered = True
                reason = (
                    f"Killswitch: {drop_pct:.1f}% drop from initial capital "
                    f"(${self._initial_capital:.2f} -> ${equity:.2f})"
                )
                logger.critical(reason)
                asyncio.create_task(self._trigger_killswitch(reason))
            return False, reason

        # Check capital preservation threshold (default 10% drop)
        if drop_pct >= settings.capital_preservation_drawdown_threshold:
            return True, "Capital preservation mode recommended"

        return True, "OK"

    async def _trigger_killswitch(self, reason: str) -> None:
        """Trigger killswitch event.

        Publishes events to stop all trading immediately.

        Args:
            reason: Reason for killswitch trigger
        """
        await self._bus.publish(
            Event(
                event_type=EventType.KILLSWITCH_TRIGGERED,
                data={"reason": reason},
            )
        )
        await self._bus.publish(
            Event(
                event_type=EventType.STOP_TRADING,
                data={"reason": reason},
            )
        )
        logger.critical(f"Killswitch triggered: {reason}")

    def is_triggered(self) -> bool:
        """Check if deposit protector is triggered.

        Returns:
            True if protector has been triggered (trading should stop)
        """
        return self._triggered

    def reset(self) -> None:
        """Reset deposit protector state.

        Note: Should only be called manually after review.
        """
        self._triggered = False
        logger.info("Deposit protector reset")

    def update_initial_capital(self, new_capital: float) -> None:
        """Update initial capital after exchange balance sync.

        Args:
            new_capital: Real balance from exchange
        """
        old = self._initial_capital
        self._initial_capital = new_capital
        self._triggered = False  # Reset since baseline changed
        logger.info(f"Deposit protector capital updated: ${old:.2f} -> ${new_capital:.2f}")

    @property
    def initial_capital(self) -> float:
        """Get initial capital amount."""
        return self._initial_capital
