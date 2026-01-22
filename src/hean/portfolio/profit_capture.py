"""Profit capture system - locks profits when target is reached."""

from datetime import datetime
from typing import Any, Literal

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class ProfitCapture:
    """Profit capture system that locks profits when target is reached.
    
    Tracks start_equity (session start) and peak_equity.
    When growth >= target_pct: triggers profit capture.
    When drawdown from peak >= trail_pct: triggers trail protection.
    """

    def __init__(self, bus: EventBus, start_equity: float) -> None:
        """Initialize profit capture.
        
        Args:
            bus: Event bus for publishing events
            start_equity: Starting equity for the session
        """
        self._bus = bus
        self._enabled = settings.profit_capture_enabled
        self._target_pct = settings.profit_capture_target_pct
        self._trail_pct = settings.profit_capture_trail_pct
        self._mode: Literal["partial", "full"] = settings.profit_capture_mode
        self._after_action: Literal["pause", "continue"] = settings.profit_capture_after_action
        self._continue_risk_mult = settings.profit_capture_continue_risk_mult
        
        self._start_equity = start_equity
        self._peak_equity = start_equity
        self._armed = False
        self._triggered = False
        self._cleared = False
        self._last_action: str | None = None
        self._last_reason: str | None = None
        self._last_action_ts: datetime | None = None
        
        logger.info(
            f"Profit capture initialized: enabled={self._enabled}, "
            f"target={self._target_pct}%, trail={self._trail_pct}%, "
            f"mode={self._mode}, after_action={self._after_action}"
        )

    def get_state(self) -> dict[str, Any]:
        """Get current profit capture state for /trading/why endpoint."""
        return {
            "enabled": self._enabled,
            "armed": self._armed,
            "triggered": self._triggered,
            "cleared": self._cleared,
            "mode": self._mode,
            "start_equity": self._start_equity,
            "peak_equity": self._peak_equity,
            "target_pct": self._target_pct,
            "trail_pct": self._trail_pct,
            "after_action": self._after_action,
            "continue_risk_mult": self._continue_risk_mult,
            "last_action": self._last_action,
            "last_reason": self._last_reason,
        }

    async def check_and_trigger(self, current_equity: float, trading_system: Any) -> bool:
        """Check if profit capture should trigger and execute if needed.
        
        Args:
            current_equity: Current portfolio equity
            trading_system: TradingSystem instance for closing positions/orders
        
        Returns:
            True if profit capture was triggered, False otherwise
        """
        if not self._enabled:
            return False
        
        # Update peak equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        
        # Check target trigger
        growth_pct = ((current_equity - self._start_equity) / self._start_equity) * 100 if self._start_equity > 0 else 0.0
        if growth_pct >= self._target_pct and not self._triggered:
            logger.info(
                f"Profit capture TARGET triggered: equity=${current_equity:.2f}, "
                f"growth={growth_pct:.2f}% >= {self._target_pct}%"
            )
            await self._execute_profit_capture(
                trading_system,
                reason="PROFIT_CAPTURE_REACHED",
                growth_pct=growth_pct,
            )
            return True
        
        # Check trail trigger
        if self._peak_equity > self._start_equity:
            drawdown_from_peak = ((self._peak_equity - current_equity) / self._peak_equity) * 100
            if drawdown_from_peak >= self._trail_pct and not self._triggered:
                logger.info(
                    f"Profit capture TRAIL triggered: equity=${current_equity:.2f}, "
                    f"drawdown from peak={drawdown_from_peak:.2f}% >= {self._trail_pct}%"
                )
                await self._execute_profit_capture(
                    trading_system,
                    reason="PROFIT_CAPTURE_TRAIL_TRIGGERED",
                    drawdown_pct=drawdown_from_peak,
                )
                return True
        
        return False

    async def _execute_profit_capture(
        self,
        trading_system: Any,
        reason: str,
        growth_pct: float | None = None,
        drawdown_pct: float | None = None,
    ) -> None:
        """Execute profit capture action.
        
        Args:
            trading_system: TradingSystem instance
            reason: Reason for trigger (PROFIT_CAPTURE_REACHED or PROFIT_CAPTURE_TRAIL_TRIGGERED)
            growth_pct: Growth percentage (for target trigger)
            drawdown_pct: Drawdown percentage (for trail trigger)
        """
        self._triggered = True
        self._last_action = "EXECUTED"
        self._last_reason = reason
        self._last_action_ts = datetime.utcnow()
        
        # Publish event
        await self._bus.publish(
            Event(
                event_type=EventType.STOP_TRADING,  # Use existing event type
                data={
                    "type": "PROFIT_CAPTURE_EXECUTED",
                    "reason": reason,
                    "mode": self._mode,
                    "after_action": self._after_action,
                    "growth_pct": growth_pct,
                    "drawdown_pct": drawdown_pct,
                    "start_equity": self._start_equity,
                    "peak_equity": self._peak_equity,
                    "current_equity": trading_system._accounting.get_equity(),
                },
            )
        )
        
        if self._mode == "full":
            # Close all positions and cancel all orders
            try:
                # Close all positions
                positions = list(trading_system._accounting.get_positions().values())
                for position in positions:
                    await trading_system.close_position(position.position_id, reason=reason)
                
                # Cancel all orders
                from hean.core.types import OrderStatus
                open_orders = trading_system._order_manager.get_open_orders()
                for order in open_orders:
                    order.status = OrderStatus.CANCELLED
                    await trading_system._bus.publish(
                        Event(
                            event_type=EventType.ORDER_CANCELLED,
                            data={"order": order, "reason": reason},
                        )
                    )
                
                logger.info(f"Profit capture FULL: closed {len(positions)} positions, cancelled {len(open_orders)} orders")
            except Exception as e:
                logger.error(f"Error executing profit capture (full): {e}", exc_info=True)
        elif self._mode == "partial":
            # Reduce-only partial close (minimal safe implementation)
            try:
                positions = list(trading_system._accounting.get_positions().values())
                # Close 50% of positions (reduce exposure)
                close_count = max(1, len(positions) // 2)
                for position in positions[:close_count]:
                    await trading_system.close_position(position.position_id, reason=reason)
                
                # Cancel risky orders (orders with high notional)
                open_orders = trading_system._order_manager.get_open_orders()
                cancelled = 0
                from hean.core.types import OrderStatus
                for order in open_orders:
                    notional = order.size * (order.price or 0)
                    if notional > 100:  # Cancel orders > $100 notional
                        order.status = OrderStatus.CANCELLED
                        await trading_system._bus.publish(
                            Event(
                                event_type=EventType.ORDER_CANCELLED,
                                data={"order": order, "reason": reason},
                            )
                        )
                        cancelled += 1
                
                logger.info(f"Profit capture PARTIAL: closed {close_count} positions, cancelled {cancelled} risky orders")
            except Exception as e:
                logger.error(f"Error executing profit capture (partial): {e}", exc_info=True)
        
        # Handle after_action
        if self._after_action == "pause":
            trading_system._stop_trading = True
            logger.info("Profit capture: Trading PAUSED after capture")
        elif self._after_action == "continue":
            # Apply reduced risk multiplier (must be visible in /trading/why + decisions)
            # This will be handled by position sizer or risk limits
            logger.info(f"Profit capture: Trading CONTINUES with {self._continue_risk_mult}x risk multiplier")
            # Store risk multiplier in trading system for position sizer to use
            if hasattr(trading_system, "_profit_capture_risk_mult"):
                trading_system._profit_capture_risk_mult = self._continue_risk_mult

    def arm(self) -> None:
        """Arm profit capture (enable monitoring)."""
        if not self._enabled:
            return
        self._armed = True
        logger.info("Profit capture ARMED")

    def disarm(self) -> None:
        """Disarm profit capture (disable monitoring)."""
        self._armed = False
        self._triggered = False
        self._cleared = True
        self._last_action = "DISARMED"
        self._last_reason = "MANUAL_DISARM"
        self._last_action_ts = datetime.utcnow()
        logger.info("Profit capture DISARMED")

    def clear(self) -> None:
        """Clear profit capture state (reset after manual intervention)."""
        self._cleared = True
        self._triggered = False
        self._last_action = "CLEARED"
        self._last_reason = "MANUAL_CLEAR"
        self._last_action_ts = datetime.utcnow()
        logger.info("Profit capture CLEARED")
