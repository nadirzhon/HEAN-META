"""Profit capture system - locks profits when target is reached.

Enhanced with Intra-Session Compounding for reinvesting profits
within the trading session to maximize compound growth.
"""

from datetime import datetime
from typing import Any, Literal

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class IntraSessionCompounding:
    """Intra-session profit compounding for accelerated growth.

    Reinvests a portion of realized profits within the session:
    - Every +5% gain, reinvest 50% of the profit
    - Creates compound effect over 10-15 trades
    - Expected impact: +10-15% daily profit increase
    """

    def __init__(self, session_start_equity: float) -> None:
        """Initialize intra-session compounding.

        Args:
            session_start_equity: Starting equity at session begin
        """
        self._session_start_equity = session_start_equity
        self._last_compound_threshold = 0.0  # Last threshold we compounded at
        self._compound_step_pct = 5.0  # Compound every 5% gain
        self._reinvest_ratio = 0.5  # Reinvest 50% of profits
        self._total_reinvested = 0.0
        self._compound_events: list[dict[str, Any]] = []
        self._enabled = True

        logger.info(
            f"Intra-session compounding initialized: "
            f"step={self._compound_step_pct}%, reinvest_ratio={self._reinvest_ratio}"
        )

    def check_compound(self, current_equity: float) -> float:
        """Check if compounding should occur and return reinvestment amount.

        Args:
            current_equity: Current portfolio equity

        Returns:
            Amount to reinvest (add to available capital), or 0.0 if no compounding
        """
        if not self._enabled:
            return 0.0

        if self._session_start_equity <= 0:
            return 0.0

        # Calculate current growth percentage
        growth_pct = (
            (current_equity - self._session_start_equity) / self._session_start_equity
        ) * 100

        # Check if we've crossed a new compound threshold
        # Thresholds: 5%, 10%, 15%, 20%, 25%, ...
        current_threshold = (growth_pct // self._compound_step_pct) * self._compound_step_pct

        if current_threshold > self._last_compound_threshold and current_threshold > 0:
            # New threshold crossed! Calculate reinvestment
            profit = current_equity - self._session_start_equity
            reinvest_amount = profit * self._reinvest_ratio

            # Update tracking
            self._last_compound_threshold = current_threshold
            self._total_reinvested += reinvest_amount

            # Log compound event
            event = {
                "timestamp": datetime.utcnow().isoformat(),
                "threshold_pct": current_threshold,
                "growth_pct": growth_pct,
                "profit": profit,
                "reinvest_amount": reinvest_amount,
                "total_reinvested": self._total_reinvested,
            }
            self._compound_events.append(event)

            logger.info(
                f"INTRA-SESSION COMPOUND: Growth {growth_pct:.1f}% crossed {current_threshold:.0f}% threshold. "
                f"Reinvesting ${reinvest_amount:.2f} (50% of ${profit:.2f} profit). "
                f"Total reinvested: ${self._total_reinvested:.2f}"
            )

            return reinvest_amount

        return 0.0

    def get_state(self) -> dict[str, Any]:
        """Get current compounding state."""
        return {
            "enabled": self._enabled,
            "session_start_equity": self._session_start_equity,
            "last_compound_threshold": self._last_compound_threshold,
            "compound_step_pct": self._compound_step_pct,
            "reinvest_ratio": self._reinvest_ratio,
            "total_reinvested": self._total_reinvested,
            "compound_events_count": len(self._compound_events),
        }

    def reset(self, new_session_equity: float) -> None:
        """Reset for a new session.

        Args:
            new_session_equity: New session starting equity
        """
        self._session_start_equity = new_session_equity
        self._last_compound_threshold = 0.0
        self._total_reinvested = 0.0
        self._compound_events.clear()
        logger.info(f"Intra-session compounding reset: new session equity ${new_session_equity:.2f}")


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
        self._target_usd = settings.profit_capture_target_usd
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

        # Capture history (how many times we've locked profit this session)
        self._capture_count = 0
        self._total_captured_usd = 0.0

        # Intra-session compounding for accelerated profit growth
        self._intra_session_compounding = IntraSessionCompounding(start_equity)
        self._compounding_enabled = True  # Enable by default

        logger.info(
            f"Profit capture initialized: enabled={self._enabled}, "
            f"target_usd=${self._target_usd:.0f}, target_pct={self._target_pct}%, "
            f"trail={self._trail_pct}%, mode={self._mode}, "
            f"after_action={self._after_action}"
        )

    def get_state(self) -> dict[str, Any]:
        """Get current profit capture state for /trading/why endpoint."""
        state = {
            "enabled": self._enabled,
            "armed": self._armed,
            "triggered": self._triggered,
            "cleared": self._cleared,
            "mode": self._mode,
            "start_equity": self._start_equity,
            "peak_equity": self._peak_equity,
            "target_usd": self._target_usd,
            "target_pct": self._target_pct,
            "trail_pct": self._trail_pct,
            "after_action": self._after_action,
            "continue_risk_mult": self._continue_risk_mult,
            "last_action": self._last_action,
            "last_reason": self._last_reason,
            "capture_count": self._capture_count,
            "total_captured_usd": self._total_captured_usd,
        }

        # Include intra-session compounding state
        if self._compounding_enabled:
            state["intra_session_compounding"] = self._intra_session_compounding.get_state()

        return state

    def check_intra_session_compound(self, current_equity: float) -> float:
        """Check and apply intra-session compounding.

        Should be called after each profitable trade to potentially
        reinvest profits for compound growth.

        Args:
            current_equity: Current portfolio equity

        Returns:
            Amount to add to available capital (reinvestment), or 0.0
        """
        if not self._compounding_enabled:
            return 0.0

        return self._intra_session_compounding.check_compound(current_equity)

    async def check_and_trigger(self, current_equity: float, trading_system: Any) -> bool:
        """Check if profit capture should trigger and execute if needed.

        Checks three conditions (in order):
        1. Dollar-based: unrealized profit >= target_usd (e.g. $1000)
        2. Percent-based: equity growth >= target_pct from start
        3. Trail protection: drawdown from peak >= trail_pct

        After capture with after_action="continue", resets baseline
        so the cycle repeats automatically.

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

        # 1. Dollar-based trigger: unrealized profit >= target_usd
        unrealized_profit = current_equity - self._start_equity
        if unrealized_profit >= self._target_usd and not self._triggered:
            logger.info(
                f"[PROFIT LOCK] ${unrealized_profit:,.2f} profit reached "
                f"(target: ${self._target_usd:,.0f}). Locking profits! "
                f"(capture #{self._capture_count + 1})"
            )
            await self._execute_profit_capture(
                trading_system,
                reason="PROFIT_TARGET_USD_REACHED",
                growth_pct=None,
                captured_usd=unrealized_profit,
            )
            return True

        # 2. Percent-based trigger
        growth_pct = ((current_equity - self._start_equity) / self._start_equity) * 100 if self._start_equity > 0 else 0.0
        if growth_pct >= self._target_pct and not self._triggered:
            logger.info(
                f"[PROFIT LOCK] {growth_pct:.1f}% growth reached "
                f"(target: {self._target_pct}%). Locking profits!"
            )
            await self._execute_profit_capture(
                trading_system,
                reason="PROFIT_TARGET_PCT_REACHED",
                growth_pct=growth_pct,
            )
            return True

        # 3. Trail protection
        if self._peak_equity > self._start_equity:
            drawdown_from_peak = ((self._peak_equity - current_equity) / self._peak_equity) * 100
            if drawdown_from_peak >= self._trail_pct and not self._triggered:
                logger.info(
                    f"[PROFIT LOCK] Trail triggered: {drawdown_from_peak:.1f}% "
                    f"drawdown from peak ${self._peak_equity:,.2f}"
                )
                await self._execute_profit_capture(
                    trading_system,
                    reason="PROFIT_TRAIL_TRIGGERED",
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
        captured_usd: float | None = None,
    ) -> None:
        """Execute profit capture: close positions, lock profit, optionally continue.

        Args:
            trading_system: TradingSystem instance
            reason: Reason for trigger
            growth_pct: Growth percentage (for pct trigger)
            drawdown_pct: Drawdown percentage (for trail trigger)
            captured_usd: Dollar amount captured (for USD trigger)
        """
        self._triggered = True
        self._capture_count += 1
        self._last_action = "EXECUTED"
        self._last_reason = reason
        self._last_action_ts = datetime.utcnow()

        current_equity = trading_system._accounting.get_equity()
        profit_amount = captured_usd or (current_equity - self._start_equity)
        self._total_captured_usd += max(0, profit_amount)

        # Publish event
        event_data = {
            "type": "PROFIT_CAPTURE_EXECUTED",
            "reason": reason,
            "mode": self._mode,
            "after_action": self._after_action,
            "growth_pct": growth_pct,
            "drawdown_pct": drawdown_pct,
            "captured_usd": profit_amount,
            "capture_count": self._capture_count,
            "total_captured_usd": self._total_captured_usd,
            "start_equity": self._start_equity,
            "peak_equity": self._peak_equity,
            "current_equity": current_equity,
        }
        if self._after_action != "continue":
            await self._bus.publish(
                Event(event_type=EventType.STOP_TRADING, data=event_data)
            )

        # Close positions to realize profit
        closed_count = 0
        try:
            positions = trading_system._accounting.get_positions()

            if self._mode == "full":
                # Close ALL positions
                for position in positions:
                    price = position.current_price or position.entry_price
                    await trading_system._close_position_at_price(
                        position, price, reason=reason
                    )
                    closed_count += 1
            else:
                # Partial: close only profitable positions first
                profitable = [p for p in positions if p.unrealized_pnl > 0]
                losing = [p for p in positions if p.unrealized_pnl <= 0]

                for position in profitable:
                    price = position.current_price or position.entry_price
                    await trading_system._close_position_at_price(
                        position, price, reason=reason
                    )
                    closed_count += 1

                # If no profitable positions, close the least losing ones
                if not profitable and losing:
                    losing.sort(key=lambda p: p.unrealized_pnl, reverse=True)
                    for position in losing[:max(1, len(losing) // 2)]:
                        price = position.current_price or position.entry_price
                        await trading_system._close_position_at_price(
                            position, price, reason=reason
                        )
                        closed_count += 1

            # Cancel open orders
            cancelled = 0
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
                cancelled += 1

            logger.info(
                f"[PROFIT LOCK] Capture #{self._capture_count}: "
                f"closed {closed_count} positions, cancelled {cancelled} orders. "
                f"Locked ${profit_amount:,.2f} profit. "
                f"Session total: ${self._total_captured_usd:,.2f}"
            )
        except Exception as e:
            logger.error(f"Error executing profit capture: {e}", exc_info=True)

        # Handle after_action
        if self._after_action == "pause":
            trading_system._stop_trading = True
            logger.info("[PROFIT LOCK] Trading PAUSED after capture")
        elif self._after_action == "continue":
            # Reset baseline for next cycle — reinvest and keep trading
            new_equity = trading_system._accounting.get_equity()
            self._start_equity = new_equity
            self._peak_equity = new_equity
            self._triggered = False  # Allow next capture cycle
            self._intra_session_compounding.reset(new_equity)
            trading_system._stop_trading = False

            logger.info(
                f"[PROFIT LOCK] Reinvesting: new capital base ${new_equity:,.2f}. "
                f"Trading continues. Next capture at +${self._target_usd:,.0f}"
            )
            # Store risk multiplier
            if hasattr(trading_system, "_profit_capture_risk_mult"):
                trading_system._profit_capture_risk_mult = self._continue_risk_mult

    def sync_start_equity(self, exchange_balance: float) -> None:
        """Sync start equity with actual exchange balance.

        Must be called after fetching the real balance from the exchange,
        so profit capture doesn't see the difference between config
        initial_capital and actual balance as a "gain".

        Args:
            exchange_balance: Actual wallet balance from the exchange
        """
        old = self._start_equity
        self._start_equity = exchange_balance
        self._peak_equity = exchange_balance
        self._intra_session_compounding.reset(exchange_balance)
        logger.info(
            f"Profit capture start equity synced: ${old:.2f} -> ${exchange_balance:.2f}"
        )

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
