"""Position Monitor - Force-close stale positions.

Monitors all open positions and force-closes them if they exceed max_hold_seconds.
This prevents positions from being stuck open for hours.

Enhanced features:
- Profit protection: Close profitable positions before they reverse
- Dynamic TTL based on position state (profitable vs losing)
- Per-strategy TTL configuration
- Trailing profit protection
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType, OrderRequest, Position
from hean.logging import get_logger
from hean.portfolio.accounting import PortfolioAccounting

logger = get_logger(__name__)


@dataclass
class PositionState:
    """Tracks extended state for a position."""
    position_id: str
    peak_pnl: float = 0.0  # Highest PnL reached
    peak_pnl_time: datetime | None = None
    last_pnl: float = 0.0
    pnl_updates: int = 0
    trailing_stop_triggered: bool = False


class PositionMonitor:
    """Monitors positions and force-closes stale ones.

    Enhanced with:
    - Profit protection: Close positions when profit pulls back significantly
    - Dynamic TTL: Shorter TTL for losing positions, longer for profitable
    - Peak PnL tracking: Trail the high-water mark of each position
    """

    # Profit protection thresholds
    PROFIT_PULLBACK_THRESHOLD = 0.5  # Close if profit drops 50% from peak
    MIN_PROFIT_TO_PROTECT = 0.005  # Only protect if profit > 0.5%
    PROFITABLE_POSITION_EXTENDED_TTL = 1.5  # 1.5x TTL for profitable positions
    LOSING_POSITION_REDUCED_TTL = 0.7  # 0.7x TTL for losing positions

    def __init__(
        self,
        bus: EventBus,
        accounting: PortfolioAccounting,
    ) -> None:
        """Initialize the position monitor.

        Args:
            bus: Event bus for publishing close requests
            accounting: Portfolio accounting for position tracking
        """
        self._bus = bus
        self._accounting = accounting
        self._running = False
        self._monitor_task: asyncio.Task[None] | None = None

        # Configuration from settings
        self._check_interval_seconds = settings.position_monitor_check_interval
        self._max_hold_seconds = settings.max_hold_seconds
        self._force_close_enabled = settings.position_monitor_enabled

        # Enhanced: Per-position state tracking
        self._position_states: dict[str, PositionState] = {}

        # Enhanced: Profit protection settings
        self._profit_protection_enabled = True
        self._trailing_stop_enabled = True

        # Statistics
        self._positions_force_closed = 0
        self._positions_profit_protected = 0  # Closed to protect profit
        self._positions_trailing_stopped = 0  # Closed by trailing stop
        self._force_close_history: list[dict] = []

        logger.info(
            f"PositionMonitor initialized: max_hold={self._max_hold_seconds}s, "
            f"check_interval={self._check_interval_seconds}s, "
            f"profit_protection={self._profit_protection_enabled}"
        )

    async def start(self) -> None:
        """Start the position monitor."""
        if self._running:
            logger.warning("PositionMonitor already running")
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("PositionMonitor started")

    async def stop(self) -> None:
        """Stop the position monitor."""
        if not self._running:
            return

        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("PositionMonitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop - checks positions periodically."""
        logger.info("PositionMonitor loop started")

        while self._running:
            try:
                await self._check_positions()
                await asyncio.sleep(self._check_interval_seconds)

            except asyncio.CancelledError:
                logger.info("PositionMonitor loop cancelled")
                break

            except Exception as e:
                logger.error(f"Error in PositionMonitor loop: {e}", exc_info=True)
                # Continue running despite errors
                await asyncio.sleep(self._check_interval_seconds)

    async def _check_positions(self) -> None:
        """Check all open positions for TTL and profit protection."""
        if not self._force_close_enabled:
            return

        positions = self._accounting.get_positions()
        now = datetime.utcnow()

        stale_positions: list[tuple[Position, float, str]] = []  # (position, age, reason)
        profit_protected: list[tuple[Position, float, str]] = []

        for position in positions:
            # Initialize or update position state
            state = self._get_or_create_state(position)

            # Calculate PnL if price data available
            current_pnl_pct = self._calculate_position_pnl_pct(position)
            if current_pnl_pct is not None:
                self._update_position_state(state, current_pnl_pct, now)

                # Check profit protection
                if self._profit_protection_enabled:
                    close_reason = self._check_profit_protection(position, state, current_pnl_pct)
                    if close_reason:
                        profit_protected.append((position, state.peak_pnl * 100, close_reason))
                        continue  # Don't also check TTL

            # Calculate position age and dynamic TTL
            if position.opened_at:
                age = now - position.opened_at
                age_seconds = age.total_seconds()

                # Dynamic TTL based on position profitability
                dynamic_ttl = self._calculate_dynamic_ttl(state, current_pnl_pct)

                # Check if position is stale
                if age_seconds > dynamic_ttl:
                    reason = f"max_hold_exceeded ({age_seconds:.0f}s > {dynamic_ttl:.0f}s)"
                    stale_positions.append((position, age_seconds, reason))

        # Handle profit protection closes
        for position, _peak_pnl_pct, reason in profit_protected:
            await self._force_close_position(
                position,
                age_seconds=0,  # Not age-based
                reason=reason,
                close_type="profit_protection"
            )
            self._positions_profit_protected += 1

        # Force-close stale positions
        for position, age_seconds, reason in stale_positions:
            await self._force_close_position(
                position,
                age_seconds,
                reason=reason,
                close_type="ttl_exceeded"
            )

        # Cleanup state for closed positions
        self._cleanup_closed_positions(positions)

        # Log summary
        if stale_positions or profit_protected:
            logger.warning(
                f"PositionMonitor: TTL closed={len(stale_positions)}, "
                f"profit_protected={len(profit_protected)}"
            )

    def _get_or_create_state(self, position: Position) -> PositionState:
        """Get or create state tracking for a position."""
        if position.position_id not in self._position_states:
            self._position_states[position.position_id] = PositionState(
                position_id=position.position_id
            )
        return self._position_states[position.position_id]

    def _calculate_position_pnl_pct(self, position: Position) -> float | None:
        """Calculate position PnL as percentage.

        Returns:
            PnL percentage or None if cannot calculate
        """
        # Try to get from position metadata or calculate from prices
        if hasattr(position, 'unrealized_pnl_pct'):
            return position.unrealized_pnl_pct

        # If position has entry_price and we have current price
        if position.entry_price and position.entry_price > 0:
            # Try to get mark price from position
            mark_price = getattr(position, 'mark_price', None)
            if mark_price and mark_price > 0:
                if position.side == "buy":
                    return (mark_price - position.entry_price) / position.entry_price
                else:
                    return (position.entry_price - mark_price) / position.entry_price

        return None

    def _update_position_state(
        self,
        state: PositionState,
        current_pnl_pct: float,
        now: datetime
    ) -> None:
        """Update position state with current PnL."""
        state.last_pnl = current_pnl_pct
        state.pnl_updates += 1

        # Update peak PnL (high water mark)
        if current_pnl_pct > state.peak_pnl:
            state.peak_pnl = current_pnl_pct
            state.peak_pnl_time = now

    def _check_profit_protection(
        self,
        position: Position,
        state: PositionState,
        current_pnl_pct: float
    ) -> str | None:
        """Check if position should be closed to protect profit.

        Returns:
            Close reason string if should close, None otherwise
        """
        # Only protect if peak profit was significant
        if state.peak_pnl < self.MIN_PROFIT_TO_PROTECT:
            return None

        # Check if profit has pulled back significantly from peak
        if state.peak_pnl > 0 and current_pnl_pct >= 0:
            pullback_ratio = 1.0 - (current_pnl_pct / state.peak_pnl)

            if pullback_ratio >= self.PROFIT_PULLBACK_THRESHOLD:
                logger.info(
                    f"[PROFIT PROTECTION] {position.symbol} peak={state.peak_pnl*100:.2f}% "
                    f"current={current_pnl_pct*100:.2f}% pullback={pullback_ratio*100:.1f}%"
                )
                return f"profit_pullback_{pullback_ratio*100:.0f}pct"

        # Check if position turned from profit to loss
        if state.peak_pnl > self.MIN_PROFIT_TO_PROTECT and current_pnl_pct < 0:
            logger.info(
                f"[PROFIT PROTECTION] {position.symbol} turned negative: "
                f"peak={state.peak_pnl*100:.2f}% current={current_pnl_pct*100:.2f}%"
            )
            return "profit_turned_loss"

        return None

    def _calculate_dynamic_ttl(
        self,
        state: PositionState,
        current_pnl_pct: float | None
    ) -> float:
        """Calculate dynamic TTL based on position state.

        - Profitable positions get extended TTL (let winners run)
        - Losing positions get reduced TTL (cut losers quickly)
        """
        base_ttl = self._max_hold_seconds

        if current_pnl_pct is None:
            return base_ttl

        if current_pnl_pct > 0.01:  # > 1% profit
            # Extend TTL for profitable positions
            return base_ttl * self.PROFITABLE_POSITION_EXTENDED_TTL
        elif current_pnl_pct < -0.005:  # > 0.5% loss
            # Reduce TTL for losing positions
            return base_ttl * self.LOSING_POSITION_REDUCED_TTL

        return base_ttl

    def _cleanup_closed_positions(self, open_positions: list[Position]) -> None:
        """Remove state for positions that are no longer open."""
        open_ids = {p.position_id for p in open_positions}
        closed_ids = [
            pid for pid in self._position_states
            if pid not in open_ids
        ]
        for pid in closed_ids:
            del self._position_states[pid]

    async def _force_close_position(
        self,
        position: Position,
        age_seconds: float,
        reason: str = "max_hold_time_exceeded",
        close_type: str = "ttl_exceeded"
    ) -> None:
        """Force-close a position using market order.

        Args:
            position: Position to close
            age_seconds: Age of position in seconds
            reason: Reason for closing
            close_type: Type of close (ttl_exceeded, profit_protection, etc.)
        """
        # Get position state for logging
        state = self._position_states.get(position.position_id)
        peak_pnl = state.peak_pnl if state else 0.0
        last_pnl = state.last_pnl if state else 0.0

        logger.warning(
            f"FORCE-CLOSING position: {position.position_id} "
            f"({position.symbol} {position.side} {position.size:.6f}) "
            f"- type={close_type}, reason={reason}, "
            f"age={age_seconds:.0f}s, peak_pnl={peak_pnl*100:.2f}%, last_pnl={last_pnl*100:.2f}%"
        )

        # Determine close side (opposite of position side)
        close_side = "sell" if position.side == "buy" else "buy"

        # Create market order request to close position
        close_request = OrderRequest(
            strategy_id=position.strategy_id,
            symbol=position.symbol,
            side=close_side,
            size=position.size,
            price=None,  # Market order - no price
            order_type="market",
            stop_loss=None,
            take_profit=None,
            signal_id=f"force_close_{position.position_id}",
            metadata={
                "force_close": True,
                "close_type": close_type,
                "reason": reason,
                "position_id": position.position_id,
                "age_seconds": age_seconds,
                "max_hold_seconds": self._max_hold_seconds,
                "peak_pnl_pct": peak_pnl * 100,
                "last_pnl_pct": last_pnl * 100,
            },
        )

        # Publish order request
        try:
            await self._bus.publish(
                Event(
                    event_type=EventType.ORDER_REQUEST,
                    data={"order_request": close_request},
                )
            )

            # Record statistics
            self._positions_force_closed += 1
            self._force_close_history.append(
                {
                    "position_id": position.position_id,
                    "symbol": position.symbol,
                    "side": position.side,
                    "size": position.size,
                    "age_seconds": age_seconds,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            # Keep only last 100 force-close records
            if len(self._force_close_history) > 100:
                self._force_close_history = self._force_close_history[-100:]

            logger.info(
                f"Force-close order sent: {close_request.symbol} {close_side} {close_request.size:.6f} "
                f"(position_id={position.position_id})"
            )

        except Exception as e:
            logger.error(
                f"Failed to send force-close order for position {position.position_id}: {e}",
                exc_info=True,
            )

    def get_statistics(self) -> dict[str, Any]:
        """Get position monitor statistics."""
        return {
            "positions_force_closed": self._positions_force_closed,
            "positions_profit_protected": self._positions_profit_protected,
            "positions_trailing_stopped": self._positions_trailing_stopped,
            "force_close_enabled": self._force_close_enabled,
            "profit_protection_enabled": self._profit_protection_enabled,
            "max_hold_seconds": self._max_hold_seconds,
            "check_interval_seconds": self._check_interval_seconds,
            "tracked_positions": len(self._position_states),
            "recent_force_closes": self._force_close_history[-10:],  # Last 10
            # Position state summary
            "position_states": [
                {
                    "position_id": s.position_id,
                    "peak_pnl_pct": s.peak_pnl * 100,
                    "last_pnl_pct": s.last_pnl * 100,
                    "pnl_updates": s.pnl_updates,
                }
                for s in list(self._position_states.values())[:5]  # Top 5
            ],
        }

    def set_max_hold_seconds(self, seconds: int) -> None:
        """Update max hold time.

        Args:
            seconds: New max hold time in seconds
        """
        old_value = self._max_hold_seconds
        self._max_hold_seconds = max(60, seconds)  # Minimum 1 minute
        logger.info(f"Updated max_hold_seconds: {old_value}s -> {self._max_hold_seconds}s")

    def enable_force_close(self, enabled: bool = True) -> None:
        """Enable or disable force-close functionality.

        Args:
            enabled: True to enable, False to disable
        """
        self._force_close_enabled = enabled
        logger.info(f"Force-close {'ENABLED' if enabled else 'DISABLED'}")

    def enable_profit_protection(self, enabled: bool = True) -> None:
        """Enable or disable profit protection.

        Args:
            enabled: True to enable, False to disable
        """
        self._profit_protection_enabled = enabled
        logger.info(f"Profit protection {'ENABLED' if enabled else 'DISABLED'}")

    def set_profit_protection_threshold(self, pullback_threshold: float) -> None:
        """Set the profit pullback threshold for protection.

        Args:
            pullback_threshold: Threshold (0.0-1.0) e.g., 0.5 = close if profit drops 50% from peak
        """
        self.PROFIT_PULLBACK_THRESHOLD = max(0.1, min(0.9, pullback_threshold))
        logger.info(f"Profit pullback threshold set to {self.PROFIT_PULLBACK_THRESHOLD*100:.0f}%")
