"""Position Reconciliation Module.

Ensures local position state matches exchange state.
Critical for preventing ghost positions and missed fills.

Features:
- Periodic reconciliation with exchange
- Drift detection and alerting
- Automatic correction of minor discrepancies
- Emergency halt on major discrepancies
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, OrderRequest
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReconciliationResult:
    """Result of position reconciliation."""

    timestamp: datetime
    local_positions: int
    exchange_positions: int
    matched: int
    missing_locally: list[str]  # Position IDs missing in local state
    missing_on_exchange: list[str]  # Position IDs missing on exchange
    size_mismatches: list[dict[str, Any]]  # Positions with size discrepancies
    is_healthy: bool
    action_taken: str


class PositionReconciler:
    """Reconciles local position state with exchange.

    Periodically fetches positions from exchange and compares
    with local accounting to detect and resolve drift.
    """

    def __init__(
        self,
        bus: EventBus,
        bybit_http: Any,  # BybitHTTPClient
        accounting: Any,  # PortfolioAccounting
    ) -> None:
        """Initialize position reconciler.

        Args:
            bus: Event bus for publishing alerts
            bybit_http: Bybit HTTP client for fetching exchange state
            accounting: Portfolio accounting for local state
        """
        self._bus = bus
        self._bybit_http = bybit_http
        self._accounting = accounting
        self._running = False
        self._reconciliation_task: asyncio.Task[None] | None = None

        # Configuration
        self._reconciliation_interval = timedelta(seconds=30)  # Check every 30s
        self._size_tolerance_pct = 0.02  # 2% relative tolerance for size mismatches
        self._size_tolerance_abs = 0.001  # Absolute tolerance (handles tiny positions)
        self._max_drift_before_halt = 10  # Halt after 10 unresolved drifts (was 3)

        # State tracking
        self._consecutive_drifts = 0
        self._start_time = datetime.utcnow()
        self._startup_grace_seconds = 120  # Don't close orphans in first 2 minutes
        self._last_reconciliation: ReconciliationResult | None = None
        self._reconciliation_history: list[ReconciliationResult] = []

    async def start(self) -> None:
        """Start position reconciliation."""
        if self._running:
            return

        self._running = True
        self._reconciliation_task = asyncio.create_task(self._reconciliation_loop())
        logger.info("Position reconciliation started")

    async def stop(self) -> None:
        """Stop position reconciliation."""
        self._running = False
        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass
        logger.info("Position reconciliation stopped")

    async def _reconciliation_loop(self) -> None:
        """Main reconciliation loop."""
        while self._running:
            try:
                result = await self.reconcile()
                self._last_reconciliation = result
                self._reconciliation_history.append(result)

                # Keep only last 100 results
                if len(self._reconciliation_history) > 100:
                    self._reconciliation_history = self._reconciliation_history[-100:]

                if not result.is_healthy:
                    self._consecutive_drifts += 1
                    logger.warning(
                        f"Position drift detected (consecutive: {self._consecutive_drifts}): "
                        f"missing_locally={result.missing_locally}, "
                        f"missing_on_exchange={result.missing_on_exchange}, "
                        f"size_mismatches={len(result.size_mismatches)}"
                    )

                    if self._consecutive_drifts >= self._max_drift_before_halt:
                        await self._trigger_emergency_halt()
                else:
                    self._consecutive_drifts = 0

                await asyncio.sleep(self._reconciliation_interval.total_seconds())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reconciliation error: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait longer on error

    async def reconcile(self) -> ReconciliationResult:
        """Perform position reconciliation.

        Returns:
            ReconciliationResult with details of the reconciliation
        """
        timestamp = datetime.utcnow()

        try:
            # Fetch positions from exchange
            exchange_positions = await self._fetch_exchange_positions()

            # Get local positions
            local_positions = self._accounting.get_positions()

            # Build lookup maps
            exchange_map = {p["symbol"]: p for p in exchange_positions}
            local_map = {p.symbol: p for p in local_positions}

            # Find discrepancies
            missing_locally: list[str] = []
            missing_on_exchange: list[str] = []
            size_mismatches: list[dict[str, Any]] = []
            matched = 0

            # Check exchange positions against local
            for symbol, ex_pos in exchange_map.items():
                if symbol not in local_map:
                    missing_locally.append(symbol)
                else:
                    local_pos = local_map[symbol]
                    # Check size match within tolerance (use max of absolute and relative)
                    ex_size = float(ex_pos.get("size", 0))
                    local_size = local_pos.size
                    size_diff = abs(ex_size - local_size)
                    ref_size = max(ex_size, local_size)
                    # Pass if within absolute tolerance OR within relative tolerance
                    within_tolerance = (
                        size_diff <= self._size_tolerance_abs
                        or (ref_size > 0 and size_diff / ref_size <= self._size_tolerance_pct)
                    )
                    if not within_tolerance:
                        drift_pct = (size_diff / ref_size * 100) if ref_size > 0 else 0.0
                        size_mismatches.append({
                            "symbol": symbol,
                            "exchange_size": ex_size,
                            "local_size": local_size,
                            "drift_pct": drift_pct,
                        })
                    else:
                        matched += 1

            # Check local positions against exchange
            for symbol in local_map:
                if symbol not in exchange_map:
                    missing_on_exchange.append(symbol)

            # Determine health
            is_healthy = (
                len(missing_locally) == 0 and
                len(missing_on_exchange) == 0 and
                len(size_mismatches) == 0
            )

            # Take corrective action if needed
            action_taken = "none"
            if not is_healthy:
                action_taken = await self._take_corrective_action(
                    missing_locally,
                    missing_on_exchange,
                    size_mismatches,
                )

            return ReconciliationResult(
                timestamp=timestamp,
                local_positions=len(local_positions),
                exchange_positions=len(exchange_positions),
                matched=matched,
                missing_locally=missing_locally,
                missing_on_exchange=missing_on_exchange,
                size_mismatches=size_mismatches,
                is_healthy=is_healthy,
                action_taken=action_taken,
            )

        except Exception as e:
            logger.error(f"Reconciliation failed: {e}", exc_info=True)
            return ReconciliationResult(
                timestamp=timestamp,
                local_positions=0,
                exchange_positions=0,
                matched=0,
                missing_locally=[],
                missing_on_exchange=[],
                size_mismatches=[],
                is_healthy=False,
                action_taken=f"error: {str(e)}",
            )

    async def _fetch_exchange_positions(self) -> list[dict[str, Any]]:
        """Fetch positions from exchange (filters out zero-size closed positions)."""
        try:
            positions = await self._bybit_http.get_positions()
            # Bybit returns closed positions with size=0; filter them out
            return [p for p in (positions or []) if float(p.get("size", 0)) > 0]
        except Exception as e:
            logger.error(f"Failed to fetch exchange positions: {e}")
            return []

    async def _take_corrective_action(
        self,
        missing_locally: list[str],
        missing_on_exchange: list[str],
        size_mismatches: list[dict[str, Any]],
    ) -> str:
        """Take corrective action for position drift.

        Returns:
            Description of action taken
        """
        actions = []

        # Handle positions missing locally (orphaned positions on exchange)
        # Skip orphan cleanup during startup grace period (system may still be registering fills)
        elapsed = (datetime.utcnow() - self._start_time).total_seconds()
        if missing_locally and elapsed > self._startup_grace_seconds:
            for symbol in missing_locally:
                await self._close_orphaned_position(symbol)
            actions.append(f"closed_orphans:{missing_locally}")
        elif missing_locally:
            logger.info(
                f"Startup grace period ({int(self._startup_grace_seconds - elapsed)}s remaining): "
                f"skipping orphan cleanup for {missing_locally}"
            )
            actions.append(f"grace_period_skip:{missing_locally}")

        # Handle positions missing on exchange (ghost positions locally)
        # These can be auto-cleaned
        if missing_on_exchange:
            for symbol in missing_on_exchange:
                logger.warning(f"Removing ghost position (not on exchange): {symbol}")
                # Find and remove the position
                local_positions = self._accounting.get_positions()
                for pos in local_positions:
                    if pos.symbol == symbol:
                        self._accounting.remove_position(pos.position_id)
                        break
            actions.append(f"removed_ghost:{missing_on_exchange}")

        # Handle size mismatches - publish alert for manual review
        if size_mismatches:
            await self._publish_alert(
                level="warning",
                message=f"Position size mismatches detected: {size_mismatches}",
                action="manual_review_required",
            )
            actions.append(f"alert_size_mismatch:{len(size_mismatches)}")

        return ", ".join(actions) if actions else "none"

    async def _close_orphaned_position(self, symbol: str) -> None:
        """Close an orphaned position on exchange not tracked locally.

        Called when reconciliation finds positions on exchange that don't
        exist in local accounting (e.g. survived a crash or restart).
        """
        try:
            positions = await self._bybit_http.get_positions(symbol=symbol)
            if not positions:
                logger.warning(f"Orphan cleanup: position for {symbol} already closed")
                return

            for pos in positions:
                size = float(pos.get("size", 0))
                if size <= 0:
                    continue

                side = pos.get("side", "")  # "Buy" or "Sell"
                close_side = "sell" if side == "Buy" else "buy"

                logger.warning(
                    f"[ORPHAN CLEANUP] Closing orphaned position: {symbol} "
                    f"side={side} size={size}"
                )

                close_request = OrderRequest(
                    signal_id=f"orphan_cleanup_{symbol}_{int(datetime.utcnow().timestamp())}",
                    strategy_id="orphan_cleanup",
                    symbol=symbol,
                    side=close_side,
                    size=size,
                    price=None,
                    order_type="market",
                    metadata={"orphan_cleanup": True},
                )

                await self._bybit_http.place_order(close_request)
                logger.warning(f"[ORPHAN CLEANUP] Orphaned position closed: {symbol}")

                await self._publish_alert(
                    level="warning",
                    message=f"Orphaned position closed: {symbol} {side} size={size}",
                    action="orphan_closed_automatically",
                )

        except Exception as e:
            logger.error(f"[ORPHAN CLEANUP] Failed to close orphan {symbol}: {e}")
            await self._publish_alert(
                level="critical",
                message=f"FAILED to close orphaned {symbol}: {e}. Manual intervention required.",
                action="manual_close_required",
            )

    async def _publish_alert(
        self,
        level: str,
        message: str,
        action: str,
    ) -> None:
        """Publish a risk alert event."""
        await self._bus.publish(
            Event(
                event_type=EventType.RISK_ALERT,
                data={
                    "level": level,
                    "source": "position_reconciliation",
                    "message": message,
                    "action": action,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
        )

    async def _trigger_emergency_halt(self) -> None:
        """Trigger emergency trading halt due to persistent drift."""
        logger.error(
            f"EMERGENCY HALT: {self._consecutive_drifts} consecutive position drifts detected"
        )

        await self._bus.publish(
            Event(
                event_type=EventType.STOP_TRADING,
                data={
                    "reason": "position_reconciliation_failure",
                    "consecutive_drifts": self._consecutive_drifts,
                    "last_result": (
                        self._last_reconciliation.__dict__
                        if self._last_reconciliation else None
                    ),
                },
            )
        )

    def get_status(self) -> dict[str, Any]:
        """Get current reconciliation status."""
        return {
            "running": self._running,
            "consecutive_drifts": self._consecutive_drifts,
            "last_reconciliation": (
                {
                    "timestamp": self._last_reconciliation.timestamp.isoformat(),
                    "is_healthy": self._last_reconciliation.is_healthy,
                    "local_positions": self._last_reconciliation.local_positions,
                    "exchange_positions": self._last_reconciliation.exchange_positions,
                    "matched": self._last_reconciliation.matched,
                    "action_taken": self._last_reconciliation.action_taken,
                }
                if self._last_reconciliation else None
            ),
            "reconciliation_count": len(self._reconciliation_history),
        }

    async def force_reconcile(self) -> ReconciliationResult:
        """Force immediate reconciliation (for manual trigger)."""
        result = await self.reconcile()
        self._last_reconciliation = result
        self._reconciliation_history.append(result)
        return result
