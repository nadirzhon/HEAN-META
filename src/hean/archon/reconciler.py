"""ARCHON Reconciler — Periodic state reconciliation between local and exchange."""

import asyncio
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class ArchonReconciler:
    """Periodic state reconciliation between local and exchange.

    Three reconciliation loops:
    1. Position reconciliation (configurable interval, default 30s)
    2. Balance reconciliation (every 60s)
    3. Order reconciliation (every 15s)

    On discrepancy: logs WARNING and publishes RECONCILIATION_ALERT event.
    """

    def __init__(
        self,
        bus: EventBus,
        accounting: Any,
        order_manager: Any,
        bybit_http: Any,
        interval_sec: int = 30,
    ) -> None:
        """Initialize reconciler.

        Args:
            bus: EventBus for publishing alerts
            accounting: PortfolioAccounting instance
            order_manager: OrderManager instance
            bybit_http: BybitHTTPClient instance
            interval_sec: Position reconciliation interval in seconds
        """
        self._bus = bus
        self._accounting = accounting
        self._order_manager = order_manager
        self._bybit_http = bybit_http
        self._interval = interval_sec

        self._running = False
        self._position_task: asyncio.Task[None] | None = None
        self._balance_task: asyncio.Task[None] | None = None
        self._order_task: asyncio.Task[None] | None = None

        # Metrics
        self._discrepancies: list[dict[str, Any]] = []
        self._reconciliation_count = 0
        self._discrepancy_count = 0

    async def start(self) -> None:
        """Start reconciliation loops as background tasks."""
        self._running = True
        self._position_task = asyncio.create_task(self._position_reconciliation_loop())
        self._balance_task = asyncio.create_task(self._balance_reconciliation_loop())
        self._order_task = asyncio.create_task(self._order_reconciliation_loop())
        logger.info("[Reconciler] Started position/balance/order reconciliation loops")

    async def stop(self) -> None:
        """Cancel all loops."""
        self._running = False
        tasks = [self._position_task, self._balance_task, self._order_task]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info(
            f"[Reconciler] Stopped — "
            f"{self._reconciliation_count} reconciliations, "
            f"{self._discrepancy_count} discrepancies found"
        )

    async def _position_reconciliation_loop(self) -> None:
        """Compare local positions with exchange positions."""
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                await self._reconcile_positions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Reconciler] Position reconciliation error: {e}", exc_info=True)

    async def _balance_reconciliation_loop(self) -> None:
        """Compare local equity with exchange balance."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Every 60 seconds
                await self._reconcile_balance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Reconciler] Balance reconciliation error: {e}", exc_info=True)

    async def _order_reconciliation_loop(self) -> None:
        """Compare local open orders with exchange open orders."""
        while self._running:
            try:
                await asyncio.sleep(15)  # Every 15 seconds
                await self._reconcile_orders()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Reconciler] Order reconciliation error: {e}", exc_info=True)

    async def _reconcile_positions(self) -> None:
        """Reconcile positions between local and exchange."""
        try:
            # Get exchange positions (wrapped in try/except — testnet may fail)
            exchange_positions = []
            try:
                exchange_positions = await self._bybit_http.get_positions()
            except Exception as e:
                logger.debug(f"[Reconciler] Failed to fetch exchange positions: {e}")
                return

            # Get local positions
            local_positions = self._accounting.get_positions()

            # Compare
            discrepancies = []
            exchange_by_symbol = {p.get("symbol"): p for p in exchange_positions}
            local_by_symbol = {p.symbol: p for p in local_positions}

            # Check for missing local positions
            for symbol in exchange_by_symbol:
                if symbol not in local_by_symbol:
                    discrepancies.append(
                        {
                            "type": "missing_local_position",
                            "symbol": symbol,
                            "exchange_size": exchange_by_symbol[symbol].get("size", 0),
                        }
                    )

            # Check for missing exchange positions
            for symbol in local_by_symbol:
                if symbol not in exchange_by_symbol:
                    discrepancies.append(
                        {
                            "type": "missing_exchange_position",
                            "symbol": symbol,
                            "local_size": local_by_symbol[symbol].size,
                        }
                    )

            # Check for size mismatches
            for symbol in set(local_by_symbol.keys()) & set(exchange_by_symbol.keys()):
                local_size = local_by_symbol[symbol].size
                exchange_size = exchange_by_symbol[symbol].get("size", 0)
                if abs(local_size - exchange_size) > 0.001:  # Tolerance for float comparison
                    discrepancies.append(
                        {
                            "type": "position_size_mismatch",
                            "symbol": symbol,
                            "local_size": local_size,
                            "exchange_size": exchange_size,
                        }
                    )

            self._reconciliation_count += 1

            if discrepancies:
                self._discrepancy_count += len(discrepancies)
                self._discrepancies.extend(discrepancies)
                # Keep only last 100 discrepancies
                if len(self._discrepancies) > 100:
                    self._discrepancies = self._discrepancies[-100:]

                logger.warning(f"[Reconciler] Position discrepancies found: {discrepancies}")

                # Publish event
                await self._bus.publish(
                    Event(
                        event_type=EventType.RECONCILIATION_ALERT,
                        data={
                            "category": "positions",
                            "discrepancies": discrepancies,
                        },
                    )
                )

        except Exception as e:
            logger.error(f"[Reconciler] Position reconciliation failed: {e}", exc_info=True)

    async def _reconcile_balance(self) -> None:
        """Reconcile balance between local and exchange."""
        try:
            # Get exchange balance (wrapped in try/except)
            exchange_balance = None
            try:
                balance_data = await self._bybit_http.get_wallet_balance()
                exchange_balance = balance_data.get("totalEquity", 0.0)
            except Exception as e:
                logger.debug(f"[Reconciler] Failed to fetch exchange balance: {e}")
                return

            # Get local equity
            local_equity = self._accounting.get_equity()

            # Compare (allow 1% tolerance due to mark price differences)
            tolerance = 0.01
            diff_pct = (
                abs(local_equity - exchange_balance) / max(exchange_balance, 1)
                if exchange_balance > 0
                else 0
            )

            self._reconciliation_count += 1

            if diff_pct > tolerance:
                discrepancy = {
                    "type": "balance_mismatch",
                    "local_equity": local_equity,
                    "exchange_balance": exchange_balance,
                    "diff_pct": round(diff_pct * 100, 2),
                }
                self._discrepancy_count += 1
                self._discrepancies.append(discrepancy)
                if len(self._discrepancies) > 100:
                    self._discrepancies = self._discrepancies[-100:]

                logger.warning(f"[Reconciler] Balance discrepancy: {discrepancy}")

                await self._bus.publish(
                    Event(
                        event_type=EventType.RECONCILIATION_ALERT,
                        data={
                            "category": "balance",
                            "discrepancies": [discrepancy],
                        },
                    )
                )

        except Exception as e:
            logger.error(f"[Reconciler] Balance reconciliation failed: {e}", exc_info=True)

    async def _reconcile_orders(self) -> None:
        """Reconcile orders between local and exchange."""
        try:
            # Get exchange open orders (wrapped in try/except)
            exchange_orders = []
            try:
                exchange_orders = await self._bybit_http.get_open_orders()
            except Exception as e:
                logger.debug(f"[Reconciler] Failed to fetch exchange orders: {e}")
                return

            # Get local open orders
            local_orders = getattr(self._order_manager, "get_open_orders", lambda: [])()

            # Compare
            discrepancies = []
            exchange_ids = {o.get("orderId") for o in exchange_orders}
            local_ids = {o.order_id for o in local_orders if hasattr(o, "order_id")}

            # Check for orders in exchange but not local
            missing_local = exchange_ids - local_ids
            if missing_local:
                discrepancies.append(
                    {
                        "type": "missing_local_orders",
                        "order_ids": list(missing_local),
                    }
                )

            # Check for orders in local but not exchange
            missing_exchange = local_ids - exchange_ids
            if missing_exchange:
                discrepancies.append(
                    {
                        "type": "missing_exchange_orders",
                        "order_ids": list(missing_exchange),
                    }
                )

            self._reconciliation_count += 1

            if discrepancies:
                self._discrepancy_count += len(discrepancies)
                self._discrepancies.extend(discrepancies)
                if len(self._discrepancies) > 100:
                    self._discrepancies = self._discrepancies[-100:]

                logger.warning(f"[Reconciler] Order discrepancies: {discrepancies}")

                await self._bus.publish(
                    Event(
                        event_type=EventType.RECONCILIATION_ALERT,
                        data={
                            "category": "orders",
                            "discrepancies": discrepancies,
                        },
                    )
                )

        except Exception as e:
            logger.error(f"[Reconciler] Order reconciliation failed: {e}", exc_info=True)

    def get_status(self) -> dict[str, Any]:
        """Get reconciliation status for API."""
        return {
            "running": self._running,
            "reconciliation_count": self._reconciliation_count,
            "discrepancy_count": self._discrepancy_count,
            "recent_discrepancies": self._discrepancies[-10:],
            "intervals": {
                "positions_sec": self._interval,
                "balance_sec": 60,
                "orders_sec": 15,
            },
        }
