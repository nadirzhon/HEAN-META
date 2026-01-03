"""Reconcile positions and orders with exchange state."""

from datetime import datetime
from typing import Any

from hean.config import settings
from hean.exchange.bybit.http import BybitHTTPClient
from hean.logging import get_logger

logger = get_logger(__name__)


class ReconcileService:
    """Service for reconciling internal state with exchange state."""

    def __init__(self, bybit_client: BybitHTTPClient | None = None) -> None:
        """Initialize reconcile service.

        Args:
            bybit_client: Optional Bybit HTTP client (creates if not provided)
        """
        self._bybit_client = bybit_client
        self._last_reconcile: datetime | None = None
        self._reconcile_lag_seconds: float = 0.0

    async def reconcile_positions(
        self, internal_positions: dict[str, Any]
    ) -> dict[str, Any]:
        """Reconcile internal positions with exchange positions.

        Args:
            internal_positions: Internal position state (symbol -> position dict)

        Returns:
            Reconciliation result with discrepancies
        """
        if not settings.is_live or not self._bybit_client:
            # In paper mode, no reconciliation needed
            return {
                "status": "skipped",
                "reason": "paper_mode",
                "discrepancies": [],
            }

        try:
            # Get exchange positions
            exchange_positions = await self._bybit_client.get_positions()

            # Convert to dict by symbol
            exchange_by_symbol: dict[str, dict[str, Any]] = {}
            for pos in exchange_positions:
                symbol = pos.get("symbol", "")
                if symbol:
                    exchange_by_symbol[symbol] = pos

            # Find discrepancies
            discrepancies = []
            all_symbols = set(internal_positions.keys()) | set(exchange_by_symbol.keys())

            for symbol in all_symbols:
                internal = internal_positions.get(symbol)
                exchange = exchange_by_symbol.get(symbol)

                if internal is None and exchange is not None:
                    discrepancies.append(
                        {
                            "symbol": symbol,
                            "type": "missing_internal",
                            "exchange": exchange,
                        }
                    )
                elif internal is not None and exchange is None:
                    discrepancies.append(
                        {
                            "symbol": symbol,
                            "type": "missing_exchange",
                            "internal": internal,
                        }
                    )
                elif internal is not None and exchange is not None:
                    # Compare sizes
                    internal_size = float(internal.get("size", 0))
                    exchange_size = float(exchange.get("size", 0))

                    if abs(internal_size - exchange_size) > 0.0001:  # Tolerance
                        discrepancies.append(
                            {
                                "symbol": symbol,
                                "type": "size_mismatch",
                                "internal_size": internal_size,
                                "exchange_size": exchange_size,
                                "difference": exchange_size - internal_size,
                            }
                        )

            self._last_reconcile = datetime.utcnow()
            self._reconcile_lag_seconds = 0.0  # Reset on successful reconcile

            return {
                "status": "completed",
                "timestamp": self._last_reconcile.isoformat(),
                "internal_count": len(internal_positions),
                "exchange_count": len(exchange_by_symbol),
                "discrepancies": discrepancies,
                "discrepancy_count": len(discrepancies),
            }

        except Exception as e:
            logger.error(f"Reconcile failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "discrepancies": [],
            }

    async def reconcile_orders(
        self, internal_orders: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Reconcile internal orders with exchange orders.

        Args:
            internal_orders: List of internal order dictionaries

        Returns:
            Reconciliation result
        """
        if not settings.is_live or not self._bybit_client:
            return {
                "status": "skipped",
                "reason": "paper_mode",
                "discrepancies": [],
            }

        try:
            # Get exchange open orders
            exchange_orders = await self._bybit_client.get_open_orders()

            # Convert to dict by order_id
            exchange_by_id: dict[str, dict[str, Any]] = {}
            for order in exchange_orders:
                order_id = order.get("orderId", "")
                if order_id:
                    exchange_by_id[order_id] = order

            # Find discrepancies
            discrepancies = []
            internal_by_id = {o.get("order_id", ""): o for o in internal_orders if o.get("order_id")}

            all_order_ids = set(internal_by_id.keys()) | set(exchange_by_id.keys())

            for order_id in all_order_ids:
                internal = internal_by_id.get(order_id)
                exchange = exchange_by_id.get(order_id)

                if internal is None and exchange is not None:
                    discrepancies.append(
                        {
                            "order_id": order_id,
                            "type": "missing_internal",
                            "exchange": exchange,
                        }
                    )
                elif internal is not None and exchange is None:
                    # Check if order is filled/cancelled (expected)
                    status = internal.get("status", "")
                    if status not in ("FILLED", "CANCELLED"):
                        discrepancies.append(
                            {
                                "order_id": order_id,
                                "type": "missing_exchange",
                                "internal": internal,
                            }
                        )

            self._last_reconcile = datetime.utcnow()

            return {
                "status": "completed",
                "timestamp": self._last_reconcile.isoformat(),
                "internal_count": len(internal_orders),
                "exchange_count": len(exchange_orders),
                "discrepancies": discrepancies,
                "discrepancy_count": len(discrepancies),
            }

        except Exception as e:
            logger.error(f"Order reconcile failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "discrepancies": [],
            }

    def get_reconcile_lag(self) -> float:
        """Get time since last reconcile in seconds."""
        if self._last_reconcile is None:
            return 0.0
        return (datetime.utcnow() - self._last_reconcile).total_seconds()

