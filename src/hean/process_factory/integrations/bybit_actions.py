"""Bybit Action Adapter: Safe interface for Bybit actions (gated by config).

This module provides an abstract interface for Bybit actions that is:
- Safe-by-default (raises NotEnabledError unless explicitly enabled)
- Gated by config (process_factory.allow_actions=false default)
- No new trading logic (uses existing execution interfaces if available)
"""

from abc import ABC, abstractmethod
from typing import Any

from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)


class NotEnabledError(Exception):
    """Raised when actions are not enabled."""

    pass


class BybitActions(ABC):
    """Interface for Bybit actions (renamed from BybitActionAdapter for clarity)."""

    @abstractmethod
    async def place_limit_postonly(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> dict[str, Any]:
        """Place a post-only limit order on Bybit.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            side: Order side ('Buy' or 'Sell')
            qty: Order quantity
            price: Limit price

        Returns:
            Order response dictionary with order_id

        Raises:
            NotEnabledError: If actions are not enabled
        """
        pass

    @abstractmethod
    async def cancel_order(
        self, symbol: str, order_id: str
    ) -> dict[str, Any]:
        """Cancel an order on Bybit.

        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel

        Returns:
            Cancellation response dictionary

        Raises:
            NotEnabledError: If actions are not enabled
        """
        pass

    @abstractmethod
    async def get_min_notional(self, symbol: str) -> float:
        """Get minimum notional value for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Minimum notional value in USD

        Raises:
            NotEnabledError: If actions are not enabled
        """
        pass

    @abstractmethod
    async def get_symbol_rules(
        self, symbol: str
    ) -> dict[str, Any]:
        """Get trading rules for a symbol (min qty, step size, etc.).

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with symbol rules (minQty, qtyStep, minNotional, etc.)

        Raises:
            NotEnabledError: If actions are not enabled
        """
        pass


# Keep old name for backward compatibility
BybitActionAdapter = BybitActions


class DefaultBybitActions(BybitActions):
    """Default implementation that raises NotEnabledError."""

    async def place_limit_postonly(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> dict[str, Any]:
        """Place limit post-only order (disabled by default)."""
        raise NotEnabledError(
            "Bybit actions are disabled by default. "
            "Set PROCESS_FACTORY_ALLOW_ACTIONS=true and DRY_RUN=false to enable."
        )

    async def cancel_order(
        self, symbol: str, order_id: str
    ) -> dict[str, Any]:
        """Cancel order (disabled by default)."""
        raise NotEnabledError(
            "Bybit actions are disabled by default. "
            "Set PROCESS_FACTORY_ALLOW_ACTIONS=true and DRY_RUN=false to enable."
        )

    async def get_min_notional(self, symbol: str) -> float:
        """Get min notional (disabled by default)."""
        raise NotEnabledError(
            "Bybit actions are disabled by default. "
            "Set PROCESS_FACTORY_ALLOW_ACTIONS=true and DRY_RUN=false to enable."
        )

    async def get_symbol_rules(
        self, symbol: str
    ) -> dict[str, Any]:
        """Get symbol rules (disabled by default)."""
        raise NotEnabledError(
            "Bybit actions are disabled by default. "
            "Set PROCESS_FACTORY_ALLOW_ACTIONS=true and DRY_RUN=false to enable."
        )


class BybitHTTPAdapter(BybitActions):
    """Adapter that uses BybitHTTPClient for execution."""

    def __init__(self, http_client: Any) -> None:
        """Initialize adapter with HTTP client.

        Args:
            http_client: BybitHTTPClient instance
        """
        self._client = http_client

    async def place_limit_postonly(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> dict[str, Any]:
        """Place limit post-only order via BybitHTTPClient."""
        from hean.core.types import OrderRequest

        # Map side to OrderRequest format
        side_lower = side.upper()
        order_request = OrderRequest(
            signal_id="smoke_test",
            strategy_id="smoke_test",
            symbol=symbol,
            side="buy" if side_lower == "BUY" else "sell",
            size=qty,
            price=price,
            order_type="limit",
        )
        order = await self._client.place_order(order_request)
        return {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "qty": str(order.size),
            "price": str(order.price) if order.price else None,
            "status": order.status.value,
        }

    async def cancel_order(
        self, symbol: str, order_id: str
    ) -> dict[str, Any]:
        """Cancel order via BybitHTTPClient."""
        await self._client.cancel_order(order_id, symbol)
        return {"order_id": order_id, "symbol": symbol, "status": "cancelled"}

    async def get_min_notional(self, symbol: str) -> float:
        """Get min notional from symbol rules.
        
        CRITICAL: Fetches from API, no hardcoded defaults.
        """
        rules = await self.get_symbol_rules(symbol)
        min_notional = rules.get("minNotional")
        if min_notional is None:
            raise ValueError(f"minNotional not found in symbol rules for {symbol}")
        return float(min_notional)

    async def get_symbol_rules(
        self, symbol: str
    ) -> dict[str, Any]:
        """Get symbol rules from Bybit API.
        
        CRITICAL: Fetches real rules from API. Hardcoded defaults are forbidden in production.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with symbol rules (minQty, qtyStep, minNotional, etc.)
        """
        # CRITICAL: Fetch real rules from Bybit API, never use hardcoded defaults
        try:
            rules = await self._client.get_instrument_info(symbol)
            return rules
        except Exception as e:
            logger.error(
                f"Failed to fetch symbol rules for {symbol} from Bybit API: {e}. "
                "Cannot proceed with trading - symbol rules are required for order validation."
            )
            # CRITICAL: Do not return defaults - fail explicitly
            raise ValueError(
                f"Cannot fetch symbol rules for {symbol}. "
                "Symbol rules are required for safe order execution. "
                f"API error: {e}"
            ) from e


class GatedBybitActions(BybitActions):
    """Gated adapter that checks config before allowing actions."""

    def __init__(self, adapter: BybitActions | None = None) -> None:
        """Initialize gated adapter.

        Args:
            adapter: Underlying adapter to use if enabled (creates default if not provided)
        """
        self._adapter = adapter or DefaultBybitActions()
        self._allow_actions = getattr(
            settings, "process_factory_allow_actions", False
        )
        self._dry_run = getattr(settings, "dry_run", True)

    def _check_enabled(self) -> None:
        """Check if actions are enabled.

        Raises:
            NotEnabledError: If actions are not enabled
        """
        if not self._allow_actions:
            raise NotEnabledError(
                "Bybit actions are disabled. "
                "Set PROCESS_FACTORY_ALLOW_ACTIONS=true in config to enable."
            )
        if self._dry_run:
            raise NotEnabledError(
                "Dry run mode is enabled. "
                "Set DRY_RUN=false to allow real orders."
            )

    async def place_limit_postonly(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> dict[str, Any]:
        """Place limit post-only order (gated)."""
        self._check_enabled()
        logger.warning(
            f"PLACING POST-ONLY LIMIT ORDER: {side} {qty} {symbol} @ {price}"
        )
        return await self._adapter.place_limit_postonly(symbol, side, qty, price)

    async def cancel_order(
        self, symbol: str, order_id: str
    ) -> dict[str, Any]:
        """Cancel order (gated)."""
        self._check_enabled()
        logger.warning(f"CANCELLING ORDER: {order_id} on {symbol}")
        return await self._adapter.cancel_order(symbol, order_id)

    async def get_min_notional(self, symbol: str) -> float:
        """Get min notional (gated)."""
        self._check_enabled()
        return await self._adapter.get_min_notional(symbol)

    async def get_symbol_rules(
        self, symbol: str
    ) -> dict[str, Any]:
        """Get symbol rules (gated)."""
        self._check_enabled()
        return await self._adapter.get_symbol_rules(symbol)


# Keep old name for backward compatibility
GatedBybitActionAdapter = GatedBybitActions
DefaultBybitActionAdapter = DefaultBybitActions


def create_bybit_actions(http_client: Any | None = None) -> BybitActions:
    """Create a Bybit actions instance.

    Args:
        http_client: Optional BybitHTTPClient to use for execution

    Returns:
        BybitActions instance (gated by default)
    """
    adapter: BybitActions | None = None
    if http_client:
        adapter = BybitHTTPAdapter(http_client)
    return GatedBybitActions(adapter)


# Keep old name for backward compatibility
def create_bybit_action_adapter(http_client: Any | None = None) -> BybitActions:
    """Create a Bybit action adapter instance (backward compatibility).

    Returns:
        BybitActionAdapter instance (gated by default)
    """
    return create_bybit_actions(http_client)

