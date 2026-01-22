"""Execution smoke test for Process Factory.

Tests that execution adapter is properly configured and can place/cancel orders.
"""

import asyncio
from typing import Any

from hean.config import settings
from hean.exchange.bybit.http import BybitHTTPClient
from hean.logging import get_logger
from hean.process_factory.integrations.bybit_actions import (
    NotEnabledError,
    create_bybit_actions,
)

logger = get_logger(__name__)


async def run_smoke_test() -> dict[str, Any]:
    """Run execution smoke test.

    Returns:
        Dictionary with test results:
        - success: bool
        - order_id: str | None
        - symbol: str
        - side: str
        - qty: float
        - price: float
        - error: str | None
        - error_type: str | None

    Raises:
        ValueError: If required flags are not enabled
    """
    # Validate flags
    if not settings.process_factory_enabled:
        raise ValueError(
            "PROCESS_FACTORY_ENABLED must be true. "
            "Set PROCESS_FACTORY_ENABLED=true in config."
        )
    if not settings.process_factory_allow_actions:
        raise ValueError(
            "PROCESS_FACTORY_ALLOW_ACTIONS must be true for smoke test. "
            "Set PROCESS_FACTORY_ALLOW_ACTIONS=true in config."
        )
    if settings.dry_run:
        raise ValueError(
            "DRY_RUN must be false for smoke test. "
            "Set DRY_RUN=false in config."
        )

    symbol = settings.execution_smoke_test_symbol
    notional_usd = settings.execution_smoke_test_notional_usd
    side = settings.execution_smoke_test_side

    result: dict[str, Any] = {
        "success": False,
        "order_id": None,
        "symbol": symbol,
        "side": side,
        "qty": 0.0,
        "price": 0.0,
        "error": None,
        "error_type": None,
    }

    http_client: BybitHTTPClient | None = None
    try:
        # Create HTTP client for read-only operations (ticker, symbol rules)
        http_client = BybitHTTPClient()
        if settings.is_live:
            await http_client.connect()

        # Use default symbol rules (read-only, no gating needed for validation)
        symbol_rules = {
            "minNotional": 5.0,  # Default minimum
            "minQty": 0.001,
            "qtyStep": 0.001,
            "pricePrecision": 2,
        }
        min_notional = symbol_rules.get("minNotional", 5.0)
        min_qty = symbol_rules.get("minQty", 0.001)
        qty_step = symbol_rules.get("qtyStep", 0.001)

        # Validate notional meets minimum
        if notional_usd < min_notional:
            result["error"] = (
                f"Notional ${notional_usd} below minimum ${min_notional}"
            )
            result["error_type"] = "min_notional"
            return result

        # Get current ticker for price (read-only, works in paper mode too)
        if settings.is_live:
            ticker = await http_client.get_ticker(symbol)
            last_price = float(ticker.get("lastPrice", 0))
            bid_price = float(ticker.get("bid1Price", 0))
            ask_price = float(ticker.get("ask1Price", 0))
        else:
            # Paper mode: use synthetic prices
            last_price = 50000.0  # Default BTC price
            bid_price = last_price * 0.9999
            ask_price = last_price * 1.0001

        if not last_price or last_price <= 0:
            result["error"] = f"Invalid price from ticker: {ticker if settings.is_live else 'paper mode'}"
            result["error_type"] = "ticker_error"
            return result

        # Calculate quantity
        qty = notional_usd / last_price
        # Round to qty_step
        qty = round(qty / qty_step) * qty_step
        qty = max(qty, min_qty)  # Ensure minimum

        # Calculate limit price (around best bid/ask for post-only)
        if side.upper() == "BUY":
            # For BUY post-only, place slightly below best ask
            price = ask_price * 0.9995 if ask_price > 0 else last_price * 0.9995
        else:
            # For SELL post-only, place slightly above best bid
            price = bid_price * 1.0005 if bid_price > 0 else last_price * 1.0005

        # Round price to reasonable precision (2 decimal places for BTCUSDT)
        price = round(price, 2)

        result["qty"] = qty
        result["price"] = price

        # Create actions adapter (gated)
        actions = create_bybit_actions(http_client if settings.is_live else None)

        # Place order
        order_response = await actions.place_limit_postonly(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
        )

        order_id = order_response.get("order_id")
        result["order_id"] = order_id

        # Immediately cancel
        if order_id:
            await asyncio.sleep(0.5)  # Brief delay to ensure order is placed
            await actions.cancel_order(symbol=symbol, order_id=order_id)

        result["success"] = True
        logger.info(
            f"SUCCESS: order_id={order_id}, symbol={symbol}, "
            f"side={side}, qty={qty:.6f}, price={price:.2f}"
        )

    except ValueError as e:
        result["error"] = str(e)
        result["error_type"] = "validation_error"
        logger.error(f"FAIL: {e}")
    except NotEnabledError as e:
        result["error"] = str(e)
        result["error_type"] = "not_enabled"
        logger.error(f"FAIL: {e}")
    except Exception as e:
        result["error"] = str(e)
        result["error_type"] = type(e).__name__
        logger.error(f"FAIL: {e}", exc_info=True)
    finally:
        if http_client:
            try:
                await http_client.disconnect()
            except Exception:
                pass

    return result
