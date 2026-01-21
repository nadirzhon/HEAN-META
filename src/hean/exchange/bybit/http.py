"""Bybit HTTP API client with full implementation."""

import asyncio
import hashlib
import hmac
import time
from typing import Any

import httpx

from hean.config import settings
from hean.core.types import Order, OrderRequest, OrderStatus
from hean.logging import get_logger

logger = get_logger(__name__)


class BybitHTTPClient:
    """Bybit HTTP API client with full implementation."""

    def __init__(self) -> None:
        """Initialize the Bybit HTTP client."""
        self._connected = False
        self._api_key = settings.bybit_api_key
        self._api_secret = settings.bybit_api_secret
        self._testnet = settings.bybit_testnet

        # Base URLs
        if self._testnet:
            self._base_url = "https://api-testnet.bybit.com"
        else:
            self._base_url = "https://api.bybit.com"

        self._client: httpx.AsyncClient | None = None
        
        # Phase 16: Dynamic endpoint switching support
        self._dynamic_endpoint: str | None = None

    def _sign_request(self, params: dict[str, Any], timestamp: int) -> str:
        """Sign request using HMAC SHA256 for Bybit API v5.

        Args:
            params: Request parameters (will NOT be modified)
            timestamp: Request timestamp in milliseconds

        Returns:
            Signature string
        """
        # For Bybit API v5, signature string format is: timestamp + api_key + recv_window + sorted_params
        import urllib.parse

        recv_window = "5000"

        # Sort parameters and create query string (URL-encoded, without timestamp/recv_window/api_key)
        sorted_params = sorted(params.items())
        param_str = "&".join(
            [f"{k}={urllib.parse.quote(str(v), safe='')}" for k, v in sorted_params]
        )

        # Build signature string: timestamp + api_key + recv_window + params
        sign_string = f"{timestamp}{self._api_key}{recv_window}{param_str}"

        # Create signature (HMAC SHA256)
        signature = hmac.new(
            self._api_secret.encode("utf-8"), sign_string.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return signature

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make authenticated HTTP request to Bybit API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/v5/order/create")
            params: Query parameters
            data: Request body data

        Returns:
            Response JSON as dict

        Raises:
            ValueError: If API credentials not configured
            httpx.HTTPError: If request fails
        """
        if not self._api_key or not self._api_secret:
            raise ValueError("Bybit API credentials not configured")

        if not self._client:
            self._client = httpx.AsyncClient(timeout=10.0)

        # Prepare parameters
        if params is None:
            params = {}
        if data is None:
            data = {}

        # Generate timestamp
        timestamp = int(time.time() * 1000)

        # For signing: merge params and data (for POST, data is in body but included in signature)
        sign_params = {**params, **data}

        # Create signature (doesn't include api_key, only timestamp + recv_window + params)
        signature = self._sign_request(sign_params, timestamp)

        # Prepare headers (Bybit API v5 uses headers for auth)
        headers = {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",  # Signature type 2 = HMAC SHA256
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": "5000",
        }

        # For POST requests, add Content-Type
        if method.upper() == "POST":
            headers["Content-Type"] = "application/json"

        # Phase 16: Use dynamic endpoint if set (from API Scouter)
        base_url = self._dynamic_endpoint if self._dynamic_endpoint else self._base_url
        url = f"{base_url}{endpoint}"

        # Retry logic with exponential backoff
        max_retries = 3
        base_delay = 1.0  # seconds

        for attempt in range(max_retries):
            try:
                if method.upper() == "GET":
                    # For GET, params go in query string
                    response = await self._client.get(url, params=params, headers=headers)
                elif method.upper() == "POST":
                    # For POST, data goes in body as JSON, params in query string
                    response = await self._client.post(
                        url, json=data, params=params, headers=headers
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()
                result = response.json()

                # Check Bybit API response format
                ret_code = result.get("retCode")
                if ret_code != 0:
                    error_msg = result.get("retMsg", "Unknown error")
                    # Don't retry on authentication or validation errors
                    if ret_code in (10003, 10004, 10005, 10006):  # Auth errors
                        logger.error(
                            f"Bybit API authentication error: {error_msg} (code: {ret_code})"
                        )
                        raise ValueError(f"Bybit API error: {error_msg}")
                    # Retry on rate limit or server errors
                    if (
                        ret_code in (10002, 10017) and attempt < max_retries - 1
                    ):  # Rate limit, server error
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            f"Bybit API error {ret_code}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.error(f"Bybit API error: {error_msg} (code: {ret_code})")
                    raise ValueError(f"Bybit API error: {error_msg}")

                return result.get("result", {})

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Network error, retrying in {delay}s (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"HTTP request failed after {max_retries} attempts: {e}")
                raise
            except httpx.HTTPStatusError as e:
                # Retry on 429 (rate limit) and 5xx server errors
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    # HTTP 429 rate limit - use exponential backoff
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Rate limited (429), retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                elif e.response.status_code >= 500 and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Server error {e.response.status_code}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"HTTP request failed: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in API request: {e}")
                raise

        # Should not reach here, but just in case
        raise RuntimeError(f"Request failed after {max_retries} attempts")

    async def connect(self) -> None:
        """Connect to Bybit API."""
        if not settings.is_live:
            logger.info("Bybit HTTP client: Paper mode - no connection needed")
            return

        if not self._api_key or not self._api_secret:
            raise ValueError("Bybit API credentials not configured")

        # Test connection by getting account info
        try:
            await self.get_account_info()
            self._connected = True
            logger.info(
                f"Bybit HTTP client connected to {'testnet' if self._testnet else 'mainnet'}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Bybit API: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Bybit API."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        logger.info("Bybit HTTP client disconnected")

    async def get_account_info(self) -> dict[str, Any]:
        """Get account information.

        Returns:
            Account information dict
        """
        return await self._request(
            "GET", "/v5/account/wallet-balance", params={"accountType": "UNIFIED"}
        )

    async def place_order(self, order_request: OrderRequest) -> Order:
        """Place an order on Bybit.

        CRITICAL: This method MUST NOT be called in simulation/dry_run mode.
        ExecutionRouter should route to PaperBroker instead.

        Args:
            order_request: Order request to place

        Returns:
            Created Order object

        Raises:
            RuntimeError: If called in simulation mode or without live trading enabled
        """
        # CRITICAL: Defensive check - prevent real API calls in simulation mode
        if settings.dry_run:
            raise RuntimeError(
                "CRITICAL: Cannot place real orders when DRY_RUN=true. "
                "This is a simulation mode - use PaperBroker instead. "
                "If you see this error, there is a bug in ExecutionRouter routing logic."
            )
        if not settings.is_live:
            raise RuntimeError(
                "CRITICAL: Cannot place orders when live trading is not enabled. "
                "Set LIVE_CONFIRM=YES and trading_mode=live to enable live trading. "
                "Use PaperBroker for simulation."
            )

        # CRITICAL: Generate unique orderLinkId for idempotency
        # Format: {strategy_id}_{symbol}_{timestamp_ms}_{random}
        import uuid
        import time
        order_link_id = (
            f"{order_request.strategy_id}_{order_request.symbol}_"
            f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        )

        # Map order request to Bybit API format
        bybit_params = {
            "category": "linear",  # Linear perpetual futures
            "symbol": order_request.symbol,
            "side": order_request.side.capitalize(),  # Buy or Sell
            "orderType": "Market" if order_request.order_type == "market" else "Limit",
            "qty": str(order_request.size),
            "timeInForce": "PostOnly" if order_request.order_type == "limit" else "IOC",
            "orderLinkId": order_link_id,  # CRITICAL: Idempotency key
        }

        # Add price for limit orders
        if order_request.order_type == "limit" and order_request.price:
            bybit_params["price"] = str(order_request.price)

        # Add stop loss and take profit
        if order_request.stop_loss:
            bybit_params["stopLoss"] = str(order_request.stop_loss)
        if order_request.take_profit:
            bybit_params["takeProfit"] = str(order_request.take_profit)

        # Place order
        response = await self._request("POST", "/v5/order/create", data=bybit_params)

        # Parse response
        order_data = response.get("orderId")
        if not order_data:
            raise ValueError("Invalid response from Bybit API: missing orderId")

        # Create Order object
        order = Order(
            order_id=order_data if isinstance(order_data, str) else str(order_data),
            strategy_id=order_request.strategy_id,
            symbol=order_request.symbol,
            side=order_request.side,
            size=order_request.size,
            price=order_request.price,
            order_type=order_request.order_type,
            status=OrderStatus.PLACED,
            stop_loss=order_request.stop_loss,
            take_profit=order_request.take_profit,
            metadata={
                **(order_request.metadata or {}),
                "signal_id": order_request.signal_id,
                "orderLinkId": order_link_id,  # Store for idempotency tracking
            },
        )

        logger.info(
            f"Order placed on Bybit: {order.order_id} {order.symbol} {order.side} {order.size}"
        )
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> None:
        """Cancel an order on Bybit.

        CRITICAL: This method MUST NOT be called in simulation/dry_run mode.

        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol

        Raises:
            RuntimeError: If called in simulation mode or without live trading enabled
        """
        # CRITICAL: Defensive check - prevent real API calls in simulation mode
        if settings.dry_run:
            raise RuntimeError(
                "CRITICAL: Cannot cancel real orders when DRY_RUN=true. "
                "This is a simulation mode. If you see this error, there is a bug in routing logic."
            )
        if not settings.is_live:
            raise RuntimeError("Bybit HTTP client: Cannot cancel orders in paper mode")

        params = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }

        await self._request("POST", "/v5/order/cancel", data=params)
        logger.info(f"Order cancelled on Bybit: {order_id}")

    async def get_order_status(self, order_id: str, symbol: str) -> dict[str, Any]:
        """Get order status from Bybit.

        Args:
            order_id: Order ID
            symbol: Trading symbol

        Returns:
            Order status information
        """
        params = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }

        response = await self._request("GET", "/v5/order/history", params=params)
        orders = response.get("list", [])

        if not orders:
            raise ValueError(f"Order not found: {order_id}")

        return orders[0]

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        """Get current ticker for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker information
        """
        params = {
            "category": "linear",
            "symbol": symbol,
        }

        response = await self._request("GET", "/v5/market/tickers", params=params)
        tickers = response.get("list", [])

        if not tickers:
            raise ValueError(f"Ticker not found for symbol: {symbol}")

        return tickers[0]

    async def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get open positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of position information
        """
        params = {
            "category": "linear",
            "settleCoin": "USDT",  # Required parameter for Bybit API
        }

        if symbol:
            params["symbol"] = symbol

        response = await self._request("GET", "/v5/position/list", params=params)
        return response.get("list", [])

    async def get_funding_rate(self, symbol: str) -> dict[str, Any]:
        """Get current funding rate for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Funding rate information
        """
        params = {
            "category": "linear",
            "symbol": symbol,
        }

        response = await self._request("GET", "/v5/market/funding/history", params=params)
        # Get the most recent funding rate
        funding_list = response.get("list", [])
        if funding_list:
            return funding_list[0]  # Most recent
        return {}

    async def get_order_book(self, symbol: str, limit: int = 25) -> dict[str, Any]:
        """Get order book for a symbol.

        Args:
            symbol: Trading symbol
            limit: Order book depth (1, 25, 50, 100, 200)

        Returns:
            Order book data
        """
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": limit,
        }

        response = await self._request("GET", "/v5/market/orderbook", params=params)
        return response

    async def get_instrument_info(self, symbol: str) -> dict[str, Any]:
        """Get instrument information including trading rules.

        CRITICAL: This fetches real symbol rules (minQty, qtyStep, minNotional) from Bybit API.
        Hardcoded defaults are NOT acceptable for production trading.

        Args:
            symbol: Trading symbol

        Returns:
            Instrument information dict with lotSizeFilter, priceFilter, etc.
        """
        params = {
            "category": "linear",
            "symbol": symbol,
        }

        response = await self._request("GET", "/v5/market/instruments-info", params=params)
        instruments = response.get("list", [])
        if not instruments:
            raise ValueError(f"Instrument info not found for symbol: {symbol}")
        
        instrument = instruments[0]
        
        # Extract trading rules
        lot_size_filter = instrument.get("lotSizeFilter", {})
        price_filter = instrument.get("priceFilter", {})
        
        rules = {
            "minQty": float(lot_size_filter.get("minQty", "0.001")),
            "qtyStep": float(lot_size_filter.get("qtyStep", "0.001")),
            "maxQty": float(lot_size_filter.get("maxQty", "100")),
            "minNotional": float(lot_size_filter.get("minNotional", "5.0")),
            "tickSize": float(price_filter.get("tickSize", "0.01")),
            "pricePrecision": len(str(price_filter.get("tickSize", "0.01")).split(".")[-1]) if "." in str(price_filter.get("tickSize", "0.01")) else 0,
        }
        
        logger.info(
            f"Fetched symbol rules for {symbol}: minQty={rules['minQty']}, "
            f"qtyStep={rules['qtyStep']}, minNotional={rules['minNotional']}"
        )
        
        return rules

    async def get_earn_products(
        self,
        category: str | None = None,
        coin: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get available Earn products.

        Args:
            category: Optional product category filter (e.g., "FlexibleSaving", "FixedSaving", "OnChain")
            coin: Optional coin filter (e.g., "USDT", "BTC")

        Returns:
            List of product information dicts
        """
        params: dict[str, Any] = {}
        if category:
            params["category"] = category
        if coin:
            params["coin"] = coin

        response = await self._request("GET", "/v5/earn/products", params=params)
        return response.get("rows", [])

    async def get_earn_holdings(
        self,
        category: str | None = None,
        coin: str | None = None,
        product_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get current Earn holdings (investments).

        Args:
            category: Optional product category filter
            coin: Optional coin filter
            product_id: Optional product ID filter

        Returns:
            List of holding information dicts
        """
        params: dict[str, Any] = {}
        if category:
            params["category"] = category
        if coin:
            params["coin"] = coin
        if product_id:
            params["productId"] = product_id

        response = await self._request("GET", "/v5/earn/holding", params=params)
        return response.get("rows", [])
    
    def set_endpoint(self, rest_url: str) -> None:
        """Phase 16: Set dynamic REST endpoint (called by API Scouter).
        
        Args:
            rest_url: New REST API base URL (e.g., "https://api.bybit.com")
        """
        if rest_url and rest_url != self._base_url:
            logger.info(f"Switching REST endpoint to: {rest_url}")
            self._dynamic_endpoint = rest_url
            # Recreate HTTP client with new endpoint
            if self._client:
                # Client will use new endpoint on next request
                pass
        else:
            # Reset to default
            self._dynamic_endpoint = None

    async def place_earn_order(
        self,
        category: str,
        order_type: str,
        account_type: str,
        amount: str,
        coin: str,
        product_id: str,
    ) -> dict[str, Any]:
        """Place an Earn order (Stake or Redeem).

        Args:
            category: Product category (e.g., "FlexibleSaving", "OnChain")
            order_type: Order type ("Stake" for investing, "Redeem" for withdrawal)
            account_type: Account type ("FUND" or "UNIFIED")
            amount: Amount to stake/redeem (as string)
            coin: Coin symbol (e.g., "USDT", "BTC")
            product_id: Product ID

        Returns:
            Order response dict
        """
        if not settings.is_live:
            raise RuntimeError("Bybit HTTP client: Cannot place Earn orders in paper mode")

        data = {
            "category": category,
            "orderType": order_type,
            "accountType": account_type,
            "amount": amount,
            "coin": coin,
            "productId": product_id,
        }

        response = await self._request("POST", "/v5/earn/place-order", data=data)
        return response


# Type check: BybitHTTPClient should implement ExchangeClient
# This is verified at runtime, not compile time
def _verify_protocol() -> None:
    """Verify that BybitHTTPClient implements ExchangeClient protocol."""
    BybitHTTPClient()  # type: ignore
