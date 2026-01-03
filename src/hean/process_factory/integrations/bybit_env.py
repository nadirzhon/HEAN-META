"""Bybit environment scanner (read-only safe) with retries/backoff."""

import asyncio
from datetime import datetime
from typing import Any

from hean.exchange.bybit.http import BybitHTTPClient
from hean.logging import get_logger
from hean.process_factory.schemas import BybitEnvironmentSnapshot

logger = get_logger(__name__)


class BybitEnvScanner:
    """Scanner for Bybit environment state (read-only) with retries/backoff."""

    def __init__(
        self,
        http_client: BybitHTTPClient | None = None,
        max_retries: int = 3,
        initial_backoff_sec: float = 1.0,
        max_backoff_sec: float = 10.0,
        timeout_sec: float = 30.0,
    ) -> None:
        """Initialize Bybit environment scanner.

        Args:
            http_client: Optional Bybit HTTP client (creates new if not provided)
            max_retries: Maximum number of retries (default 3)
            initial_backoff_sec: Initial backoff in seconds (default 1.0)
            max_backoff_sec: Maximum backoff in seconds (default 10.0)
            timeout_sec: Request timeout in seconds (default 30.0)
        """
        self._http_client = http_client
        self._client_owned = http_client is None
        self.max_retries = max_retries
        self.initial_backoff_sec = initial_backoff_sec
        self.max_backoff_sec = max_backoff_sec
        self.timeout_sec = timeout_sec

    async def _retry_with_backoff(
        self, func: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute function with exponential backoff retry.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries fail
        """
        backoff = self.initial_backoff_sec
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # Execute with timeout
                return await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout_sec)
            except asyncio.TimeoutError as e:
                last_exception = e
                if attempt < self.max_retries:
                    logger.warning(
                        f"Timeout on attempt {attempt + 1}/{self.max_retries + 1}, "
                        f"retrying in {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self.max_backoff_sec)
                else:
                    logger.error(f"Timeout after {self.max_retries + 1} attempts")
            except Exception as e:
                last_exception = e
                # Check if it's a rate limit error
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    if attempt < self.max_retries:
                        logger.warning(
                            f"Rate limit on attempt {attempt + 1}/{self.max_retries + 1}, "
                            f"retrying in {backoff:.1f}s"
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, self.max_backoff_sec)
                    else:
                        logger.error(f"Rate limit after {self.max_retries + 1} attempts")
                else:
                    # Non-retryable error, re-raise immediately
                    raise

        # All retries exhausted
        raise last_exception

    async def scan(self) -> BybitEnvironmentSnapshot:
        """Scan Bybit environment and create snapshot.

        Returns:
            Environment snapshot

        Note:
            If API is not available or not configured, returns snapshot with UNKNOWN flags.
        """
        snapshot = BybitEnvironmentSnapshot(
            timestamp=datetime.now(),
            balances={},
            positions=[],
            open_orders=[],
            funding_rates={},
            fees={},
            earn_availability={},
            campaign_availability={},
            source_flags={},
        )

        if not self._http_client:
            # Try to create client (will fail if not configured)
            try:
                self._http_client = BybitHTTPClient()
                await self._http_client.connect()
            except Exception as e:
                logger.warning(f"Could not connect to Bybit API: {e}")
                # Mark all as UNKNOWN
                for key in ["balances", "positions", "open_orders", "funding_rates", "fees"]:
                    snapshot.source_flags[key] = "UNKNOWN"
                snapshot.earn_availability = {"status": "UNKNOWN", "reason": "API not configured"}
                snapshot.campaign_availability = {"status": "UNKNOWN", "reason": "API not configured"}
                return snapshot

        try:
            # Fetch balances with retry
            try:
                account_info = await self._retry_with_backoff(
                    self._http_client.get_account_info
                )
                # Parse account info (format depends on Bybit API response)
                # This is a simplified parser - adjust based on actual API response format
                if isinstance(account_info, dict):
                    # Extract balances from account info
                    # Actual parsing depends on Bybit API response structure
                    balances = account_info.get("list", [{}])[0].get("coin", [])
                    if balances:
                        for coin in balances:
                            asset = coin.get("coin", "")
                            balance = float(coin.get("walletBalance", 0))
                            if asset and balance > 0:
                                snapshot.balances[asset] = balance
                snapshot.source_flags["balances"] = "API"
            except Exception as e:
                logger.warning(f"Could not fetch balances: {e}")
                snapshot.source_flags["balances"] = "UNKNOWN"

            # Fetch positions with retry
            try:
                positions = await self._retry_with_backoff(
                    self._http_client.get_positions
                )
                snapshot.positions = positions if isinstance(positions, list) else []
                snapshot.source_flags["positions"] = "API"
            except Exception as e:
                logger.warning(f"Could not fetch positions: {e}")
                snapshot.source_flags["positions"] = "UNKNOWN"

            # Open orders - not directly available in current HTTP client, mark as UNKNOWN
            snapshot.source_flags["open_orders"] = "UNKNOWN"

            # Fetch funding rates for major symbols with retry
            try:
                for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
                    try:
                        funding_rate_info = await self._retry_with_backoff(
                            self._http_client.get_funding_rate, symbol
                        )
                        if funding_rate_info and "fundingRate" in funding_rate_info:
                            snapshot.funding_rates[symbol] = float(funding_rate_info["fundingRate"])
                    except Exception as e:
                        logger.debug(f"Could not fetch funding rate for {symbol}: {e}")
                if snapshot.funding_rates:
                    snapshot.source_flags["funding_rates"] = "API"
                else:
                    snapshot.source_flags["funding_rates"] = "UNKNOWN"
            except Exception as e:
                logger.warning(f"Could not fetch funding rates: {e}")
                snapshot.source_flags["funding_rates"] = "UNKNOWN"

            # Fees - not directly available, mark as UNKNOWN
            snapshot.source_flags["fees"] = "UNKNOWN"

            # Earn availability - try to fetch via API
            try:
                earn_products = await self._retry_with_backoff(
                    self._http_client.get_earn_products
                )
                if earn_products:
                    # Group products by category
                    categories = set()
                    coins = set()
                    for product in earn_products:
                        if "category" in product:
                            categories.add(product["category"])
                        if "coin" in product:
                            coins.add(product["coin"])
                    
                    snapshot.earn_availability = {
                        "status": "AVAILABLE",
                        "reason": f"API accessible, {len(earn_products)} products found",
                        "categories": list(categories),
                        "coins": list(coins),
                        "product_count": len(earn_products),
                    }
                    snapshot.source_flags["earn_availability"] = "API"
                else:
                    snapshot.earn_availability = {
                        "status": "AVAILABLE",
                        "reason": "API accessible but no products found",
                        "product_count": 0,
                    }
                    snapshot.source_flags["earn_availability"] = "API"
            except Exception as e:
                error_msg = str(e).lower()
                # Check if it's a permissions/authentication error
                if "permission" in error_msg or "auth" in error_msg or "10003" in error_msg or "10004" in error_msg:
                    snapshot.earn_availability = {
                        "status": "RESTRICTED",
                        "reason": f"API accessible but permission denied: {e}",
                        "suggestion": "Ensure API key has Earn permissions enabled",
                    }
                    snapshot.source_flags["earn_availability"] = "API_PERMISSION_DENIED"
                else:
                    logger.warning(f"Could not fetch Earn products: {e}")
                    snapshot.earn_availability = {
                        "status": "UNKNOWN",
                        "reason": f"API error: {e}",
                        "suggestion": "Check API connectivity and permissions",
                    }
                    snapshot.source_flags["earn_availability"] = "UNKNOWN"

            # Campaign availability - not available via API, mark as UNKNOWN with HUMAN_TASK hint
            snapshot.campaign_availability = {
                "status": "UNKNOWN",
                "reason": "No API available, requires manual check via UI",
                "suggestion": "Use HUMAN_TASK step to check Campaigns section manually",
            }
            snapshot.source_flags["campaign_availability"] = "UNKNOWN"

        except Exception as e:
            logger.error(f"Error during environment scan: {e}", exc_info=True)

        finally:
            # Clean up if we created the client
            if self._client_owned and self._http_client:
                try:
                    await self._http_client.disconnect()
                except Exception:
                    pass
                self._http_client = None

        return snapshot

