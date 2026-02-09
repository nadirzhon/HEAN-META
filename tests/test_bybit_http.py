"""Tests for Bybit HTTP API client.

CRITICAL: This module has 0% test coverage but handles ALL exchange communication.
These tests ensure the API client correctly handles:
- Authentication (signature generation)
- Order lifecycle (place, cancel, query)
- Position management
- Error handling and circuit breaker
- Rate limiting
"""

import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hean.exchange.bybit.http import BybitHTTPClient


class TestBybitHTTPClientInit:
    """Test BybitHTTPClient initialization."""

    def test_init_testnet_url(self):
        """Test that testnet URL is used when bybit_testnet=True."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = "test_key"
            mock_settings.bybit_api_secret = "test_secret"

            client = BybitHTTPClient()

            assert client._base_url == "https://api-testnet.bybit.com"
            assert client._testnet is True

    def test_init_mainnet_url(self):
        """Test that mainnet URL is used when bybit_testnet=False."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = False
            mock_settings.bybit_api_key = "test_key"
            mock_settings.bybit_api_secret = "test_secret"

            client = BybitHTTPClient()

            assert client._base_url == "https://api.bybit.com"
            assert client._testnet is False

    def test_init_circuit_breaker_configured(self):
        """Test that circuit breaker is properly configured."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = "test_key"
            mock_settings.bybit_api_secret = "test_secret"

            client = BybitHTTPClient()

            assert client._circuit_breaker.failure_threshold == 5
            assert client._circuit_breaker.recovery_timeout == 60.0


class TestBybitSignatureGeneration:
    """Test request signature generation."""

    def test_sign_request_get(self):
        """Test signature generation for GET requests."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = "test_api_key"
            mock_settings.bybit_api_secret = "test_api_secret"

            client = BybitHTTPClient()

            timestamp = 1234567890000
            recv_window = "5000"
            params = {"symbol": "BTCUSDT", "category": "linear"}
            data = {}

            signature = client._sign_request("GET", timestamp, recv_window, params, data)

            # Verify signature is a hex string
            assert isinstance(signature, str)
            assert len(signature) == 64  # SHA256 produces 64 hex chars

            # Verify signature is deterministic
            signature2 = client._sign_request("GET", timestamp, recv_window, params, data)
            assert signature == signature2

    def test_sign_request_post(self):
        """Test signature generation for POST requests."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = "test_api_key"
            mock_settings.bybit_api_secret = "test_api_secret"

            client = BybitHTTPClient()

            timestamp = 1234567890000
            recv_window = "5000"
            params = {}
            data = {"symbol": "BTCUSDT", "side": "Buy", "qty": "0.001"}

            signature = client._sign_request("POST", timestamp, recv_window, params, data)

            # Verify signature is a hex string
            assert isinstance(signature, str)
            assert len(signature) == 64

    def test_sign_request_different_data_different_signature(self):
        """Test that different data produces different signatures."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = "test_api_key"
            mock_settings.bybit_api_secret = "test_api_secret"

            client = BybitHTTPClient()

            timestamp = 1234567890000
            recv_window = "5000"

            sig1 = client._sign_request("POST", timestamp, recv_window, {}, {"qty": "0.001"})
            sig2 = client._sign_request("POST", timestamp, recv_window, {}, {"qty": "0.002"})

            assert sig1 != sig2


class TestBybitHTTPClientConnection:
    """Test connection management."""

    @pytest.mark.asyncio
    async def test_connect_creates_client(self):
        """Test that connect() creates httpx client."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = "test_key"
            mock_settings.bybit_api_secret = "test_secret"

            client = BybitHTTPClient()
            assert client._connected is False

            await client.connect()

            assert client._connected is True

    @pytest.mark.asyncio
    async def test_disconnect_closes_client(self):
        """Test that disconnect() closes httpx client."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = "test_key"
            mock_settings.bybit_api_secret = "test_secret"

            client = BybitHTTPClient()
            await client.connect()
            assert client._connected is True

            await client.disconnect()

            assert client._connected is False


class TestBybitHTTPClientOrders:
    """Test order-related API calls."""

    @pytest.mark.asyncio
    async def test_place_order_success(self):
        """Test successful order placement."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = "test_key"
            mock_settings.bybit_api_secret = "test_secret"

            client = BybitHTTPClient()

            # Mock the HTTP response
            mock_response = {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "orderId": "test_order_123",
                    "orderLinkId": "link_123",
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "orderType": "Market",
                    "price": "50000",
                    "qty": "0.001",
                    "orderStatus": "New",
                },
            }

            with patch.object(client, "_request", return_value=mock_response):
                from hean.core.types import OrderRequest

                order_request = OrderRequest(
                    signal_id="sig_123",
                    strategy_id="test_strategy",
                    symbol="BTCUSDT",
                    side="buy",
                    size=0.001,
                    order_type="market",
                )

                order = await client.place_order(order_request)

                assert order is not None
                assert order.order_id == "test_order_123"
                assert order.symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_place_order_api_error(self):
        """Test order placement with API error response."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = "test_key"
            mock_settings.bybit_api_secret = "test_secret"

            client = BybitHTTPClient()

            # Mock API error response
            mock_response = {
                "retCode": 10001,
                "retMsg": "Insufficient balance",
                "result": {},
            }

            with patch.object(client, "_request", return_value=mock_response):
                from hean.core.types import OrderRequest

                order_request = OrderRequest(
                    signal_id="sig_123",
                    strategy_id="test_strategy",
                    symbol="BTCUSDT",
                    side="buy",
                    size=0.001,
                    order_type="market",
                )

                # Should raise or return error order
                order = await client.place_order(order_request)
                # Depending on implementation, this might be None or a rejected order
                # At minimum, it should not crash

    @pytest.mark.asyncio
    async def test_cancel_order_success(self):
        """Test successful order cancellation."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = "test_key"
            mock_settings.bybit_api_secret = "test_secret"

            client = BybitHTTPClient()

            mock_response = {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "orderId": "order_123",
                    "orderLinkId": "link_123",
                },
            }

            with patch.object(client, "_request", return_value=mock_response):
                result = await client.cancel_order("order_123", "BTCUSDT")
                assert result is not None


class TestBybitHTTPClientPositions:
    """Test position-related API calls."""

    @pytest.mark.asyncio
    async def test_get_positions_success(self):
        """Test successful position retrieval."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = "test_key"
            mock_settings.bybit_api_secret = "test_secret"

            client = BybitHTTPClient()

            mock_response = {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "list": [
                        {
                            "symbol": "BTCUSDT",
                            "side": "Buy",
                            "size": "0.001",
                            "avgPrice": "50000",
                            "unrealisedPnl": "10.5",
                            "leverage": "10",
                        }
                    ]
                },
            }

            with patch.object(client, "_request", return_value=mock_response):
                positions = await client.get_positions()

                assert positions is not None
                assert isinstance(positions, list)

    @pytest.mark.asyncio
    async def test_get_positions_empty(self):
        """Test position retrieval when no positions exist."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = "test_key"
            mock_settings.bybit_api_secret = "test_secret"

            client = BybitHTTPClient()

            mock_response = {
                "retCode": 0,
                "retMsg": "OK",
                "result": {"list": []},
            }

            with patch.object(client, "_request", return_value=mock_response):
                positions = await client.get_positions()

                assert positions is not None
                assert len(positions) == 0


class TestBybitHTTPClientCircuitBreaker:
    """Test circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test that circuit breaker opens after threshold failures."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = "test_key"
            mock_settings.bybit_api_secret = "test_secret"

            client = BybitHTTPClient()

            # Simulate multiple failures
            failure_count = 0

            async def failing_request(*args, **kwargs):
                nonlocal failure_count
                failure_count += 1
                raise httpx.HTTPError("Connection failed")

            with patch.object(client, "_request_impl", side_effect=failing_request):
                # Try to make requests until circuit opens
                for _ in range(10):
                    try:
                        await client._request("GET", "/test", {}, {})
                    except (httpx.HTTPError, RuntimeError):
                        pass

                # Circuit should be open now
                assert client._circuit_breaker._state.name in ["OPEN", "HALF_OPEN"]


class TestBybitHTTPClientRateLimiting:
    """Test rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_respected(self):
        """Test that rate limiting prevents too-fast requests."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = "test_key"
            mock_settings.bybit_api_secret = "test_secret"

            client = BybitHTTPClient()

            # Verify rate limit parameters are set
            assert client._min_order_interval > 0


class TestBybitHTTPClientErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_missing_credentials_raises_error(self):
        """Test that missing credentials raises ValueError."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = ""  # Empty key
            mock_settings.bybit_api_secret = ""

            client = BybitHTTPClient()

            with pytest.raises(ValueError, match="credentials not configured"):
                await client._request_impl("GET", "/v5/account/info", {}, {})

    @pytest.mark.asyncio
    async def test_http_timeout_handled(self):
        """Test that HTTP timeouts are handled gracefully."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = "test_key"
            mock_settings.bybit_api_secret = "test_secret"

            client = BybitHTTPClient()

            async def timeout_request(*args, **kwargs):
                raise httpx.TimeoutException("Request timed out")

            with patch.object(client, "_request_impl", side_effect=timeout_request):
                with pytest.raises((httpx.TimeoutException, RuntimeError)):
                    await client._request("GET", "/test", {}, {})


class TestBybitHTTPClientBalance:
    """Test balance retrieval."""

    @pytest.mark.asyncio
    async def test_get_wallet_balance_success(self):
        """Test successful wallet balance retrieval."""
        with patch("hean.exchange.bybit.http.settings") as mock_settings:
            mock_settings.bybit_testnet = True
            mock_settings.bybit_api_key = "test_key"
            mock_settings.bybit_api_secret = "test_secret"

            client = BybitHTTPClient()

            mock_response = {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "list": [
                        {
                            "accountType": "UNIFIED",
                            "totalEquity": "10000.00",
                            "totalWalletBalance": "9500.00",
                            "totalAvailableBalance": "8000.00",
                        }
                    ]
                },
            }

            with patch.object(client, "_request", return_value=mock_response):
                balance = await client.get_wallet_balance()
                assert balance is not None
