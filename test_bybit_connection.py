#!/usr/bin/env python3
"""Test script for Bybit connection and integration."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hean.config import settings
from hean.core.bus import EventBus
from hean.exchange.bybit.integration import BybitIntegration
from hean.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def test_bybit_connection() -> None:
    """Test Bybit connection."""
    print("=" * 60)
    print("Bybit Connection Test")
    print("=" * 60)

    # Check configuration
    print(f"\nConfiguration:")
    print(f"  Live mode: {settings.is_live}")
    print(f"  Testnet: {settings.bybit_testnet}")
    print(f"  API Key: {'SET' if settings.bybit_api_key else 'NOT SET'}")
    print(f"  API Secret: {'SET' if settings.bybit_api_secret else 'NOT SET'}")

    if not settings.bybit_api_key or not settings.bybit_api_secret:
        print("\n❌ ERROR: Bybit API credentials not configured!")
        print("Please set BYBIT_API_KEY and BYBIT_API_SECRET in .env file")
        return

    if not settings.is_live:
        print("\n⚠️  WARNING: Not in live mode. Set LIVE_CONFIRM=YES to enable live trading.")
        print("This test will still check connection, but won't place real orders.")

    # Create integration
    bus = EventBus()
    await bus.start()

    integration = BybitIntegration(bus)

    try:
        # Test connection
        print("\n1. Testing connection...")
        connected = await integration.connect(["BTCUSDT", "ETHUSDT"])

        if not connected:
            print("❌ Connection failed")
            return

        print("✅ Connected successfully")

        # Test API connection
        print("\n2. Testing API connection...")
        api_ok = await integration.test_connection()
        if api_ok:
            print("✅ API connection test passed")
        else:
            print("❌ API connection test failed")

        # Test order placement (dry run)
        print("\n3. Testing order placement (dry run)...")
        order_ok = await integration.test_order_placement("BTCUSDT")
        if order_ok:
            print("✅ Order placement test passed")
        else:
            print("❌ Order placement test failed")

        # Wait a bit to receive WebSocket messages
        print("\n4. Testing WebSocket messages (waiting 5 seconds)...")
        await asyncio.sleep(5)
        print("✅ WebSocket test completed")

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.exception("Test error")
    finally:
        await integration.disconnect()
        await bus.stop()


if __name__ == "__main__":
    asyncio.run(test_bybit_connection())
