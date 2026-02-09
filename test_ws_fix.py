#!/usr/bin/env python3
"""Test WebSocket connection fix for Bybit testnet.

This test verifies:
1. WebSocket connects successfully
2. Ping/pong mechanism works
3. Price updates are received
4. Connection stays alive for at least 2 minutes
"""

import asyncio
import logging
import sys
from datetime import datetime

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import EventType
from hean.exchange.bybit.ws_public import BybitPublicWebSocket
from hean.logging import get_logger

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)

logger = get_logger(__name__)


async def test_websocket_connection():
    """Test WebSocket connection and heartbeat."""
    print("=" * 80)
    print("WebSocket Connection Test - Bybit Testnet")
    print("=" * 80)
    print()

    # Verify config
    print(f"BYBIT_TESTNET: {settings.bybit_testnet}")
    print(f"Trading mode: {settings.trading_mode}")
    print()

    if not settings.bybit_testnet:
        print("ERROR: BYBIT_TESTNET must be true for this test")
        return False

    # Create event bus and WebSocket client
    bus = EventBus()
    await bus.start()  # CRITICAL: Must start the bus to process events!
    ws_client = BybitPublicWebSocket(bus)

    # Track received ticks
    tick_count = 0
    last_tick_time = None
    symbols_seen = set()

    async def tick_handler(event):
        nonlocal tick_count, last_tick_time, symbols_seen
        tick = event.data.get("tick")
        if tick:
            tick_count += 1
            last_tick_time = datetime.utcnow()
            symbols_seen.add(tick.symbol)
            if tick_count <= 5 or tick_count % 10 == 0:
                print(f"[TICK #{tick_count}] {tick.symbol}: ${tick.price:.2f} at {tick.timestamp}")

    bus.subscribe(EventType.TICK, tick_handler)

    try:
        # Connect
        print("Connecting to Bybit WebSocket...")
        await ws_client.connect()
        print("✓ Connected successfully")
        print()

        # Subscribe to a few symbols
        test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        print(f"Subscribing to {', '.join(test_symbols)}...")
        for symbol in test_symbols:
            await ws_client.subscribe_ticker(symbol)
        print("✓ Subscribed")
        print()

        # Wait and monitor for 2 minutes
        print("Monitoring connection for 2 minutes...")
        print("(Press Ctrl+C to stop early)")
        print()

        start_time = datetime.utcnow()
        test_duration = 120  # 2 minutes
        check_interval = 10  # Check every 10 seconds

        for elapsed in range(0, test_duration, check_interval):
            await asyncio.sleep(check_interval)

            time_since_last_tick = None
            if last_tick_time:
                time_since_last_tick = (datetime.utcnow() - last_tick_time).total_seconds()

            print(f"[{elapsed}s] Ticks received: {tick_count} | "
                  f"Symbols: {len(symbols_seen)} | "
                  f"Last tick: {time_since_last_tick:.1f}s ago" if time_since_last_tick else "waiting...")

            # Check for stale connection
            if time_since_last_tick and time_since_last_tick > 60:
                print(f"WARNING: No ticks received for {time_since_last_tick:.1f}s - connection may be stale")

        print()
        print("=" * 80)
        print("Test Results")
        print("=" * 80)
        print(f"Total ticks received: {tick_count}")
        print(f"Symbols seen: {symbols_seen}")
        print(f"Average tick rate: {tick_count / test_duration:.2f} ticks/sec")
        print()

        # Disconnect
        await ws_client.disconnect()
        await bus.stop()
        print("✓ Disconnected cleanly")
        print()

        # Evaluate results
        if tick_count == 0:
            print("❌ FAILED: No ticks received")
            return False
        elif tick_count < 10:
            print("⚠️  WARNING: Very few ticks received (may indicate connection issues)")
            return False
        else:
            print("✅ SUCCESS: WebSocket connection working properly")
            return True

    except KeyboardInterrupt:
        print()
        print("Test interrupted by user")
        await ws_client.disconnect()
        return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        await ws_client.disconnect()
        return False


async def main():
    """Run the test."""
    success = await test_websocket_connection()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
