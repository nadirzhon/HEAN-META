#!/usr/bin/env python3
"""
Test connection to Bybit Testnet for SYMBIONT X

This script tests both WebSocket and REST API connections to ensure
everything is configured correctly before running the bot.

Usage:
    python test_bybit_connection.py
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Load environment variables
load_dotenv()


async def test_websocket_connection():
    """Test WebSocket connection to Bybit Testnet"""

    print("🔌 Testing WebSocket connection to Bybit Testnet...")

    try:
        import websockets
        import json

        url = "wss://stream-testnet.bybit.com/v5/public/linear"

        print(f"   Connecting to: {url}")

        async with websockets.connect(url) as ws:
            print("   ✅ WebSocket connected!")

            # Subscribe to ticker
            subscribe_msg = {
                "op": "subscribe",
                "args": ["tickers.BTCUSDT"]
            }
            await ws.send(json.dumps(subscribe_msg))
            print("   📡 Subscribed to BTCUSDT ticker")

            # Receive first few messages
            for i in range(3):
                message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(message)

                if data.get('topic') == 'tickers.BTCUSDT':
                    ticker = data['data']
                    price = ticker.get('lastPrice', 'N/A')
                    print(f"   ✅ Received ticker: BTC price = ${price}")
                elif data.get('op') == 'subscribe':
                    print(f"   ✅ Subscription confirmed")

            return True

    except ImportError:
        print("   ❌ websockets library not installed")
        print("      Please run: pip install websockets")
        return False
    except asyncio.TimeoutError:
        print("   ❌ Timeout waiting for WebSocket response")
        return False
    except Exception as e:
        print(f"   ❌ WebSocket connection failed: {e}")
        return False


def test_rest_api():
    """Test REST API connection to Bybit Testnet"""

    print("\n🔌 Testing REST API connection to Bybit Testnet...")

    try:
        from pybit.unified_trading import HTTP
    except ImportError:
        print("   ❌ pybit library not installed")
        print("      Please run: pip install pybit")
        return False

    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')

    if not api_key or not api_secret:
        print("   ❌ API credentials not found in environment")
        print("      Please set BYBIT_API_KEY and BYBIT_API_SECRET in .env file")
        print("\n   📝 To get API keys:")
        print("      1. Go to https://testnet.bybit.com")
        print("      2. Login and go to API Management")
        print("      3. Create new API key with Read + Trade permissions")
        print("      4. Add to .env file:")
        print("         BYBIT_API_KEY=your_key_here")
        print("         BYBIT_API_SECRET=your_secret_here")
        return False

    try:
        # Create client (testnet)
        client = HTTP(
            testnet=True,
            api_key=api_key,
            api_secret=api_secret
        )

        # Test 1: Get server time
        print("   📡 Testing server time...")
        result = client.get_server_time()

        if result['retCode'] == 0:
            server_time = result['result']['timeNano']
            print(f"   ✅ Server time: {server_time}")
        else:
            print(f"   ❌ Failed to get server time: {result}")
            return False

        # Test 2: Get wallet balance
        print("   📡 Testing wallet balance...")
        balance_result = client.get_wallet_balance(accountType="UNIFIED")

        if balance_result['retCode'] == 0:
            print("   ✅ Wallet balance retrieved successfully")

            # Show balances
            coins = balance_result['result']['list'][0]['coin']
            print("\n   💰 Account Balances:")
            for coin_data in coins:
                coin = coin_data['coin']
                balance = float(coin_data.get('walletBalance', 0))
                if balance > 0:
                    print(f"      {coin}: {balance}")

        else:
            print(f"   ❌ Failed to get wallet balance: {balance_result}")
            return False

        # Test 3: Get position info
        print("\n   📡 Testing position info...")
        position_result = client.get_positions(
            category="linear",
            symbol="BTCUSDT"
        )

        if position_result['retCode'] == 0:
            print("   ✅ Position info retrieved successfully")
            positions = position_result['result']['list']
            if positions:
                print(f"      Current positions: {len(positions)}")
            else:
                print("      No open positions")
        else:
            print(f"   ❌ Failed to get position info: {position_result}")

        print("\n   ✅ REST API connection successful!")
        return True

    except Exception as e:
        print(f"   ❌ REST API connection failed: {e}")
        return False


def check_dependencies():
    """Check if all required dependencies are installed"""

    print("📦 Checking dependencies...")

    dependencies = {
        'websockets': 'pip install websockets',
        'pybit': 'pip install pybit',
        'dotenv': 'pip install python-dotenv',
        'pydantic': 'pip install pydantic'
    }

    all_ok = True

    for module, install_cmd in dependencies.items():
        try:
            __import__(module if module != 'dotenv' else 'dotenv')
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} - Install with: {install_cmd}")
            all_ok = False

    return all_ok


async def main():
    """Main function"""

    print("="*60)
    print("🧪 BYBIT TESTNET CONNECTION TEST")
    print("="*60)

    # Check dependencies
    deps_ok = check_dependencies()

    if not deps_ok:
        print("\n❌ Please install missing dependencies first")
        return 1

    print()

    # Test WebSocket
    ws_ok = await test_websocket_connection()

    # Test REST API
    rest_ok = test_rest_api()

    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"WebSocket Connection: {'✅ PASS' if ws_ok else '❌ FAIL'}")
    print(f"REST API Connection:  {'✅ PASS' if rest_ok else '❌ FAIL'}")

    if ws_ok and rest_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYour system is ready to connect to Bybit Testnet.")
        print("You can now run SYMBIONT X in paper trading mode.")
        print("\n📝 Next steps:")
        print("   1. Review configuration in .env file")
        print("   2. Run paper trading: python examples/symbiont_x_example.py")
        print("   3. Monitor logs and performance")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nPlease fix the issues above before running SYMBIONT X.")

        if not rest_ok and not (os.getenv('BYBIT_API_KEY') and os.getenv('BYBIT_API_SECRET')):
            print("\n💡 Tip: You need to:")
            print("   1. Register at https://testnet.bybit.com")
            print("   2. Create API keys (Read + Trade permissions)")
            print("   3. Add them to .env file")

        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
