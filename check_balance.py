#!/usr/bin/env python3
"""Script to check Bybit account balance."""

import asyncio
import sys

from hean.config import settings
from hean.exchange.bybit.http import BybitHTTPClient
from hean.logging import setup_logging

setup_logging()


async def check_balance() -> None:
    """Check and display account balance."""
    if not settings.is_live:
        print("⚠️  WARNING: Not in live mode (LIVE_CONFIRM != 'YES')")
        print("   This script requires live mode to check real balance.")
        sys.exit(1)

    if not settings.bybit_api_key or not settings.bybit_api_secret:
        print("❌ ERROR: Bybit API credentials not configured")
        print("   Please set BYBIT_API_KEY and BYBIT_API_SECRET in .env file")
        sys.exit(1)

    client = BybitHTTPClient()

    try:
        print("🔌 Connecting to Bybit API...")
        await client.connect()

        print("\n📊 Fetching account balance (UNIFIED account)...")
        account_info = await client.get_account_info()

        # Parse account info (format depends on Bybit API response)
        print("\n" + "=" * 70)
        print("💰 ACCOUNT BALANCE (UNIFIED)")
        print("=" * 70)

        if isinstance(account_info, dict):
            # Extract balances from account info
            accounts = account_info.get("list", [])
            if not accounts:
                print("❌ No account data found")
                return

            unified_account = accounts[0]  # First account (UNIFIED)
            coins = unified_account.get("coin", [])

            if not coins:
                print("❌ No coin balances found")
                return

            print(f"\n{'Asset':<10} {'Available':<20} {'Total Balance':<20}")
            print("-" * 70)

            total_usdt = 0.0
            has_balances = False

            for coin in coins:
                asset = coin.get("coin", "")
                available = float(coin.get("availableToWithdraw", 0) or 0)
                wallet_balance = float(coin.get("walletBalance", 0) or 0)

                if wallet_balance > 0 or available > 0:
                    has_balances = True
                    print(f"{asset:<10} {available:>18.8f} {wallet_balance:>18.8f}")

                    # Convert to USDT equivalent (simplified - in real system would use prices)
                    if asset == "USDT":
                        total_usdt += wallet_balance

            if not has_balances:
                print("💰 No balances found (all assets are zero)")

            print("-" * 70)

            # Try to get equity if available
            equity = unified_account.get("totalEquity")
            if equity:
                print(f"\n📈 Total Equity: {equity} USDT")

            # Get positions
            print("\n📊 Checking open positions...")
            positions = await client.get_positions()

            if positions:
                print(
                    f"\n{'Symbol':<15} {'Side':<8} {'Size':<15} {'Entry Price':<15} {'Mark Price':<15}"
                )
                print("-" * 70)
                for pos in positions:
                    if float(pos.get("size", 0) or 0) != 0:
                        symbol = pos.get("symbol", "")
                        side = pos.get("side", "")
                        size = pos.get("size", "")
                        entry_price = pos.get("avgPrice", "")
                        mark_price = pos.get("markPrice", "")
                        print(
                            f"{symbol:<15} {side:<8} {size:<15} {entry_price:<15} {mark_price:<15}"
                        )
            else:
                print("✅ No open positions")

        else:
            print(f"⚠️  Unexpected response format: {type(account_info)}")
            print(f"   Response: {account_info}")

        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n❌ ERROR: Failed to get account balance: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        await client.disconnect()
        print("\n✅ Disconnected")


if __name__ == "__main__":
    asyncio.run(check_balance())
