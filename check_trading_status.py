#!/usr/bin/env python3
"""Script to check trading status: orders, positions, and PnL."""

import asyncio
import sys

from hean.config import settings
from hean.exchange.bybit.http import BybitHTTPClient
from hean.logging import setup_logging

setup_logging()


async def check_trading_status() -> None:
    """Check trading status: orders, positions, PnL."""
    if not settings.is_live:
        print("⚠️  WARNING: Not in live mode (LIVE_CONFIRM != 'YES')")
        sys.exit(1)

    if not settings.bybit_api_key or not settings.bybit_api_secret:
        print("❌ ERROR: Bybit API credentials not configured")
        sys.exit(1)

    client = BybitHTTPClient()
    
    try:
        print("🔌 Connecting to Bybit API...")
        await client.connect()
        
        print("\n" + "=" * 70)
        print("📊 TRADING STATUS")
        print("=" * 70)
        
        # Get account balance
        print("\n💰 ACCOUNT BALANCE:")
        account_info = await client.get_account_info()
        accounts = account_info.get("list", [])
        if accounts:
            unified_account = accounts[0]
            coins = unified_account.get("coin", [])
            total_equity = unified_account.get("totalEquity", "0")
            
            print(f"  Total Equity: {total_equity} USDT")
            
            for coin in coins:
                asset = coin.get("coin", "")
                available = float(coin.get("availableToWithdraw", 0) or 0)
                wallet_balance = float(coin.get("walletBalance", 0) or 0)
                if wallet_balance > 0 or available > 0:
                    print(f"  {asset}: {wallet_balance:.8f} (available: {available:.8f})")
        
        # Get open positions
        print("\n📈 OPEN POSITIONS:")
        positions = await client.get_positions()
        
        if positions:
            total_unrealized_pnl = 0.0
            print(f"\n{'Symbol':<15} {'Side':<8} {'Size':<15} {'Entry Price':<15} {'Mark Price':<15} {'Unrealized PnL':<15}")
            print("-" * 100)
            for pos in positions:
                size = float(pos.get("size", 0) or 0)
                if size != 0:
                    symbol = pos.get("symbol", "")
                    side = pos.get("side", "")
                    entry_price = pos.get("avgPrice", "")
                    mark_price = pos.get("markPrice", "")
                    unrealized_pnl = float(pos.get("unrealisedPnl", 0) or 0)
                    total_unrealized_pnl += unrealized_pnl
                    print(f"{symbol:<15} {side:<8} {size:<15} {entry_price:<15} {mark_price:<15} {unrealized_pnl:>13.2f} USDT")
            print("-" * 100)
            print(f"Total Unrealized PnL: {total_unrealized_pnl:.2f} USDT")
        else:
            print("  ✅ No open positions")
        
        # Note: Bybit API v5 doesn't have a simple "open orders" endpoint
        # Orders are tracked via WebSocket or order history
        print("\n📝 NOTE:")
        print("  • Open orders are tracked via WebSocket")
        print("  • Check system logs for order activity")
        print("  • System must be running to place orders")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        await client.disconnect()
        print("\n✅ Disconnected")


if __name__ == "__main__":
    asyncio.run(check_trading_status())

