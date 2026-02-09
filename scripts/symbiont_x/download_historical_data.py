#!/usr/bin/env python3
"""
Download historical OHLCV data from Bybit for backtesting

Usage:
    python download_historical_data.py --symbol BTCUSDT --days 180
"""

import argparse
import pandas as pd
import time
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def download_klines_from_bybit(symbol: str, interval: str, days: int = 180):
    """
    Download historical kline data from Bybit

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Timeframe ('1', '5', '15', '60', '240', 'D')
        days: Number of days to download

    Returns:
        DataFrame with OHLCV data
    """

    try:
        from pybit.unified_trading import HTTP
    except ImportError:
        print("❌ Error: pybit library not installed")
        print("   Please run: pip install pybit")
        return None

    client = HTTP(testnet=False)  # Use mainnet for historical data

    # Calculate timestamps
    end_time = int(time.time() * 1000)
    start_time = int((time.time() - days * 24 * 3600) * 1000)

    print(f"📥 Downloading {symbol} {interval}min data for {days} days...")
    print(f"   Start: {pd.to_datetime(start_time, unit='ms')}")
    print(f"   End:   {pd.to_datetime(end_time, unit='ms')}")

    all_data = []
    current_start = start_time
    request_count = 0

    while current_start < end_time:
        try:
            result = client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                start=current_start,
                end=min(current_start + 200 * 60 * 1000, end_time),  # Max 200 candles
                limit=200
            )

            request_count += 1

            if result['retCode'] == 0 and result['result']['list']:
                data = result['result']['list']
                all_data.extend(data)

                # Update start time for next batch
                current_start = int(data[-1][0]) + 1

                print(f"  📊 Downloaded {len(data)} candles... Total: {len(all_data)} (Request #{request_count})")
                time.sleep(0.1)  # Rate limiting
            else:
                print(f"⚠️  Warning: No data returned for request #{request_count}")
                break

        except Exception as e:
            print(f"❌ Error on request #{request_count}: {e}")
            break

    if not all_data:
        print("❌ No data downloaded")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Sort by timestamp (ascending)
    df = df.sort_values('timestamp')

    print(f"✅ Successfully downloaded {len(df)} candles")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Download historical data from Bybit')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading pair (default: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='60',
                        help='Timeframe in minutes: 1, 5, 15, 60, 240, D (default: 60)')
    parser.add_argument('--days', type=int, default=180,
                        help='Number of days to download (default: 180)')
    parser.add_argument('--output-dir', type=str, default='data/historical',
                        help='Output directory (default: data/historical)')

    args = parser.parse_args()

    print("="*60)
    print("📥 BYBIT HISTORICAL DATA DOWNLOADER")
    print("="*60)
    print(f"Symbol: {args.symbol}")
    print(f"Interval: {args.interval} minutes")
    print(f"Days: {args.days}")
    print("="*60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Download data
    df = download_klines_from_bybit(args.symbol, args.interval, args.days)

    if df is not None:
        # Save to CSV
        filename = f"{args.symbol}_{args.interval}min_{args.days}days.csv"
        filepath = os.path.join(args.output_dir, filename)

        df.to_csv(filepath, index=False)
        print(f"\n💾 Saved to: {filepath}")

        # Show statistics
        print("\n" + "="*60)
        print("📊 DATA STATISTICS")
        print("="*60)
        print(f"Total candles: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"Average volume: {df['volume'].mean():.2f}")
        print("="*60)

        return 0
    else:
        print("\n❌ Failed to download data")
        return 1


if __name__ == "__main__":
    sys.exit(main())
