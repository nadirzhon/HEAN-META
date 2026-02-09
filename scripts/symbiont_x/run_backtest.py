#!/usr/bin/env python3
"""
Run backtest on historical data with SYMBIONT X strategies

Usage:
    python run_backtest.py --data data/historical/BTCUSDT_60min_180days.csv --population-size 100
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hean.symbiont_x.genome_lab import create_random_genome
from hean.symbiont_x.backtesting import BacktestEngine, BacktestConfig


def load_historical_data(filepath: str):
    """Load historical data from CSV"""

    print(f"📥 Loading historical data from {filepath}...")

    try:
        df = pd.read_csv(filepath)

        # Convert to list of dicts
        data = []
        for _, row in df.iterrows():
            # Parse timestamp
            if 'timestamp' in row:
                timestamp = pd.to_datetime(row['timestamp'])
                timestamp_ms = int(timestamp.timestamp() * 1000)
            else:
                timestamp_ms = int(row.name)  # Use index if no timestamp column

            data.append({
                'timestamp': timestamp_ms,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })

        print(f"✅ Loaded {len(data)} candles")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return data

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def run_backtest(args):
    """Run backtest with given parameters"""

    # Load historical data
    historical_data = load_historical_data(args.data)

    if historical_data is None:
        return 1

    # Create backtest configuration
    config = BacktestConfig(
        initial_capital=args.capital,
        position_size_pct=args.position_size / 100.0,
        commission_pct=args.commission / 100.0,
        slippage_pct=args.slippage / 100.0
    )

    # Create backtest engine
    engine = BacktestEngine(config=config)

    # Create population
    print(f"\n🧬 Creating population of {args.population_size} strategies...")
    population = [
        create_random_genome(f"Strategy_{i}")
        for i in range(args.population_size)
    ]
    print(f"✅ Population created")

    # Run backtest
    print(f"\n🏃 Running backtest on {len(historical_data)} candles...")
    print("   This may take a few minutes...")

    results = engine.run_population_backtest(population, historical_data)

    # Sort by Sharpe ratio
    results.sort(key=lambda x: x.sharpe_ratio, reverse=True)

    # Print results
    print("\n" + "="*80)
    print("📊 BACKTEST RESULTS")
    print("="*80)

    print(f"\n🏆 Top {min(args.top_n, len(results))} Strategies:\n")
    print(f"{'Rank':<6} {'Name':<25} {'Return %':<12} {'Sharpe':<10} {'Win Rate':<10} {'Trades':<8} {'Max DD %':<10}")
    print("-" * 80)

    for i, result in enumerate(results[:args.top_n]):
        print(f"{i+1:<6} {result.genome_name:<25} "
              f"{result.return_pct:>10.2f}% "
              f"{result.sharpe_ratio:>9.2f} "
              f"{result.win_rate*100:>8.1f}% "
              f"{result.total_trades:>7} "
              f"{result.max_drawdown_pct:>9.2f}%")

    # Statistics
    print("\n" + "="*80)
    print("📈 POPULATION STATISTICS")
    print("="*80)

    profitable = sum(1 for r in results if r.return_pct > 0)
    avg_return = sum(r.return_pct for r in results) / len(results)
    avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results)
    avg_trades = sum(r.total_trades for r in results) / len(results)
    avg_win_rate = sum(r.win_rate for r in results) / len(results)

    print(f"\nTotal Strategies: {len(results)}")
    print(f"Profitable: {profitable} ({profitable/len(results)*100:.1f}%)")
    print(f"Average Return: {avg_return:.2f}%")
    print(f"Average Sharpe: {avg_sharpe:.2f}")
    print(f"Average Trades: {avg_trades:.1f}")
    print(f"Average Win Rate: {avg_win_rate*100:.1f}%")

    print(f"\nBest Strategy: {results[0].genome_name}")
    print(f"  Return: {results[0].return_pct:.2f}%")
    print(f"  Sharpe: {results[0].sharpe_ratio:.2f}")
    print(f"  Win Rate: {results[0].win_rate*100:.1f}%")
    print(f"  Max Drawdown: {results[0].max_drawdown_pct:.2f}%")

    # Save results
    if args.save_results:
        output_dir = os.path.dirname(args.data)
        results_file = os.path.join(output_dir, "backtest_results.csv")

        results_df = pd.DataFrame([{
            'rank': i+1,
            'name': r.genome_name,
            'return_pct': r.return_pct,
            'sharpe_ratio': r.sharpe_ratio,
            'win_rate': r.win_rate,
            'total_trades': r.total_trades,
            'winning_trades': r.winning_trades,
            'losing_trades': r.losing_trades,
            'max_drawdown_pct': r.max_drawdown_pct,
            'final_capital': r.final_capital
        } for i, r in enumerate(results)])

        results_df.to_csv(results_file, index=False)
        print(f"\n💾 Results saved to: {results_file}")

    return 0


def main():
    parser = argparse.ArgumentParser(description='Run backtest on historical data')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to historical data CSV file')
    parser.add_argument('--population-size', type=int, default=100,
                        help='Number of strategies to test (default: 100)')
    parser.add_argument('--capital', type=float, default=10000.0,
                        help='Initial capital (default: 10000)')
    parser.add_argument('--position-size', type=float, default=10.0,
                        help='Position size as percentage of capital (default: 10)')
    parser.add_argument('--commission', type=float, default=0.1,
                        help='Commission percentage (default: 0.1)')
    parser.add_argument('--slippage', type=float, default=0.05,
                        help='Slippage percentage (default: 0.05)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top strategies to display (default: 10)')
    parser.add_argument('--save-results', action='store_true',
                        help='Save results to CSV file')

    args = parser.parse_args()

    print("="*80)
    print("🧬 SYMBIONT X - BACKTEST ENGINE")
    print("="*80)
    print(f"Data file: {args.data}")
    print(f"Population size: {args.population_size}")
    print(f"Initial capital: ${args.capital:,.2f}")
    print(f"Position size: {args.position_size}%")
    print(f"Commission: {args.commission}%")
    print(f"Slippage: {args.slippage}%")
    print("="*80)

    return run_backtest(args)


if __name__ == "__main__":
    sys.exit(main())
