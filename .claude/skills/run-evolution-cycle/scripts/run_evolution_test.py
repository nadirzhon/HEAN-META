"""
Evolutionary Backtest Runner

This script demonstrates the full evolutionary backtesting loop:
1. Loads historical OHLCV data from DuckDB.
2. Creates a population of random strategy genomes.
3. Runs the backtester to evaluate the fitness of each genome.
4. Prints a summary of the results.
"""

import time
from datetime import datetime, timedelta

# Mock EventBus for DuckDBStore initialization
class MockEventBus:
    def subscribe(self, *args, **kwargs): pass
    def unsubscribe(self, *args, **kwargs): pass

# Adjust path to import from the root of the project
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from hean.storage.duckdb_store import DuckDBStore
from hean.symbiont_x.backtesting.backtest_engine import BacktestEngine
from hean.symbiont_x.genome_lab.genome_types import create_random_genome

def run_evolution_test():
    """Main function to run the evolutionary test."""
    print("🧬 Starting Evolutionary Backtest Runner...")

    # 1. Setup
    print("🔌 Initializing components...")
    mock_bus = MockEventBus()
    # Assuming default db path 'data/hean.duckdb'
    db_store = DuckDBStore(bus=mock_bus)
    backtest_engine = BacktestEngine()

    # 2. Data Loading
    print("📊 Loading historical data...")
    now = datetime.utcnow()
    # Load last 3 days of 5-minute candles for testing
    start_ts = (now - timedelta(days=3)).timestamp()
    
    historical_data = db_store.get_ohlcv_candles(
        symbol='BTCUSDT',
        timeframe='5m',
        start_ts=start_ts,
    )

    if not historical_data:
        print("❌ No historical data found. Please ensure the main application has been running to collect data.")
        print("💡 You might need to run 'make run' or 'docker-compose up' to start the data collector.")
        return

    print(f"📈 Loaded {len(historical_data)} candles for backtesting.")

    # 3. Population Creation
    print("👶 Creating initial population of 15 random genomes...")
    population_size = 15
    population = [create_random_genome(name=f"RandomStrategy_{i+1}") for i in range(population_size)]

    # 4. Backtesting
    print("🔬 Running backtests for the entire population... (this may take a moment)")
    start_time = time.time()
    results = backtest_engine.run_population_backtest(population, historical_data)
    end_time = time.time()
    print(f"✅ Backtesting complete in {end_time - start_time:.2f} seconds.")

    # 5. Results Display
    print("
--- 🏆 EVOLUTIONARY FITNESS RESULTS ---")
    
    # Sort by Sharpe Ratio (a good measure of risk-adjusted return)
    sorted_results = sorted(results, key=lambda r: r.sharpe_ratio, reverse=True)
    
    print("-" * 100)
    print(f"{'Rank':<5} | {'Genome Name':<20} | {'Return %':<12} | {'Win Rate %':<12} | {'Sharpe Ratio':<15} | {'Max Drawdown %':<18} | {'Trades':<7}")
    print("-" * 100)

    for i, result in enumerate(sorted_results):
        print(f"{i+1:<5} | {result.genome_name:<20} | {result.return_pct:<12.2f} | {result.win_rate * 100:<12.2f} | {result.sharpe_ratio:<15.2f} | {result.max_drawdown_pct:<18.2f} | {result.total_trades:<7}")

    print("-" * 100)
    print("
💡 Top-ranked genomes are 'fitter' and will be selected for the next generation's evolution.")

if __name__ == "__main__":
    run_evolution_test()
