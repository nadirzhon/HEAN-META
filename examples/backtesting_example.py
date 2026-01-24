"""
Complete example of vectorbt + Optuna backtesting system.

This example demonstrates:
1. Simple backtesting
2. Parameter optimization with Optuna
3. Walk-forward analysis
4. Visualization

Requirements:
    pip install vectorbt optuna matplotlib seaborn
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from hean.backtesting import (
    VectorBTEngine,
    BacktestConfig,
    OptunaOptimizer,
    OptimizationConfig,
    WalkForwardAnalysis,
    WalkForwardConfig,
    BacktestVisualizer,
)
from hean.backtesting.vectorbt_engine import (
    create_simple_ma_crossover_signals,
    create_rsi_mean_reversion_signals,
)


def generate_sample_data(
    symbol: str = "BTCUSDT",
    days: int = 1095,  # 3 years
    start_date: datetime = None,
) -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing.

    In production, you would fetch real data from exchange.
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)

    # Generate dates (hourly bars)
    dates = pd.date_range(start=start_date, periods=days * 24, freq='1h')

    # Generate realistic price data with trend + noise
    np.random.seed(42)
    trend = np.linspace(30000, 45000, len(dates))
    noise = np.random.randn(len(dates)) * 500
    close_prices = trend + noise

    # Generate OHLC from close
    data = pd.DataFrame(
        {
            'open': close_prices + np.random.randn(len(dates)) * 100,
            'high': close_prices + np.abs(np.random.randn(len(dates)) * 200),
            'low': close_prices - np.abs(np.random.randn(len(dates)) * 200),
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, len(dates)),
        },
        index=dates,
    )

    # Ensure OHLC consistency
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data


def example_1_simple_backtest():
    """Example 1: Simple backtesting of MA crossover strategy."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Backtesting")
    print("=" * 80)

    # Generate data
    data = generate_sample_data(days=365)  # 1 year
    print(f"Generated {len(data)} bars of data")

    # Create backtesting engine
    config = BacktestConfig(
        initial_capital=10000,
        commission=0.0006,
        slippage=0.0002,
    )
    engine = VectorBTEngine(config)

    # Generate signals (MA crossover)
    entries, exits = create_simple_ma_crossover_signals(
        data, fast_period=10, slow_period=50
    )

    # Run backtest
    result = engine.backtest(data, entries, exits)

    # Print results
    print("\n" + result.__repr__())

    # Visualize
    visualizer = BacktestVisualizer()
    visualizer.plot_backtest_summary(result, title="MA Crossover Strategy")
    print("\nPlot saved to backtest_results/backtest_summary.png")


def example_2_parameter_optimization():
    """Example 2: Optimize strategy parameters with Optuna."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Parameter Optimization with Optuna")
    print("=" * 80)

    # Generate data
    data = generate_sample_data(days=730)  # 2 years
    print(f"Generated {len(data)} bars of data")

    # Create engine
    config = BacktestConfig(initial_capital=10000)
    engine = VectorBTEngine(config)

    # Create optimizer
    opt_config = OptimizationConfig(
        n_trials=50,  # Increase for better results
        n_jobs=-1,  # Use all CPUs
        show_progress_bar=True,
    )
    optimizer = OptunaOptimizer(engine, data, opt_config)

    # Define parameter space
    param_space = {
        'fast_period': ('int', 5, 30),
        'slow_period': ('int', 20, 100),
    }

    # Optimize
    print("\nOptimizing MA crossover parameters...")
    opt_result = optimizer.optimize_strategy(
        create_simple_ma_crossover_signals,
        param_space,
        objective_metric='sharpe_ratio',
    )

    print(f"\n{opt_result}")
    print(f"Best parameters: {opt_result.best_params}")

    # Visualize optimization
    visualizer = BacktestVisualizer()
    visualizer.plot_optimization_results(opt_result)
    print("\nOptimization plot saved to backtest_results/optimization_results.png")

    # Test with best parameters
    print("\nRunning backtest with optimized parameters...")
    entries, exits = create_simple_ma_crossover_signals(data, **opt_result.best_params)
    result = engine.backtest(data, entries, exits)
    print(result)


def example_3_multi_objective_optimization():
    """Example 3: Multi-objective optimization (maximize profit, minimize drawdown)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Multi-Objective Optimization")
    print("=" * 80)

    # Generate data
    data = generate_sample_data(days=730)
    print(f"Generated {len(data)} bars of data")

    # Create engine and optimizer
    engine = VectorBTEngine(BacktestConfig(initial_capital=10000))
    optimizer = OptunaOptimizer(
        engine,
        data,
        OptimizationConfig(n_trials=50, show_progress_bar=True),
    )

    # Define parameter space for RSI strategy
    param_space = {
        'rsi_period': ('int', 7, 21),
        'oversold': ('float', 20, 40),
        'overbought': ('float', 60, 80),
    }

    # Multi-objective: maximize Sharpe, minimize drawdown
    print("\nOptimizing RSI strategy (multi-objective)...")
    opt_result = optimizer.optimize_multi_objective(
        create_rsi_mean_reversion_signals,
        param_space,
        objectives=['sharpe_ratio', 'max_drawdown'],
    )

    print(f"\n{opt_result}")

    # Get Pareto-optimal solutions
    print("\nPareto-optimal solutions:")
    pareto_trials = opt_result.study.best_trials
    for i, trial in enumerate(pareto_trials[:5]):  # Show top 5
        print(f"  {i+1}. Sharpe: {trial.values[0]:.3f}, "
              f"Drawdown: {trial.values[1]:.2%}, "
              f"Params: {trial.params}")


def example_4_walk_forward_analysis():
    """Example 4: Walk-forward analysis to prevent overfitting."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Walk-Forward Analysis")
    print("=" * 80)

    # Generate 3 years of data
    data = generate_sample_data(days=1095)
    print(f"Generated {len(data)} bars of data")

    # Create engine
    engine = VectorBTEngine(BacktestConfig(initial_capital=10000))

    # Configure walk-forward analysis
    wf_config = WalkForwardConfig(
        train_window_months=6,  # 6 months training
        test_window_months=2,  # 2 months testing
        step_months=1,  # 1 month step
        anchored=False,  # Rolling window
        reoptimize_every_n_windows=1,  # Re-optimize every window
        optimization_config=OptimizationConfig(
            n_trials=30,  # Fewer trials per window
            show_progress_bar=False,
        ),
    )

    # Create walk-forward analyzer
    wfa = WalkForwardAnalysis(engine, wf_config)

    # Define parameter space
    param_space = {
        'fast_period': ('int', 5, 30),
        'slow_period': ('int', 20, 100),
    }

    # Run walk-forward analysis
    print("\nRunning walk-forward analysis...")
    print("This will take a few minutes...\n")

    wf_result = wfa.run(
        data,
        create_simple_ma_crossover_signals,
        param_space,
    )

    print(f"\n{wf_result}")

    # Analyze overfitting
    print("\nOverfitting Analysis:")
    print(f"  Average overfitting ratio: {wf_result.avg_overfitting_ratio:.3f}")
    print(f"  (Ideal: close to 1.0, < 0.5 indicates severe overfitting)")

    # Show parameter stability
    print("\nParameter Stability (lower = more stable):")
    for param, std in wf_result.param_stability.items():
        print(f"  {param}: {std:.2f}")

    # Visualize
    visualizer = BacktestVisualizer()
    visualizer.plot_walk_forward_results(wf_result)
    print("\nWalk-forward plot saved to backtest_results/walk_forward_results.png")

    # Get summary DataFrame
    summary_df = wfa.get_summary_df(wf_result)
    print("\nWindow Summary:")
    print(summary_df[['window_id', 'test_return', 'test_sharpe', 'overfitting_ratio']])


def example_5_grid_search_heatmap():
    """Example 5: Grid search with parameter heatmap."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Grid Search with Heatmap")
    print("=" * 80)

    # Generate data
    data = generate_sample_data(days=365)
    print(f"Generated {len(data)} bars of data")

    # Create engine and optimizer
    engine = VectorBTEngine(BacktestConfig(initial_capital=10000))
    optimizer = OptunaOptimizer(engine, data)

    # Define grid
    param_grid = {
        'fast_period': [5, 10, 15, 20, 25, 30],
        'slow_period': [30, 50, 70, 90, 110, 130],
    }

    # Run grid search
    print("\nRunning grid search...")
    results_df = optimizer.grid_search(
        create_simple_ma_crossover_signals,
        param_grid,
        objective_metric='sharpe_ratio',
    )

    print("\nTop 5 parameter combinations:")
    print(results_df.head()[['fast_period', 'slow_period', 'sharpe_ratio', 'total_return']])

    # Create heatmap
    visualizer = BacktestVisualizer()
    visualizer.plot_parameter_heatmap(
        results_df,
        'fast_period',
        'slow_period',
        metric='sharpe_ratio',
    )
    print("\nHeatmap saved to backtest_results/heatmap_fast_period_slow_period.png")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("VECTORBT + OPTUNA BACKTESTING SYSTEM")
    print("Complete Example Suite")
    print("=" * 80)

    # Run examples
    example_1_simple_backtest()
    example_2_parameter_optimization()
    example_3_multi_objective_optimization()
    example_4_walk_forward_analysis()
    example_5_grid_search_heatmap()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 80)
    print("\nCheck the 'backtest_results' directory for visualizations.")


if __name__ == '__main__':
    main()
