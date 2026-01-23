"""
VectorBT Backtesting Engine

Ultra-fast vectorized backtesting using vectorbt.

Performance:
- 100-1000x faster than traditional loop-based backtesting
- Can test 1000+ parameter combinations in minutes
- Parallel execution support

Expected Usage:
- Parameter optimization (find best RSI periods, MA crossovers, etc.)
- Strategy validation
- Walk-forward testing

Author: HEAN Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    logger.warning(
        "Vectorbt not installed. Install with: pip install vectorbt"
    )


@dataclass
class BacktestConfig:
    """Backtesting configuration."""

    # Capital
    initial_capital: float = 10000
    fees: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage

    # Position sizing
    size_type: str = "percent"  # "percent" or "fixed"
    size: float = 1.0  # 100% of capital or fixed amount

    # Risk management
    stop_loss_pct: Optional[float] = None  # e.g., 0.02 for 2% SL
    take_profit_pct: Optional[float] = None  # e.g., 0.05 for 5% TP

    # Execution
    freq: str = "5T"  # Data frequency (5T = 5 minutes)
    direction: str = "both"  # "long", "short", "both"


@dataclass
class BacktestResult:
    """Backtesting results."""

    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    avg_trade_duration: str

    # Equity curve
    equity_curve: pd.Series
    drawdown_curve: pd.Series

    # Trades
    trades: pd.DataFrame

    # Raw portfolio object
    portfolio: Any

    def __str__(self) -> str:
        return f"""BacktestResult:
  Total Return: {self.total_return:.2%}
  Annual Return: {self.annual_return:.2%}
  Sharpe Ratio: {self.sharpe_ratio:.2f}
  Max Drawdown: {self.max_drawdown:.2%}
  Win Rate: {self.win_rate:.2%}
  Total Trades: {self.total_trades}
  Profit Factor: {self.profit_factor:.2f}
"""


class VectorBTBacktester:
    """
    Ultra-fast vectorized backtesting engine.

    Usage:
        # Basic backtest
        config = BacktestConfig(initial_capital=10000, fees=0.001)
        backtester = VectorBTBacktester(config)

        # Run backtest with signals
        result = backtester.backtest(
            prices=price_series,
            entries=entry_signals,  # Boolean series
            exits=exit_signals,     # Boolean series
        )

        # Print results
        print(result)

        # Optimize parameters
        results = backtester.optimize_rsi_strategy(
            prices,
            rsi_periods=[10, 14, 21, 28],
            oversold=[20, 25, 30],
            overbought=[70, 75, 80],
        )
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        """Initialize backtester."""
        if not VBT_AVAILABLE:
            raise ImportError(
                "Vectorbt required. Install with: pip install vectorbt"
            )

        self.config = config or BacktestConfig()
        logger.info("VectorBTBacktester initialized", config=self.config)

    def backtest(
        self,
        prices: pd.Series | pd.DataFrame,
        entries: pd.Series | pd.DataFrame,
        exits: Optional[pd.Series | pd.DataFrame] = None,
        short_entries: Optional[pd.Series | pd.DataFrame] = None,
        short_exits: Optional[pd.Series | pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run backtest with given signals.

        Args:
            prices: Price series
            entries: Long entry signals (True/False)
            exits: Long exit signals (optional, will hold until opposite signal)
            short_entries: Short entry signals (optional)
            short_exits: Short exit signals (optional)

        Returns:
            Backtest results
        """
        logger.info("Running backtest...")

        # Create portfolio
        portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=entries,
            exits=exits if exits is not None else ~entries,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=self.config.initial_capital,
            fees=self.config.fees,
            slippage=self.config.slippage,
            size=self.config.size,
            size_type=self.config.size_type,
            freq=self.config.freq,
        )

        # Extract metrics
        result = self._extract_metrics(portfolio)

        logger.info("Backtest complete", return_pct=f"{result.total_return:.2%}")
        return result

    def backtest_from_strategy(
        self,
        prices: pd.DataFrame,
        strategy_func: callable,
        **kwargs: Any,
    ) -> BacktestResult:
        """
        Backtest a custom strategy function.

        Args:
            prices: OHLCV DataFrame
            strategy_func: Function that returns (entries, exits) signals
            **kwargs: Arguments to pass to strategy function

        Returns:
            Backtest results
        """
        entries, exits = strategy_func(prices, **kwargs)
        return self.backtest(prices['close'], entries, exits)

    def optimize_parameters(
        self,
        prices: pd.Series,
        signal_func: callable,
        param_grid: Dict[str, List[Any]],
        metric: str = "sharpe_ratio",
    ) -> pd.DataFrame:
        """
        Optimize strategy parameters using grid search.

        Args:
            prices: Price series
            signal_func: Function(prices, **params) -> (entries, exits)
            param_grid: Parameter grid {"param": [val1, val2, ...]}
            metric: Metric to optimize ("sharpe_ratio", "total_return", etc.)

        Returns:
            DataFrame with all parameter combinations and metrics
        """
        logger.info("Starting parameter optimization...")

        # Generate parameter combinations
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        logger.info(f"Testing {len(combinations)} parameter combinations")

        results = []

        for combo in combinations:
            params = dict(zip(param_names, combo))

            try:
                # Generate signals
                entries, exits = signal_func(prices, **params)

                # Backtest
                result = self.backtest(prices, entries, exits)

                # Store results
                results.append({
                    **params,
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades,
                })

            except Exception as e:
                logger.warning(f"Failed for params {params}: {e}")
                continue

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Sort by metric
        results_df = results_df.sort_values(metric, ascending=False)

        logger.info(
            f"Optimization complete. Best {metric}: "
            f"{results_df[metric].iloc[0]:.3f}"
        )

        return results_df

    def optimize_rsi_strategy(
        self,
        prices: pd.Series,
        rsi_periods: List[int] = [10, 14, 21],
        oversold_levels: List[int] = [20, 25, 30],
        overbought_levels: List[int] = [70, 75, 80],
    ) -> pd.DataFrame:
        """
        Optimize RSI strategy parameters.

        Returns:
            DataFrame with best parameter combinations
        """
        def rsi_strategy(
            prices: pd.Series,
            period: int,
            oversold: int,
            overbought: int,
        ) -> Tuple[pd.Series, pd.Series]:
            """RSI mean reversion strategy."""
            # Calculate RSI
            rsi = vbt.RSI.run(prices, window=period).rsi

            # Signals
            entries = rsi < oversold
            exits = rsi > overbought

            return entries, exits

        param_grid = {
            'period': rsi_periods,
            'oversold': oversold_levels,
            'overbought': overbought_levels,
        }

        return self.optimize_parameters(prices, rsi_strategy, param_grid)

    def optimize_ma_crossover(
        self,
        prices: pd.Series,
        fast_periods: List[int] = [10, 20, 30],
        slow_periods: List[int] = [50, 100, 200],
    ) -> pd.DataFrame:
        """Optimize MA crossover strategy."""
        def ma_crossover(
            prices: pd.Series,
            fast: int,
            slow: int,
        ) -> Tuple[pd.Series, pd.Series]:
            """MA crossover strategy."""
            fast_ma = vbt.MA.run(prices, window=fast).ma
            slow_ma = vbt.MA.run(prices, window=slow).ma

            entries = vbt.crossover(fast_ma, slow_ma)
            exits = vbt.crossover(slow_ma, fast_ma)

            return entries, exits

        param_grid = {
            'fast': fast_periods,
            'slow': slow_periods,
        }

        return self.optimize_parameters(prices, ma_crossover, param_grid)

    def walk_forward_analysis(
        self,
        prices: pd.Series,
        signal_func: callable,
        param_grid: Dict[str, List[Any]],
        train_period: int = 90,  # days
        test_period: int = 30,   # days
        step: int = 30,          # days
    ) -> Dict[str, Any]:
        """
        Perform walk-forward analysis to prevent overfitting.

        Process:
        1. Train on first N days, optimize parameters
        2. Test on next M days with best parameters
        3. Slide window forward and repeat

        Args:
            prices: Price series
            signal_func: Strategy function
            param_grid: Parameters to optimize
            train_period: Training window (days)
            test_period: Test window (days)
            step: Step size (days)

        Returns:
            Walk-forward results
        """
        logger.info("Starting walk-forward analysis...")

        # Convert to daily if needed
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("Prices must have DatetimeIndex")

        results = []
        start_date = prices.index[0]
        end_date = prices.index[-1]

        current_date = start_date

        while current_date + pd.Timedelta(days=train_period + test_period) <= end_date:
            # Define windows
            train_start = current_date
            train_end = train_start + pd.Timedelta(days=train_period)
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=test_period)

            # Split data
            train_data = prices[train_start:train_end]
            test_data = prices[test_start:test_end]

            logger.info(
                f"Train: {train_start.date()} to {train_end.date()}, "
                f"Test: {test_start.date()} to {test_end.date()}"
            )

            # Optimize on train
            train_results = self.optimize_parameters(
                train_data, signal_func, param_grid
            )

            # Get best parameters
            best_params = train_results.iloc[0].to_dict()
            # Remove metrics, keep only parameters
            param_names = list(param_grid.keys())
            best_params = {k: v for k, v in best_params.items() if k in param_names}

            # Test on test data
            entries, exits = signal_func(test_data, **best_params)
            test_result = self.backtest(test_data, entries, exits)

            results.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'best_params': best_params,
                'test_return': test_result.total_return,
                'test_sharpe': test_result.sharpe_ratio,
                'test_drawdown': test_result.max_drawdown,
                'test_trades': test_result.total_trades,
            })

            # Move forward
            current_date += pd.Timedelta(days=step)

        logger.info(f"Walk-forward complete. {len(results)} windows tested")

        return {
            'results': pd.DataFrame(results),
            'avg_return': np.mean([r['test_return'] for r in results]),
            'avg_sharpe': np.mean([r['test_sharpe'] for r in results]),
            'avg_drawdown': np.mean([r['test_drawdown'] for r in results]),
        }

    def _extract_metrics(self, portfolio: Any) -> BacktestResult:
        """Extract metrics from vectorbt portfolio."""
        stats = portfolio.stats()

        return BacktestResult(
            total_return=stats.get('Total Return [%]', 0) / 100,
            annual_return=stats.get('Annualized Return [%]', 0) / 100,
            sharpe_ratio=stats.get('Sharpe Ratio', 0),
            sortino_ratio=stats.get('Sortino Ratio', 0),
            max_drawdown=stats.get('Max Drawdown [%]', 0) / 100,
            win_rate=stats.get('Win Rate [%]', 0) / 100,
            profit_factor=stats.get('Profit Factor', 0),
            total_trades=stats.get('Total Trades', 0),
            winning_trades=stats.get('Total Trades', 0) * stats.get('Win Rate [%]', 0) / 100,
            losing_trades=stats.get('Total Trades', 0) * (1 - stats.get('Win Rate [%]', 0) / 100),
            avg_win=stats.get('Avg Winning Trade [%]', 0) / 100,
            avg_loss=stats.get('Avg Losing Trade [%]', 0) / 100,
            avg_trade_duration=str(stats.get('Avg Winning Trade Duration', 'N/A')),
            equity_curve=portfolio.value(),
            drawdown_curve=portfolio.drawdowns.drawdown,
            trades=portfolio.trades.records_readable,
            portfolio=portfolio,
        )

    def plot_results(
        self,
        result: BacktestResult,
        show: bool = True,
    ) -> Any:
        """
        Plot backtest results.

        Args:
            result: Backtest result
            show: Whether to display plot

        Returns:
            Figure object
        """
        fig = result.portfolio.plot()
        if show:
            fig.show()
        return fig

    def create_heatmap(
        self,
        optimization_results: pd.DataFrame,
        x_param: str,
        y_param: str,
        metric: str = "sharpe_ratio",
    ) -> Any:
        """
        Create heatmap of optimization results.

        Args:
            optimization_results: Results from optimize_parameters
            x_param: Parameter for x-axis
            y_param: Parameter for y-axis
            metric: Metric to visualize

        Returns:
            Heatmap figure
        """
        # Pivot table
        pivot = optimization_results.pivot_table(
            values=metric,
            index=y_param,
            columns=x_param,
            aggfunc='mean'
        )

        # Create heatmap
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
        ))

        fig.update_layout(
            title=f'{metric} Heatmap',
            xaxis_title=x_param,
            yaxis_title=y_param,
        )

        return fig
