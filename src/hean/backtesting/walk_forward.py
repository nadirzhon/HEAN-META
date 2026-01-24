"""Walk-forward analysis to prevent overfitting."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from hean.backtesting.optuna_optimizer import OptunaOptimizer, OptimizationConfig
from hean.backtesting.vectorbt_engine import BacktestConfig, BacktestResult, VectorBTEngine
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""

    # Window settings
    train_window_months: int = 6  # Training window size
    test_window_months: int = 2  # Testing (validation) window size
    step_months: int = 1  # How much to step forward each iteration

    # Optimization
    optimization_config: Optional[OptimizationConfig] = None
    objective_metric: str | list[str] = 'sharpe_ratio'

    # Anchored vs rolling
    anchored: bool = False  # If True, training window grows (anchored walk-forward)

    # Re-optimization frequency
    reoptimize_every_n_windows: int = 1  # How often to re-optimize (1 = every window)

    def __post_init__(self) -> None:
        """Set defaults."""
        if self.optimization_config is None:
            self.optimization_config = OptimizationConfig(
                n_trials=50,  # Fewer trials for WFA
                show_progress_bar=False,
            )


@dataclass
class WalkForwardWindow:
    """Single walk-forward window result."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Optimization results
    optimized_params: dict[str, Any]
    train_result: Optional[BacktestResult] = None

    # Out-of-sample results
    test_result: Optional[BacktestResult] = None

    # Metrics
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_return: float = 0.0
    test_return: float = 0.0
    overfitting_ratio: float = 0.0  # test_sharpe / train_sharpe

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WalkForwardWindow({self.window_id})\n"
            f"  Train: {self.train_start.date()} to {self.train_end.date()}\n"
            f"  Test:  {self.test_start.date()} to {self.test_end.date()}\n"
            f"  Train Sharpe: {self.train_sharpe:.3f}, Test Sharpe: {self.test_sharpe:.3f}\n"
            f"  Overfitting Ratio: {self.overfitting_ratio:.3f}\n"
            f"  Params: {self.optimized_params}"
        )


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis results."""

    windows: list[WalkForwardWindow]
    combined_test_result: BacktestResult  # Combined out-of-sample results

    # Aggregate metrics
    avg_train_sharpe: float
    avg_test_sharpe: float
    avg_overfitting_ratio: float
    total_test_return: float
    total_test_trades: int

    # Consistency metrics
    win_rate_across_windows: float  # % of windows with positive returns
    sharpe_stability: float  # Std of test Sharpes

    # Parameter stability
    param_stability: dict[str, float]  # Std of each parameter across windows

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WalkForwardResult(\n"
            f"  Windows: {len(self.windows)}\n"
            f"  Avg Train Sharpe: {self.avg_train_sharpe:.3f}\n"
            f"  Avg Test Sharpe: {self.avg_test_sharpe:.3f}\n"
            f"  Overfitting Ratio: {self.avg_overfitting_ratio:.3f}\n"
            f"  Total Test Return: {self.total_test_return:.2%}\n"
            f"  Win Rate (windows): {self.win_rate_across_windows:.2%}\n"
            f"  Sharpe Stability: {self.sharpe_stability:.3f}\n"
            f")"
        )


class WalkForwardAnalysis:
    """
    Walk-forward analysis engine.

    Prevents overfitting by:
    1. Training on historical window
    2. Testing on unseen future window
    3. Rolling forward and repeating

    Supports both rolling and anchored windows.

    Example:
        >>> wfa = WalkForwardAnalysis(engine, config)
        >>> result = wfa.run(data, strategy_func, param_space)
        >>> print(result.avg_overfitting_ratio)
    """

    def __init__(
        self,
        engine: VectorBTEngine,
        config: Optional[WalkForwardConfig] = None,
    ):
        """Initialize walk-forward analysis."""
        self.engine = engine
        self.config = config or WalkForwardConfig()

        logger.info(
            f"WalkForwardAnalysis initialized:\n"
            f"  Train window: {self.config.train_window_months} months\n"
            f"  Test window: {self.config.test_window_months} months\n"
            f"  Step: {self.config.step_months} months\n"
            f"  Anchored: {self.config.anchored}"
        )

    def run(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_space: dict[str, tuple],
        **backtest_kwargs: Any,
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis.

        Args:
            data: Full OHLCV dataset
            strategy_func: Strategy function(data, **params) -> (entries, exits)
            param_space: Parameter space for optimization
            **backtest_kwargs: Additional backtest arguments

        Returns:
            WalkForwardResult with all windows and aggregate metrics
        """
        logger.info("Starting walk-forward analysis...")

        # Generate windows
        windows = self._generate_windows(data.index)
        logger.info(f"Generated {len(windows)} walk-forward windows")

        results = []
        for i, (train_idx, test_idx) in enumerate(windows):
            logger.info(f"\n=== Window {i + 1}/{len(windows)} ===")

            # Split data
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            logger.info(
                f"Train: {train_data.index[0]} to {train_data.index[-1]} "
                f"({len(train_data)} bars)"
            )
            logger.info(
                f"Test:  {test_data.index[0]} to {test_data.index[-1]} "
                f"({len(test_data)} bars)"
            )

            # Optimize on training data (if needed)
            if i % self.config.reoptimize_every_n_windows == 0:
                optimizer = OptunaOptimizer(
                    self.engine,
                    train_data,
                    self.config.optimization_config,
                )

                opt_result = optimizer.optimize_strategy(
                    strategy_func,
                    param_space,
                    objective_metric=self.config.objective_metric,
                    **backtest_kwargs,
                )

                best_params = opt_result.best_params
                logger.info(f"Optimized params: {best_params}")
            else:
                # Reuse previous params
                best_params = results[-1].optimized_params
                logger.info(f"Reusing params: {best_params}")

            # Backtest on training data
            train_entries, train_exits = strategy_func(train_data, **best_params)
            train_result = self.engine.backtest(
                train_data, train_entries, train_exits, **backtest_kwargs
            )

            # Backtest on test data (out-of-sample)
            test_entries, test_exits = strategy_func(test_data, **best_params)
            test_result = self.engine.backtest(
                test_data, test_entries, test_exits, **backtest_kwargs
            )

            # Calculate overfitting ratio
            overfitting_ratio = (
                test_result.sharpe_ratio / train_result.sharpe_ratio
                if train_result.sharpe_ratio != 0
                else 0.0
            )

            # Store window result
            window_result = WalkForwardWindow(
                window_id=i,
                train_start=train_data.index[0],
                train_end=train_data.index[-1],
                test_start=test_data.index[0],
                test_end=test_data.index[-1],
                optimized_params=best_params,
                train_result=train_result,
                test_result=test_result,
                train_sharpe=train_result.sharpe_ratio,
                test_sharpe=test_result.sharpe_ratio,
                train_return=train_result.total_return,
                test_return=test_result.total_return,
                overfitting_ratio=overfitting_ratio,
            )

            results.append(window_result)

            logger.info(
                f"Train: Return={train_result.total_return:.2%}, "
                f"Sharpe={train_result.sharpe_ratio:.3f}"
            )
            logger.info(
                f"Test:  Return={test_result.total_return:.2%}, "
                f"Sharpe={test_result.sharpe_ratio:.3f}"
            )
            logger.info(f"Overfitting Ratio: {overfitting_ratio:.3f}")

        # Combine all test results
        combined_test_result = self._combine_test_results(results)

        # Calculate aggregate metrics
        avg_train_sharpe = np.mean([w.train_sharpe for w in results])
        avg_test_sharpe = np.mean([w.test_sharpe for w in results])
        avg_overfitting_ratio = np.mean([w.overfitting_ratio for w in results])

        # Consistency metrics
        positive_windows = sum(1 for w in results if w.test_return > 0)
        win_rate = positive_windows / len(results)
        sharpe_stability = np.std([w.test_sharpe for w in results])

        # Parameter stability
        param_stability = self._calculate_param_stability(results)

        wf_result = WalkForwardResult(
            windows=results,
            combined_test_result=combined_test_result,
            avg_train_sharpe=avg_train_sharpe,
            avg_test_sharpe=avg_test_sharpe,
            avg_overfitting_ratio=avg_overfitting_ratio,
            total_test_return=combined_test_result.total_return,
            total_test_trades=combined_test_result.total_trades,
            win_rate_across_windows=win_rate,
            sharpe_stability=sharpe_stability,
            param_stability=param_stability,
        )

        logger.info(f"\n=== Walk-Forward Analysis Complete ===")
        logger.info(str(wf_result))

        return wf_result

    def _generate_windows(
        self,
        index: pd.DatetimeIndex,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test windows."""
        windows = []

        # Convert months to approximate days
        train_days = self.config.train_window_months * 30
        test_days = self.config.test_window_months * 30
        step_days = self.config.step_months * 30

        start_date = index[0]
        anchor_start = start_date if self.config.anchored else None

        current_pos = 0

        while True:
            # Determine train window
            if self.config.anchored and anchor_start is not None:
                train_start_date = anchor_start
            else:
                train_start_date = start_date + pd.Timedelta(days=current_pos)

            train_end_date = train_start_date + pd.Timedelta(days=train_days)
            test_end_date = train_end_date + pd.Timedelta(days=test_days)

            # Check if we have enough data
            if test_end_date > index[-1]:
                break

            # Get indices
            train_mask = (index >= train_start_date) & (index < train_end_date)
            test_mask = (index >= train_end_date) & (index < test_end_date)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) > 0 and len(test_idx) > 0:
                windows.append((train_idx, test_idx))

            # Step forward
            current_pos += step_days

        return windows

    def _combine_test_results(
        self,
        windows: list[WalkForwardWindow],
    ) -> BacktestResult:
        """Combine all test results into single result."""
        # Concatenate all test equity curves
        equity_curves = []
        returns_list = []
        trades_list = []

        for window in windows:
            if window.test_result is not None:
                equity_curves.append(window.test_result.equity_curve)
                returns_list.append(window.test_result.returns)
                trades_list.append(window.test_result.trades)

        # Combine
        combined_equity = pd.concat(equity_curves)
        combined_returns = pd.concat(returns_list)
        combined_trades = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()

        # Calculate combined metrics
        total_return = (combined_equity.iloc[-1] / combined_equity.iloc[0]) - 1

        # Use first window's portfolio config for combined result
        first_result = windows[0].test_result

        # Create combined result (reuse structure from first window)
        from hean.backtesting.vectorbt_engine import BacktestResult
        from datetime import timedelta

        combined_result = BacktestResult(
            portfolio=None,  # No single portfolio for combined
            returns=combined_returns,
            equity_curve=combined_equity,
            drawdown=pd.Series(),  # Calculate if needed
            trades=combined_trades,
            total_return=total_return,
            annualized_return=0.0,  # Calculate if needed
            sharpe_ratio=np.mean([w.test_sharpe for w in windows]),
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            win_rate=np.mean([w.test_result.win_rate for w in windows if w.test_result]),
            profit_factor=0.0,
            total_trades=len(combined_trades),
            avg_trade_duration=timedelta(0),
            execution_time=sum(w.test_result.execution_time for w in windows if w.test_result),
        )

        return combined_result

    def _calculate_param_stability(
        self,
        windows: list[WalkForwardWindow],
    ) -> dict[str, float]:
        """Calculate parameter stability across windows."""
        param_values = {}

        for window in windows:
            for param_name, param_value in window.optimized_params.items():
                if param_name not in param_values:
                    param_values[param_name] = []
                param_values[param_name].append(param_value)

        stability = {}
        for param_name, values in param_values.items():
            # Convert to numeric if possible
            try:
                numeric_values = [float(v) for v in values]
                stability[param_name] = np.std(numeric_values)
            except (ValueError, TypeError):
                # Categorical parameter - count unique values
                stability[param_name] = len(set(values)) / len(values)

        return stability

    def get_summary_df(self, result: WalkForwardResult) -> pd.DataFrame:
        """Get summary DataFrame of all windows."""
        data = []
        for window in result.windows:
            data.append(
                {
                    'window_id': window.window_id,
                    'train_start': window.train_start,
                    'train_end': window.train_end,
                    'test_start': window.test_start,
                    'test_end': window.test_end,
                    'train_return': window.train_return,
                    'test_return': window.test_return,
                    'train_sharpe': window.train_sharpe,
                    'test_sharpe': window.test_sharpe,
                    'overfitting_ratio': window.overfitting_ratio,
                    **window.optimized_params,
                }
            )

        return pd.DataFrame(data)
