"""Vectorbt-based backtesting engine for super-fast strategy testing."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Backtesting configuration."""

    initial_capital: float = 10000.0
    commission: float = 0.0006  # 0.06% taker fee on Bybit
    slippage: float = 0.0002  # 0.02% slippage
    leverage: float = 1.0  # Max leverage to use

    # Data parameters
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    timeframe: str = "1h"  # 1m, 5m, 15m, 1h, 4h, 1d

    # Risk parameters
    max_position_size: float = 1.0  # Max % of capital per position
    stop_loss_pct: Optional[float] = None  # Stop loss %
    take_profit_pct: Optional[float] = None  # Take profit %

    # Performance
    use_numba: bool = True  # Use Numba JIT compilation for speed
    n_jobs: int = -1  # Parallel jobs (-1 = all CPUs)


@dataclass
class BacktestResult:
    """Backtesting result container."""

    portfolio: Any  # vectorbt Portfolio object
    returns: pd.Series
    equity_curve: pd.Series
    drawdown: pd.Series
    trades: pd.DataFrame

    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: timedelta

    # Execution stats
    execution_time: float  # seconds

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BacktestResult(\n"
            f"  Total Return: {self.total_return:.2%}\n"
            f"  Annual Return: {self.annualized_return:.2%}\n"
            f"  Sharpe: {self.sharpe_ratio:.2f}\n"
            f"  Sortino: {self.sortino_ratio:.2f}\n"
            f"  Calmar: {self.calmar_ratio:.2f}\n"
            f"  Max DD: {self.max_drawdown:.2%}\n"
            f"  Win Rate: {self.win_rate:.2%}\n"
            f"  Profit Factor: {self.profit_factor:.2f}\n"
            f"  Total Trades: {self.total_trades}\n"
            f"  Avg Trade Duration: {self.avg_trade_duration}\n"
            f"  Execution Time: {self.execution_time:.2f}s\n"
            f")"
        )


class VectorBTEngine:
    """
    Vectorized backtesting engine using VectorBT.

    Provides 100x+ speed improvements over event-driven backtesting
    through vectorization and Numba JIT compilation.

    Example:
        >>> engine = VectorBTEngine(config)
        >>> result = engine.backtest(data, signals)
        >>> print(result.sharpe_ratio)
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """Initialize the backtesting engine."""
        self.config = config or BacktestConfig()

        # Configure vectorbt settings
        if self.config.use_numba:
            vbt.settings.numba["check_func_type"] = False
            vbt.settings.numba["check_func_suffix"] = False

        logger.info(f"VectorBT Engine initialized with config: {self.config}")

    def backtest(
        self,
        data: pd.DataFrame,
        entries: pd.Series | pd.DataFrame,
        exits: pd.Series | pd.DataFrame,
        short_entries: Optional[pd.Series | pd.DataFrame] = None,
        short_exits: Optional[pd.Series | pd.DataFrame] = None,
        size: float | pd.Series | pd.DataFrame = 1.0,
        **kwargs: Any,
    ) -> BacktestResult:
        """
        Run backtest on given data and signals.

        Args:
            data: OHLCV data (must have 'open', 'high', 'low', 'close', 'volume')
            entries: Long entry signals (boolean)
            exits: Long exit signals (boolean)
            short_entries: Short entry signals (optional)
            short_exits: Short exit signals (optional)
            size: Position size (constant or dynamic)
            **kwargs: Additional portfolio arguments

        Returns:
            BacktestResult with all metrics and portfolio object
        """
        import time
        start_time = time.time()

        logger.info(f"Running backtest on {len(data)} bars...")

        # Validate data
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Build portfolio
        portfolio_kwargs = {
            'close': data['close'],
            'entries': entries,
            'exits': exits,
            'init_cash': self.config.initial_capital,
            'fees': self.config.commission,
            'slippage': self.config.slippage,
            'size': size,
            'size_type': 'targetpercent',  # Size as % of portfolio value
            'direction': 'both' if short_entries is not None else 'longonly',
            **kwargs,
        }

        # Add short signals if provided
        if short_entries is not None and short_exits is not None:
            portfolio_kwargs['short_entries'] = short_entries
            portfolio_kwargs['short_exits'] = short_exits

        # Add SL/TP if configured
        if self.config.stop_loss_pct is not None:
            portfolio_kwargs['sl_stop'] = self.config.stop_loss_pct
        if self.config.take_profit_pct is not None:
            portfolio_kwargs['tp_stop'] = self.config.take_profit_pct

        # Run vectorized backtest
        portfolio = vbt.Portfolio.from_signals(**portfolio_kwargs)

        execution_time = time.time() - start_time

        # Extract metrics
        result = self._build_result(portfolio, execution_time)

        logger.info(
            f"Backtest completed in {execution_time:.2f}s\n"
            f"Total Return: {result.total_return:.2%}, "
            f"Sharpe: {result.sharpe_ratio:.2f}, "
            f"Max DD: {result.max_drawdown:.2%}"
        )

        return result

    def backtest_custom_strategy(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        params: dict[str, Any],
        **kwargs: Any,
    ) -> BacktestResult:
        """
        Backtest a custom strategy function.

        Args:
            data: OHLCV data
            strategy_func: Function that takes (data, **params) and returns (entries, exits)
            params: Strategy parameters
            **kwargs: Additional portfolio arguments

        Returns:
            BacktestResult
        """
        # Generate signals
        entries, exits = strategy_func(data, **params)

        # Run backtest
        return self.backtest(data, entries, exits, **kwargs)

    def backtest_indicator(
        self,
        data: pd.DataFrame,
        indicator: str,
        entry_threshold: float,
        exit_threshold: float,
        **indicator_params: Any,
    ) -> BacktestResult:
        """
        Backtest using a VectorBT indicator.

        Args:
            data: OHLCV data
            indicator: Indicator name (e.g., 'RSI', 'MACD', 'BB')
            entry_threshold: Entry signal threshold
            exit_threshold: Exit signal threshold
            **indicator_params: Indicator parameters

        Returns:
            BacktestResult
        """
        # Get indicator
        if indicator.upper() == 'RSI':
            rsi = vbt.RSI.run(data['close'], **indicator_params)
            entries = rsi.rsi_below(entry_threshold)
            exits = rsi.rsi_above(exit_threshold)

        elif indicator.upper() == 'MACD':
            macd = vbt.MACD.run(data['close'], **indicator_params)
            entries = macd.macd_above(macd.signal)
            exits = macd.macd_below(macd.signal)

        elif indicator.upper() == 'BB':
            bb = vbt.BBANDS.run(data['close'], **indicator_params)
            entries = data['close'] < bb.lower
            exits = data['close'] > bb.upper

        else:
            raise ValueError(f"Unsupported indicator: {indicator}")

        return self.backtest(data, entries, exits)

    def _build_result(
        self,
        portfolio: Any,
        execution_time: float,
    ) -> BacktestResult:
        """Build BacktestResult from portfolio."""
        stats = portfolio.stats()

        # Extract key metrics with fallbacks
        total_return = stats.get('Total Return [%]', 0.0) / 100.0
        sharpe = stats.get('Sharpe Ratio', 0.0)
        sortino = stats.get('Sortino Ratio', 0.0)
        calmar = stats.get('Calmar Ratio', 0.0)
        max_dd = stats.get('Max Drawdown [%]', 0.0) / 100.0
        win_rate = stats.get('Win Rate [%]', 0.0) / 100.0
        profit_factor = stats.get('Profit Factor', 0.0)
        total_trades = int(stats.get('Total Trades', 0))

        # Calculate annualized return
        years = (portfolio.wrapper.index[-1] - portfolio.wrapper.index[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

        # Average trade duration
        trades_df = portfolio.trades.records_readable
        if len(trades_df) > 0 and 'Duration' in trades_df.columns:
            avg_duration = trades_df['Duration'].mean()
        else:
            avg_duration = timedelta(0)

        return BacktestResult(
            portfolio=portfolio,
            returns=portfolio.returns(),
            equity_curve=portfolio.value(),
            drawdown=portfolio.drawdowns.drawdown(),
            trades=trades_df if len(trades_df) > 0 else pd.DataFrame(),
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_duration=avg_duration,
            execution_time=execution_time,
        )

    def run_monte_carlo(
        self,
        result: BacktestResult,
        n_simulations: int = 1000,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run Monte Carlo simulation on backtest results.

        Args:
            result: BacktestResult from previous backtest
            n_simulations: Number of simulations
            **kwargs: Additional simulation parameters

        Returns:
            Dict with simulation results
        """
        logger.info(f"Running Monte Carlo with {n_simulations} simulations...")

        # Run Monte Carlo on trades
        mc = result.portfolio.trades.apply_monte_carlo(
            n_simulations,
            **kwargs,
        )

        return {
            'simulations': mc,
            'mean_total_return': mc['total_return'].mean(),
            'std_total_return': mc['total_return'].std(),
            'percentile_5': mc['total_return'].quantile(0.05),
            'percentile_95': mc['total_return'].quantile(0.95),
        }


def create_simple_ma_crossover_signals(
    data: pd.DataFrame,
    fast_period: int = 10,
    slow_period: int = 50,
) -> tuple[pd.Series, pd.Series]:
    """
    Create simple moving average crossover signals.

    Args:
        data: OHLCV data
        fast_period: Fast MA period
        slow_period: Slow MA period

    Returns:
        Tuple of (entries, exits)
    """
    fast_ma = vbt.MA.run(data['close'], fast_period, short_name='fast')
    slow_ma = vbt.MA.run(data['close'], slow_period, short_name='slow')

    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    return entries, exits


def create_rsi_mean_reversion_signals(
    data: pd.DataFrame,
    rsi_period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
) -> tuple[pd.Series, pd.Series]:
    """
    Create RSI mean reversion signals.

    Args:
        data: OHLCV data
        rsi_period: RSI period
        oversold: Oversold threshold
        overbought: Overbought threshold

    Returns:
        Tuple of (entries, exits)
    """
    rsi = vbt.RSI.run(data['close'], rsi_period)

    entries = rsi.rsi_below(oversold)
    exits = rsi.rsi_above(overbought)

    return entries, exits
