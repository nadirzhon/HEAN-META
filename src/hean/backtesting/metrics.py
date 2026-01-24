"""Advanced performance metrics for backtesting."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestMetrics:
    """Container for comprehensive backtest performance metrics."""

    # Return metrics
    total_return: float
    annualized_return: float
    cagr: float  # Compound Annual Growth Rate
    monthly_returns: pd.Series

    # Risk metrics
    volatility: float  # Annualized volatility
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    avg_drawdown: float
    avg_drawdown_duration: int

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: Optional[float] = None  # vs benchmark

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    largest_win: float
    largest_loss: float
    profit_factor: float  # Total wins / Total losses

    # Trade duration
    avg_trade_duration_hours: float
    avg_winning_trade_duration_hours: float
    avg_losing_trade_duration_hours: float

    # Exposure
    exposure_time: float  # % of time in market
    avg_position_size: float

    # Advanced metrics
    expectancy: float  # Expected value per trade
    payoff_ratio: float  # Avg win / Avg loss
    recovery_factor: float  # Net profit / Max drawdown
    risk_of_ruin: float  # Probability of losing all capital
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR (95%)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BacktestMetrics(\n"
            f"  === RETURNS ===\n"
            f"  Total Return: {self.total_return:.2%}\n"
            f"  Annual Return: {self.annualized_return:.2%}\n"
            f"  CAGR: {self.cagr:.2%}\n"
            f"\n"
            f"  === RISK ===\n"
            f"  Volatility: {self.volatility:.2%}\n"
            f"  Max Drawdown: {self.max_drawdown:.2%}\n"
            f"  Avg Drawdown: {self.avg_drawdown:.2%}\n"
            f"  VaR (95%): {self.var_95:.2%}\n"
            f"  CVaR (95%): {self.cvar_95:.2%}\n"
            f"\n"
            f"  === RISK-ADJUSTED ===\n"
            f"  Sharpe: {self.sharpe_ratio:.3f}\n"
            f"  Sortino: {self.sortino_ratio:.3f}\n"
            f"  Calmar: {self.calmar_ratio:.3f}\n"
            f"  Omega: {self.omega_ratio:.3f}\n"
            f"\n"
            f"  === TRADES ===\n"
            f"  Total Trades: {self.total_trades}\n"
            f"  Win Rate: {self.win_rate:.2%}\n"
            f"  Profit Factor: {self.profit_factor:.2f}\n"
            f"  Avg Win: {self.avg_win:.2%}\n"
            f"  Avg Loss: {self.avg_loss:.2%}\n"
            f"  Payoff Ratio: {self.payoff_ratio:.2f}\n"
            f"  Expectancy: {self.expectancy:.2%}\n"
            f"\n"
            f"  === OTHER ===\n"
            f"  Exposure: {self.exposure_time:.2%}\n"
            f"  Recovery Factor: {self.recovery_factor:.2f}\n"
            f"  Risk of Ruin: {self.risk_of_ruin:.4%}\n"
            f")"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'cagr': self.cagr,
            'volatility': self.volatility,
            'downside_deviation': self.downside_deviation,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'avg_drawdown': self.avg_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'omega_ratio': self.omega_ratio,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'payoff_ratio': self.payoff_ratio,
            'recovery_factor': self.recovery_factor,
            'risk_of_ruin': self.risk_of_ruin,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
        }


def calculate_metrics(
    returns: pd.Series,
    trades: pd.DataFrame,
    equity_curve: pd.Series,
    initial_capital: float = 10000.0,
    risk_free_rate: float = 0.02,
    benchmark_returns: Optional[pd.Series] = None,
) -> BacktestMetrics:
    """
    Calculate comprehensive backtest metrics.

    Args:
        returns: Series of returns (% or decimal)
        trades: DataFrame of trades with columns: PnL, Duration, etc.
        equity_curve: Series of portfolio value over time
        initial_capital: Initial capital
        risk_free_rate: Annual risk-free rate
        benchmark_returns: Optional benchmark returns for information ratio

    Returns:
        BacktestMetrics object
    """
    # Ensure returns are in decimal form
    if returns.abs().max() > 1.0:
        returns = returns / 100.0

    # Time-based calculations
    periods_per_year = _infer_periods_per_year(returns.index)
    total_days = (returns.index[-1] - returns.index[0]).days
    total_years = total_days / 365.25

    # === RETURNS ===
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    cagr = (1 + total_return) ** (1 / total_years) - 1 if total_years > 0 else 0.0
    annualized_return = returns.mean() * periods_per_year

    # Monthly returns
    monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    # === RISK ===
    volatility = returns.std() * np.sqrt(periods_per_year)
    downside_deviation = _downside_deviation(returns, risk_free_rate / periods_per_year)
    downside_deviation_annual = downside_deviation * np.sqrt(periods_per_year)

    # Drawdown analysis
    dd_info = _calculate_drawdown_info(equity_curve)

    # === RISK-ADJUSTED ===
    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe_ratio = (
        np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
        if returns.std() > 0
        else 0.0
    )
    sortino_ratio = (
        np.sqrt(periods_per_year) * excess_returns.mean() / downside_deviation
        if downside_deviation > 0
        else 0.0
    )
    calmar_ratio = (
        annualized_return / abs(dd_info['max_drawdown'])
        if dd_info['max_drawdown'] != 0
        else 0.0
    )
    omega_ratio = _omega_ratio(returns, risk_free_rate / periods_per_year)

    # Information ratio (vs benchmark)
    information_ratio = None
    if benchmark_returns is not None:
        tracking_error = (returns - benchmark_returns).std() * np.sqrt(periods_per_year)
        if tracking_error > 0:
            information_ratio = (annualized_return - benchmark_returns.mean()) / tracking_error

    # === TRADE STATISTICS ===
    trade_stats = _calculate_trade_stats(trades, initial_capital)

    # === ADVANCED METRICS ===
    var_95 = _value_at_risk(returns, confidence=0.95)
    cvar_95 = _conditional_var(returns, confidence=0.95)
    risk_of_ruin = _risk_of_ruin(returns, initial_capital)

    # Exposure
    exposure_time = (returns != 0).sum() / len(returns)

    # Recovery factor
    recovery_factor = (
        total_return / abs(dd_info['max_drawdown'])
        if dd_info['max_drawdown'] != 0
        else 0.0
    )

    return BacktestMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        cagr=cagr,
        monthly_returns=monthly_returns,
        volatility=volatility,
        downside_deviation=downside_deviation_annual,
        max_drawdown=dd_info['max_drawdown'],
        max_drawdown_duration=dd_info['max_drawdown_duration'],
        avg_drawdown=dd_info['avg_drawdown'],
        avg_drawdown_duration=dd_info['avg_drawdown_duration'],
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        omega_ratio=omega_ratio,
        information_ratio=information_ratio,
        total_trades=trade_stats['total_trades'],
        winning_trades=trade_stats['winning_trades'],
        losing_trades=trade_stats['losing_trades'],
        win_rate=trade_stats['win_rate'],
        avg_win=trade_stats['avg_win'],
        avg_loss=trade_stats['avg_loss'],
        avg_trade=trade_stats['avg_trade'],
        largest_win=trade_stats['largest_win'],
        largest_loss=trade_stats['largest_loss'],
        profit_factor=trade_stats['profit_factor'],
        avg_trade_duration_hours=trade_stats['avg_trade_duration_hours'],
        avg_winning_trade_duration_hours=trade_stats['avg_winning_trade_duration_hours'],
        avg_losing_trade_duration_hours=trade_stats['avg_losing_trade_duration_hours'],
        exposure_time=exposure_time,
        avg_position_size=trade_stats.get('avg_position_size', 1.0),
        expectancy=trade_stats['expectancy'],
        payoff_ratio=trade_stats['payoff_ratio'],
        recovery_factor=recovery_factor,
        risk_of_ruin=risk_of_ruin,
        var_95=var_95,
        cvar_95=cvar_95,
    )


def _infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Infer number of periods per year from index."""
    if len(index) < 2:
        return 252  # Default to daily

    avg_delta = (index[-1] - index[0]) / (len(index) - 1)
    hours = avg_delta.total_seconds() / 3600

    if hours < 0.5:  # Minutes
        return 252 * 24 * 60  # Assume 1-minute bars
    elif hours < 2:  # Hours
        return 252 * 24
    elif hours < 12:  # 4h or less
        return 252 * 6
    else:  # Daily or more
        return 252


def _downside_deviation(returns: pd.Series, target: float = 0.0) -> float:
    """Calculate downside deviation."""
    downside_returns = returns[returns < target] - target
    return np.sqrt((downside_returns**2).mean()) if len(downside_returns) > 0 else 0.0


def _calculate_drawdown_info(equity_curve: pd.Series) -> dict:
    """Calculate detailed drawdown information."""
    # Calculate drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max

    max_dd = drawdown.min()

    # Find drawdown periods
    is_drawdown = drawdown < 0
    dd_periods = (is_drawdown != is_drawdown.shift()).cumsum()

    if is_drawdown.any():
        dd_durations = dd_periods[is_drawdown].value_counts()
        max_dd_duration = dd_durations.max()
        avg_dd_duration = dd_durations.mean()

        # Average drawdown (excluding zero)
        dd_values = drawdown[is_drawdown].groupby(dd_periods[is_drawdown]).min()
        avg_dd = dd_values.mean()
    else:
        max_dd_duration = 0
        avg_dd_duration = 0
        avg_dd = 0.0

    return {
        'max_drawdown': max_dd,
        'max_drawdown_duration': max_dd_duration,
        'avg_drawdown': avg_dd,
        'avg_drawdown_duration': int(avg_dd_duration),
    }


def _omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Calculate Omega ratio."""
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    return gains / losses if losses > 0 else 0.0


def _value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Value at Risk."""
    return returns.quantile(1 - confidence)


def _conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Conditional VaR (Expected Shortfall)."""
    var = _value_at_risk(returns, confidence)
    return returns[returns <= var].mean()


def _risk_of_ruin(returns: pd.Series, initial_capital: float) -> float:
    """
    Estimate risk of ruin using simplified formula.

    This is a rough approximation based on win rate and payoff ratio.
    """
    winning_returns = returns[returns > 0]
    losing_returns = returns[returns < 0]

    if len(winning_returns) == 0 or len(losing_returns) == 0:
        return 0.0

    win_rate = len(winning_returns) / len(returns)
    avg_win = winning_returns.mean()
    avg_loss = abs(losing_returns.mean())

    if avg_loss == 0:
        return 0.0

    payoff_ratio = avg_win / avg_loss

    # Simplified risk of ruin formula
    if win_rate * payoff_ratio > (1 - win_rate):
        # Positive expectancy - low risk
        return max(0.0, 1.0 - (win_rate * payoff_ratio / (1 - win_rate)))
    else:
        # Negative expectancy - high risk
        return 1.0


def _calculate_trade_stats(trades: pd.DataFrame, initial_capital: float) -> dict:
    """Calculate trade statistics."""
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_trade': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'profit_factor': 0.0,
            'avg_trade_duration_hours': 0.0,
            'avg_winning_trade_duration_hours': 0.0,
            'avg_losing_trade_duration_hours': 0.0,
            'expectancy': 0.0,
            'payoff_ratio': 0.0,
        }

    # Try to get PnL column (different names possible)
    pnl_col = None
    for col in ['PnL', 'pnl', 'P&L', 'Return', 'return']:
        if col in trades.columns:
            pnl_col = col
            break

    if pnl_col is None:
        logger.warning("Could not find PnL column in trades DataFrame")
        return _calculate_trade_stats(pd.DataFrame(), initial_capital)

    pnl = trades[pnl_col]

    winning_trades = pnl[pnl > 0]
    losing_trades = pnl[pnl < 0]

    total_trades = len(trades)
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0.0
    avg_trade = pnl.mean()

    largest_win = winning_trades.max() if len(winning_trades) > 0 else 0.0
    largest_loss = losing_trades.min() if len(losing_trades) > 0 else 0.0

    profit_factor = (
        abs(winning_trades.sum() / losing_trades.sum())
        if len(losing_trades) > 0 and losing_trades.sum() != 0
        else 0.0
    )

    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # Duration stats
    if 'Duration' in trades.columns:
        durations = pd.to_timedelta(trades['Duration']).dt.total_seconds() / 3600
        avg_duration = durations.mean()

        winning_durations = durations[pnl > 0]
        avg_win_duration = winning_durations.mean() if len(winning_durations) > 0 else 0.0

        losing_durations = durations[pnl < 0]
        avg_loss_duration = losing_durations.mean() if len(losing_durations) > 0 else 0.0
    else:
        avg_duration = 0.0
        avg_win_duration = 0.0
        avg_loss_duration = 0.0

    return {
        'total_trades': total_trades,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_trade': avg_trade,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'profit_factor': profit_factor,
        'avg_trade_duration_hours': avg_duration,
        'avg_winning_trade_duration_hours': avg_win_duration,
        'avg_losing_trade_duration_hours': avg_loss_duration,
        'expectancy': expectancy,
        'payoff_ratio': payoff_ratio,
    }
