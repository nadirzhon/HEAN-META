"""
Backtest Engine for testing strategies on historical data
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hean.symbiont_x.genome_lab import StrategyGenome


@dataclass
class BacktestConfig:
    """Configuration for backtest"""
    initial_capital: float = 10000.0
    position_size_pct: float = 0.1  # 10% of capital per position
    max_positions: int = 3
    commission_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005  # 0.05% slippage


@dataclass
class Trade:
    """Single trade in backtest"""
    entry_time: int
    entry_price: float
    exit_time: int | None = None
    exit_price: float | None = None
    size: float = 0.0
    side: str = "LONG"  # LONG or SHORT
    pnl: float | None = None
    pnl_pct: float | None = None


@dataclass
class BacktestResult:
    """Result of a backtest run"""
    genome_name: str
    initial_capital: float
    final_capital: float
    total_return: float  # Absolute return
    return_pct: float  # Percentage return
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)


class BacktestEngine:
    """
    Backtest engine for testing strategy genomes on historical data
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run_backtest(
        self,
        genome: StrategyGenome,
        historical_data: list[dict[str, Any]]
    ) -> BacktestResult:
        """
        Run backtest for a single strategy genome

        Args:
            genome: Strategy genome to test
            historical_data: List of OHLCV candles with keys:
                - timestamp: unix timestamp in ms
                - open, high, low, close: prices
                - volume: trading volume

        Returns:
            BacktestResult with metrics
        """

        capital = self.config.initial_capital
        position = None  # Current position
        trades = []
        equity_curve = []

        # Extract genome parameters
        _entry_threshold = genome.genes.get('entry_threshold', 0.5)  # noqa: F841
        _exit_threshold = genome.genes.get('exit_threshold', 0.3)  # noqa: F841
        stop_loss_pct = genome.genes.get('stop_loss_pct', 2.0)
        take_profit_pct = genome.genes.get('take_profit_pct', 5.0)
        position_size_pct = genome.genes.get('position_size_pct', 10.0) / 100.0

        for i, candle in enumerate(historical_data):
            timestamp = candle['timestamp']
            price = candle['close']

            # Check if we have an open position
            if position is not None:
                # Check stop loss
                pnl_pct = ((price - position.entry_price) / position.entry_price) * 100

                should_exit = False
                _exit_reason = None  # noqa: F841

                if pnl_pct <= -stop_loss_pct:
                    should_exit = True
                    _exit_reason = "STOP_LOSS"  # noqa: F841
                elif pnl_pct >= take_profit_pct:
                    should_exit = True
                    _exit_reason = "TAKE_PROFIT"  # noqa: F841
                elif self._should_exit(genome, historical_data, i):
                    should_exit = True
                    _exit_reason = "SIGNAL"  # noqa: F841

                if should_exit:
                    # Close position
                    position.exit_time = timestamp
                    position.exit_price = price
                    position.pnl = (price - position.entry_price) * position.size
                    position.pnl_pct = pnl_pct

                    # Apply commission
                    commission = abs(position.pnl) * self.config.commission_pct
                    position.pnl -= commission

                    # Update capital
                    capital += position.pnl

                    trades.append(position)
                    position = None

            # Check for entry signal (if no position)
            elif position is None and self._should_enter(genome, historical_data, i):
                # Open position
                position_size_value = capital * position_size_pct
                position_size = position_size_value / price

                position = Trade(
                    entry_time=timestamp,
                    entry_price=price,
                    size=position_size,
                    side="LONG"
                )

            # Record equity
            current_equity = capital
            if position is not None:
                unrealized_pnl = (price - position.entry_price) * position.size
                current_equity += unrealized_pnl

            equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'price': price
            })

        # Close any remaining position at end
        if position is not None:
            last_price = historical_data[-1]['close']
            position.exit_time = historical_data[-1]['timestamp']
            position.exit_price = last_price
            position.pnl = (last_price - position.entry_price) * position.size
            position.pnl_pct = ((last_price - position.entry_price) / position.entry_price) * 100

            # Apply commission
            commission = abs(position.pnl) * self.config.commission_pct
            position.pnl -= commission

            capital += position.pnl
            trades.append(position)

        # Calculate metrics
        return self._calculate_metrics(genome, trades, equity_curve, capital)

    def _should_enter(
        self,
        genome: StrategyGenome,
        historical_data: list[dict[str, Any]],
        current_index: int
    ) -> bool:
        """
        Determine if strategy should enter position

        This is a simplified version - real implementation would use
        proper indicators and signals based on genome parameters
        """

        if current_index < 10:
            return False  # Need history

        # Simple momentum strategy as example
        current_price = historical_data[current_index]['close']
        past_price = historical_data[current_index - 10]['close']

        momentum = (current_price - past_price) / past_price

        entry_threshold = genome.genes.get('entry_threshold', 0.5) / 100.0

        return momentum > entry_threshold

    def _should_exit(
        self,
        genome: StrategyGenome,
        historical_data: list[dict[str, Any]],
        current_index: int
    ) -> bool:
        """
        Determine if strategy should exit position
        """

        if current_index < 5:
            return False

        # Simple mean reversion exit
        current_price = historical_data[current_index]['close']
        avg_price = statistics.mean([
            historical_data[i]['close']
            for i in range(current_index - 5, current_index)
        ])

        exit_threshold = genome.genes.get('exit_threshold', 0.3) / 100.0

        # Exit if price reverted to mean
        return abs(current_price - avg_price) / avg_price < exit_threshold

    def _calculate_metrics(
        self,
        genome: StrategyGenome,
        trades: list[Trade],
        equity_curve: list[dict[str, Any]],
        final_capital: float
    ) -> BacktestResult:
        """Calculate backtest metrics"""

        initial_capital = self.config.initial_capital

        # Basic metrics
        total_return = final_capital - initial_capital
        return_pct = (total_return / initial_capital) * 100

        # Trade statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl <= 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Sharpe ratio
        if len(trades) > 1:
            returns = [t.pnl_pct for t in trades]
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        peak_equity = initial_capital

        for point in equity_curve:
            equity = point['equity']
            if equity > peak_equity:
                peak_equity = equity

            drawdown = peak_equity - equity
            drawdown_pct = (drawdown / peak_equity) * 100 if peak_equity > 0 else 0.0

            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct

        return BacktestResult(
            genome_name=genome.name,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            return_pct=return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            trades=trades,
            equity_curve=equity_curve
        )

    def run_population_backtest(
        self,
        population: list[StrategyGenome],
        historical_data: list[dict[str, Any]]
    ) -> list[BacktestResult]:
        """
        Run backtest for entire population

        Args:
            population: List of strategy genomes
            historical_data: Historical OHLCV data

        Returns:
            List of BacktestResults, one per genome
        """

        results = []

        for genome in population:
            result = self.run_backtest(genome, historical_data)
            results.append(result)

        return results
