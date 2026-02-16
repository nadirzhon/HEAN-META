"""
Backtest Engine for testing strategies on historical data
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from hean.symbiont_x.genome_lab.genome_types import GeneType, StrategyGenome

from . import indicators


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
        self._indicator_cache: dict[str, list[float | None]] = {}
        self._price_data: list[float] = []

    def _prepare_indicators(self, genome: StrategyGenome, historical_data: list[dict[str, Any]]):
        """Pre-calculates all indicators needed for the genome."""
        self._indicator_cache = {}
        self._price_data = [c['close'] for c in historical_data]

        indicator_genes = genome.get_genes_by_type(GeneType.INDICATOR_PARAM)
        for gene in indicator_genes:
            if 'period' in gene.name:
                period = int(gene.value)
                if 'sma' in gene.name:
                    self._indicator_cache[f'sma_{period}'] = indicators.calculate_sma(self._price_data, period)
                elif 'ema' in gene.name:
                    self._indicator_cache[f'ema_{period}'] = indicators.calculate_ema(self._price_data, period)
                elif 'rsi' in gene.name:
                    self._indicator_cache[f'rsi_{period}'] = indicators.calculate_rsi(self._price_data, period)

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

        if not historical_data:
            raise ValueError("Historical data cannot be empty")

        capital = self.config.initial_capital
        position = None  # Current position
        trades = []
        equity_curve = []

        # Pre-calculate indicators for efficiency
        self._prepare_indicators(genome, historical_data)

        # Extract risk parameters from genome
        stop_loss_gene = genome.get_gene(GeneType.STOP_LOSS)
        take_profit_gene = genome.get_gene(GeneType.TAKE_PROFIT)
        position_size_gene = genome.get_gene(GeneType.POSITION_SIZE)

        stop_loss_pct = stop_loss_gene.value if stop_loss_gene else 2.0
        take_profit_pct = take_profit_gene.value if take_profit_gene else 5.0
        position_size_pct = (position_size_gene.value / 100.0) if position_size_gene else 0.1

        for i, candle in enumerate(historical_data):
            timestamp = candle['timestamp']
            price = candle['close']

            # Check if we have an open position
            if position is not None:
                pnl_pct = ((price - position.entry_price) / position.entry_price) * 100
                should_exit = False

                if pnl_pct <= -stop_loss_pct:
                    should_exit = True
                elif pnl_pct >= take_profit_pct:
                    should_exit = True
                elif self._should_exit(genome, i):
                    should_exit = True

                if should_exit:
                    position.exit_time = timestamp
                    position.exit_price = price
                    position.pnl = (price - position.entry_price) * position.size
                    position.pnl_pct = pnl_pct
                    commission = abs(position.pnl) * self.config.commission_pct
                    position.pnl -= commission
                    capital += position.pnl
                    trades.append(position)
                    position = None

            # Check for entry signal (if no position)
            elif position is None and self._should_enter(genome, i):
                position_size_value = capital * position_size_pct
                position_size = position_size_value / price
                position = Trade(entry_time=timestamp, entry_price=price, size=position_size, side="LONG")

            # Record equity
            current_equity = capital
            if position is not None:
                unrealized_pnl = (price - position.entry_price) * position.size
                current_equity += unrealized_pnl
            equity_curve.append({'timestamp': timestamp, 'equity': current_equity, 'price': price})

        # Close any remaining position at end
        if position is not None:
            last_price = self._price_data[-1]
            position.exit_time = historical_data[-1]['timestamp']
            position.exit_price = last_price
            position.pnl = (last_price - position.entry_price) * position.size
            position.pnl_pct = ((last_price - position.entry_price) / position.entry_price) * 100
            commission = abs(position.pnl) * self.config.commission_pct
            position.pnl -= commission
            capital += position.pnl
            trades.append(position)

        return self._calculate_metrics(genome, trades, equity_curve, capital)

    def _should_enter(self, genome: StrategyGenome, current_index: int) -> bool:
        """Determine if strategy should enter based on genome."""
        entry_signal_gene = genome.get_gene(GeneType.ENTRY_SIGNAL)
        if not entry_signal_gene:
            return False

        signal_type = entry_signal_gene.value
        current_price = self._price_data[current_index]

        if signal_type == 'momentum':
            momentum_period_gene = genome.get_gene(GeneType.INDICATOR_PARAM, 'momentum_period')
            period = int(momentum_period_gene.value) if momentum_period_gene else 10
            momentum_threshold_gene = genome.get_gene(GeneType.THRESHOLD, 'momentum_threshold')
            threshold = momentum_threshold_gene.value if momentum_threshold_gene else 0.005
            if current_index < period:
                return False

            past_price = self._price_data[current_index - period]
            momentum = (current_price - past_price) / past_price
            return momentum > threshold

        elif signal_type == 'mean_reversion':
            ema_period_gene = genome.get_gene(GeneType.INDICATOR_PARAM, 'ema_period')
            period = int(ema_period_gene.value) if ema_period_gene else 20
            reversion_threshold_gene = genome.get_gene(GeneType.THRESHOLD, 'reversion_threshold_pct')
            threshold_pct = reversion_threshold_gene.value if reversion_threshold_gene else 1.0
            if f'ema_{period}' not in self._indicator_cache:
                return False

            ema_values = self._indicator_cache[f'ema_{period}']
            if not ema_values or ema_values[current_index] is None:
                return False

            ema = ema_values[current_index]
            delta_pct = (current_price - ema) / ema * 100
            return delta_pct < -threshold_pct  # Enter when price is below EMA by threshold

        elif signal_type == 'rsi_oversold':
            rsi_period_gene = genome.get_gene(GeneType.INDICATOR_PARAM, 'rsi_period')
            period = int(rsi_period_gene.value) if rsi_period_gene else 14
            rsi_oversold_gene = genome.get_gene(GeneType.THRESHOLD, 'rsi_oversold_level')
            threshold = rsi_oversold_gene.value if rsi_oversold_gene else 30.0
            if f'rsi_{period}' not in self._indicator_cache:
                return False

            rsi_values = self._indicator_cache[f'rsi_{period}']
            if not rsi_values or rsi_values[current_index] is None:
                return False

            return rsi_values[current_index] < threshold

        return False

    def _should_exit(self, genome: StrategyGenome, current_index: int) -> bool:
        """Determine if strategy should exit based on genome."""
        exit_signal_gene = genome.get_gene(GeneType.EXIT_SIGNAL)
        if not exit_signal_gene:
            return False

        signal_type = exit_signal_gene.value
        current_price = self._price_data[current_index]

        if signal_type == 'momentum_reverse':
            momentum_period_gene = genome.get_gene(GeneType.INDICATOR_PARAM, 'momentum_period')
            period = int(momentum_period_gene.value) if momentum_period_gene else 10
            momentum_threshold_gene = genome.get_gene(GeneType.THRESHOLD, 'momentum_threshold')
            threshold = momentum_threshold_gene.value if momentum_threshold_gene else 0.005
            if current_index < period:
                return False

            past_price = self._price_data[current_index - period]
            momentum = (current_price - past_price) / past_price
            return momentum < -threshold  # Exit on reversal

        elif signal_type == 'revert_to_mean':
            ema_period_gene = genome.get_gene(GeneType.INDICATOR_PARAM, 'ema_period')
            period = int(ema_period_gene.value) if ema_period_gene else 20
            if f'ema_{period}' not in self._indicator_cache:
                return False

            ema_values = self._indicator_cache[f'ema_{period}']
            if not ema_values or ema_values[current_index] is None:
                return False

            ema = ema_values[current_index]
            # Exit if price has crossed back above the moving average
            return current_price > ema

        return False

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
