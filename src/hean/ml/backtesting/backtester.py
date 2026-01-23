"""
Backtesting Module for ML Models

Simulates trading based on ML predictions on historical data.
Calculates realistic performance metrics including:
- Returns
- Sharpe ratio
- Max drawdown
- Win rate
- And more...
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime


class Backtester:
    """
    Backtests ML model predictions on historical data.

    Simulates realistic trading with fees and slippage.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize backtester.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Trading parameters
        self.trading_fee = self.config.get('trading_fee', 0.001)  # 0.1%
        self.slippage = self.config.get('slippage', 0.0005)  # 0.05%
        self.initial_capital = self.config.get('initial_capital', 10000.0)

        # Position sizing
        self.position_size = self.config.get('position_size', 1.0)  # 100% of capital
        self.use_kelly = self.config.get('use_kelly', False)

        # Results
        self.backtest_results = None

    def backtest(
        self,
        predictions: np.ndarray,
        prices: pd.Series,
        timestamps: Optional[pd.Series] = None,
        probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Run backtest simulation.

        Args:
            predictions: Binary predictions (0=down, 1=up)
            prices: Price series
            timestamps: Timestamps for each price
            probabilities: Prediction probabilities (optional, for Kelly criterion)

        Returns:
            Dictionary with backtest results
        """
        # Convert to numpy arrays
        predictions = np.array(predictions)
        prices = np.array(prices)

        if timestamps is not None:
            timestamps = pd.to_datetime(timestamps)

        # Initialize tracking
        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long
        entry_price = 0

        # Track history
        capital_history = [capital]
        returns_history = []
        trades = []
        positions_history = [0]

        # Simulate trading
        for i in range(len(predictions)):
            current_price = prices[i]

            # Calculate position size
            if self.use_kelly and probabilities is not None:
                position_size = self._kelly_criterion(probabilities[i])
            else:
                position_size = self.position_size

            # Trading logic
            if position == 0:  # No position
                if predictions[i] == 1:  # Buy signal
                    # Enter long position
                    position = 1
                    entry_price = current_price * (1 + self.slippage)
                    trade_cost = capital * position_size * self.trading_fee

                    capital -= trade_cost

                    trades.append({
                        'timestamp': timestamps[i] if timestamps is not None else i,
                        'type': 'BUY',
                        'price': entry_price,
                        'capital': capital,
                        'fee': trade_cost
                    })

            else:  # In long position
                if predictions[i] == 0:  # Sell signal
                    # Exit long position
                    exit_price = current_price * (1 - self.slippage)

                    # Calculate PnL
                    price_change = (exit_price - entry_price) / entry_price
                    position_value = capital * position_size
                    pnl = position_value * price_change

                    # Apply exit fee
                    trade_cost = position_value * self.trading_fee

                    capital += pnl - trade_cost

                    returns_history.append(price_change)

                    trades.append({
                        'timestamp': timestamps[i] if timestamps is not None else i,
                        'type': 'SELL',
                        'price': exit_price,
                        'pnl': pnl,
                        'capital': capital,
                        'fee': trade_cost,
                        'return_pct': price_change * 100
                    })

                    position = 0
                    entry_price = 0

            capital_history.append(capital)
            positions_history.append(position)

        # Close any open position at the end
        if position == 1:
            exit_price = prices[-1] * (1 - self.slippage)
            price_change = (exit_price - entry_price) / entry_price
            position_value = capital * position_size
            pnl = position_value * price_change
            trade_cost = position_value * self.trading_fee

            capital += pnl - trade_cost

            returns_history.append(price_change)

            trades.append({
                'timestamp': timestamps[-1] if timestamps is not None else len(predictions) - 1,
                'type': 'SELL (CLOSE)',
                'price': exit_price,
                'pnl': pnl,
                'capital': capital,
                'fee': trade_cost,
                'return_pct': price_change * 100
            })

            capital_history.append(capital)

        # Calculate metrics
        metrics = self._calculate_metrics(
            capital_history,
            returns_history,
            trades
        )

        # Store results
        self.backtest_results = {
            'metrics': metrics,
            'capital_history': capital_history,
            'trades': trades,
            'positions_history': positions_history,
            'final_capital': capital,
            'initial_capital': self.initial_capital
        }

        return self.backtest_results

    def _calculate_metrics(
        self,
        capital_history: List[float],
        returns_history: List[float],
        trades: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate performance metrics."""
        capital_history = np.array(capital_history)
        returns_history = np.array(returns_history) if returns_history else np.array([])

        # Total return
        total_return = (capital_history[-1] - self.initial_capital) / self.initial_capital
        total_return_pct = total_return * 100

        # Number of trades
        num_trades = len([t for t in trades if t['type'] in ['SELL', 'SELL (CLOSE)']])

        # Win rate
        if returns_history.size > 0:
            winning_trades = np.sum(returns_history > 0)
            win_rate = winning_trades / len(returns_history) if len(returns_history) > 0 else 0
        else:
            win_rate = 0

        # Average return per trade
        avg_return = np.mean(returns_history) if returns_history.size > 0 else 0
        avg_return_pct = avg_return * 100

        # Sharpe ratio (annualized, assuming 5-min candles)
        if returns_history.size > 1:
            # 288 five-minute periods per day, 365 days per year
            periods_per_year = 288 * 365
            sharpe_ratio = (
                np.mean(returns_history) / (np.std(returns_history) + 1e-8) *
                np.sqrt(periods_per_year)
            )
        else:
            sharpe_ratio = 0

        # Max drawdown
        cumulative = capital_history / capital_history[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        max_drawdown_pct = max_drawdown * 100

        # Sortino ratio (downside deviation)
        if returns_history.size > 1:
            downside_returns = returns_history[returns_history < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                periods_per_year = 288 * 365
                sortino_ratio = (
                    np.mean(returns_history) / (downside_std + 1e-8) *
                    np.sqrt(periods_per_year)
                )
            else:
                sortino_ratio = float('inf') if np.mean(returns_history) > 0 else 0
        else:
            sortino_ratio = 0

        # Profit factor
        if returns_history.size > 0:
            gross_profit = np.sum(returns_history[returns_history > 0])
            gross_loss = abs(np.sum(returns_history[returns_history < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            profit_factor = 0

        # Calmar ratio (return / max drawdown)
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'total_return_pct': total_return_pct,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_return_pct': avg_return_pct,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown_pct,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'final_capital': capital_history[-1],
            'initial_capital': self.initial_capital
        }

    def _kelly_criterion(self, win_prob: float) -> float:
        """
        Calculate Kelly criterion for position sizing.

        Args:
            win_prob: Probability of winning trade

        Returns:
            Optimal position size (0-1)
        """
        # Simplified Kelly: f = p - q = 2p - 1
        # Where p is win probability
        kelly = 2 * win_prob - 1

        # Apply Kelly fraction (typically use 25-50% of full Kelly)
        kelly_fraction = self.config.get('kelly_fraction', 0.25)
        kelly *= kelly_fraction

        # Clamp to [0, 1]
        return np.clip(kelly, 0, 1)

    def print_results(self) -> None:
        """Print backtest results in a readable format."""
        if self.backtest_results is None:
            print("No backtest results available. Run backtest() first.")
            return

        metrics = self.backtest_results['metrics']

        print(f"\n{'='*60}")
        print(f"Backtest Results")
        print(f"{'='*60}")

        print(f"\nCapital:")
        print(f"  Initial:  ${metrics['initial_capital']:,.2f}")
        print(f"  Final:    ${metrics['final_capital']:,.2f}")
        print(f"  Return:   {metrics['total_return_pct']:,.2f}%")

        print(f"\nTrading:")
        print(f"  Trades:     {metrics['num_trades']}")
        print(f"  Win Rate:   {metrics['win_rate']:.2%}")
        print(f"  Avg Return: {metrics['avg_return_pct']:.3f}%")

        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"  Max Drawdown:  {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Calmar Ratio:  {metrics['calmar_ratio']:.2f}")

        print(f"{'='*60}\n")

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as a DataFrame."""
        if self.backtest_results is None:
            return pd.DataFrame()

        return pd.DataFrame(self.backtest_results['trades'])

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as a DataFrame."""
        if self.backtest_results is None:
            return pd.DataFrame()

        return pd.DataFrame({
            'capital': self.backtest_results['capital_history'],
            'position': self.backtest_results['positions_history']
        })
