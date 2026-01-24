"""Visualization utilities for backtesting results."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

from hean.backtesting.optuna_optimizer import OptimizationResult
from hean.backtesting.vectorbt_engine import BacktestResult
from hean.backtesting.walk_forward import WalkForwardResult
from hean.logging import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class BacktestVisualizer:
    """
    Visualization toolkit for backtesting results.

    Provides:
    - Equity curves
    - Drawdown plots
    - Trade analysis
    - Parameter heatmaps
    - Walk-forward analysis charts
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize visualizer."""
        self.output_dir = output_dir or Path('backtest_results')
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def plot_backtest_summary(
        self,
        result: BacktestResult,
        title: str = "Backtest Summary",
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot comprehensive backtest summary.

        Includes:
        - Equity curve
        - Drawdown
        - Monthly returns heatmap
        - Trade distribution
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_curve(ax1, result)
        ax1.set_title(f"{title} - Equity Curve", fontsize=14, fontweight='bold')

        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_drawdown(ax2, result)
        ax2.set_title("Drawdown", fontsize=12)

        # 3. Monthly returns heatmap
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_monthly_returns_heatmap(ax3, result)
        ax3.set_title("Monthly Returns (%)", fontsize=12)

        # 4. Trade distribution
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_trade_distribution(ax4, result)
        ax4.set_title("Trade P&L Distribution", fontsize=12)

        # Add summary stats as text
        stats_text = self._format_stats(result)
        fig.text(
            0.02,
            0.98,
            stats_text,
            verticalalignment='top',
            fontfamily='monospace',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
        )

        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.savefig(self.output_dir / 'backtest_summary.png', dpi=150, bbox_inches='tight')

        plt.close()

    def plot_optimization_results(
        self,
        result: OptimizationResult,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot optimization results.

        Includes:
        - Optimization history
        - Parameter importances
        - Parameter relationships (scatter matrix)
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Optimization history
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_optimization_history(ax1, result)
        ax1.set_title("Optimization History", fontsize=14, fontweight='bold')

        # 2. Parameter importances
        if result.param_importances:
            ax2 = fig.add_subplot(gs[1, 0])
            self._plot_param_importances(ax2, result)
            ax2.set_title("Parameter Importances", fontsize=12)

        # 3. Best trials (for multi-objective)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_trials_scatter(ax3, result)
        ax3.set_title("Trials Performance", fontsize=12)

        plt.suptitle("Optimization Results", fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.savefig(self.output_dir / 'optimization_results.png', dpi=150, bbox_inches='tight')

        plt.close()

    def plot_walk_forward_results(
        self,
        result: WalkForwardResult,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot walk-forward analysis results.

        Includes:
        - Combined equity curve
        - Train vs Test Sharpe per window
        - Overfitting ratio
        - Parameter stability
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Combined equity curve
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_curve(ax1, result.combined_test_result)
        ax1.set_title("Walk-Forward: Combined Out-of-Sample Equity", fontsize=14, fontweight='bold')

        # 2. Sharpe ratios per window
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_wf_sharpe_comparison(ax2, result)
        ax2.set_title("Train vs Test Sharpe per Window", fontsize=12)

        # 3. Overfitting ratios
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_wf_overfitting_ratios(ax3, result)
        ax3.set_title("Overfitting Ratios (Test/Train)", fontsize=12)

        # 4. Returns per window
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_wf_returns(ax4, result)
        ax4.set_title("Returns per Window", fontsize=12)

        # 5. Parameter stability
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_wf_param_stability(ax5, result)
        ax5.set_title("Parameter Stability (Std)", fontsize=12)

        # Add summary stats
        stats_text = self._format_wf_stats(result)
        fig.text(
            0.02,
            0.98,
            stats_text,
            verticalalignment='top',
            fontfamily='monospace',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
        )

        plt.suptitle("Walk-Forward Analysis", fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.savefig(self.output_dir / 'walk_forward_results.png', dpi=150, bbox_inches='tight')

        plt.close()

    def plot_parameter_heatmap(
        self,
        results_df: pd.DataFrame,
        param1: str,
        param2: str,
        metric: str = 'sharpe_ratio',
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot 2D heatmap of metric vs two parameters.

        Args:
            results_df: DataFrame with parameters and metrics
            param1: First parameter name
            param2: Second parameter name
            metric: Metric to visualize
            save_path: Optional save path
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Pivot for heatmap
        pivot = results_df.pivot_table(
            values=metric, index=param2, columns=param1, aggfunc='mean'
        )

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            ax=ax,
            cbar_kws={'label': metric},
        )

        ax.set_title(f"{metric} vs {param1} and {param2}", fontsize=14, fontweight='bold')
        ax.set_xlabel(param1, fontsize=12)
        ax.set_ylabel(param2, fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved heatmap to {save_path}")
        else:
            plt.savefig(
                self.output_dir / f'heatmap_{param1}_{param2}.png',
                dpi=150,
                bbox_inches='tight',
            )

        plt.close()

    # === Helper plotting methods ===

    def _plot_equity_curve(self, ax: plt.Axes, result: BacktestResult) -> None:
        """Plot equity curve."""
        equity = result.equity_curve
        ax.plot(equity.index, equity.values, linewidth=2, label='Portfolio Value')
        ax.fill_between(equity.index, equity.values, alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_drawdown(self, ax: plt.Axes, result: BacktestResult) -> None:
        """Plot drawdown."""
        if len(result.drawdown) > 0:
            dd = result.drawdown * 100  # Convert to percentage
            ax.fill_between(dd.index, dd.values, 0, color='red', alpha=0.3)
            ax.plot(dd.index, dd.values, color='darkred', linewidth=1)
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.grid(True, alpha=0.3)
        else:
            # Calculate from equity curve
            equity = result.equity_curve
            running_max = equity.expanding().max()
            drawdown = ((equity - running_max) / running_max) * 100
            ax.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
            ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.grid(True, alpha=0.3)

    def _plot_monthly_returns_heatmap(self, ax: plt.Axes, result: BacktestResult) -> None:
        """Plot monthly returns heatmap."""
        returns = result.returns
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100

        if len(monthly) > 0:
            # Reshape into year x month
            monthly_df = monthly.to_frame('return')
            monthly_df['year'] = monthly_df.index.year
            monthly_df['month'] = monthly_df.index.month

            pivot = monthly_df.pivot_table(
                values='return', index='year', columns='month', aggfunc='mean'
            )

            sns.heatmap(
                pivot,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                ax=ax,
                cbar_kws={'label': 'Return (%)'},
            )
            ax.set_xlabel('Month')
            ax.set_ylabel('Year')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')

    def _plot_trade_distribution(self, ax: plt.Axes, result: BacktestResult) -> None:
        """Plot trade P&L distribution."""
        if len(result.trades) > 0 and 'PnL' in result.trades.columns:
            pnl = result.trades['PnL']
            ax.hist(pnl, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('P&L per Trade')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No trade data', ha='center', va='center')

    def _plot_optimization_history(self, ax: plt.Axes, result: OptimizationResult) -> None:
        """Plot optimization history."""
        trials_df = result.trials_df

        if 'value' in trials_df.columns:
            # Single objective
            ax.plot(trials_df['number'], trials_df['value'], marker='o', alpha=0.6)
            # Best value so far
            best_values = trials_df['value'].cummax() if result.study.directions[0].name == 'MAXIMIZE' else trials_df['value'].cummin()
            ax.plot(trials_df['number'], best_values, color='red', linewidth=2, label='Best')
            ax.set_ylabel('Objective Value')
        else:
            # Multi-objective - plot first objective
            ax.plot(trials_df['number'], trials_df['values_0'], marker='o', alpha=0.6)
            ax.set_ylabel('Objective 1 Value')

        ax.set_xlabel('Trial Number')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_param_importances(self, ax: plt.Axes, result: OptimizationResult) -> None:
        """Plot parameter importances."""
        importances = result.param_importances
        if importances:
            params = list(importances.keys())
            values = list(importances.values())

            ax.barh(params, values, color='skyblue', edgecolor='black')
            ax.set_xlabel('Importance')
            ax.grid(True, alpha=0.3, axis='x')

    def _plot_trials_scatter(self, ax: plt.Axes, result: OptimizationResult) -> None:
        """Plot trials scatter."""
        trials_df = result.trials_df

        if 'value' in trials_df.columns:
            # Single objective - plot value vs trial number
            scatter = ax.scatter(
                trials_df['number'],
                trials_df['value'],
                c=trials_df['value'],
                cmap='viridis',
                alpha=0.6,
            )
            ax.set_xlabel('Trial Number')
            ax.set_ylabel('Objective Value')
            plt.colorbar(scatter, ax=ax)
        else:
            # Multi-objective - plot objective 1 vs objective 2
            scatter = ax.scatter(
                trials_df['values_0'],
                trials_df['values_1'],
                alpha=0.6,
                c=trials_df['number'],
                cmap='viridis',
            )
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            plt.colorbar(scatter, ax=ax, label='Trial #')

        ax.grid(True, alpha=0.3)

    def _plot_wf_sharpe_comparison(self, ax: plt.Axes, result: WalkForwardResult) -> None:
        """Plot train vs test Sharpe ratios."""
        windows = result.windows
        window_ids = [w.window_id for w in windows]
        train_sharpes = [w.train_sharpe for w in windows]
        test_sharpes = [w.test_sharpe for w in windows]

        x = np.arange(len(window_ids))
        width = 0.35

        ax.bar(x - width / 2, train_sharpes, width, label='Train', alpha=0.8)
        ax.bar(x + width / 2, test_sharpes, width, label='Test', alpha=0.8)

        ax.set_xlabel('Window')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_xticks(x)
        ax.set_xticklabels(window_ids)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.8)

    def _plot_wf_overfitting_ratios(self, ax: plt.Axes, result: WalkForwardResult) -> None:
        """Plot overfitting ratios."""
        windows = result.windows
        window_ids = [w.window_id for w in windows]
        ratios = [w.overfitting_ratio for w in windows]

        ax.bar(window_ids, ratios, alpha=0.7, color='orange', edgecolor='black')
        ax.axhline(1.0, color='green', linestyle='--', linewidth=2, label='Perfect (1.0)')
        ax.axhline(0.8, color='yellow', linestyle='--', linewidth=1, label='Good (0.8)')
        ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Poor (0.5)')

        ax.set_xlabel('Window')
        ax.set_ylabel('Overfitting Ratio (Test/Train)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_wf_returns(self, ax: plt.Axes, result: WalkForwardResult) -> None:
        """Plot returns per window."""
        windows = result.windows
        window_ids = [w.window_id for w in windows]
        train_returns = [w.train_return * 100 for w in windows]
        test_returns = [w.test_return * 100 for w in windows]

        x = np.arange(len(window_ids))
        width = 0.35

        ax.bar(x - width / 2, train_returns, width, label='Train', alpha=0.8)
        ax.bar(x + width / 2, test_returns, width, label='Test', alpha=0.8)

        ax.set_xlabel('Window')
        ax.set_ylabel('Return (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(window_ids)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.8)

    def _plot_wf_param_stability(self, ax: plt.Axes, result: WalkForwardResult) -> None:
        """Plot parameter stability."""
        stability = result.param_stability

        if stability:
            params = list(stability.keys())
            values = list(stability.values())

            ax.barh(params, values, color='lightcoral', edgecolor='black')
            ax.set_xlabel('Std Deviation')
            ax.grid(True, alpha=0.3, axis='x')

    def _format_stats(self, result: BacktestResult) -> str:
        """Format stats as text."""
        return (
            f"Total Return: {result.total_return:.2%}\n"
            f"Annual Return: {result.annualized_return:.2%}\n"
            f"Sharpe Ratio: {result.sharpe_ratio:.3f}\n"
            f"Sortino Ratio: {result.sortino_ratio:.3f}\n"
            f"Calmar Ratio: {result.calmar_ratio:.3f}\n"
            f"Max Drawdown: {result.max_drawdown:.2%}\n"
            f"Win Rate: {result.win_rate:.2%}\n"
            f"Profit Factor: {result.profit_factor:.2f}\n"
            f"Total Trades: {result.total_trades}\n"
        )

    def _format_wf_stats(self, result: WalkForwardResult) -> str:
        """Format walk-forward stats."""
        return (
            f"Windows: {len(result.windows)}\n"
            f"Avg Train Sharpe: {result.avg_train_sharpe:.3f}\n"
            f"Avg Test Sharpe: {result.avg_test_sharpe:.3f}\n"
            f"Avg Overfitting: {result.avg_overfitting_ratio:.3f}\n"
            f"Total Return: {result.total_test_return:.2%}\n"
            f"Win Rate: {result.win_rate_across_windows:.2%}\n"
            f"Sharpe Stability: {result.sharpe_stability:.3f}\n"
        )
