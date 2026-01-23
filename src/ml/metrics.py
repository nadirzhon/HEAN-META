"""
Metrics for evaluating Bitcoin price prediction models.

Includes:
- Classification metrics (accuracy, precision, recall, F1)
- Trading-specific metrics (profit, Sharpe ratio, max drawdown)
- Confusion matrix and ROC curves
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ClassificationMetrics:
    """Classification metrics container."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    confusion_matrix: np.ndarray

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
        }


@dataclass
class TradingMetrics:
    """Trading performance metrics."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
        }


class MetricsCalculator:
    """Calculate various metrics for model evaluation."""

    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ClassificationMetrics:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted labels (0 or 1)

        Returns:
            ClassificationMetrics object
        """
        # Confusion matrix
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        confusion_matrix = np.array([[tn, fp], [fn, tp]])

        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            confusion_matrix=confusion_matrix,
        )

    @staticmethod
    def calculate_roc_auc(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """
        Calculate ROC AUC score.

        Args:
            y_true: True labels (0 or 1)
            y_proba: Predicted probabilities (0-1)

        Returns:
            ROC AUC score
        """
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_proba)
        except ImportError:
            # Manual calculation
            return MetricsCalculator._calculate_roc_auc_manual(y_true, y_proba)

    @staticmethod
    def _calculate_roc_auc_manual(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """Manual ROC AUC calculation (if sklearn not available)."""
        # Sort by predicted probability
        sorted_indices = np.argsort(y_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]

        # Calculate TPR and FPR at different thresholds
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr = 0
        fpr = 0
        auc = 0.0

        for label in y_true_sorted:
            if label == 1:
                tpr += 1 / n_pos
            else:
                fpr += 1 / n_neg
                auc += tpr * (1 / n_neg)

        return auc

    @staticmethod
    def calculate_trading_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prices: np.ndarray,
        transaction_cost: float = 0.001,  # 0.1% per trade
    ) -> TradingMetrics:
        """
        Calculate trading performance metrics.

        Args:
            y_true: True price directions (0=down, 1=up)
            y_pred: Predicted price directions (0=down, 1=up)
            prices: Actual prices
            transaction_cost: Transaction cost as fraction (default: 0.1%)

        Returns:
            TradingMetrics object
        """
        # Calculate returns based on predictions
        actual_returns = np.diff(prices) / prices[:-1]

        # Align arrays (we lose one element due to diff)
        y_pred_aligned = y_pred[:-1]
        y_true_aligned = y_true[:-1]

        # Trading strategy: long if predict up (1), short if predict down (0)
        # Position: 1 for long, -1 for short
        positions = np.where(y_pred_aligned == 1, 1, -1)

        # Strategy returns
        strategy_returns = positions * actual_returns

        # Apply transaction costs (when position changes)
        position_changes = np.diff(np.concatenate([[0], positions]))
        transaction_costs = np.abs(position_changes) * transaction_cost
        strategy_returns -= np.concatenate([[0], transaction_costs])

        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + strategy_returns) - 1
        total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0.0

        # Sharpe ratio (annualized, assuming 5-min intervals)
        # 288 periods per day (24 * 60 / 5)
        periods_per_year = 288 * 365
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        sharpe_ratio = (mean_return * np.sqrt(periods_per_year) / std_return) if std_return > 0 else 0.0

        # Max drawdown
        cumulative_max = np.maximum.accumulate(1 + cumulative_returns)
        drawdown = (1 + cumulative_returns) / cumulative_max - 1
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

        # Win/loss statistics
        winning_trades = strategy_returns > 0
        losing_trades = strategy_returns < 0

        total_trades = len(strategy_returns)
        n_winning = np.sum(winning_trades)
        n_losing = np.sum(losing_trades)
        win_rate = n_winning / total_trades if total_trades > 0 else 0.0

        # Average win/loss
        avg_win = np.mean(strategy_returns[winning_trades]) if n_winning > 0 else 0.0
        avg_loss = np.mean(strategy_returns[losing_trades]) if n_losing > 0 else 0.0

        # Profit factor
        gross_profit = np.sum(strategy_returns[winning_trades]) if n_winning > 0 else 0.0
        gross_loss = abs(np.sum(strategy_returns[losing_trades])) if n_losing > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Consecutive wins/losses
        max_consecutive_wins = MetricsCalculator._max_consecutive(winning_trades)
        max_consecutive_losses = MetricsCalculator._max_consecutive(losing_trades)

        return TradingMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=n_winning,
            losing_trades=n_losing,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
        )

    @staticmethod
    def _max_consecutive(boolean_array: np.ndarray) -> int:
        """Calculate maximum consecutive True values."""
        if len(boolean_array) == 0:
            return 0

        max_count = 0
        current_count = 0

        for val in boolean_array:
            if val:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        prices: Optional[np.ndarray] = None,
        transaction_cost: float = 0.001,
    ) -> Dict:
        """
        Calculate all metrics (classification + trading).

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            prices: Actual prices (optional, for trading metrics)
            transaction_cost: Transaction cost fraction

        Returns:
            Dictionary with all metrics
        """
        # Classification metrics
        classification = MetricsCalculator.calculate_classification_metrics(y_true, y_pred)
        roc_auc = MetricsCalculator.calculate_roc_auc(y_true, y_proba)

        metrics = {
            **classification.to_dict(),
            'roc_auc': roc_auc,
        }

        # Trading metrics (if prices provided)
        if prices is not None and len(prices) > 0:
            trading = MetricsCalculator.calculate_trading_metrics(
                y_true, y_pred, prices, transaction_cost
            )
            metrics.update(trading.to_dict())

        return metrics

    @staticmethod
    def print_metrics_report(metrics: Dict) -> None:
        """
        Print formatted metrics report.

        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)

        # Classification metrics
        print("\nClassification Metrics:")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0):.4f}")
        print(f"  F1 Score:  {metrics.get('f1_score', 0):.4f}")
        print(f"  ROC AUC:   {metrics.get('roc_auc', 0):.4f}")

        # Confusion matrix
        print("\nConfusion Matrix:")
        print(f"  True Positives:  {metrics.get('true_positives', 0)}")
        print(f"  True Negatives:  {metrics.get('true_negatives', 0)}")
        print(f"  False Positives: {metrics.get('false_positives', 0)}")
        print(f"  False Negatives: {metrics.get('false_negatives', 0)}")

        # Trading metrics (if available)
        if 'total_return' in metrics:
            print("\nTrading Performance Metrics:")
            print(f"  Total Return:    {metrics['total_return']:.2%}")
            print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown:    {metrics['max_drawdown']:.2%}")
            print(f"  Win Rate:        {metrics['win_rate']:.2%}")
            print(f"  Profit Factor:   {metrics['profit_factor']:.2f}")
            print(f"  Total Trades:    {metrics['total_trades']}")
            print(f"  Winning Trades:  {metrics['winning_trades']}")
            print(f"  Losing Trades:   {metrics['losing_trades']}")
            print(f"  Avg Win:         {metrics['avg_win']:.4f}")
            print(f"  Avg Loss:        {metrics['avg_loss']:.4f}")
            print(f"  Max Consecutive Wins:   {metrics['max_consecutive_wins']}")
            print(f"  Max Consecutive Losses: {metrics['max_consecutive_losses']}")

        print("="*60 + "\n")

    @staticmethod
    def save_metrics_to_csv(
        metrics: Dict,
        file_path: str,
        append: bool = False,
    ) -> None:
        """
        Save metrics to CSV file.

        Args:
            metrics: Metrics dictionary
            file_path: Output file path
            append: Whether to append to existing file
        """
        # Add timestamp
        metrics['timestamp'] = pd.Timestamp.now().isoformat()

        # Convert to DataFrame
        df = pd.DataFrame([metrics])

        # Save
        if append:
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)

        print(f"Metrics saved to {file_path}")
