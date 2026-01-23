"""
Model Evaluation Module

Provides comprehensive metrics for ML model evaluation:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix
- Classification Report
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report,
        precision_recall_curve,
        roc_curve
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")


class ModelEvaluator:
    """
    Comprehensive model evaluation with standard ML metrics.

    Computes accuracy, precision, recall, F1, ROC AUC, and more.
    """

    def __init__(self):
        """Initialize evaluator."""
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required. Install with: pip install scikit-learn"
            )

        self.evaluation_history = []

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        dataset_name: str = "test"
    ) -> Dict[str, Any]:
        """
        Evaluate model predictions with comprehensive metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            dataset_name: Name of the dataset being evaluated

        Returns:
            Dictionary with all evaluation metrics
        """
        results = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(y_true)
        }

        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, zero_division=0)
        results['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }

        # Derived metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()

        # Specificity (True Negative Rate)
        results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        # False Positive Rate
        results['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0

        # False Negative Rate
        results['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Positive Predictive Value (same as precision)
        results['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Negative Predictive Value
        results['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0

        # Matthews Correlation Coefficient
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        results['mcc'] = mcc_num / mcc_den if mcc_den > 0 else 0

        # ROC AUC (if probabilities are provided)
        if y_pred_proba is not None:
            try:
                results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                results['roc_auc'] = None

        # Classification report as dict
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        results['classification_report'] = report

        # Store in history
        self.evaluation_history.append(results)

        return results

    def print_evaluation(self, results: Dict[str, Any]) -> None:
        """
        Print evaluation results in a readable format.

        Args:
            results: Results dictionary from evaluate()
        """
        print(f"\n{'='*60}")
        print(f"Model Evaluation Results - {results['dataset']}")
        print(f"{'='*60}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Samples: {results['n_samples']}")
        print(f"\n{'Core Metrics':-^60}")
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1 Score:  {results['f1']:.4f}")

        if results.get('roc_auc') is not None:
            print(f"ROC AUC:   {results['roc_auc']:.4f}")

        print(f"\n{'Confusion Matrix':-^60}")
        cm = results['confusion_matrix']
        print(f"True Negatives:  {cm['tn']:6d}  |  False Positives: {cm['fp']:6d}")
        print(f"False Negatives: {cm['fn']:6d}  |  True Positives:  {cm['tp']:6d}")

        print(f"\n{'Additional Metrics':-^60}")
        print(f"Specificity (TNR): {results['specificity']:.4f}")
        print(f"FPR:               {results['fpr']:.4f}")
        print(f"FNR:               {results['fnr']:.4f}")
        print(f"PPV:               {results['ppv']:.4f}")
        print(f"NPV:               {results['npv']:.4f}")
        print(f"MCC:               {results['mcc']:.4f}")
        print(f"{'='*60}\n")

    def compare_models(
        self,
        results_list: list,
        metric: str = 'f1'
    ) -> pd.DataFrame:
        """
        Compare multiple model results.

        Args:
            results_list: List of results dictionaries
            metric: Metric to sort by (default: f1)

        Returns:
            DataFrame with comparison
        """
        comparison = []

        for result in results_list:
            comparison.append({
                'dataset': result['dataset'],
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1': result['f1'],
                'roc_auc': result.get('roc_auc', None),
                'n_samples': result['n_samples']
            })

        df = pd.DataFrame(comparison)

        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)

        return df

    def calculate_profit_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        price_changes: np.ndarray,
        trading_fee: float = 0.001
    ) -> Dict[str, float]:
        """
        Calculate profit-based metrics for trading.

        Args:
            y_true: True price directions
            y_pred: Predicted price directions
            price_changes: Actual price changes (in percentage)
            trading_fee: Trading fee per trade (default: 0.1%)

        Returns:
            Dictionary with profit metrics
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        price_changes = np.array(price_changes)

        # Calculate profit for each trade
        # If prediction is correct, we profit from price change
        # If prediction is wrong, we lose from price change
        profits = []

        for true, pred, change in zip(y_true, y_pred, price_changes):
            # Apply trading fee
            net_change = abs(change) - trading_fee

            if pred == true:
                # Correct prediction
                profits.append(net_change)
            else:
                # Wrong prediction
                profits.append(-net_change)

        profits = np.array(profits)

        # Calculate metrics
        total_profit = np.sum(profits)
        avg_profit = np.mean(profits)
        win_rate = np.sum(profits > 0) / len(profits) if len(profits) > 0 else 0

        # Cumulative returns
        cumulative_returns = np.cumsum(profits)
        max_drawdown = np.min(cumulative_returns) if len(cumulative_returns) > 0 else 0

        # Sharpe ratio (simplified)
        sharpe_ratio = (
            np.mean(profits) / (np.std(profits) + 1e-8)
            if len(profits) > 0 else 0
        )

        return {
            'total_profit_pct': total_profit,
            'avg_profit_per_trade_pct': avg_profit,
            'win_rate': win_rate,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(profits),
            'winning_trades': int(np.sum(profits > 0)),
            'losing_trades': int(np.sum(profits < 0))
        }

    def evaluate_with_profit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        price_changes: np.ndarray,
        dataset_name: str = "test",
        trading_fee: float = 0.001
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation including both ML metrics and profit metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            price_changes: Actual price changes
            dataset_name: Dataset name
            trading_fee: Trading fee

        Returns:
            Combined metrics
        """
        # Standard ML metrics
        ml_metrics = self.evaluate(y_true, y_pred, y_pred_proba, dataset_name)

        # Profit metrics
        profit_metrics = self.calculate_profit_metrics(
            y_true, y_pred, price_changes, trading_fee
        )

        # Combine results
        results = {**ml_metrics, **profit_metrics}

        return results

    def get_evaluation_summary(self) -> pd.DataFrame:
        """
        Get summary of all evaluations in history.

        Returns:
            DataFrame with evaluation history
        """
        if not self.evaluation_history:
            return pd.DataFrame()

        summary = []
        for result in self.evaluation_history:
            summary.append({
                'timestamp': result['timestamp'],
                'dataset': result['dataset'],
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1': result['f1'],
                'roc_auc': result.get('roc_auc'),
                'n_samples': result['n_samples']
            })

        return pd.DataFrame(summary)
