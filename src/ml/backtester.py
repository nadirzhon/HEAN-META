"""
Backtesting module for Bitcoin price prediction models.

Features:
- Walk-forward backtesting
- Performance visualization
- Trading simulation with transaction costs
- Rolling window evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta
import logging

from ml.price_predictor import BitcoinPricePredictor, PredictorConfig
from ml.features import FeatureConfig
from ml.metrics import MetricsCalculator
from ml.data_loader import DataLoader

logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtesting engine for price prediction models.

    Supports walk-forward testing and performance analysis.
    """

    def __init__(
        self,
        predictor: Optional[BitcoinPricePredictor] = None,
        predictor_config: Optional[PredictorConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
    ):
        """
        Initialize backtester.

        Args:
            predictor: Pre-trained predictor (optional)
            predictor_config: Predictor configuration
            feature_config: Feature configuration
        """
        if predictor is not None:
            self.predictor = predictor
        else:
            self.predictor = BitcoinPricePredictor(
                config=predictor_config or PredictorConfig(),
                feature_config=feature_config or FeatureConfig(),
            )

        self.backtest_results: List[Dict[str, Any]] = []

    def run_backtest(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        orderbook_data: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Run simple backtest with train/test split.

        Args:
            df: OHLCV dataframe
            train_size: Training data proportion
            orderbook_data: Optional orderbook data
            sentiment_data: Optional sentiment data

        Returns:
            Backtest results dictionary
        """
        logger.info(f"Running backtest on {len(df)} samples")

        # Prepare data
        X, y = self.predictor.prepare_data(df, orderbook_data, sentiment_data)

        # Split
        n = len(X)
        train_end = int(n * train_size)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[train_end:], y.iloc[train_end:]

        logger.info(f"Split: train={len(X_train)}, test={len(X_test)}")

        # Train
        logger.info("Training model...")
        self.predictor.train(X_train, y_train)

        # Predict
        logger.info("Generating predictions...")
        y_pred = self.predictor.predict(X_test)
        y_proba = self.predictor.predict_proba(X_test)

        # Get prices for trading metrics
        prices = df.iloc[train_end:]['close'].values

        # Calculate metrics
        metrics = MetricsCalculator.calculate_all_metrics(
            y_test.values,
            y_pred,
            y_proba,
            prices,
            transaction_cost=0.001,
        )

        # Add metadata
        results = {
            'backtest_type': 'simple',
            'start_date': df.iloc[train_end]['timestamp'],
            'end_date': df.iloc[-1]['timestamp'],
            'n_train': len(X_train),
            'n_test': len(X_test),
            **metrics,
        }

        self.backtest_results.append(results)

        return results

    def run_walk_forward_backtest(
        self,
        df: pd.DataFrame,
        train_window: int = 5000,  # Number of samples for training
        test_window: int = 1000,   # Number of samples for testing
        step_size: int = 500,      # Step size between windows
        orderbook_data: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtesting.

        Trains model on sliding window and tests on next window.

        Args:
            df: OHLCV dataframe
            train_window: Training window size
            test_window: Test window size
            step_size: Step size between windows
            orderbook_data: Optional orderbook data
            sentiment_data: Optional sentiment data

        Returns:
            Aggregated backtest results
        """
        logger.info(
            f"Running walk-forward backtest: "
            f"train_window={train_window}, test_window={test_window}, step={step_size}"
        )

        # Prepare all data
        X, y = self.predictor.prepare_data(df, orderbook_data, sentiment_data)

        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        all_prices = []

        n = len(X)
        start_idx = 0
        window_count = 0

        while start_idx + train_window + test_window <= n:
            window_count += 1
            train_start = start_idx
            train_end = start_idx + train_window
            test_start = train_end
            test_end = train_end + test_window

            logger.info(
                f"Window {window_count}: train[{train_start}:{train_end}], "
                f"test[{test_start}:{test_end}]"
            )

            # Get train/test data
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]

            # Train model
            self.predictor.train(X_train, y_train)

            # Predict
            y_pred = self.predictor.predict(X_test)
            y_proba = self.predictor.predict_proba(X_test)

            # Store results
            all_predictions.extend(y_pred)
            all_true_labels.extend(y_test.values)
            all_probabilities.extend(y_proba)

            # Get prices
            test_prices = df.iloc[test_start:test_end]['close'].values
            all_prices.extend(test_prices)

            # Move to next window
            start_idx += step_size

        # Convert to arrays
        all_predictions = np.array(all_predictions)
        all_true_labels = np.array(all_true_labels)
        all_probabilities = np.array(all_probabilities)
        all_prices = np.array(all_prices)

        # Calculate overall metrics
        metrics = MetricsCalculator.calculate_all_metrics(
            all_true_labels,
            all_predictions,
            all_probabilities,
            all_prices,
            transaction_cost=0.001,
        )

        # Add metadata
        results = {
            'backtest_type': 'walk_forward',
            'train_window': train_window,
            'test_window': test_window,
            'step_size': step_size,
            'n_windows': window_count,
            'total_predictions': len(all_predictions),
            **metrics,
        }

        self.backtest_results.append(results)

        return results

    def run_rolling_window_backtest(
        self,
        df: pd.DataFrame,
        window_days: int = 30,
        step_days: int = 7,
        train_ratio: float = 0.7,
        orderbook_data: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Run rolling window backtesting (time-based).

        Args:
            df: OHLCV dataframe (must have 'timestamp' column)
            window_days: Window size in days
            step_days: Step size in days
            train_ratio: Proportion of window for training
            orderbook_data: Optional orderbook data
            sentiment_data: Optional sentiment data

        Returns:
            Aggregated backtest results
        """
        logger.info(
            f"Running rolling window backtest: "
            f"window={window_days}d, step={step_days}d"
        )

        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Prepare all data
        X, y = self.predictor.prepare_data(df, orderbook_data, sentiment_data)

        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        all_prices = []

        # Get date range
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()

        window_start = start_date
        window_count = 0

        while window_start + timedelta(days=window_days) <= end_date:
            window_count += 1
            window_end = window_start + timedelta(days=window_days)

            logger.info(f"Window {window_count}: {window_start} to {window_end}")

            # Get window data
            window_mask = (df['timestamp'] >= window_start) & (df['timestamp'] < window_end)
            window_indices = np.where(window_mask)[0]

            if len(window_indices) == 0:
                logger.warning("No data in window, skipping")
                window_start += timedelta(days=step_days)
                continue

            # Split into train/test
            n_window = len(window_indices)
            train_size = int(n_window * train_ratio)

            train_indices = window_indices[:train_size]
            test_indices = window_indices[train_size:]

            if len(test_indices) == 0:
                logger.warning("No test data in window, skipping")
                window_start += timedelta(days=step_days)
                continue

            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_test = y.iloc[test_indices]

            # Train model
            self.predictor.train(X_train, y_train)

            # Predict
            y_pred = self.predictor.predict(X_test)
            y_proba = self.predictor.predict_proba(X_test)

            # Store results
            all_predictions.extend(y_pred)
            all_true_labels.extend(y_test.values)
            all_probabilities.extend(y_proba)

            # Get prices
            test_prices = df.iloc[test_indices]['close'].values
            all_prices.extend(test_prices)

            # Move to next window
            window_start += timedelta(days=step_days)

        # Convert to arrays
        all_predictions = np.array(all_predictions)
        all_true_labels = np.array(all_true_labels)
        all_probabilities = np.array(all_probabilities)
        all_prices = np.array(all_prices)

        # Calculate overall metrics
        metrics = MetricsCalculator.calculate_all_metrics(
            all_true_labels,
            all_predictions,
            all_probabilities,
            all_prices,
            transaction_cost=0.001,
        )

        # Add metadata
        results = {
            'backtest_type': 'rolling_window',
            'window_days': window_days,
            'step_days': step_days,
            'train_ratio': train_ratio,
            'n_windows': window_count,
            'total_predictions': len(all_predictions),
            **metrics,
        }

        self.backtest_results.append(results)

        return results

    def compare_models(
        self,
        df: pd.DataFrame,
        configs: List[Tuple[str, PredictorConfig]],
        train_size: float = 0.7,
    ) -> pd.DataFrame:
        """
        Compare multiple model configurations.

        Args:
            df: OHLCV dataframe
            configs: List of (name, config) tuples
            train_size: Training data proportion

        Returns:
            Comparison results as DataFrame
        """
        logger.info(f"Comparing {len(configs)} model configurations")

        comparison_results = []

        for name, config in configs:
            logger.info(f"Testing configuration: {name}")

            # Create predictor with this config
            predictor = BitcoinPricePredictor(config=config)

            # Prepare data
            X, y = predictor.prepare_data(df)

            # Split
            n = len(X)
            train_end = int(n * train_size)
            X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
            X_test, y_test = X.iloc[train_end:], y.iloc[train_end:]

            # Train
            predictor.train(X_train, y_train)

            # Predict
            y_pred = predictor.predict(X_test)
            y_proba = predictor.predict_proba(X_test)

            # Metrics
            prices = df.iloc[train_end:]['close'].values
            metrics = MetricsCalculator.calculate_all_metrics(
                y_test.values, y_pred, y_proba, prices
            )

            # Store results
            result = {'config_name': name, **metrics}
            comparison_results.append(result)

        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_results)

        # Sort by F1 score
        comparison_df = comparison_df.sort_values('f1_score', ascending=False)

        return comparison_df

    def get_backtest_history(self) -> List[Dict[str, Any]]:
        """Get all backtest results."""
        return self.backtest_results

    def save_results(self, file_path: str) -> None:
        """
        Save backtest results to CSV.

        Args:
            file_path: Output file path
        """
        if not self.backtest_results:
            logger.warning("No backtest results to save")
            return

        df = pd.DataFrame(self.backtest_results)
        df.to_csv(file_path, index=False)
        logger.info(f"Results saved to {file_path}")


# CLI interface for running backtests
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest Bitcoin price prediction model")
    parser.add_argument(
        "--data-source",
        choices=["csv", "synthetic"],
        default="synthetic",
        help="Data source"
    )
    parser.add_argument("--csv-path", type=str, help="Path to CSV file")
    parser.add_argument(
        "--backtest-type",
        choices=["simple", "walk_forward", "rolling"],
        default="simple",
        help="Backtest type"
    )
    parser.add_argument("--train-window", type=int, default=5000, help="Training window size")
    parser.add_argument("--test-window", type=int, default=1000, help="Test window size")
    parser.add_argument("--step-size", type=int, default=500, help="Step size")
    parser.add_argument("--output", type=str, help="Output CSV file path")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load data
    data_loader = DataLoader()

    if args.data_source == "csv":
        if args.csv_path is None:
            raise ValueError("--csv-path required for CSV data source")
        df = data_loader.load_from_csv(args.csv_path)
    else:
        df = data_loader.generate_synthetic_data(n_samples=10000)

    # Create backtester
    backtester = Backtester()

    # Run backtest
    if args.backtest_type == "simple":
        results = backtester.run_backtest(df)
    elif args.backtest_type == "walk_forward":
        results = backtester.run_walk_forward_backtest(
            df,
            train_window=args.train_window,
            test_window=args.test_window,
            step_size=args.step_size,
        )
    else:  # rolling
        results = backtester.run_rolling_window_backtest(df)

    # Print results
    MetricsCalculator.print_metrics_report(results)

    # Save results
    if args.output:
        backtester.save_results(args.output)
