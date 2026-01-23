"""
Data Splitting Utilities

Provides different methods for splitting time series data:
- Time-based split (recommended for time series)
- Random split
- K-fold cross-validation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List


class DataSplitter:
    """Utility class for splitting data into train/val/test sets."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data splitter.

        Args:
            config: Configuration with split ratios
        """
        self.config = config or {}

        # Default split ratios
        self.train_ratio = self.config.get('train_ratio', 0.7)
        self.val_ratio = self.config.get('val_ratio', 0.15)
        self.test_ratio = self.config.get('test_ratio', 0.15)

        # Validate ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio}"
            )

    def time_series_split(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically (recommended for time series).

        Maintains temporal order: train -> validation -> test

        Args:
            df: DataFrame to split

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(df)

        # Calculate split indices
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        # Split data
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        return train_df, val_df, test_df

    def random_split(
        self,
        df: pd.DataFrame,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Randomly split data (not recommended for time series).

        Args:
            df: DataFrame to split
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Shuffle data
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Use time series split on shuffled data
        return self.time_series_split(df_shuffled)

    def walk_forward_split(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        gap: int = 0
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Walk-forward (expanding window) cross-validation.

        Each fold uses all previous data for training and next chunk for validation.

        Args:
            df: DataFrame to split
            n_splits: Number of splits
            gap: Gap between train and validation (in samples)

        Returns:
            List of (train_df, val_df) tuples
        """
        n = len(df)
        val_size = n // (n_splits + 1)

        splits = []

        for i in range(1, n_splits + 1):
            train_end = val_size * i - gap
            val_start = val_size * i
            val_end = val_size * (i + 1)

            if train_end <= 0 or val_end > n:
                continue

            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[val_start:val_end].copy()

            splits.append((train_df, val_df))

        return splits

    def sliding_window_split(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        window_size: Optional[int] = None,
        gap: int = 0
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Sliding window cross-validation.

        Uses a fixed-size training window that slides forward.

        Args:
            df: DataFrame to split
            n_splits: Number of splits
            window_size: Size of training window (if None, uses auto-calculated size)
            gap: Gap between train and validation

        Returns:
            List of (train_df, val_df) tuples
        """
        n = len(df)

        # Auto-calculate window size if not provided
        if window_size is None:
            window_size = n // (n_splits + 1)

        val_size = n // (n_splits + 2)

        splits = []

        for i in range(n_splits):
            val_start = window_size + (val_size * i) + gap
            val_end = val_start + val_size

            if val_end > n:
                break

            train_start = val_start - window_size - gap
            train_end = val_start - gap

            if train_start < 0:
                continue

            train_df = df.iloc[train_start:train_end].copy()
            val_df = df.iloc[val_start:val_end].copy()

            splits.append((train_df, val_df))

        return splits

    def stratified_split(
        self,
        df: pd.DataFrame,
        target_col: str = 'target',
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified split maintaining class distribution.

        Not ideal for time series but useful when class imbalance is severe.

        Args:
            df: DataFrame to split
            target_col: Name of target column
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split

        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_ratio,
            stratify=df[target_col],
            random_state=random_state
        )

        # Second split: train vs val
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            stratify=train_val_df[target_col],
            random_state=random_state
        )

        return train_df, val_df, test_df
