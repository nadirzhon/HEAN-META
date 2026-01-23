"""
Orderbook-based Features for ML

Implements orderbook-related indicators:
- Bid-Ask spread
- Order imbalance
- Liquidity metrics
- Depth analysis
- And more...
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class OrderbookFeatures:
    """Generator for orderbook-based features."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize with configuration."""
        self.config = config or {}
        self.depth_levels = self.config.get('depth_levels', 10)

    def add_orderbook_features(
        self,
        df: pd.DataFrame,
        orderbook_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add all orderbook-based features.

        Args:
            df: Main price dataframe
            orderbook_data: Orderbook snapshots with bid/ask levels
        """
        # Merge orderbook data with price data
        if 'timestamp' in df.columns and 'timestamp' in orderbook_data.columns:
            df = df.merge(orderbook_data, on='timestamp', how='left', suffixes=('', '_ob'))

        df = self._add_spread_features(df)
        df = self._add_imbalance_features(df)
        df = self._add_liquidity_features(df)
        df = self._add_depth_features(df)
        df = self._add_pressure_features(df)

        return df

    def _add_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bid-ask spread features."""
        # Assume orderbook has 'best_bid' and 'best_ask' columns
        if 'best_bid' in df.columns and 'best_ask' in df.columns:
            # Absolute spread
            df['bid_ask_spread'] = df['best_ask'] - df['best_bid']

            # Relative spread (in basis points)
            mid_price = (df['best_bid'] + df['best_ask']) / 2
            df['bid_ask_spread_bps'] = (df['bid_ask_spread'] / mid_price) * 10000

            # Spread moving average
            df['spread_sma_10'] = df['bid_ask_spread'].rolling(window=10).mean()

            # Spread volatility
            df['spread_volatility'] = df['bid_ask_spread'].rolling(window=20).std()

            # Wide spread indicator
            spread_mean = df['bid_ask_spread'].rolling(window=50).mean()
            df['wide_spread'] = (df['bid_ask_spread'] > spread_mean * 1.5).astype(int)

            # Spread change
            df['spread_change'] = df['bid_ask_spread'].pct_change()

        return df

    def _add_imbalance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add order imbalance features."""
        # Assume orderbook has volume at different levels
        if 'bid_volume_total' in df.columns and 'ask_volume_total' in df.columns:
            # Order imbalance ratio
            total_volume = df['bid_volume_total'] + df['ask_volume_total']
            df['order_imbalance'] = (
                (df['bid_volume_total'] - df['ask_volume_total']) /
                (total_volume + 1e-8)
            )

            # Imbalance direction
            df['imbalance_bullish'] = (df['order_imbalance'] > 0).astype(int)

            # Imbalance strength
            df['imbalance_strength'] = abs(df['order_imbalance'])

            # Imbalance moving average
            df['imbalance_sma_5'] = df['order_imbalance'].rolling(window=5).mean()
            df['imbalance_sma_10'] = df['order_imbalance'].rolling(window=10).mean()

            # Imbalance change
            df['imbalance_change'] = df['order_imbalance'].diff()

            # Extreme imbalance
            imbalance_std = df['order_imbalance'].rolling(window=20).std()
            imbalance_mean = df['order_imbalance'].rolling(window=20).mean()
            df['extreme_imbalance'] = (
                abs(df['order_imbalance'] - imbalance_mean) > 2 * imbalance_std
            ).astype(int)

        return df

    def _add_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity-related features."""
        if 'bid_volume_total' in df.columns and 'ask_volume_total' in df.columns:
            # Total liquidity
            df['total_liquidity'] = df['bid_volume_total'] + df['ask_volume_total']

            # Liquidity ratio
            df['liquidity_ratio'] = (
                df['bid_volume_total'] /
                (df['ask_volume_total'] + 1e-8)
            )

            # Liquidity change
            df['liquidity_change'] = df['total_liquidity'].pct_change()

            # Liquidity moving average
            df['liquidity_sma_10'] = df['total_liquidity'].rolling(window=10).mean()

            # Low liquidity indicator
            liquidity_mean = df['total_liquidity'].rolling(window=50).mean()
            df['low_liquidity'] = (
                df['total_liquidity'] < liquidity_mean * 0.5
            ).astype(int)

            # Liquidity concentration (top levels vs all levels)
            if 'bid_volume_l1' in df.columns and 'ask_volume_l1' in df.columns:
                top_level_liquidity = df['bid_volume_l1'] + df['ask_volume_l1']
                df['liquidity_concentration'] = (
                    top_level_liquidity / (df['total_liquidity'] + 1e-8)
                )

        return df

    def _add_depth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market depth features."""
        # Assume we have bid/ask volumes at different levels
        levels_to_check = min(self.depth_levels, 10)

        bid_volumes = []
        ask_volumes = []

        for i in range(1, levels_to_check + 1):
            bid_col = f'bid_volume_l{i}'
            ask_col = f'ask_volume_l{i}'

            if bid_col in df.columns:
                bid_volumes.append(df[bid_col])
            if ask_col in df.columns:
                ask_volumes.append(df[ask_col])

        if bid_volumes and ask_volumes:
            # Cumulative depth
            for i, (bid_vol, ask_vol) in enumerate(zip(bid_volumes, ask_volumes), 1):
                df[f'depth_imbalance_l{i}'] = (
                    (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
                )

            # Average depth imbalance across levels
            depth_cols = [f'depth_imbalance_l{i}' for i in range(1, len(bid_volumes) + 1)]
            if depth_cols:
                df['avg_depth_imbalance'] = df[depth_cols].mean(axis=1)
                df['depth_imbalance_std'] = df[depth_cols].std(axis=1)

        # Depth pressure (weighted by distance from mid price)
        if 'bid_price_l1' in df.columns and 'ask_price_l1' in df.columns:
            mid_price = (df['bid_price_l1'] + df['ask_price_l1']) / 2

            weighted_bid_pressure = 0
            weighted_ask_pressure = 0

            for i in range(1, min(5, levels_to_check) + 1):
                bid_price_col = f'bid_price_l{i}'
                ask_price_col = f'ask_price_l{i}'
                bid_vol_col = f'bid_volume_l{i}'
                ask_vol_col = f'ask_volume_l{i}'

                if all(col in df.columns for col in [
                    bid_price_col, ask_price_col, bid_vol_col, ask_vol_col
                ]):
                    # Weight by inverse distance from mid price
                    bid_weight = 1 / (abs(mid_price - df[bid_price_col]) + 1e-8)
                    ask_weight = 1 / (abs(mid_price - df[ask_price_col]) + 1e-8)

                    weighted_bid_pressure += df[bid_vol_col] * bid_weight
                    weighted_ask_pressure += df[ask_vol_col] * ask_weight

            df['weighted_depth_pressure'] = (
                (weighted_bid_pressure - weighted_ask_pressure) /
                (weighted_bid_pressure + weighted_ask_pressure + 1e-8)
            )

        return df

    def _add_pressure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add buying/selling pressure features."""
        if 'bid_volume_total' in df.columns and 'ask_volume_total' in df.columns:
            # Buying pressure (bid volume normalized)
            total_volume = df['bid_volume_total'] + df['ask_volume_total']
            df['buying_pressure'] = df['bid_volume_total'] / (total_volume + 1e-8)

            # Selling pressure (ask volume normalized)
            df['selling_pressure'] = df['ask_volume_total'] / (total_volume + 1e-8)

            # Net pressure
            df['net_pressure'] = df['buying_pressure'] - df['selling_pressure']

            # Pressure momentum
            df['pressure_momentum_5'] = df['net_pressure'].diff(5)

            # Pressure acceleration
            df['pressure_acceleration'] = df['pressure_momentum_5'].diff()

            # Extreme pressure
            pressure_mean = df['net_pressure'].rolling(window=20).mean()
            pressure_std = df['net_pressure'].rolling(window=20).std()
            df['extreme_buy_pressure'] = (
                df['net_pressure'] > pressure_mean + 2 * pressure_std
            ).astype(int)
            df['extreme_sell_pressure'] = (
                df['net_pressure'] < pressure_mean - 2 * pressure_std
            ).astype(int)

        return df

    def create_synthetic_orderbook(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic orderbook features when real orderbook data is not available.

        Uses price and volume data to estimate orderbook characteristics.
        """
        # Estimate bid-ask spread from high-low range
        df['best_bid'] = df['low'] * 0.9995  # Estimate
        df['best_ask'] = df['high'] * 1.0005  # Estimate

        # Estimate volumes based on actual volume and price movement
        df['bid_volume_total'] = df['volume'] * (1 + ((df['close'] - df['open']) / df['open']))
        df['ask_volume_total'] = df['volume'] * (1 - ((df['close'] - df['open']) / df['open']))

        # Ensure non-negative volumes
        df['bid_volume_total'] = df['bid_volume_total'].clip(lower=0)
        df['ask_volume_total'] = df['ask_volume_total'].clip(lower=0)

        # Add basic orderbook features
        df = self._add_spread_features(df)
        df = self._add_imbalance_features(df)
        df = self._add_liquidity_features(df)

        return df
