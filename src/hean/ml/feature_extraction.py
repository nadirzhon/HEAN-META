"""ML Feature Extraction for Signal Quality Assessment.

Extracts features from market data and signals to predict signal quality.
Supports both real-time feature extraction and batch processing.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from hean.core.types import Signal
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MarketFeatures:
    """Extracted market features for ML model."""

    # Price features
    price_momentum_5m: float = 0.0
    price_momentum_15m: float = 0.0
    price_momentum_1h: float = 0.0
    price_volatility_5m: float = 0.0
    price_volatility_15m: float = 0.0

    # Volume features
    volume_ratio_5m: float = 1.0  # Current vs 5min average
    volume_trend: float = 0.0  # Increasing or decreasing
    volume_spike: bool = False  # Volume > 2x average

    # Orderbook features
    bid_ask_spread_bps: float = 0.0
    orderbook_imbalance: float = 0.0  # -1 to 1
    depth_ratio: float = 1.0  # Bid depth / ask depth

    # Microstructure features
    tick_direction: int = 0  # Last tick direction (+1, 0, -1)
    aggressive_buy_ratio: float = 0.5  # Ratio of aggressive buys
    trade_intensity: float = 0.0  # Trades per minute

    # Temporal features
    hour_of_day: int = 0  # 0-23
    is_high_liquidity_hour: bool = False
    minutes_to_funding: float = 480.0  # Minutes to next funding

    # Regime features
    regime_impulse: bool = False
    regime_range: bool = False
    regime_normal: bool = True
    volatility_percentile: float = 50.0

    # Signal features
    signal_confidence: float = 0.5
    signal_strength: float = 0.0  # Distance from entry to TP
    risk_reward_ratio: float = 1.0  # TP distance / SL distance

    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML model.

        Returns:
            Feature array of shape (n_features,)
        """
        return np.array([
            # Price (5 features)
            self.price_momentum_5m,
            self.price_momentum_15m,
            self.price_momentum_1h,
            self.price_volatility_5m,
            self.price_volatility_15m,

            # Volume (4 features)
            self.volume_ratio_5m,
            self.volume_trend,
            float(self.volume_spike),
            self.trade_intensity,

            # Orderbook (3 features)
            self.bid_ask_spread_bps,
            self.orderbook_imbalance,
            self.depth_ratio,

            # Microstructure (3 features)
            float(self.tick_direction),
            self.aggressive_buy_ratio,
            self.trade_intensity,

            # Temporal (3 features)
            self.hour_of_day / 24.0,  # Normalize to [0, 1]
            float(self.is_high_liquidity_hour),
            self.minutes_to_funding / 480.0,  # Normalize to [0, 1]

            # Regime (4 features)
            float(self.regime_impulse),
            float(self.regime_range),
            float(self.regime_normal),
            self.volatility_percentile / 100.0,  # Normalize to [0, 1]

            # Signal (3 features)
            self.signal_confidence,
            self.signal_strength,
            self.risk_reward_ratio,
        ])

    @classmethod
    def get_feature_names(cls) -> list[str]:
        """Get feature names for interpretation."""
        return [
            "price_momentum_5m", "price_momentum_15m", "price_momentum_1h",
            "price_volatility_5m", "price_volatility_15m",
            "volume_ratio_5m", "volume_trend", "volume_spike", "trade_intensity",
            "bid_ask_spread_bps", "orderbook_imbalance", "depth_ratio",
            "tick_direction", "aggressive_buy_ratio", "trade_intensity_micro",
            "hour_of_day_norm", "is_high_liquidity_hour", "minutes_to_funding_norm",
            "regime_impulse", "regime_range", "regime_normal", "volatility_percentile_norm",
            "signal_confidence", "signal_strength", "risk_reward_ratio",
        ]


class FeatureExtractor:
    """Extracts ML features from market data and signals.

    Maintains rolling windows of market data to compute features
    in real-time.
    """

    # High liquidity hours (UTC) - same as OrderTimingOptimizer
    HIGH_LIQUIDITY_HOURS = [
        (7, 10),   # London open
        (13, 16),  # London/NY overlap
        (20, 23),  # Asia open
    ]

    # Funding times (UTC)
    FUNDING_TIMES = [0, 8, 16]

    def __init__(self, window_size: int = 100):
        """Initialize feature extractor.

        Args:
            window_size: Size of rolling windows for historical data
        """
        self._window_size = window_size

        # Price history per symbol
        self._price_history: dict[str, deque] = {}
        self._price_timestamps: dict[str, deque] = {}

        # Volume history per symbol
        self._volume_history: dict[str, deque] = {}

        # Tick data
        self._last_tick_direction: dict[str, int] = {}

        # Orderbook snapshots
        self._bid_history: dict[str, deque] = {}
        self._ask_history: dict[str, deque] = {}

        # Trade flow
        self._aggressive_buys: dict[str, deque] = {}
        self._trade_timestamps: dict[str, deque] = {}

        logger.info(f"FeatureExtractor initialized with window_size={window_size}")

    def update_price(self, symbol: str, price: float, timestamp: datetime | None = None) -> None:
        """Update price history.

        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Price timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self._window_size)
            self._price_timestamps[symbol] = deque(maxlen=self._window_size)

        self._price_history[symbol].append(price)
        self._price_timestamps[symbol].append(timestamp)

        # Update tick direction
        if len(self._price_history[symbol]) >= 2:
            prev_price = self._price_history[symbol][-2]
            if price > prev_price:
                self._last_tick_direction[symbol] = 1
            elif price < prev_price:
                self._last_tick_direction[symbol] = -1
            else:
                self._last_tick_direction[symbol] = 0

    def update_volume(self, symbol: str, volume: float) -> None:
        """Update volume history.

        Args:
            symbol: Trading symbol
            volume: Current volume
        """
        if symbol not in self._volume_history:
            self._volume_history[symbol] = deque(maxlen=self._window_size)

        self._volume_history[symbol].append(volume)

    def update_orderbook(self, symbol: str, bid: float, ask: float) -> None:
        """Update orderbook history.

        Args:
            symbol: Trading symbol
            bid: Best bid price
            ask: Best ask price
        """
        if symbol not in self._bid_history:
            self._bid_history[symbol] = deque(maxlen=self._window_size)
            self._ask_history[symbol] = deque(maxlen=self._window_size)

        self._bid_history[symbol].append(bid)
        self._ask_history[symbol].append(ask)

    def update_trade(self, symbol: str, is_aggressive_buy: bool, timestamp: datetime | None = None) -> None:
        """Update trade flow data.

        Args:
            symbol: Trading symbol
            is_aggressive_buy: Whether trade was an aggressive buy
            timestamp: Trade timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        if symbol not in self._aggressive_buys:
            self._aggressive_buys[symbol] = deque(maxlen=self._window_size)
            self._trade_timestamps[symbol] = deque(maxlen=self._window_size)

        self._aggressive_buys[symbol].append(is_aggressive_buy)
        self._trade_timestamps[symbol].append(timestamp)

    def extract_features(
        self,
        signal: Signal,
        context: dict[str, Any],
    ) -> MarketFeatures:
        """Extract features for a signal.

        Args:
            signal: Trading signal
            context: Context with market data

        Returns:
            MarketFeatures instance
        """
        symbol = signal.symbol
        now = datetime.utcnow()

        features = MarketFeatures()

        # Extract price features
        if symbol in self._price_history:
            self._extract_price_features(symbol, features, now)

        # Extract volume features
        if symbol in self._volume_history:
            self._extract_volume_features(symbol, features)

        # Extract orderbook features
        if symbol in self._bid_history and symbol in self._ask_history:
            self._extract_orderbook_features(symbol, features, signal.entry_price)

        # Extract microstructure features
        self._extract_microstructure_features(symbol, features, now)

        # Extract temporal features
        self._extract_temporal_features(features, now)

        # Extract regime features
        self._extract_regime_features(features, context)

        # Extract signal features
        self._extract_signal_features(signal, features)

        return features

    def _extract_price_features(self, symbol: str, features: MarketFeatures, now: datetime) -> None:
        """Extract price-based features."""
        prices = list(self._price_history[symbol])
        timestamps = list(self._price_timestamps[symbol])

        if len(prices) < 2:
            return

        # current_price removed - unused

        # Price momentum at different timeframes
        features.price_momentum_5m = self._calculate_momentum(
            prices, timestamps, now, minutes=5
        )
        features.price_momentum_15m = self._calculate_momentum(
            prices, timestamps, now, minutes=15
        )
        features.price_momentum_1h = self._calculate_momentum(
            prices, timestamps, now, minutes=60
        )

        # Price volatility
        features.price_volatility_5m = self._calculate_volatility(
            prices, timestamps, now, minutes=5
        )
        features.price_volatility_15m = self._calculate_volatility(
            prices, timestamps, now, minutes=15
        )

    def _extract_volume_features(self, symbol: str, features: MarketFeatures) -> None:
        """Extract volume-based features."""
        volumes = list(self._volume_history[symbol])

        if len(volumes) < 2:
            return

        current_volume = volumes[-1]
        avg_volume = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else current_volume

        # Volume ratio
        if avg_volume > 0:
            features.volume_ratio_5m = current_volume / avg_volume
            features.volume_spike = current_volume > (avg_volume * 2.0)

        # Volume trend (simple linear regression slope)
        if len(volumes) >= 5:
            recent_volumes = volumes[-5:]
            x = np.arange(len(recent_volumes))
            coeffs = np.polyfit(x, recent_volumes, 1)
            features.volume_trend = coeffs[0]  # Slope

    def _extract_orderbook_features(
        self, symbol: str, features: MarketFeatures, entry_price: float
    ) -> None:
        """Extract orderbook-based features."""
        bids = list(self._bid_history[symbol])
        asks = list(self._ask_history[symbol])

        if not bids or not asks:
            return

        current_bid = bids[-1]
        current_ask = asks[-1]

        # Bid-ask spread
        if current_bid > 0:
            spread_pct = (current_ask - current_bid) / current_bid
            features.bid_ask_spread_bps = spread_pct * 10000

        # Orderbook imbalance (simplified - would need depth data for real imbalance)
        # For now, use price position within spread
        if current_ask > current_bid:
            spread = current_ask - current_bid
            if spread > 0:
                position_in_spread = (entry_price - current_bid) / spread
                # Convert to [-1, 1] where -1 = at bid, 1 = at ask
                features.orderbook_imbalance = (position_in_spread * 2) - 1

    def _extract_microstructure_features(
        self, symbol: str, features: MarketFeatures, now: datetime
    ) -> None:
        """Extract microstructure features."""
        # Tick direction
        features.tick_direction = self._last_tick_direction.get(symbol, 0)

        # Aggressive buy ratio
        if symbol in self._aggressive_buys and len(self._aggressive_buys[symbol]) > 0:
            aggressive_buys = list(self._aggressive_buys[symbol])
            features.aggressive_buy_ratio = sum(aggressive_buys) / len(aggressive_buys)

        # Trade intensity (trades per minute)
        if symbol in self._trade_timestamps and len(self._trade_timestamps[symbol]) >= 2:
            timestamps = list(self._trade_timestamps[symbol])
            # Count trades in last minute
            cutoff = now - timedelta(minutes=1)
            recent_trades = sum(1 for ts in timestamps if ts >= cutoff)
            features.trade_intensity = float(recent_trades)

    def _extract_temporal_features(self, features: MarketFeatures, now: datetime) -> None:
        """Extract temporal features."""
        features.hour_of_day = now.hour

        # Check if in high liquidity window
        for start, end in self.HIGH_LIQUIDITY_HOURS:
            if start <= now.hour < end:
                features.is_high_liquidity_hour = True
                break

        # Minutes to next funding
        next_funding = self._next_funding_time(now)
        if next_funding:
            features.minutes_to_funding = (next_funding - now).total_seconds() / 60.0

    def _extract_regime_features(self, features: MarketFeatures, context: dict[str, Any]) -> None:
        """Extract regime features from context."""
        regime = context.get("regime")
        if regime:
            regime_str = regime.value if hasattr(regime, "value") else str(regime)
            features.regime_impulse = regime_str == "IMPULSE"
            features.regime_range = regime_str == "RANGE"
            features.regime_normal = regime_str == "NORMAL"

        # Volatility percentile
        vol_percentile = context.get("volatility_percentile", 50.0)
        features.volatility_percentile = float(vol_percentile)

    def _extract_signal_features(self, signal: Signal, features: MarketFeatures) -> None:
        """Extract signal-specific features."""
        # Signal strength (distance to TP as % of entry)
        if signal.take_profit:
            if signal.side == "buy":
                features.signal_strength = (signal.take_profit - signal.entry_price) / signal.entry_price
            else:
                features.signal_strength = (signal.entry_price - signal.take_profit) / signal.entry_price

        # Risk/reward ratio
        if signal.stop_loss and signal.take_profit:
            if signal.side == "buy":
                risk = signal.entry_price - signal.stop_loss
                reward = signal.take_profit - signal.entry_price
            else:
                risk = signal.stop_loss - signal.entry_price
                reward = signal.entry_price - signal.take_profit

            if risk > 0:
                features.risk_reward_ratio = reward / risk

    def _calculate_momentum(
        self,
        prices: list[float],
        timestamps: list[datetime],
        now: datetime,
        minutes: int,
    ) -> float:
        """Calculate price momentum over a time window.

        Args:
            prices: Price history
            timestamps: Price timestamps
            now: Current time
            minutes: Lookback window in minutes

        Returns:
            Momentum as percentage change
        """
        cutoff = now - timedelta(minutes=minutes)

        # Find prices within window
        window_prices = [
            p for p, ts in zip(prices, timestamps, strict=False)
            if ts >= cutoff
        ]

        if len(window_prices) < 2:
            return 0.0

        start_price = window_prices[0]
        end_price = window_prices[-1]

        if start_price == 0:
            return 0.0

        return (end_price - start_price) / start_price

    def _calculate_volatility(
        self,
        prices: list[float],
        timestamps: list[datetime],
        now: datetime,
        minutes: int,
    ) -> float:
        """Calculate price volatility over a time window.

        Args:
            prices: Price history
            timestamps: Price timestamps
            now: Current time
            minutes: Lookback window in minutes

        Returns:
            Volatility as standard deviation of returns
        """
        cutoff = now - timedelta(minutes=minutes)

        # Find prices within window
        window_prices = [
            p for p, ts in zip(prices, timestamps, strict=False)
            if ts >= cutoff
        ]

        if len(window_prices) < 2:
            return 0.0

        # Calculate returns
        returns = []
        for i in range(1, len(window_prices)):
            if window_prices[i - 1] > 0:
                ret = (window_prices[i] - window_prices[i - 1]) / window_prices[i - 1]
                returns.append(ret)

        if not returns:
            return 0.0

        return float(np.std(returns))

    def _next_funding_time(self, now: datetime) -> datetime | None:
        """Calculate next funding time.

        Args:
            now: Current time

        Returns:
            Next funding datetime
        """
        for hour in self.FUNDING_TIMES:
            funding_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if funding_time > now:
                return funding_time

        # Next day's first funding
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=self.FUNDING_TIMES[0], minute=0, second=0, microsecond=0)
