"""
Data models for ML price prediction
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PredictionDirection(str, Enum):
    """Predicted price direction"""
    STRONG_UP = "strong_up"      # >3% expected
    UP = "up"                     # 1-3% expected
    NEUTRAL = "neutral"           # -1% to +1%
    DOWN = "down"                 # -3% to -1%
    STRONG_DOWN = "strong_down"  # <-3% expected


@dataclass
class PricePrediction:
    """Price prediction result"""

    symbol: str
    current_price: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Predictions for different timeframes
    price_1h: float | None = None
    price_4h: float | None = None
    price_24h: float | None = None

    # Direction predictions
    direction_1h: PredictionDirection | None = None
    direction_4h: PredictionDirection | None = None
    direction_24h: PredictionDirection | None = None

    # Confidence scores (0-1)
    confidence_1h: float = 0.0
    confidence_4h: float = 0.0
    confidence_24h: float = 0.0

    # Expected returns (%)
    expected_return_1h: float = 0.0
    expected_return_4h: float = 0.0
    expected_return_24h: float = 0.0

    # Model metadata
    model_version: str = "unknown"
    features_used: list[str] = field(default_factory=list)

    @property
    def best_timeframe(self) -> tuple[str, float, PredictionDirection]:
        """Get timeframe with highest confidence"""
        timeframes = [
            ("1h", self.confidence_1h, self.direction_1h),
            ("4h", self.confidence_4h, self.direction_4h),
            ("24h", self.confidence_24h, self.direction_24h)
        ]
        # Filter out None directions
        valid = [(tf, conf, dir) for tf, conf, dir in timeframes if dir is not None]
        if not valid:
            return ("1h", 0.0, PredictionDirection.NEUTRAL)
        return max(valid, key=lambda x: x[1])

    @property
    def should_trade(self) -> bool:
        """Should we trade based on this prediction?"""
        tf, conf, direction = self.best_timeframe
        # Trade if:
        # 1. Confidence > 70%
        # 2. Direction is not neutral
        # 3. Expected return > 2%
        if conf < 0.7:
            return False
        if direction == PredictionDirection.NEUTRAL:
            return False

        # Check expected return for best timeframe
        if tf == "1h" and abs(self.expected_return_1h) < 2.0:
            return False
        elif tf == "4h" and abs(self.expected_return_4h) < 2.0:
            return False
        elif tf == "24h" and abs(self.expected_return_24h) < 2.0:
            return False

        return True

    @property
    def is_bullish(self) -> bool:
        """Is prediction bullish?"""
        tf, conf, direction = self.best_timeframe
        return direction in [PredictionDirection.UP, PredictionDirection.STRONG_UP]

    @property
    def is_bearish(self) -> bool:
        """Is prediction bearish?"""
        tf, conf, direction = self.best_timeframe
        return direction in [PredictionDirection.DOWN, PredictionDirection.STRONG_DOWN]


@dataclass
class ModelMetrics:
    """Model performance metrics"""

    # Accuracy metrics
    direction_accuracy: float  # % correct direction predictions
    mae: float  # Mean Absolute Error (price)
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error

    # Trading metrics (if backtested)
    win_rate: float | None = None
    profit_factor: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None

    # Timeframe-specific metrics
    accuracy_1h: float | None = None
    accuracy_4h: float | None = None
    accuracy_24h: float | None = None

    # Training info
    training_samples: int = 0
    validation_samples: int = 0
    epochs: int = 0
    training_time_seconds: float = 0.0

    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_good_model(self) -> bool:
        """Is this a good model for trading?"""
        # Requirements:
        # 1. Direction accuracy > 60%
        # 2. MAPE < 5%
        # 3. (If available) Win rate > 55%
        if self.direction_accuracy < 0.6:
            return False
        if self.mape > 5.0:
            return False
        if self.win_rate is not None and self.win_rate < 0.55:
            return False
        return True


@dataclass
class TrainingConfig:
    """Configuration for model training"""

    # Data parameters
    lookback_periods: int = 60  # How many periods to look back
    prediction_horizons: list[int] = field(default_factory=lambda: [1, 4, 24])  # Hours ahead

    # Model architecture
    lstm_units: list[int] = field(default_factory=lambda: [128, 64, 32])  # LSTM layers
    dropout_rate: float = 0.2
    learning_rate: float = 0.001

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 10

    # Features
    use_technical_indicators: bool = True
    use_sentiment: bool = True
    use_google_trends: bool = True
    use_funding_rates: bool = True

    # Data preprocessing
    normalize_features: bool = True
    handle_missing: str = "forward_fill"  # or "drop", "interpolate"

    # Training data
    train_start_date: datetime | None = None
    train_end_date: datetime | None = None

    # Model saving
    model_save_path: str = "models/"
    checkpoint_frequency: int = 10  # Save every N epochs


@dataclass
class FeatureSet:
    """Set of features for prediction"""

    # Raw price data
    open_prices: list[float]
    high_prices: list[float]
    low_prices: list[float]
    close_prices: list[float]
    volumes: list[float]

    # Technical indicators
    rsi: list[float] | None = None
    macd: list[float] | None = None
    macd_signal: list[float] | None = None
    bollinger_upper: list[float] | None = None
    bollinger_lower: list[float] | None = None
    sma_20: list[float] | None = None
    sma_50: list[float] | None = None
    ema_12: list[float] | None = None
    ema_26: list[float] | None = None
    atr: list[float] | None = None
    obv: list[float] | None = None

    # External data
    sentiment_scores: list[float] | None = None
    google_trends: list[float] | None = None
    funding_rates: list[float] | None = None

    # Timestamps
    timestamps: list[datetime] = field(default_factory=list)

    def to_numpy_array(self):  # -> numpy.ndarray:
        """Convert to numpy array for model input"""
        import numpy as np

        features = [
            self.open_prices,
            self.high_prices,
            self.low_prices,
            self.close_prices,
            self.volumes,
        ]

        # Add technical indicators if available
        if self.rsi:
            features.append(self.rsi)
        if self.macd:
            features.append(self.macd)
        if self.macd_signal:
            features.append(self.macd_signal)
        if self.bollinger_upper:
            features.append(self.bollinger_upper)
        if self.bollinger_lower:
            features.append(self.bollinger_lower)
        if self.sma_20:
            features.append(self.sma_20)
        if self.sma_50:
            features.append(self.sma_50)
        if self.ema_12:
            features.append(self.ema_12)
        if self.ema_26:
            features.append(self.ema_26)
        if self.atr:
            features.append(self.atr)
        if self.obv:
            features.append(self.obv)

        # Add external data if available
        if self.sentiment_scores:
            features.append(self.sentiment_scores)
        if self.google_trends:
            features.append(self.google_trends)
        if self.funding_rates:
            features.append(self.funding_rates)

        # Stack features (n_features, n_timesteps)
        return np.array(features).T  # Transpose to (n_timesteps, n_features)

    @property
    def n_features(self) -> int:
        """Number of features"""
        count = 5  # OHLCV
        if self.rsi:
            count += 1
        if self.macd:
            count += 1
        if self.macd_signal:
            count += 1
        if self.bollinger_upper:
            count += 1
        if self.bollinger_lower:
            count += 1
        if self.sma_20:
            count += 1
        if self.sma_50:
            count += 1
        if self.ema_12:
            count += 1
        if self.ema_26:
            count += 1
        if self.atr:
            count += 1
        if self.obv:
            count += 1
        if self.sentiment_scores:
            count += 1
        if self.google_trends:
            count += 1
        if self.funding_rates:
            count += 1
        return count


@dataclass
class BacktestResult:
    """Backtest results for ML model"""

    symbol: str
    start_date: datetime
    end_date: datetime

    # Performance
    total_return: float  # %
    annual_return: float  # %
    sharpe_ratio: float
    max_drawdown: float  # %

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float  # %
    avg_loss: float  # %
    profit_factor: float

    # Prediction accuracy
    direction_accuracy: float
    mae: float
    mse: float

    # By timeframe
    trades_1h: int = 0
    trades_4h: int = 0
    trades_24h: int = 0

    @property
    def is_profitable(self) -> bool:
        """Is the backtest profitable?"""
        return self.total_return > 0 and self.sharpe_ratio > 1.0 and self.win_rate > 0.5
