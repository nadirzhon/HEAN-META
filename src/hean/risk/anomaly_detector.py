"""
Market Anomaly Detection using Isolation Forest.

Detects unusual market conditions that may indicate:
- Flash crashes or spikes
- Unusual volume patterns
- Liquidity crises
- Market manipulation attempts

Used by RiskGovernor to trigger protective measures.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from hean.logging import get_logger

logger = get_logger(__name__)

# Try to import sklearn, fallback to simple detection if not available
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available, using simple anomaly detection")


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""

    is_anomaly: bool
    anomaly_score: float  # -1 (most anomalous) to 1 (most normal)
    confidence: float  # 0 to 1
    anomaly_type: str | None  # Type of anomaly if detected
    features: dict[str, float]  # Feature values used for detection
    timestamp: datetime


class MarketAnomalyDetector:
    """
    Detect market anomalies using Isolation Forest algorithm.

    Monitors multiple features:
    - Price volatility (short and long term)
    - Volume spikes
    - Spread widening
    - Order flow imbalance
    - Return distribution (fat tails)
    """

    FEATURE_NAMES = [
        "volatility_ratio",      # Short-term vol / long-term vol
        "volume_spike",          # Current volume / average volume
        "spread_percentile",     # Current spread as percentile
        "return_zscore",         # Current return in z-scores
        "imbalance_extreme",     # Order flow imbalance extremity
        "momentum_divergence",   # Short vs long momentum difference
        "price_velocity",        # Rate of price change
        "liquidity_score",       # Depth-weighted liquidity
    ]

    def __init__(
        self,
        contamination: float = 0.01,
        history_size: int = 10000,
        min_samples_for_detection: int = 100,
        anomaly_threshold: float = -0.5,
    ):
        """
        Initialize the anomaly detector.

        Args:
            contamination: Expected proportion of anomalies (default 1%)
            history_size: Maximum feature history to keep
            min_samples_for_detection: Minimum samples before detection is active
            anomaly_threshold: Threshold for anomaly score (lower = more strict)
        """
        self._contamination = contamination
        self._history_size = history_size
        self._min_samples = min_samples_for_detection
        self._anomaly_threshold = anomaly_threshold

        # Feature history per symbol
        self._feature_history: dict[str, deque] = {}
        self._price_history: dict[str, deque] = {}
        self._volume_history: dict[str, deque] = {}
        self._spread_history: dict[str, deque] = {}
        self._return_history: dict[str, deque] = {}

        # Isolation Forest models per symbol
        self._models: dict[str, object] = {}
        self._scalers: dict[str, object] = {}
        self._last_retrain: dict[str, datetime] = {}

        # Anomaly tracking
        self._recent_anomalies: deque = deque(maxlen=100)
        self._anomaly_count = 0
        self._total_checks = 0

        # Retrain interval
        self._retrain_interval_seconds = 3600  # 1 hour

    def update(
        self,
        symbol: str,
        price: float,
        volume: float,
        bid: float,
        ask: float,
        buy_volume: float = 0.0,
        sell_volume: float = 0.0,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Update detector with new market data.

        Args:
            symbol: Trading symbol
            price: Current price
            volume: Trade volume
            bid: Best bid price
            ask: Best ask price
            buy_volume: Aggressive buy volume
            sell_volume: Aggressive sell volume
            timestamp: Tick timestamp
        """
        timestamp = timestamp or datetime.utcnow()

        # Initialize histories if needed
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self._history_size)
            self._volume_history[symbol] = deque(maxlen=self._history_size)
            self._spread_history[symbol] = deque(maxlen=self._history_size)
            self._return_history[symbol] = deque(maxlen=self._history_size)
            self._feature_history[symbol] = deque(maxlen=self._history_size)

        # Calculate return
        if len(self._price_history[symbol]) > 0:
            last_price = self._price_history[symbol][-1]
            if last_price > 0:
                ret = (price - last_price) / last_price
                self._return_history[symbol].append(ret)

        # Update histories
        self._price_history[symbol].append(price)
        self._volume_history[symbol].append(volume)
        spread = (ask - bid) / price if price > 0 else 0.0
        self._spread_history[symbol].append(spread)

        # Calculate and store features
        features = self._calculate_features(
            symbol, price, volume, bid, ask, buy_volume, sell_volume
        )
        if features is not None:
            self._feature_history[symbol].append(features)

    def _calculate_features(
        self,
        symbol: str,
        price: float,
        volume: float,
        bid: float,
        ask: float,
        buy_volume: float,
        sell_volume: float,
    ) -> np.ndarray | None:
        """Calculate feature vector for anomaly detection."""
        prices = list(self._price_history[symbol])
        volumes = list(self._volume_history[symbol])
        spreads = list(self._spread_history[symbol])
        returns = list(self._return_history[symbol])

        if len(prices) < 50 or len(returns) < 20:
            return None

        # 1. Volatility ratio (short/long)
        short_returns = returns[-10:] if len(returns) >= 10 else returns
        long_returns = returns[-50:] if len(returns) >= 50 else returns
        short_vol = np.std(short_returns) if short_returns else 0.001
        long_vol = np.std(long_returns) if long_returns else 0.001
        volatility_ratio = short_vol / max(long_vol, 1e-8)

        # 2. Volume spike
        avg_volume = np.mean(volumes[-50:]) if len(volumes) >= 50 else np.mean(volumes)
        volume_spike = volume / max(avg_volume, 1e-8)

        # 3. Spread percentile
        spread = (ask - bid) / price if price > 0 else 0.0
        spread_percentile = (
            np.percentile(spreads, [np.searchsorted(np.sort(spreads), spread) / len(spreads) * 100])[0]
            if spreads else 50.0
        )
        spread_percentile = spread_percentile / 100.0  # Normalize to 0-1

        # 4. Return z-score
        if returns:
            current_return = returns[-1] if returns else 0.0
            mean_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else 0.001
            return_zscore = (current_return - mean_return) / max(std_return, 1e-8)
        else:
            return_zscore = 0.0

        # 5. Order flow imbalance extremity
        total_flow = buy_volume + sell_volume
        if total_flow > 0:
            imbalance = (buy_volume - sell_volume) / total_flow
            imbalance_extreme = abs(imbalance)
        else:
            imbalance_extreme = 0.0

        # 6. Momentum divergence
        short_momentum = (price - prices[-10]) / prices[-10] if len(prices) >= 10 and prices[-10] > 0 else 0.0
        long_momentum = (price - prices[-50]) / prices[-50] if len(prices) >= 50 and prices[-50] > 0 else 0.0
        momentum_divergence = abs(short_momentum - long_momentum)

        # 7. Price velocity (acceleration)
        if len(returns) >= 5:
            recent_returns = returns[-5:]
            price_velocity = abs(np.mean(recent_returns) / max(np.std(recent_returns), 1e-8))
        else:
            price_velocity = 0.0

        # 8. Liquidity score (inverse of spread * volume)
        liquidity_score = 1.0 / max(spread * max(volume, 1.0), 1e-8)
        liquidity_score = min(liquidity_score, 100.0)  # Cap extreme values

        features = np.array([
            volatility_ratio,
            volume_spike,
            spread_percentile,
            return_zscore,
            imbalance_extreme,
            momentum_divergence,
            price_velocity,
            liquidity_score,
        ], dtype=np.float32)

        return features

    def is_anomaly(self, symbol: str) -> AnomalyResult:
        """
        Check if current market state is anomalous.

        Args:
            symbol: Trading symbol

        Returns:
            AnomalyResult with detection details
        """
        self._total_checks += 1
        timestamp = datetime.utcnow()

        # Check if we have enough data
        if symbol not in self._feature_history:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                anomaly_type=None,
                features={},
                timestamp=timestamp,
            )

        features = list(self._feature_history[symbol])
        if len(features) < self._min_samples:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                anomaly_type="insufficient_data",
                features={},
                timestamp=timestamp,
            )

        current_features = features[-1]

        if SKLEARN_AVAILABLE:
            result = self._detect_with_isolation_forest(symbol, current_features, timestamp)
        else:
            result = self._detect_simple(symbol, current_features, timestamp)

        if result.is_anomaly:
            self._anomaly_count += 1
            self._recent_anomalies.append(result)
            logger.warning(
                f"[ANOMALY DETECTED] {symbol}: score={result.anomaly_score:.3f}, "
                f"type={result.anomaly_type}, confidence={result.confidence:.2f}"
            )

        return result

    def _detect_with_isolation_forest(
        self,
        symbol: str,
        current_features: np.ndarray,
        timestamp: datetime,
    ) -> AnomalyResult:
        """Detect anomaly using Isolation Forest."""
        features_array = np.array(list(self._feature_history[symbol]))

        # Check if model needs retraining
        should_retrain = (
            symbol not in self._models
            or symbol not in self._last_retrain
            or (timestamp - self._last_retrain[symbol]).total_seconds() > self._retrain_interval_seconds
        )

        if should_retrain:
            self._train_model(symbol, features_array)

        # Scale features
        scaler = self._scalers.get(symbol)
        model = self._models.get(symbol)

        if scaler is None or model is None:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                anomaly_type="model_not_ready",
                features={},
                timestamp=timestamp,
            )

        # Predict
        scaled_features = scaler.transform(current_features.reshape(1, -1))
        anomaly_score = model.decision_function(scaled_features)[0]
        prediction = model.predict(scaled_features)[0]

        # Determine if anomaly
        is_anomaly = prediction == -1 or anomaly_score < self._anomaly_threshold

        # Calculate confidence
        confidence = max(0.0, min(1.0, abs(anomaly_score) / 0.5))

        # Determine anomaly type based on feature values
        anomaly_type = self._classify_anomaly(current_features) if is_anomaly else None

        # Build feature dict
        feature_dict = {
            name: float(current_features[i])
            for i, name in enumerate(self.FEATURE_NAMES)
        }

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=float(anomaly_score),
            confidence=confidence,
            anomaly_type=anomaly_type,
            features=feature_dict,
            timestamp=timestamp,
        )

    def _train_model(self, symbol: str, features: np.ndarray) -> None:
        """Train Isolation Forest model for symbol."""
        try:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            model = IsolationForest(
                contamination=self._contamination,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(scaled_features)

            self._models[symbol] = model
            self._scalers[symbol] = scaler
            self._last_retrain[symbol] = datetime.utcnow()

            logger.info(f"Anomaly detection model trained for {symbol} with {len(features)} samples")

        except Exception as e:
            logger.error(f"Failed to train anomaly model for {symbol}: {e}")

    def _detect_simple(
        self,
        symbol: str,
        current_features: np.ndarray,
        timestamp: datetime,
    ) -> AnomalyResult:
        """Simple anomaly detection without sklearn."""
        features_array = np.array(list(self._feature_history[symbol]))

        # Calculate z-scores for each feature
        mean = np.mean(features_array, axis=0)
        std = np.std(features_array, axis=0) + 1e-8
        z_scores = (current_features - mean) / std

        # Max z-score indicates most anomalous feature
        max_z = np.max(np.abs(z_scores))
        anomaly_score = -max_z / 3.0  # Normalize to roughly -1 to 1

        is_anomaly = max_z > 3.0  # 3 standard deviations

        # Confidence based on z-score extremity
        confidence = min(1.0, max_z / 4.0) if is_anomaly else 0.0

        # Determine anomaly type
        anomaly_type = self._classify_anomaly(current_features) if is_anomaly else None

        feature_dict = {
            name: float(current_features[i])
            for i, name in enumerate(self.FEATURE_NAMES)
        }

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=float(anomaly_score),
            confidence=confidence,
            anomaly_type=anomaly_type,
            features=feature_dict,
            timestamp=timestamp,
        )

    def _classify_anomaly(self, features: np.ndarray) -> str:
        """Classify the type of anomaly based on dominant features."""
        volatility_ratio = features[0]
        volume_spike = features[1]
        spread_percentile = features[2]
        return_zscore = features[3]

        # Determine primary anomaly type
        if abs(return_zscore) > 3.0:
            if return_zscore > 0:
                return "flash_spike"
            else:
                return "flash_crash"
        elif volume_spike > 5.0:
            return "volume_explosion"
        elif spread_percentile > 0.95:
            return "liquidity_crisis"
        elif volatility_ratio > 3.0:
            return "volatility_regime_change"
        else:
            return "general_anomaly"

    async def on_anomaly_trigger_risk_escalation(self, risk_governor) -> None:
        """
        Trigger risk escalation when anomaly detected.

        Args:
            risk_governor: RiskGovernor instance to escalate
        """
        try:
            if hasattr(risk_governor, "escalate_to"):
                from hean.risk.risk_governor import RiskState
                await risk_governor.escalate_to(RiskState.SOFT_BRAKE)
                logger.warning("[ANOMALY] Triggered RiskGovernor escalation to SOFT_BRAKE")
        except Exception as e:
            logger.error(f"Failed to trigger risk escalation: {e}")

    def get_recent_anomalies(self, limit: int = 10) -> list[AnomalyResult]:
        """Get recent anomaly detections."""
        return list(self._recent_anomalies)[-limit:]

    def get_metrics(self) -> dict:
        """Get detector metrics."""
        anomaly_rate = (
            self._anomaly_count / self._total_checks
            if self._total_checks > 0 else 0.0
        )

        return {
            "total_checks": self._total_checks,
            "anomaly_count": self._anomaly_count,
            "anomaly_rate": anomaly_rate,
            "models_trained": len(self._models),
            "sklearn_available": SKLEARN_AVAILABLE,
        }

    def reset(self) -> None:
        """Reset detector state."""
        self._feature_history.clear()
        self._price_history.clear()
        self._volume_history.clear()
        self._spread_history.clear()
        self._return_history.clear()
        self._models.clear()
        self._scalers.clear()
        self._last_retrain.clear()
        self._recent_anomalies.clear()
        self._anomaly_count = 0
        self._total_checks = 0
        logger.info("Anomaly detector reset")
