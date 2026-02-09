"""Execution edge estimator for signal filtering."""

import pickle
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from hean.config import settings
from hean.core.regime import Regime
from hean.core.types import Signal, Tick
from hean.logging import get_logger
from hean.paper_trade_assist import (
    get_edge_threshold_reduction_pct,
    is_paper_assist_enabled,
    log_allow_reason,
    log_block_reason,
)

logger = get_logger(__name__)


class ExecutionEdgeEstimator:
    """Estimates execution edge for trading signals.

    Edge calculation includes:
    - Expected move toward take_profit (in bps)
    - Spread cost (ask-bid)
    - Maker fill probability proxy (based on spread size and regime)
    - Volatility penalty (higher volatility reduces certainty)
    - Regime adjustment (IMPULSE: higher threshold, RANGE: stricter threshold)
    """

    def __init__(self, model_path: str | None = None) -> None:
        """Initialize the edge estimator.

        Args:
            model_path: Path to saved ML model (optional)
        """
        self._volatility_history: dict[str, deque[float]] = {}
        self._window_size = 20  # Lookback window for volatility
        self._signals_blocked_by_edge = 0
        self._edge_sum = 0.0
        self._edge_count = 0

        # ML-based edge prediction components
        self._ml_enabled = False
        self._ml_model: dict[str, Any] | None = None
        self._feature_scaler: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        self._training_data: deque = deque(maxlen=10000)  # Store (features, actual_edge, outcome)
        self._model_path = model_path or str(Path.home() / ".hean" / "edge_model.pkl")

        # OFI (Order Flow Imbalance) tracking
        self._ofi_history: dict[str, deque[float]] = {}

        # Try to load existing model
        if Path(self._model_path).exists():
            self.load_model(self._model_path)

    def estimate_edge(self, signal: Signal, tick: Tick, regime: Regime) -> float:
        """Estimate execution edge in basis points.

        Args:
            signal: Trading signal
            tick: Current market tick
            regime: Current market regime

        Returns:
            Estimated edge in basis points (bps)
        """
        if not tick.bid or not tick.ask:
            return -1000.0  # No edge if no bid/ask

        # Calculate spread cost in bps
        spread = tick.ask - tick.bid
        spread_bps = (spread / tick.price) * 10000 if tick.price > 0 else 0

        # Calculate expected move toward take_profit in bps
        expected_move_bps = 0.0
        if signal.take_profit:
            if signal.side == "buy":
                move = (signal.take_profit - signal.entry_price) / signal.entry_price if signal.entry_price != 0 else 0.0
            else:  # sell
                move = (signal.entry_price - signal.take_profit) / signal.entry_price if signal.entry_price != 0 else 0.0
            expected_move_bps = move * 10000

        # Calculate maker fill probability proxy
        # Improved model: more realistic fill probability based on spread and offset
        # With maker_price_offset_bps=2, orders are placed 2 bps away from best bid/ask
        # This gives reasonable fill probability (typically 60-80% depending on spread)
        base_fill_prob = 0.75  # Increased from 0.8 to 0.75 (more realistic)

        # Spread penalty: larger spreads reduce fill probability
        # Normalize spread penalty: spread of 8 bps = 50% penalty, spread of 4 bps = 25% penalty
        spread_penalty = min(spread_bps / 16.0, 0.6)  # Max 60% penalty for very large spreads

        # Regime adjustments: RANGE has tighter spreads (higher fill prob), IMPULSE has wider spreads
        if regime == Regime.RANGE:
            fill_prob = base_fill_prob * (1.0 - spread_penalty * 0.4)  # Less penalty in RANGE
        elif regime == Regime.IMPULSE:
            fill_prob = base_fill_prob * (1.0 - spread_penalty * 1.2)  # More penalty in IMPULSE
        else:  # NORMAL
            fill_prob = base_fill_prob * (1.0 - spread_penalty * 0.8)  # Moderate penalty

        # Clamp between 0.3 and 0.95 (more realistic range)
        fill_prob = max(0.3, min(0.95, fill_prob))

        # Calculate volatility penalty
        volatility_penalty = self._get_volatility_penalty(tick.symbol, tick.price)

        # Calculate raw edge (expected move adjusted for fill probability)
        raw_edge_bps = expected_move_bps * fill_prob - spread_bps

        # Apply volatility penalty
        edge_bps = raw_edge_bps * (1.0 - volatility_penalty)

        # Regime adjustment (already factored into fill_prob, but add small adjustment)
        if regime == Regime.IMPULSE:
            # Allow slightly higher edge threshold (reduce penalty by 5%)
            edge_bps *= 1.05
        elif regime == Regime.RANGE:
            # Stricter threshold (add 5% penalty)
            edge_bps *= 0.95

        return edge_bps

    def _get_volatility_penalty(self, symbol: str, current_price: float) -> float:
        """Calculate volatility penalty based on recent price history.

        Higher volatility reduces certainty, thus reducing edge.

        Returns:
            Volatility penalty (0.0 to 1.0)
        """
        if symbol not in self._volatility_history:
            self._volatility_history[symbol] = deque(maxlen=self._window_size)
            return 0.1  # Default low penalty if no history

        history = list(self._volatility_history[symbol])
        if len(history) < 5:
            return 0.1  # Low penalty if insufficient data

        # Calculate rolling volatility
        returns = []
        for i in range(1, len(history)):
            if history[i - 1] > 0:
                ret = abs((history[i] - history[i - 1]) / history[i - 1])
                returns.append(ret)

        if not returns:
            return 0.1

        volatility = sum(returns) / len(returns)

        # Convert volatility to penalty (0.0 to 0.5 max)
        # High volatility (e.g., 0.01 = 1%) -> higher penalty
        penalty = min(volatility * 50, 0.5)  # Cap at 50% penalty

        return penalty

    def update_price_history(self, symbol: str, price: float) -> None:
        """Update price history for volatility calculation."""
        if symbol not in self._volatility_history:
            self._volatility_history[symbol] = deque(maxlen=self._window_size)
        self._volatility_history[symbol].append(price)

    def get_min_edge_threshold(self, regime: Regime) -> float:
        """Get minimum edge threshold for a regime.

        Args:
            regime: Current market regime

        Returns:
            Minimum edge threshold in bps
        """
        # REDUCED BY 50% FOR DEBUG
        if regime == Regime.IMPULSE:
            base_threshold = settings.min_edge_bps_impulse * 0.5
        elif regime == Regime.RANGE:
            base_threshold = settings.min_edge_bps_range * 0.5
        else:  # NORMAL
            base_threshold = settings.min_edge_bps_normal * 0.5

        # Apply paper assist reduction
        if is_paper_assist_enabled():
            reduction_pct = get_edge_threshold_reduction_pct()
            base_threshold = base_threshold * (1.0 - reduction_pct / 100.0)

        return base_threshold

    def should_emit_signal(self, signal: Signal, tick: Tick, regime: Regime) -> bool:
        """Check if signal should be emitted based on edge estimation.

        Args:
            signal: Trading signal
            tick: Current market tick
            regime: Current market regime

        Returns:
            True if signal should be emitted, False otherwise
        """
        edge_bps = self.estimate_edge(signal, tick, regime)
        min_threshold = self.get_min_edge_threshold(regime)

        # Track metrics
        self._edge_sum += edge_bps
        self._edge_count += 1

        if edge_bps < min_threshold:
            self._signals_blocked_by_edge += 1
            log_block_reason(
                "edge_reject",
                measured_value=edge_bps,
                threshold=min_threshold,
                symbol=tick.symbol,
                strategy_id=signal.strategy_id,
                agent_name=signal.strategy_id,
            )
            logger.debug(
                f"Signal blocked by edge: edge={edge_bps:.1f} bps < "
                f"threshold={min_threshold} bps (regime={regime.value})"
            )
            return False

        log_allow_reason("edge_ok", symbol=tick.symbol, strategy_id=signal.strategy_id)
        return True

    def get_metrics(self) -> dict[str, float]:
        """Get edge estimator metrics.

        Returns:
            Dictionary with metrics:
            - signals_blocked_by_edge: Number of signals blocked
            - avg_edge_bps: Average edge in bps
        """
        avg_edge = self._edge_sum / self._edge_count if self._edge_count > 0 else 0.0

        return {
            "signals_blocked_by_edge": float(self._signals_blocked_by_edge),
            "avg_edge_bps": avg_edge,
            "total_signals_evaluated": float(self._edge_count),
        }

    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        self._signals_blocked_by_edge = 0
        self._edge_sum = 0.0
        self._edge_count = 0

    def _extract_features(self, signal: Signal, tick: Tick, regime: Regime) -> np.ndarray:
        """
        Extract features for ML-based edge prediction.

        Features:
        - spread_bps: Bid-ask spread in basis points
        - volatility: Recent price volatility
        - ofi: Order flow imbalance
        - time_of_day: Hour of day (normalized)
        - regime_impulse: 1 if IMPULSE, 0 otherwise
        - regime_range: 1 if RANGE, 0 otherwise
        - expected_move_bps: Expected move to TP in bps
        - volume: Current volume (log-normalized)

        Args:
            signal: Trading signal
            tick: Current tick
            regime: Market regime

        Returns:
            Feature array (8 features)
        """
        # Spread in bps
        spread = (tick.ask - tick.bid) if tick.bid and tick.ask else 0.0
        spread_bps = (spread / tick.price) * 10000 if tick.price > 0 else 0.0

        # Volatility
        volatility = self._get_volatility_penalty(tick.symbol, tick.price)

        # OFI (Order Flow Imbalance) - simplified proxy
        ofi = self._get_ofi(tick)

        # Time of day (hour normalized to 0-1)
        hour = datetime.utcnow().hour
        time_of_day = hour / 24.0

        # Regime indicators
        regime_impulse = 1.0 if regime == Regime.IMPULSE else 0.0
        regime_range = 1.0 if regime == Regime.RANGE else 0.0

        # Expected move to TP in bps
        expected_move_bps = 0.0
        if signal.take_profit:
            if signal.side == "buy":
                move = (signal.take_profit - signal.entry_price) / signal.entry_price if signal.entry_price != 0 else 0.0
            else:
                move = (signal.entry_price - signal.take_profit) / signal.entry_price if signal.entry_price != 0 else 0.0
            expected_move_bps = move * 10000

        # Volume (log-normalized)
        volume = np.log1p(tick.volume) if tick.volume > 0 else 0.0

        features = np.array([
            spread_bps,
            volatility,
            ofi,
            time_of_day,
            regime_impulse,
            regime_range,
            expected_move_bps,
            volume,
        ], dtype=np.float32)

        return features

    def _get_ofi(self, tick: Tick) -> float:
        """
        Calculate Order Flow Imbalance (OFI) proxy.

        OFI = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        For now, use simplified version based on spread asymmetry.

        Args:
            tick: Current tick

        Returns:
            OFI value (-1 to 1)
        """
        if not tick.bid or not tick.ask:
            return 0.0

        # Simplified OFI: if price is closer to ask, buyers are aggressive (positive OFI)
        # If price closer to bid, sellers are aggressive (negative OFI)
        mid = (tick.bid + tick.ask) / 2.0

        # If price above mid, positive OFI (buy pressure)
        # If price below mid, negative OFI (sell pressure)
        if mid == 0:
            return 0.0

        ofi = (tick.price - mid) / (tick.ask - tick.bid) if (tick.ask - tick.bid) > 0 else 0.0

        # Clamp to [-1, 1]
        ofi = max(-1.0, min(1.0, ofi))

        # Store in history
        if tick.symbol not in self._ofi_history:
            self._ofi_history[tick.symbol] = deque(maxlen=20)
        self._ofi_history[tick.symbol].append(ofi)

        # Return smoothed OFI
        return float(np.mean(list(self._ofi_history[tick.symbol])))

    def estimate_edge_ml(self, signal: Signal, tick: Tick, regime: Regime) -> float:
        """
        Estimate edge using ML model.

        Args:
            signal: Trading signal
            tick: Current tick
            regime: Market regime

        Returns:
            Estimated edge in bps
        """
        if not self._ml_enabled or not self._ml_model:
            # Fallback to rule-based
            return self.estimate_edge(signal, tick, regime)

        # Extract features
        features = self._extract_features(signal, tick, regime)

        # Normalize features
        if self._feature_scaler:
            mean, std = self._feature_scaler["mean"], self._feature_scaler["std"]
            features = (features - mean) / (std + 1e-8)

        # Predict edge using simple gradient boosting (simulated with weighted average for now)
        # In production, use sklearn GradientBoostingRegressor or XGBoost
        predicted_edge = self._predict_with_model(features)

        return predicted_edge

    def _predict_with_model(self, features: np.ndarray) -> float:
        """
        Predict edge using trained model.

        Args:
            features: Normalized feature array

        Returns:
            Predicted edge in bps
        """
        if not self._ml_model:
            return 0.0

        # Simple linear model for now (in production, use GradientBoosting)
        weights = self._ml_model.get("weights", np.zeros(len(features)))
        bias = self._ml_model.get("bias", 0.0)

        prediction = float(np.dot(features, weights) + bias)
        return prediction

    def update_ml_model(self, signal: Signal, tick: Tick, regime: Regime, actual_outcome: float) -> None:
        """
        Update ML model with actual trade outcome for online learning.

        Args:
            signal: Original signal
            tick: Tick at signal time
            regime: Regime at signal time
            actual_outcome: Actual edge achieved (in bps)
        """
        features = self._extract_features(signal, tick, regime)
        rule_based_edge = self.estimate_edge(signal, tick, regime)

        # Store training sample
        self._training_data.append({
            "features": features,
            "predicted_edge": rule_based_edge,
            "actual_edge": actual_outcome,
            "timestamp": datetime.utcnow(),
        })

        # Retrain model if we have enough samples
        if len(self._training_data) >= 100 and len(self._training_data) % 50 == 0:
            self._train_ml_model()

    def _train_ml_model(self) -> None:
        """
        Train ML model using accumulated training data.
        Uses simple gradient descent for online learning.
        """
        if len(self._training_data) < 100:
            return

        # Prepare training data
        X = np.array([sample["features"] for sample in self._training_data])
        y = np.array([sample["actual_edge"] for sample in self._training_data])

        # Normalize features
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8
        X_normalized = (X - mean) / std

        # Store scaler
        self._feature_scaler = {"mean": mean, "std": std}

        # Simple linear regression with L2 regularization
        # In production, use sklearn.ensemble.GradientBoostingRegressor
        n_features = X_normalized.shape[1]

        # Initialize or update weights
        if not self._ml_model:
            self._ml_model = {
                "weights": np.zeros(n_features),
                "bias": 0.0,
            }

        # Gradient descent update
        learning_rate = 0.01
        l2_lambda = 0.001

        # Predictions
        predictions = X_normalized @ self._ml_model["weights"] + self._ml_model["bias"]

        # Gradients
        errors = predictions - y
        grad_weights = (X_normalized.T @ errors) / len(y) + l2_lambda * self._ml_model["weights"]
        grad_bias = np.mean(errors)

        # Update
        self._ml_model["weights"] -= learning_rate * grad_weights
        self._ml_model["bias"] -= learning_rate * grad_bias

        # Enable ML
        self._ml_enabled = True

        logger.info(
            f"ML edge model updated: {len(self._training_data)} samples, "
            f"MSE: {np.mean(errors**2):.2f}"
        )

    def save_model(self, path: str | None = None) -> bool:
        """
        Save ML model to disk.

        Args:
            path: Path to save model (default: ~/.hean/edge_model.pkl)

        Returns:
            True if saved successfully
        """
        save_path = path or self._model_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(save_path, "wb") as f:
                pickle.dump({
                    "model": self._ml_model,
                    "scaler": self._feature_scaler,
                    "ml_enabled": self._ml_enabled,
                    "training_samples": len(self._training_data),
                }, f)
            logger.info(f"ML edge model saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save ML model: {e}")
            return False

    def load_model(self, path: str | None = None) -> bool:
        """
        Load ML model from disk.

        Args:
            path: Path to load model from (default: ~/.hean/edge_model.pkl)

        Returns:
            True if loaded successfully
        """
        load_path = path or self._model_path

        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)
                self._ml_model = data.get("model")
                self._feature_scaler = data.get("scaler")
                self._ml_enabled = data.get("ml_enabled", False)
            logger.info(
                f"ML edge model loaded from {load_path}: "
                f"{data.get('training_samples', 0)} training samples"
            )
            return True
        except Exception as e:
            logger.debug(f"No ML model loaded: {e}")
            return False

    def enable_ml(self, enabled: bool = True) -> None:
        """
        Enable or disable ML-based edge estimation.

        Args:
            enabled: True to enable ML, False to use rule-based only
        """
        if enabled and not self._ml_model:
            logger.warning("ML model not trained yet, continuing with rule-based")
            return

        self._ml_enabled = enabled
        logger.info(f"ML edge estimation {'enabled' if enabled else 'disabled'}")
