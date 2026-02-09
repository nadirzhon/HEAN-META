"""
Transformer-based Price Predictor for Multi-Horizon Forecasting.

Uses Temporal Fusion Transformer (TFT) architecture for:
- Multi-step price prediction (1s, 5s, 30s, 1m horizons)
- Variable importance learning
- Attention-based interpretability
- Multi-asset support
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PricePrediction:
    """Price prediction with confidence."""

    horizon_name: str  # e.g., "1s", "5s", "30s", "1m"
    horizon_ms: int
    predicted_price: float
    predicted_return_pct: float
    confidence: float
    direction: str  # "up", "down", "neutral"
    features_importance: dict[str, float]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VariableSelectionNetwork(nn.Module):
    """Variable selection for feature importance."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.grn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Linear(hidden_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.elu(self.hidden(x))
        grn_out = self.grn(hidden)
        weights = self.softmax(self.gate(grn_out))
        selected = x * weights
        return selected, weights


class TransformerPricePredictor(nn.Module):
    """Transformer model for price prediction."""

    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_horizons: int = 4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_horizons = num_horizons

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Variable selection
        self.var_selection = VariableSelectionNetwork(input_dim, d_model, dropout)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-horizon output heads
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 2),  # [return_prediction, confidence]
            )
            for _ in range(num_horizons)
        ])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            predictions: (batch, num_horizons, 2) - return and confidence per horizon
            attention_weights: (batch, input_dim) - feature importance
            hidden: (batch, d_model) - hidden representation
        """
        # Variable selection
        selected, var_weights = self.var_selection(x)

        # Project to model dimension
        x_proj = self.input_proj(selected)

        # Add positional encoding
        x_pos = self.pos_encoder(x_proj)

        # Transformer encoding
        hidden = self.transformer(x_pos)

        # Use last timestep for predictions
        last_hidden = hidden[:, -1, :]

        # Multi-horizon predictions
        predictions = []
        for head in self.horizon_heads:
            pred = head(last_hidden)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=1)  # (batch, num_horizons, 2)

        # Average variable weights across sequence
        attention_weights = var_weights.mean(dim=1)  # (batch, input_dim)

        return predictions, attention_weights, last_hidden


class MultiHorizonPricePredictor:
    """
    Multi-horizon price predictor using Transformer architecture.

    Predicts price movements at multiple time horizons:
    - 1 second (ultra-short for scalping)
    - 5 seconds (short-term momentum)
    - 30 seconds (trend confirmation)
    - 1 minute (position sizing)
    """

    HORIZONS = [
        ("1s", 1000),
        ("5s", 5000),
        ("30s", 30000),
        ("1m", 60000),
    ]

    FEATURE_NAMES = [
        "price_return",
        "volume_ratio",
        "bid_ask_spread",
        "trade_imbalance",
        "volatility_short",
        "volatility_long",
        "momentum_short",
        "momentum_long",
    ]

    def __init__(
        self,
        sequence_length: int = 100,
        device: str = "cpu",
        model_path: str | None = None,
    ):
        """
        Initialize the predictor.

        Args:
            sequence_length: Number of ticks to use for prediction
            device: Device to run model on
            model_path: Optional path to load pre-trained model
        """
        self.sequence_length = sequence_length
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = TransformerPricePredictor(
            input_dim=len(self.FEATURE_NAMES),
            num_horizons=len(self.HORIZONS),
        ).to(self.device)
        self.model.eval()

        # Feature buffers per symbol
        self._feature_buffers: dict[str, deque] = {}
        self._price_history: dict[str, deque] = {}
        self._volume_history: dict[str, deque] = {}
        self._trade_history: dict[str, deque] = {}

        # Training buffer for online learning
        self._training_buffer: deque = deque(maxlen=10000)
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # Prediction thresholds
        self._min_confidence = 0.6
        self._min_return_threshold = 0.0001  # 1 bps minimum

        # Load model if path provided
        if model_path:
            self.load_model(model_path)
        else:
            self._initialize_weights()

        # Metrics
        self._predictions_made = 0
        self._correct_directions = 0

    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for param in self.model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def update_tick(
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
        Update with new tick data.

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
        # Initialize buffers if needed
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.sequence_length * 2)
            self._volume_history[symbol] = deque(maxlen=self.sequence_length * 2)
            self._trade_history[symbol] = deque(maxlen=self.sequence_length * 2)
            self._feature_buffers[symbol] = deque(maxlen=self.sequence_length)

        # Update price history
        self._price_history[symbol].append(price)
        self._volume_history[symbol].append(volume)
        self._trade_history[symbol].append({
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "bid": bid,
            "ask": ask,
        })

        # Calculate features
        features = self._calculate_features(symbol, price, volume, bid, ask)
        if features is not None:
            self._feature_buffers[symbol].append(features)

    def _calculate_features(
        self,
        symbol: str,
        price: float,
        volume: float,
        bid: float,
        ask: float,
    ) -> np.ndarray | None:
        """Calculate feature vector for current tick."""
        prices = list(self._price_history[symbol])
        volumes = list(self._volume_history[symbol])
        trades = list(self._trade_history[symbol])

        if len(prices) < 20:
            return None

        # Price return
        price_return = (price - prices[-2]) / prices[-2] if prices[-2] > 0 else 0.0

        # Volume ratio (current vs average)
        avg_volume = np.mean(volumes[-20:]) if volumes else 1.0
        volume_ratio = volume / max(avg_volume, 1e-8)

        # Bid-ask spread
        spread = (ask - bid) / price if price > 0 else 0.0

        # Trade imbalance
        recent_trades = trades[-10:] if len(trades) >= 10 else trades
        total_buy = sum(t["buy_volume"] for t in recent_trades)
        total_sell = sum(t["sell_volume"] for t in recent_trades)
        total_volume = total_buy + total_sell
        trade_imbalance = (total_buy - total_sell) / max(total_volume, 1e-8)

        # Short-term volatility (10 ticks)
        short_returns = [
            (prices[i] - prices[i-1]) / prices[i-1]
            for i in range(-min(10, len(prices)-1), 0)
            if prices[i-1] > 0
        ]
        volatility_short = np.std(short_returns) if short_returns else 0.0

        # Long-term volatility (50 ticks)
        long_returns = [
            (prices[i] - prices[i-1]) / prices[i-1]
            for i in range(-min(50, len(prices)-1), 0)
            if prices[i-1] > 0
        ]
        volatility_long = np.std(long_returns) if long_returns else 0.0

        # Short-term momentum (10 ticks)
        momentum_short = (price - prices[-10]) / prices[-10] if len(prices) >= 10 and prices[-10] > 0 else 0.0

        # Long-term momentum (50 ticks)
        momentum_long = (price - prices[-50]) / prices[-50] if len(prices) >= 50 and prices[-50] > 0 else 0.0

        features = np.array([
            price_return,
            volume_ratio,
            spread,
            trade_imbalance,
            volatility_short,
            volatility_long,
            momentum_short,
            momentum_long,
        ], dtype=np.float32)

        return features

    def predict(self, symbol: str) -> list[PricePrediction] | None:
        """
        Generate price predictions for all horizons.

        Args:
            symbol: Trading symbol

        Returns:
            List of predictions for each horizon, or None if insufficient data
        """
        if symbol not in self._feature_buffers:
            return None

        features = list(self._feature_buffers[symbol])
        if len(features) < self.sequence_length:
            return None

        # Prepare input
        features_array = np.array(features[-self.sequence_length:], dtype=np.float32)

        # Normalize features
        mean = np.mean(features_array, axis=0, keepdims=True)
        std = np.std(features_array, axis=0, keepdims=True) + 1e-8
        features_normalized = (features_array - mean) / std

        # Convert to tensor
        x = torch.FloatTensor(features_normalized).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            predictions, attention_weights, _ = self.model(x)

        # Get current price
        current_price = self._price_history[symbol][-1]

        # Process predictions
        results = []
        attention = attention_weights[0].cpu().numpy()
        feature_importance = {
            name: float(attention[i])
            for i, name in enumerate(self.FEATURE_NAMES)
        }

        for i, (horizon_name, horizon_ms) in enumerate(self.HORIZONS):
            pred = predictions[0, i].cpu().numpy()
            predicted_return = float(pred[0])
            raw_confidence = float(torch.sigmoid(torch.tensor(pred[1])))

            # Adjust confidence based on return magnitude
            confidence = raw_confidence * min(abs(predicted_return) / 0.001, 1.0)
            confidence = max(0.0, min(1.0, confidence))

            # Determine direction
            if abs(predicted_return) < self._min_return_threshold:
                direction = "neutral"
            elif predicted_return > 0:
                direction = "up"
            else:
                direction = "down"

            # Calculate predicted price
            predicted_price = current_price * (1 + predicted_return)

            results.append(PricePrediction(
                horizon_name=horizon_name,
                horizon_ms=horizon_ms,
                predicted_price=predicted_price,
                predicted_return_pct=predicted_return * 100,
                confidence=confidence,
                direction=direction,
                features_importance=feature_importance,
            ))

        self._predictions_made += 1
        return results

    def get_best_prediction(
        self,
        symbol: str,
        min_confidence: float = 0.6,
    ) -> PricePrediction | None:
        """
        Get the highest-confidence directional prediction.

        Args:
            symbol: Trading symbol
            min_confidence: Minimum confidence threshold

        Returns:
            Best prediction above threshold, or None
        """
        predictions = self.predict(symbol)
        if not predictions:
            return None

        # Filter by confidence and direction
        valid_predictions = [
            p for p in predictions
            if p.confidence >= min_confidence and p.direction != "neutral"
        ]

        if not valid_predictions:
            return None

        # Return highest confidence
        return max(valid_predictions, key=lambda p: p.confidence)

    def update_from_outcome(
        self,
        symbol: str,
        horizon_name: str,
        predicted_direction: str,
        actual_direction: str,
    ) -> None:
        """
        Update model based on prediction outcome.

        Args:
            symbol: Trading symbol
            horizon_name: Which horizon was predicted
            predicted_direction: What was predicted
            actual_direction: What actually happened
        """
        correct = predicted_direction == actual_direction
        if correct:
            self._correct_directions += 1

        # Store in training buffer for online learning
        if symbol in self._feature_buffers and len(self._feature_buffers[symbol]) >= self.sequence_length:
            features = np.array(
                list(self._feature_buffers[symbol])[-self.sequence_length:],
                dtype=np.float32
            )
            horizon_idx = next(
                i for i, (name, _) in enumerate(self.HORIZONS) if name == horizon_name
            )
            target = 1.0 if actual_direction == "up" else 0.0
            self._training_buffer.append((features, horizon_idx, target))

    def train_step(self, batch_size: int = 32) -> float | None:
        """
        Perform one training step using buffered data.

        Args:
            batch_size: Training batch size

        Returns:
            Training loss, or None if insufficient data
        """
        if len(self._training_buffer) < batch_size:
            return None

        # Sample batch
        indices = np.random.choice(len(self._training_buffer), batch_size, replace=False)
        batch = [self._training_buffer[i] for i in indices]

        # Prepare data
        features = np.stack([b[0] for b in batch])
        horizon_indices = [b[1] for b in batch]
        targets = np.array([b[2] for b in batch], dtype=np.float32)

        # Normalize
        mean = np.mean(features, axis=(0, 1), keepdims=True)
        std = np.std(features, axis=(0, 1), keepdims=True) + 1e-8
        features_normalized = (features - mean) / std

        # Convert to tensors
        x = torch.FloatTensor(features_normalized).to(self.device)
        y = torch.FloatTensor(targets).to(self.device)

        # Forward pass
        self.model.train()
        predictions, _, _ = self.model(x)

        # Calculate loss for each sample's horizon
        losses = []
        for i, horizon_idx in enumerate(horizon_indices):
            pred_return = predictions[i, horizon_idx, 0]
            pred_direction = torch.sigmoid(pred_return)  # Convert to probability
            loss = F.binary_cross_entropy(pred_direction, y[i:i+1])
            losses.append(loss)

        total_loss = torch.stack(losses).mean()

        # Backward pass
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        self.model.eval()

        return float(total_loss)

    def get_metrics(self) -> dict:
        """Get predictor metrics."""
        accuracy = (
            self._correct_directions / self._predictions_made
            if self._predictions_made > 0 else 0.0
        )

        return {
            "predictions_made": self._predictions_made,
            "correct_directions": self._correct_directions,
            "direction_accuracy": accuracy,
            "training_buffer_size": len(self._training_buffer),
        }

    def save_model(self, path: str) -> bool:
        """Save model to disk."""
        try:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "metrics": self.get_metrics(),
            }, path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Load model from disk."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.model.eval()
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
