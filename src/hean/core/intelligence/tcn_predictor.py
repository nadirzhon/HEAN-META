"""
Temporal Convolutional Network (TCN) for Price Reversal Prediction
Processes last 10,000 micro-ticks to predict probability of immediate reversal
If probability > 85%, triggers exit or position flip
"""

from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """Temporal Convolutional Block with dilated causal convolution."""

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.chomp1 = Chomp1d((kernel_size - 1) * dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.chomp2 = Chomp1d((kernel_size - 1) * dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove padding from the end of sequence."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCNPredictor(nn.Module):
    """
    Temporal Convolutional Network for price reversal prediction.
    Lightweight model that processes last 10,000 micro-ticks.
    """

    def __init__(self, input_size: int = 4, num_channels: list = None, kernel_size: int = 3, dropout: float = 0.2):
        if num_channels is None:
            num_channels = [64, 64, 64, 64]
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        # Output: probability of reversal (0-1)
        self.fc = nn.Linear(num_channels[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, features, sequence_length)
        y = self.network(x)
        # Take last timestep
        y = y[:, :, -1]
        y = self.fc(y)
        return self.sigmoid(y)


class TCPriceReversalPredictor:
    """
    Lightweight TCN-based predictor for immediate price reversal.
    Processes last 10,000 micro-ticks to predict reversal probability.
    """

    def __init__(self, sequence_length: int = 10000, device: str = "cpu"):
        """
        Initialize TCN predictor.

        Args:
            sequence_length: Number of micro-ticks to process (default: 10,000)
            device: Device to run model on ("cpu" or "cuda")
        """
        self.sequence_length = sequence_length
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Input features: [price_change, volume, bid_ask_spread, time_delta]
        input_size = 4
        num_channels = [32, 32, 32]  # Lightweight: fewer channels for speed
        self.model = TCNPredictor(input_size=input_size, num_channels=num_channels, kernel_size=3, dropout=0.1)
        self.model.to(self.device)
        self.model.eval()  # Inference mode

        # Circular buffer for micro-ticks
        self.tick_buffer: deque = deque(maxlen=sequence_length)
        self.last_price: float | None = None
        self.last_timestamp: datetime | None = None

        # Reversal threshold
        self.reversal_threshold = 0.85  # 85% probability triggers exit/flip

        # Online learning components
        self._training_buffer: deque = deque(maxlen=1000)  # Store (features, actual_outcome) pairs
        self._training_enabled = True
        self._training_batch_size = 32
        self._optimizer: torch.optim.Adam | None = None
        self._criterion = nn.BCELoss()
        self._training_metrics = {
            "total_updates": 0,
            "total_loss": 0.0,
            "recent_accuracy": 0.0,
        }

        # Initialize optimizer for online learning
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # Initialize with random weights (in production, load trained model)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model with small random weights for inference."""
        for param in self.model.parameters():
            nn.init.normal_(param, mean=0.0, std=0.01)

    def update_tick(self, price: float, volume: float, bid: float, ask: float, timestamp: datetime) -> None:
        """
        Update with new micro-tick data.

        Args:
            price: Current price
            volume: Trade volume
            bid: Best bid price
            ask: Best ask price
            timestamp: Tick timestamp
        """
        # Calculate features
        price_change = 0.0
        time_delta = 0.0

        if self.last_price is not None:
            price_change = (price - self.last_price) / self.last_price

        if self.last_timestamp is not None:
            time_delta = (timestamp - self.last_timestamp).total_seconds()

        bid_ask_spread = (ask - bid) / bid if bid > 0 else 0.0

        # Store features: [price_change, volume, bid_ask_spread, time_delta]
        features = np.array([price_change, volume, bid_ask_spread, time_delta], dtype=np.float32)
        self.tick_buffer.append(features)

        self.last_price = price
        self.last_timestamp = timestamp

    def predict_reversal_probability(self, trigger_training: bool = False) -> tuple[float, bool]:
        """
        Predict probability of immediate reversal.

        Args:
            trigger_training: If True, trigger online training if buffer is full

        Returns:
            Tuple of (probability, should_trigger)
            - probability: Probability of reversal (0.0 to 1.0)
            - should_trigger: True if probability > threshold (85%)
        """
        if len(self.tick_buffer) < self.sequence_length:
            # Not enough data yet
            return 0.0, False

        # Prepare input: (batch=1, features=4, sequence_length)
        features_array = np.array(list(self.tick_buffer), dtype=np.float32)

        # Normalize features
        features_mean = np.mean(features_array, axis=0, keepdims=True)
        features_std = np.std(features_array, axis=0, keepdims=True) + 1e-8
        features_normalized = (features_array - features_mean) / features_std

        # Reshape to (batch, features, sequence)
        features_tensor = torch.FloatTensor(features_normalized).T.unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            prob = self.model(features_tensor)
            probability = prob.item()

        should_trigger = probability > self.reversal_threshold

        # Trigger online training if requested and buffer is full
        if trigger_training and self._training_enabled and len(self._training_buffer) >= self._training_batch_size:
            self._train_step()

        return probability, should_trigger

    def get_prediction_with_confidence(self) -> dict:
        """
        Get reversal prediction with confidence intervals.

        Returns:
            Dictionary with:
            - probability: Reversal probability (0.0 to 1.0)
            - should_trigger: True if > 85%
            - confidence_high: Upper confidence bound
            - confidence_low: Lower confidence bound
            - ticks_processed: Number of ticks in buffer
        """
        probability, should_trigger = self.predict_reversal_probability()

        # Simple confidence intervals (in production, use ensemble or dropout uncertainty)
        confidence_std = 0.05  # 5% uncertainty estimate
        confidence_high = min(1.0, probability + confidence_std)
        confidence_low = max(0.0, probability - confidence_std)

        return {
            "probability": probability,
            "should_trigger": should_trigger,
            "confidence_high": confidence_high,
            "confidence_low": confidence_low,
            "ticks_processed": len(self.tick_buffer),
            "sequence_length": self.sequence_length,
            "threshold": self.reversal_threshold,
        }

    def load_model(self, model_path: str) -> bool:
        """
        Load trained model weights.

        Args:
            model_path: Path to saved model weights

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def save_model(self, model_path: str) -> bool:
        """
        Save model weights.

        Args:
            model_path: Path to save model weights

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'sequence_length': self.sequence_length,
                'input_size': 4,
                'training_metrics': self._training_metrics,
            }, model_path)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def update_from_feedback(self, actual_outcome: bool) -> None:
        """
        Update model with actual outcome feedback for online learning.

        Args:
            actual_outcome: True if reversal occurred, False otherwise
        """
        if len(self.tick_buffer) < self.sequence_length:
            return

        # Store current features and actual outcome
        features_array = np.array(list(self.tick_buffer), dtype=np.float32)
        self._training_buffer.append((features_array, actual_outcome))

    def _train_step(self) -> None:
        """
        Perform a single training step using buffered feedback data.
        Uses mini-batch gradient descent with Adam optimizer.
        """
        if len(self._training_buffer) < self._training_batch_size:
            return

        # Sample a batch from training buffer
        batch_indices = np.random.choice(
            len(self._training_buffer),
            size=self._training_batch_size,
            replace=False
        )

        batch_features = []
        batch_labels = []

        for idx in batch_indices:
            features, label = self._training_buffer[idx]
            # Normalize features
            features_mean = np.mean(features, axis=0, keepdims=True)
            features_std = np.std(features, axis=0, keepdims=True) + 1e-8
            features_normalized = (features - features_mean) / features_std
            batch_features.append(features_normalized)
            batch_labels.append(float(label))

        # Convert to tensors
        X = torch.FloatTensor(np.array(batch_features))  # (batch, sequence, features)
        X = X.transpose(1, 2)  # (batch, features, sequence)
        X = X.to(self.device)

        y = torch.FloatTensor(batch_labels).unsqueeze(1).to(self.device)  # (batch, 1)

        # Training step
        self.model.train()
        self._optimizer.zero_grad()

        # Forward pass
        predictions = self.model(X)

        # Compute loss
        loss = self._criterion(predictions, y)

        # Backward pass
        loss.backward()
        self._optimizer.step()

        # Update metrics
        self._training_metrics["total_updates"] += 1
        self._training_metrics["total_loss"] += loss.item()

        # Calculate accuracy
        predicted_labels = (predictions > 0.5).float()
        accuracy = (predicted_labels == y).float().mean().item()
        self._training_metrics["recent_accuracy"] = accuracy

        # Back to eval mode
        self.model.eval()

    def get_training_metrics(self) -> dict[str, float]:
        """
        Get training metrics for monitoring online learning performance.

        Returns:
            Dictionary with training metrics:
            - total_updates: Number of training updates performed
            - avg_loss: Average training loss
            - recent_accuracy: Most recent batch accuracy
            - buffer_size: Current training buffer size
        """
        avg_loss = 0.0
        if self._training_metrics["total_updates"] > 0:
            avg_loss = self._training_metrics["total_loss"] / self._training_metrics["total_updates"]

        return {
            "total_updates": float(self._training_metrics["total_updates"]),
            "avg_loss": avg_loss,
            "recent_accuracy": self._training_metrics["recent_accuracy"],
            "buffer_size": float(len(self._training_buffer)),
            "training_enabled": float(self._training_enabled),
        }

    def enable_training(self, enabled: bool = True) -> None:
        """
        Enable or disable online training.

        Args:
            enabled: True to enable training, False to disable
        """
        self._training_enabled = enabled
