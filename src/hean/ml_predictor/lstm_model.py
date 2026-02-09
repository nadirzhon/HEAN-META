"""
LSTM Neural Network for Price Prediction
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class LSTMPriceModel:
    """
    LSTM model for cryptocurrency price prediction

    Architecture:
    - Multiple LSTM layers with dropout
    - Dense output layer for multi-horizon prediction
    - Adam optimizer with learning rate scheduling

    Input shape: (batch_size, lookback_periods, n_features)
    Output shape: (batch_size, n_prediction_horizons)
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int = 3,  # 1h, 4h, 24h predictions
        lstm_units: list[int] | None = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM model

        Args:
            n_features: Number of input features
            n_outputs: Number of prediction horizons
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        """
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.lstm_units = lstm_units if lstm_units is not None else [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model = None
        self._build_model()

    def _build_model(self):
        """Build LSTM model architecture"""
        try:
            import tensorflow as tf  # noqa: F401
            from tensorflow import keras
            from tensorflow.keras import layers

            # Input layer
            inputs = keras.Input(shape=(None, self.n_features))

            # LSTM layers
            x = inputs
            for i, units in enumerate(self.lstm_units):
                return_sequences = i < len(self.lstm_units) - 1
                x = layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    name=f'lstm_{i+1}'
                )(x)
                x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)

            # Dense layers
            x = layers.Dense(64, activation='relu', name='dense_1')(x)
            x = layers.Dropout(self.dropout_rate, name='dropout_dense')(x)

            # Output layer (predicting returns %)
            outputs = layers.Dense(self.n_outputs, activation='linear', name='output')(x)

            # Build model
            self.model = keras.Model(inputs=inputs, outputs=outputs, name='lstm_price_predictor')

            # Compile
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
            self.model.compile(
                optimizer=optimizer,
                loss='mse',  # Mean Squared Error
                metrics=['mae', 'mape']  # Mean Absolute Error, Mean Absolute Percentage Error
            )

            logger.info(f"LSTM model built: {len(self.lstm_units)} LSTM layers, {self.n_features} features")

        except ImportError:
            logger.error(
                "TensorFlow not installed. Install with: "
                "pip install tensorflow --break-system-packages"
            )
            raise

    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        verbose: int = 1
    ) -> dict:
        """
        Train the model

        Args:
            X_train: Training features (n_samples, lookback, n_features)
            y_train: Training targets (n_samples, n_outputs)
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping_patience: Early stopping patience
            verbose: Verbosity (0=silent, 1=progress bar, 2=one line per epoch)

        Returns:
            Training history
        """
        from tensorflow import keras

        # Callbacks
        callbacks = []

        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)

        # Learning rate reduction
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Train
        logger.info(f"Training LSTM model: {epochs} epochs, batch size {batch_size}")

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        logger.info("Training completed")

        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features (n_samples, lookback, n_features)

        Returns:
            Predictions (n_samples, n_outputs) - predicted returns %
        """
        return self.model.predict(X, verbose=0)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Evaluate model

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            (loss, mae, mape)
        """
        results = self.model.evaluate(X_test, y_test, verbose=0)
        return tuple(results)

    def save(self, filepath: str):
        """Save model to file"""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model from file"""
        from tensorflow import keras
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

    def calculate_direction_accuracy(
        self,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Calculate direction prediction accuracy

        Args:
            X: Features
            y_true: True returns %

        Returns:
            Accuracy (0-1)
        """
        y_pred = self.predict(X)

        # Check if predicted and actual have same sign (direction)
        correct = np.sign(y_pred) == np.sign(y_true)

        # Average across all predictions
        accuracy = correct.mean()

        return accuracy


# Example usage
async def main():
    """Example usage"""
    # Generate sample data
    np.random.seed(42)

    n_samples = 1000
    lookback = 60
    n_features = 15
    n_outputs = 3

    X_train = np.random.randn(n_samples, lookback, n_features)
    y_train = np.random.randn(n_samples, n_outputs)  # Returns %

    X_val = np.random.randn(200, lookback, n_features)
    y_val = np.random.randn(200, n_outputs)

    # Create model
    model = LSTMPriceModel(
        n_features=n_features,
        n_outputs=n_outputs,
        lstm_units=[128, 64, 32],
        dropout_rate=0.2,
        learning_rate=0.001
    )

    print("\nModel Architecture:")
    model.summary()

    print("\nTraining model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=10,  # Just 10 for demo
        batch_size=32,
        verbose=1
    )

    print("\nTraining completed!")
    print(f"  Final loss: {history['loss'][-1]:.4f}")
    print(f"  Final val_loss: {history['val_loss'][-1]:.4f}")

    # Make prediction
    X_test = np.random.randn(1, lookback, n_features)
    prediction = model.predict(X_test)

    print("\nSample Prediction (returns %):")
    print(f"  1h: {prediction[0][0]:+.2f}%")
    print(f"  4h: {prediction[0][1]:+.2f}%")
    print(f"  24h: {prediction[0][2]:+.2f}%")

    # Direction accuracy
    accuracy = model.calculate_direction_accuracy(X_val, y_val)
    print(f"\nDirection Accuracy: {accuracy:.1%}")

    # Save model
    model.save("btc_lstm_demo.h5")
    print("\nModel saved to btc_lstm_demo.h5")


if __name__ == "__main__":
    import sys
    print("WARNING: This is a demo script. Do not use in production.", file=sys.stderr)
    print("Models should be trained with real data via the trainer module.", file=sys.stderr)
    import asyncio
    asyncio.run(main())
