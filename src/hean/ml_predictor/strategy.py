"""
ML Price Predictor Trading Strategy

Integrates LSTM predictions with HEAN trading system
"""

import asyncio
import logging

from hean.core.bus import EventBus
from hean.core.types import Signal
from hean.strategies.base import BaseStrategy

from .models import PredictionDirection, PricePrediction
from .predictor import PricePredictor

logger = logging.getLogger(__name__)


class MLPredictorStrategy(BaseStrategy):
    """
    Trading strategy based on ML price predictions

    Uses trained LSTM model to predict future prices and generates
    trading signals based on predictions.

    Features:
    - Multi-timeframe predictions (1h, 4h, 24h)
    - Confidence-based position sizing
    - Direction-based entry/exit
    - Adaptive stop-loss and take-profit

    Usage:
        strategy = MLPredictorStrategy(
            bus=event_bus,
            model_path="models/btcusdt_v1.h5",
            symbols=["BTCUSDT"],
            min_confidence=0.7
        )
    """

    def __init__(
        self,
        bus: EventBus,
        model_path: str,
        symbols: list[str] | None = None,
        enabled: bool = True,
        min_confidence: float = 0.7,
        min_expected_return: float = 2.0,  # Minimum 2% expected return
        check_interval_seconds: int = 3600,  # Check every hour
        use_stop_loss: bool = True,
        stop_loss_pct: float = 2.0,  # 2% stop loss
        use_take_profit: bool = True,
        take_profit_multiplier: float = 2.0  # 2x expected return
    ):
        """
        Initialize ML predictor strategy

        Args:
            bus: Event bus
            model_path: Path to trained model
            symbols: Symbols to trade
            enabled: Enable strategy
            min_confidence: Minimum prediction confidence (0-1)
            min_expected_return: Minimum expected return %
            check_interval_seconds: How often to check predictions
            use_stop_loss: Use stop loss
            stop_loss_pct: Stop loss %
            use_take_profit: Use take profit
            take_profit_multiplier: Take profit as multiple of expected return
        """
        super().__init__("ml_predictor", bus)

        self.model_path = model_path
        self.symbols = symbols or ["BTCUSDT"]
        self.enabled = enabled
        self.min_confidence = min_confidence
        self.min_expected_return = min_expected_return
        self.check_interval_seconds = check_interval_seconds
        self.use_stop_loss = use_stop_loss
        self.stop_loss_pct = stop_loss_pct
        self.use_take_profit = use_take_profit
        self.take_profit_multiplier = take_profit_multiplier

        # Initialize predictor
        self.predictor = PricePredictor()

        # Track active positions
        self.active_positions: dict[str, PricePrediction] = {}

        # Stats
        self.predictions_made = 0
        self.trades_executed = 0
        self.correct_predictions = 0

        self._initialized = False

    async def initialize(self):
        """Initialize strategy"""
        if self._initialized:
            return

        # Load model
        await self.predictor.load_model(self.model_path)

        self._initialized = True

        logger.info(
            f"ML Predictor strategy initialized: "
            f"model={self.model_path}, "
            f"symbols={self.symbols}, "
            f"min_confidence={self.min_confidence:.0%}"
        )

    async def on_tick(self, event):
        """Handle tick events - not used (we check predictions periodically)"""
        pass

    async def on_funding(self, event):
        """Handle funding events - not used"""
        pass

    async def run(self):
        """
        Main strategy loop

        Periodically makes predictions and generates signals
        """
        if not self._initialized:
            await self.initialize()

        if not self.enabled:
            logger.info("ML Predictor strategy is disabled")
            return

        logger.info("Starting ML Predictor strategy")

        while self.enabled:
            try:
                await self._check_predictions()
                await asyncio.sleep(self.check_interval_seconds)

            except Exception as e:
                logger.error(f"Error in ML Predictor loop: {e}")
                await asyncio.sleep(self.check_interval_seconds)

    async def _check_predictions(self):
        """Check predictions for all symbols"""
        for symbol in self.symbols:
            try:
                # Make prediction
                prediction = await self.predictor.predict(symbol)
                self.predictions_made += 1

                # Check if we should trade
                if self._should_trade(symbol, prediction):
                    await self._execute_signal(symbol, prediction)

            except Exception as e:
                logger.error(f"Error predicting {symbol}: {e}")

    def _should_trade(self, symbol: str, prediction: PricePrediction) -> bool:
        """Check if prediction meets our criteria"""
        # Basic checks
        if not prediction.should_trade:
            return False

        # Get best timeframe
        tf, conf, direction = prediction.best_timeframe

        if conf < self.min_confidence:
            return False

        # Check expected return
        if tf == "1h" and abs(prediction.expected_return_1h) < self.min_expected_return:
            return False
        elif tf == "4h" and abs(prediction.expected_return_4h) < self.min_expected_return:
            return False
        elif tf == "24h" and abs(prediction.expected_return_24h) < self.min_expected_return:
            return False

        # Check if already in position
        if symbol in self.active_positions:
            logger.debug(f"Already in position for {symbol}")
            return False

        return True

    async def _execute_signal(self, symbol: str, prediction: PricePrediction):
        """Execute trading signal based on prediction"""
        try:
            # Get best timeframe and details
            tf, conf, direction = prediction.best_timeframe

            # Determine side
            is_long = direction in [PredictionDirection.UP, PredictionDirection.STRONG_UP]
            side = "buy" if is_long else "sell"

            # Get expected return for this timeframe
            if tf == "1h":
                expected_return = prediction.expected_return_1h
            elif tf == "4h":
                expected_return = prediction.expected_return_4h
            else:
                expected_return = prediction.expected_return_24h

            # Calculate stop loss and take profit
            entry_price = prediction.current_price

            stop_loss = None
            if self.use_stop_loss:
                if is_long:
                    stop_loss = entry_price * (1 - self.stop_loss_pct / 100)
                else:
                    stop_loss = entry_price * (1 + self.stop_loss_pct / 100)

            take_profit = None
            if self.use_take_profit:
                target_return = abs(expected_return) * self.take_profit_multiplier
                if is_long:
                    take_profit = entry_price * (1 + target_return / 100)
                else:
                    take_profit = entry_price * (1 - target_return / 100)

            logger.info(
                f"Executing ML prediction signal: {symbol} {side.upper()} "
                f"(conf: {conf:.0%}, expected: {expected_return:+.2f}%, tf: {tf})"
            )

            # Create HEAN signal
            signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    "source": "ml_predictor",
                    "model_version": prediction.model_version,
                    "timeframe": tf,
                    "confidence": conf,
                    "expected_return": expected_return,
                    "direction": direction.value,
                    "predicted_price_1h": prediction.price_1h,
                    "predicted_price_4h": prediction.price_4h,
                    "predicted_price_24h": prediction.price_24h
                }
            )

            # Publish signal
            await self._publish_signal(signal)

            # Track position
            self.active_positions[symbol] = prediction
            self.trades_executed += 1

            logger.info(f"ML prediction signal published for {symbol}")

        except Exception as e:
            logger.error(f"Error executing signal: {e}")

    async def close_position(
        self,
        symbol: str,
        actual_return: float
    ):
        """
        Close position and update statistics

        Args:
            symbol: Symbol to close
            actual_return: Actual return %
        """
        if symbol not in self.active_positions:
            return

        prediction = self.active_positions[symbol]
        del self.active_positions[symbol]

        # Check if prediction was correct
        tf, conf, direction = prediction.best_timeframe

        predicted_direction = prediction.is_bullish
        actual_direction = actual_return > 0

        if predicted_direction == actual_direction:
            self.correct_predictions += 1

        logger.info(
            f"Closed ML position: {symbol} "
            f"Return: {actual_return:+.2f}% "
            f"(predicted: {prediction.expected_return_1h:+.2f}%)"
        )

    def get_stats(self) -> dict:
        """Get strategy statistics"""
        accuracy = (
            self.correct_predictions / self.trades_executed
            if self.trades_executed > 0 else 0
        )

        return {
            "strategy": "ml_predictor",
            "enabled": self.enabled,
            "model_path": self.model_path,
            "symbols": self.symbols,
            "predictions_made": self.predictions_made,
            "trades_executed": self.trades_executed,
            "prediction_accuracy": accuracy,
            "active_positions": len(self.active_positions),
            "min_confidence": self.min_confidence,
            "min_expected_return": self.min_expected_return
        }


# Example usage
async def main():
    """Example usage"""
    from hean.core.bus import EventBus

    bus = EventBus()

    strategy = MLPredictorStrategy(
        bus=bus,
        model_path="models/btcusdt_v1_20260130.h5",
        symbols=["BTCUSDT", "ETHUSDT"],
        enabled=True,
        min_confidence=0.7,
        min_expected_return=2.0,
        check_interval_seconds=3600
    )

    await strategy.initialize()

    # Run one check
    await strategy._check_predictions()

    # Print stats
    stats = strategy.get_stats()
    print("\nStrategy Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
