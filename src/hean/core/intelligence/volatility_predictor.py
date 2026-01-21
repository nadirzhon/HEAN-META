"""ONNX-based volatility spike predictor with circuit breaker integration."""

import time
from pathlib import Path
from typing import Optional, Tuple

try:
    import graph_engine_py  # type: ignore
    _ONNX_AVAILABLE = getattr(graph_engine_py, 'ENABLE_ONNX', False)
    if _ONNX_AVAILABLE:
        from graph_engine_py import VolatilityPredictor as CPPVolatilityPredictor  # type: ignore
    else:
        CPPVolatilityPredictor = None
except (ImportError, AttributeError):
    _ONNX_AVAILABLE = False
    CPPVolatilityPredictor = None

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.execution.order_manager import OrderManager
from hean.hft.circuit_breaker import CircuitBreaker
from hean.logging import get_logger

logger = get_logger(__name__)


class VolatilitySpikePredictor:
    """Predicts volatility spikes 1 second ahead using TFT model.
    
    If probability > 85%, triggers circuit breaker to clear maker orders.
    """

    def __init__(
        self,
        bus: EventBus,
        order_manager: OrderManager,
        model_path: Optional[str] = None,
        probability_threshold: float = 0.85,
    ):
        """Initialize volatility spike predictor.

        Args:
            bus: Event bus for publishing predictions
            order_manager: Order manager for clearing orders
            model_path: Path to ONNX model file (optional, will look for default)
            probability_threshold: Probability threshold to trigger circuit breaker (default 0.85)
        """
        self._bus = bus
        self._order_manager = order_manager
        self._probability_threshold = probability_threshold
        self._predictor: Optional[CPPVolatilityPredictor] = None
        self._model_loaded = False
        
        # Circuit breaker integration
        self._circuit_breaker = CircuitBreaker()
        self._last_prediction_time = 0.0
        self._prediction_interval_seconds = 0.1  # Predict every 100ms
        
        # Find model file
        if model_path is None:
            # Look for model in project directory
            project_root = Path(__file__).parent.parent.parent.parent.parent
            model_path = project_root / "models" / "tft_volatility_predictor.onnx"
        
        self._model_path = Path(model_path) if model_path else None
        
        # Initialize ONNX predictor if available
        if _ONNX_AVAILABLE and CPPVolatilityPredictor and self._model_path and self._model_path.exists():
            try:
                self._predictor = CPPVolatilityPredictor()
                if self._predictor.load_model(str(self._model_path)):
                    self._model_loaded = True
                    logger.info(f"Volatility predictor model loaded: {self._model_path}")
                else:
                    logger.warning(f"Failed to load model: {self._model_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize ONNX predictor: {e}")
        else:
            if not _ONNX_AVAILABLE:
                logger.warning("ONNX Runtime not available. Volatility prediction disabled.")
            elif not self._model_path or not self._model_path.exists():
                logger.warning(f"Model file not found: {self._model_path}")

    async def start(self) -> None:
        """Start the volatility predictor."""
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        logger.info("Volatility spike predictor started")

    async def stop(self) -> None:
        """Stop the volatility predictor."""
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        logger.info("Volatility spike predictor stopped")

    async def _handle_tick(self, event: Event) -> None:
        """Handle tick events for prediction."""
        if not self._model_loaded:
            return
        
        current_time = time.time()
        if current_time - self._last_prediction_time < self._prediction_interval_seconds:
            return
        
        self._last_prediction_time = current_time
        
        # Get feature vector from graph engine
        # This would be injected or accessed via bus/global state
        feature_vector = await self._get_feature_vector()
        if not feature_vector:
            return
        
        # Predict volatility spike
        if self._predictor is None:
            return
        
        success, probability = self._predictor.predict_volatility_spike(feature_vector)
        
        if success and probability >= self._probability_threshold:
            logger.critical(
                f"VOLATILITY SPIKE PREDICTED: probability={probability:.2%} "
                f"(threshold={self._probability_threshold:.2%}). "
                f"Clearing maker orders to avoid being picked off."
            )
            
            # Trigger circuit breaker - clear all maker orders
            await self._clear_maker_orders()
            
            # Publish prediction event
            await self._bus.publish(
                Event(
                    event_type=EventType.SIGNAL,  # Using SIGNAL as generic event type
                    data={
                        "prediction_type": "volatility_spike",
                        "probability": probability,
                        "threshold": self._probability_threshold,
                        "action": "maker_orders_cleared",
                    }
                )
            )

    async def _get_feature_vector(self) -> Optional[list[float]]:
        """Get feature vector from graph engine.
        
        In a full implementation, this would access the graph engine instance.
        For now, return None to indicate feature vector not available.
        """
        # TODO: Integrate with GraphEngineWrapper to get feature vector
        # This requires access to the graph engine instance
        return None

    async def _clear_maker_orders(self) -> None:
        """Clear all pending maker orders to avoid being picked off during volatility spike."""
        # Cancel all pending maker orders
        pending_orders = self._order_manager.get_pending_orders()
        maker_orders = [
            order for order in pending_orders
            if order.order_type == "limit" and order.post_only
        ]
        
        for order in maker_orders:
            try:
                await self._order_manager.cancel_order(order.order_id)
                logger.info(f"Cleared maker order {order.order_id} due to volatility spike prediction")
            except Exception as e:
                logger.error(f"Failed to cancel order {order.order_id}: {e}")

    def predict(self, feature_vector: list[float]) -> Tuple[bool, float]:
        """Predict volatility spike probability.
        
        Args:
            feature_vector: High-dimensional feature vector from graph engine
            
        Returns:
            (success, probability) tuple
        """
        if not self._model_loaded or self._predictor is None:
            return False, 0.0
        
        try:
            return self._predictor.predict_volatility_spike(feature_vector)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return False, 0.0

    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._model_loaded
