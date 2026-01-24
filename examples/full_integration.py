"""Integration example: Full system with all modules connected."""

import asyncio
import os
from typing import Any

from hean.ai.factory import AIFactory
from hean.core.bus import EventBus
from hean.core.event_streams import HybridEventBus, RedisEventStream
from hean.core.orchestrator import TradingOrchestrator
from hean.core.types import Event, EventType
from hean.execution.router import ExecutionRouter
from hean.logging import get_logger
from hean.observability.health import HealthMonitor
from hean.observability.metrics import MetricsCollector
from hean.risk.risk_governor import RiskGovernor

logger = get_logger(__name__)


class FeatureEngine:
    """Feature engineering module - calculates technical indicators."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._running = False

    async def start(self) -> None:
        """Start feature engine."""
        logger.info("Starting FeatureEngine...")
        self.bus.subscribe(EventType.TICK, self.on_tick)
        self.bus.subscribe(EventType.CANDLE, self.on_candle)
        self._running = True
        logger.info("FeatureEngine started")

    async def stop(self) -> None:
        """Stop feature engine."""
        logger.info("Stopping FeatureEngine...")
        self._running = False
        logger.info("FeatureEngine stopped")

    async def on_tick(self, event: Event) -> None:
        """Process tick data and generate features.

        Args:
            event: TICK event
        """
        try:
            # Extract tick data
            data = event.data
            symbol = data.get("symbol")
            price = data.get("price")

            # TODO: Calculate technical indicators (RSI, MACD, etc.)
            # For now, just pass through
            features = {
                "symbol": symbol,
                "price": price,
                "rsi": 50.0,  # Placeholder
                "macd": 0.0,  # Placeholder
                "volume": data.get("volume", 0),
            }

            # Publish features
            await self.bus.publish(
                Event(
                    event_type=EventType.FEATURES_READY,
                    data={"symbol": symbol, "features": features},
                )
            )

        except Exception as e:
            logger.error(f"Error processing tick: {e}", exc_info=True)
            await self.bus.publish(
                Event(
                    event_type=EventType.ERROR,
                    data={"component": "FeatureEngine", "error": str(e)},
                )
            )

    async def on_candle(self, event: Event) -> None:
        """Process candle data.

        Args:
            event: CANDLE event
        """
        # Similar processing for candles
        pass

    def health_check(self) -> bool:
        """Health check."""
        return self._running


class MLInferenceEngine:
    """ML inference module - generates predictions."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._running = False
        self._model = None

    async def start(self) -> None:
        """Start ML engine."""
        logger.info("Starting MLInferenceEngine...")
        self.bus.subscribe(EventType.FEATURES_READY, self.on_features)

        # TODO: Load ML models (LightGBM, XGBoost, RL)
        # self._model = load_model()

        self._running = True
        logger.info("MLInferenceEngine started")

    async def stop(self) -> None:
        """Stop ML engine."""
        logger.info("Stopping MLInferenceEngine...")
        self._running = False
        logger.info("MLInferenceEngine stopped")

    async def on_features(self, event: Event) -> None:
        """Generate ML predictions from features.

        Args:
            event: FEATURES_READY event
        """
        try:
            data = event.data
            symbol = data.get("symbol")
            features = data.get("features")

            # TODO: Run ML inference
            # prediction = self._model.predict(features)

            # Placeholder prediction
            prediction = {"signal": 0.65, "confidence": 0.82, "direction": "BUY"}

            # Publish prediction
            await self.bus.publish(
                Event(
                    event_type=EventType.ML_PREDICTION,
                    data={
                        "symbol": symbol,
                        "prediction": prediction,
                        "features": features,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Error in ML inference: {e}", exc_info=True)
            await self.bus.publish(
                Event(
                    event_type=EventType.ERROR,
                    data={"component": "MLInferenceEngine", "error": str(e)},
                )
            )

    def health_check(self) -> bool:
        """Health check."""
        return self._running and self._model is not None


class TradingStrategy:
    """Trading strategy - combines ML + technical analysis."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._running = False

    async def start(self) -> None:
        """Start strategy."""
        logger.info("Starting TradingStrategy...")
        self.bus.subscribe(EventType.ML_PREDICTION, self.on_prediction)
        self._running = True
        logger.info("TradingStrategy started")

    async def stop(self) -> None:
        """Stop strategy."""
        logger.info("Stopping TradingStrategy...")
        self._running = False
        logger.info("TradingStrategy stopped")

    async def on_prediction(self, event: Event) -> None:
        """Process ML prediction and generate trading signal.

        Args:
            event: ML_PREDICTION event
        """
        try:
            data = event.data
            symbol = data.get("symbol")
            prediction = data.get("prediction")
            features = data.get("features")

            # Decision logic
            signal_strength = prediction.get("signal", 0)
            confidence = prediction.get("confidence", 0)

            # Only trade if high confidence
            if confidence > 0.75 and abs(signal_strength) > 0.6:
                # Generate trading signal
                signal = {
                    "symbol": symbol,
                    "side": "BUY" if signal_strength > 0 else "SELL",
                    "size": 100.0,  # USD
                    "signal_strength": signal_strength,
                    "confidence": confidence,
                }

                await self.bus.publish(
                    Event(event_type=EventType.SIGNAL, data=signal)
                )
                logger.info(f"Generated signal: {signal}")

        except Exception as e:
            logger.error(f"Error generating signal: {e}", exc_info=True)
            await self.bus.publish(
                Event(
                    event_type=EventType.ERROR,
                    data={"component": "TradingStrategy", "error": str(e)},
                )
            )

    def health_check(self) -> bool:
        """Health check."""
        return self._running


async def run_integrated_system() -> None:
    """Run complete integrated trading system."""
    logger.info("Starting Integrated Trading System...")

    # 1. Create EventBus (hybrid: in-memory + Redis)
    in_memory_bus = EventBus()
    redis_stream = RedisEventStream(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")
    )
    hybrid_bus = HybridEventBus(in_memory_bus, redis_stream)

    # 2. Create Orchestrator
    orchestrator = TradingOrchestrator(bus=hybrid_bus)

    # 3. Create and register components
    feature_engine = FeatureEngine(bus=hybrid_bus)
    ml_engine = MLInferenceEngine(bus=hybrid_bus)
    strategy = TradingStrategy(bus=hybrid_bus)

    # Initialize existing modules
    risk_governor = RiskGovernor(bus=hybrid_bus)
    execution_router = ExecutionRouter(bus=hybrid_bus)
    metrics_collector = MetricsCollector()
    health_monitor = HealthMonitor(orchestrator)

    # Register all components
    orchestrator.register_component("feature_engine", feature_engine)
    orchestrator.register_component("ml_engine", ml_engine)
    orchestrator.register_component("strategy", strategy)
    orchestrator.register_component("risk_governor", risk_governor)
    orchestrator.register_component("execution_router", execution_router)
    orchestrator.register_component(
        "metrics_collector", metrics_collector, circuit_breaker=False
    )
    orchestrator.register_component(
        "health_monitor", health_monitor, circuit_breaker=False
    )

    # 4. Start orchestrator (starts all components)
    await orchestrator.start()

    # 5. Simulate market data
    logger.info("Simulating market data...")
    for i in range(5):
        await asyncio.sleep(1)

        # Publish tick event
        await hybrid_bus.publish(
            Event(
                event_type=EventType.TICK,
                data={
                    "symbol": "BTC/USDT",
                    "price": 50000 + i * 100,
                    "volume": 1000,
                },
            )
        )

    # 6. Health check
    health = await orchestrator.health_check()
    logger.info(f"System health: {health}")

    # 7. Metrics
    metrics = await orchestrator.get_metrics()
    logger.info(f"System metrics: {metrics}")

    # 8. Graceful shutdown
    logger.info("Shutting down system...")
    await orchestrator.stop()

    logger.info("System stopped successfully")


async def main() -> None:
    """Main entry point."""
    try:
        await run_integrated_system()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
