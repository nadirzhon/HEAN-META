"""
Example: Monitoring & Event Streaming (Final Phase 3 Modules)

Demonstrates:
1. Prometheus metrics collection and export
2. Redis Streams event publishing and consuming
3. Kafka event streaming (optional)
4. Integrated monitoring + streaming system

Author: HEAN Team
"""

import asyncio
import time
from datetime import datetime

from loguru import logger


async def example_prometheus_metrics() -> None:
    """Example: Prometheus metrics collection."""
    logger.info("=== Prometheus Metrics Example ===")

    from hean.monitoring import MetricsCollector, MetricsConfig

    # Initialize metrics
    config = MetricsConfig(port=8000)
    metrics = MetricsCollector(config)

    # Start HTTP server (metrics available at http://localhost:8000/metrics)
    metrics.start_server()

    logger.info("Metrics server started at http://localhost:8000/metrics")
    logger.info("Open http://localhost:8000/metrics in your browser to see Prometheus metrics")

    # Simulate trading activity
    logger.info("\nSimulating trading activity...")

    # 1. Record trades
    trades = [
        ("BTC/USDT", "BUY", 0.1, 150.0, True),
        ("ETH/USDT", "BUY", 2.0, 80.0, True),
        ("BTC/USDT", "SELL", 0.1, -30.0, False),
        ("SOL/USDT", "BUY", 10.0, 120.0, True),
    ]

    for symbol, side, size, pnl, is_win in trades:
        metrics.record_trade(
            symbol=symbol,
            side=side,
            size=size,
            pnl=pnl,
            is_win=is_win,
            strategy="ml_ensemble",
        )
        await asyncio.sleep(0.1)

    logger.info(f"Recorded {len(trades)} trades")

    # 2. Update account metrics
    metrics.update_account_balance(10320.0)
    metrics.update_position("BTC/USDT", size=0.5, unrealized_pnl=250.0)
    metrics.update_position("ETH/USDT", size=2.0, unrealized_pnl=100.0)

    # 3. Record ML predictions
    predictions = [
        ("ensemble", "BTC/USDT", "UP", 0.68, 45.0),
        ("lstm", "BTC/USDT", "UP", 0.72, 120.0),
        ("xgboost", "ETH/USDT", "DOWN", 0.55, 30.0),
    ]

    for model, symbol, pred, conf, inf_time in predictions:
        metrics.record_prediction(
            model=model,
            symbol=symbol,
            prediction=pred,
            confidence=conf,
            inference_time_ms=inf_time,
        )

    logger.info(f"Recorded {len(predictions)} predictions")

    # 4. Update model accuracy
    metrics.update_model_accuracy("ensemble", 0.62)
    metrics.update_model_accuracy("lstm", 0.58)

    # 5. Record API calls
    for _ in range(10):
        metrics.record_api_call(
            exchange="binance",
            latency_ms=120.0,
            success=True,
            endpoint="ticker",
        )

    # 6. Record cache hits/misses
    for _ in range(20):
        metrics.record_cache_access(hit=True, cache_type="features")
    for _ in range(5):
        metrics.record_cache_access(hit=False, cache_type="features")

    logger.info("Cache stats: 20 hits, 5 misses (80% hit rate)")

    # 7. Update risk metrics
    metrics.update_drawdown(current=0.05, maximum=0.12)
    metrics.update_exposure(total_exposure=5000.0, leverage=2.0)

    # 8. Update performance metrics
    metrics.update_performance(
        sharpe_ratio=2.8,
        win_rate=0.65,
        profit_factor=2.3,
    )

    logger.info("\n✅ All metrics recorded!")
    logger.info("Check http://localhost:8000/metrics to see the data")

    # Keep server running for a bit
    logger.info("\nServer will run for 10 seconds, then shutdown...")
    await asyncio.sleep(10)


async def example_redis_streams_publishing() -> None:
    """Example: Publishing events to Redis Streams."""
    logger.info("\n=== Redis Streams Publishing Example ===")

    from hean.streaming import (
        EventPublisher,
        EventStreamConfig,
        StreamBackend,
        TradeEvent,
        SignalEvent,
        PredictionEvent,
    )

    # Initialize publisher
    config = EventStreamConfig(
        backend=StreamBackend.REDIS,
        redis_host="localhost",
        redis_port=6379,
    )

    publisher = EventPublisher(config)
    await publisher.start()

    logger.info("Redis Streams publisher started")

    # 1. Publish trade events
    logger.info("\nPublishing trade events...")
    trade_event = TradeEvent(
        symbol="BTC/USDT",
        side="BUY",
        size=0.1,
        price=50000.0,
        pnl=150.0,
        strategy="ml_ensemble",
    )
    await publisher.publish(trade_event)

    trade_event2 = TradeEvent(
        symbol="ETH/USDT",
        side="SELL",
        size=2.0,
        price=3000.0,
        pnl=-30.0,
        strategy="sentiment",
    )
    await publisher.publish(trade_event2)

    logger.info("Published 2 trade events")

    # 2. Publish signal events
    logger.info("\nPublishing signal events...")
    signal_event = SignalEvent(
        symbol="BTC/USDT",
        direction="BUY",
        strength=0.75,
        source="ml_ensemble",
    )
    await publisher.publish(signal_event)

    signal_event2 = SignalEvent(
        symbol="SOL/USDT",
        direction="SELL",
        strength=0.60,
        source="sentiment_analysis",
    )
    await publisher.publish(signal_event2)

    logger.info("Published 2 signal events")

    # 3. Publish prediction events
    logger.info("\nPublishing prediction events...")
    pred_event = PredictionEvent(
        model="ensemble",
        symbol="BTC/USDT",
        prediction="UP",
        confidence=0.68,
        features={"rsi": 45.0, "macd": 0.02, "vol": 1.8},
    )
    await publisher.publish(pred_event)

    logger.info("Published 1 prediction event")

    # 4. Batch publishing
    logger.info("\nBatch publishing events...")
    batch_events = [
        TradeEvent("ADA/USDT", "BUY", 100.0, 0.5, 20.0, "default"),
        TradeEvent("DOT/USDT", "SELL", 50.0, 7.0, -10.0, "default"),
        SignalEvent("MATIC/USDT", "BUY", 0.55, "ta_indicators"),
    ]
    await publisher.publish_batch(batch_events)

    logger.info(f"Published {len(batch_events)} events in batch")

    await publisher.stop()
    logger.info("\n✅ Publishing complete!")


async def example_redis_streams_consuming() -> None:
    """Example: Consuming events from Redis Streams."""
    logger.info("\n=== Redis Streams Consuming Example ===")

    from hean.streaming import (
        EventConsumer,
        EventStreamConfig,
        StreamBackend,
        EventType,
        Event,
    )

    # Define event handlers
    async def handle_trade(event: Event):
        """Handle trade events."""
        logger.info(f"📈 TRADE: {event.data['symbol']} {event.data['side']} "
                   f"{event.data['size']} @ ${event.data['price']:.2f} "
                   f"| PnL: ${event.data['pnl']:.2f}")

    async def handle_signal(event: Event):
        """Handle signal events."""
        logger.info(f"🎯 SIGNAL: {event.data['symbol']} {event.data['direction']} "
                   f"| Strength: {event.data['strength']:.1%} "
                   f"| Source: {event.data['source']}")

    async def handle_prediction(event: Event):
        """Handle prediction events."""
        logger.info(f"🤖 PREDICTION: {event.data['model']} -> {event.data['prediction']} "
                   f"| Symbol: {event.data['symbol']} "
                   f"| Confidence: {event.data['confidence']:.1%}")

    # Initialize consumer
    config = EventStreamConfig(
        backend=StreamBackend.REDIS,
        redis_host="localhost",
        redis_port=6379,
        consumer_group="example-group",
        consumer_name="consumer-1",
    )

    consumer = EventConsumer(config)
    await consumer.start()

    # Subscribe to events
    await consumer.subscribe(EventType.TRADE, handle_trade)
    await consumer.subscribe(EventType.SIGNAL, handle_signal)
    await consumer.subscribe(EventType.PREDICTION, handle_prediction)

    logger.info("Consumer subscribed to TRADE, SIGNAL, PREDICTION events")
    logger.info("Consuming for 5 seconds...")

    # Consume for 5 seconds
    consume_task = asyncio.create_task(consumer.consume())

    await asyncio.sleep(5)

    await consumer.stop()
    consume_task.cancel()

    logger.info("\n✅ Consuming complete!")


async def example_integrated_monitoring_streaming() -> None:
    """Example: Integrated monitoring + streaming system."""
    logger.info("\n=== Integrated Monitoring + Streaming Example ===")

    from hean.monitoring import MetricsCollector
    from hean.streaming import (
        EventPublisher,
        EventStreamConfig,
        StreamBackend,
        TradeEvent,
        SignalEvent,
    )

    # Initialize systems
    metrics = MetricsCollector()
    metrics.start_server(port=8001)

    publisher = EventPublisher(EventStreamConfig(backend=StreamBackend.REDIS))
    await publisher.start()

    logger.info("Integrated system started")
    logger.info("Metrics: http://localhost:8001/metrics")

    # Simulate trading loop
    logger.info("\nSimulating trading activity...")

    for i in range(5):
        symbol = "BTC/USDT"
        side = "BUY" if i % 2 == 0 else "SELL"
        size = 0.1
        price = 50000 + i * 100
        pnl = (i - 2) * 50  # Some wins, some losses

        # 1. Publish signal event
        signal = SignalEvent(
            symbol=symbol,
            direction=side,
            strength=0.60 + i * 0.05,
            source="integrated_system",
        )
        await publisher.publish(signal)

        # 2. Execute trade and publish event
        trade = TradeEvent(
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            pnl=pnl,
            strategy="integrated",
        )
        await publisher.publish(trade)

        # 3. Record metrics
        metrics.record_trade(
            symbol=symbol,
            side=side,
            size=size,
            pnl=pnl,
            is_win=(pnl > 0),
            strategy="integrated",
        )

        # 4. Update account balance
        metrics.update_account_balance(10000 + sum(range(i + 1)) * 50)

        logger.info(f"Trade {i+1}: {side} {size} {symbol} @ ${price:.2f} | PnL: ${pnl:.2f}")

        await asyncio.sleep(0.5)

    # Update final metrics
    metrics.update_performance(
        sharpe_ratio=2.5,
        win_rate=0.60,
        profit_factor=2.0,
    )

    logger.info("\n✅ Integrated system example complete!")
    logger.info("Check metrics at http://localhost:8001/metrics")

    await publisher.stop()
    await asyncio.sleep(2)


async def example_kafka_streaming() -> None:
    """Example: Kafka event streaming (optional, requires Kafka)."""
    logger.info("\n=== Kafka Streaming Example (Optional) ===")

    try:
        from hean.streaming import (
            EventPublisher,
            EventStreamConfig,
            StreamBackend,
            TradeEvent,
        )

        # Initialize Kafka publisher
        config = EventStreamConfig(
            backend=StreamBackend.KAFKA,
            kafka_brokers=["localhost:9092"],
        )

        publisher = EventPublisher(config)
        await publisher.start()

        logger.info("Kafka publisher started")

        # Publish event
        trade = TradeEvent(
            symbol="BTC/USDT",
            side="BUY",
            size=0.1,
            price=50000.0,
            pnl=150.0,
            strategy="kafka_test",
        )
        await publisher.publish(trade)

        logger.info("Published event to Kafka")

        await publisher.stop()
        logger.info("✅ Kafka example complete!")

    except ImportError:
        logger.warning("⚠️ aiokafka not installed. Install with: pip install aiokafka")
    except Exception as e:
        logger.warning(f"⚠️ Kafka not available: {e}")
        logger.info("This is optional - Redis Streams works great for most use cases")


async def main() -> None:
    """Run all examples."""
    # 1. Prometheus metrics
    await example_prometheus_metrics()

    # 2. Redis Streams publishing
    await example_redis_streams_publishing()

    # 3. Redis Streams consuming
    await example_redis_streams_consuming()

    # 4. Integrated system
    await example_integrated_monitoring_streaming()

    # 5. Kafka (optional)
    await example_kafka_streaming()

    logger.info("\n" + "="*60)
    logger.info("✅ All monitoring & streaming examples completed!")
    logger.info("="*60)


if __name__ == "__main__":
    asyncio.run(main())
