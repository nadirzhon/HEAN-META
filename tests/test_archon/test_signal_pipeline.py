"""Tests for Signal Pipeline Manager."""

import asyncio

import pytest

from hean.archon.signal_pipeline import SignalStage
from hean.archon.signal_pipeline_manager import SignalPipelineManager
from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Signal


async def test_happy_path():
    """Test signal lifecycle: GENERATED → ORDER_REQUEST → ORDER_PLACED → ORDER_FILLED."""
    bus = EventBus()
    await bus.start()

    pipeline = SignalPipelineManager(bus=bus, max_active=100, stage_timeout_sec=5.0)
    await pipeline.start()

    # Step 1: Publish SIGNAL event
    signal = Signal(
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        confidence=0.8,
    )
    await bus.publish(Event(event_type=EventType.SIGNAL, data={"signal": signal}))
    await asyncio.sleep(0.05)

    # Check signal is tracked
    status = pipeline.get_status()
    assert status["active_count"] == 1
    assert status["signals_tracked"] == 1

    # Step 2: Risk approves → ORDER_REQUEST
    await bus.publish(
        Event(
            event_type=EventType.ORDER_REQUEST,
            data={
                "strategy_id": "test_strategy",
                "symbol": "BTCUSDT",
                "side": "buy",
                "signal_id": "sig_123",
            },
        )
    )
    await asyncio.sleep(0.05)

    # Step 3: Order placed
    await bus.publish(
        Event(
            event_type=EventType.ORDER_PLACED,
            data={
                "strategy_id": "test_strategy",
                "symbol": "BTCUSDT",
                "side": "buy",
                "order_id": "ord_456",
            },
        )
    )
    await asyncio.sleep(0.05)

    # Step 4: Order filled
    await bus.publish(
        Event(
            event_type=EventType.ORDER_FILLED,
            data={
                "order_id": "ord_456",
                "fill_price": 50010.0,
                "fill_qty": 0.1,
            },
        )
    )
    await asyncio.sleep(0.05)

    # Check final state
    status = pipeline.get_status()
    assert status["active_count"] == 0  # Moved to completed
    assert status["signals_tracked"] == 1
    assert status["signals_completed"] == 1
    assert status["fill_rate_pct"] == 100.0
    assert status["avg_latency_ms"] > 0

    await pipeline.stop()
    await bus.stop()


async def test_risk_blocked():
    """Test signal blocked by risk layer."""
    bus = EventBus()
    await bus.start()

    pipeline = SignalPipelineManager(bus=bus, max_active=100, stage_timeout_sec=5.0)
    await pipeline.start()

    # Publish signal
    signal = Signal(
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
    )
    await bus.publish(Event(event_type=EventType.SIGNAL, data={"signal": signal}))
    await asyncio.sleep(0.05)

    # Risk blocks
    await bus.publish(
        Event(
            event_type=EventType.RISK_BLOCKED,
            data={
                "strategy_id": "test_strategy",
                "symbol": "BTCUSDT",
                "side": "buy",
                "reason": "max_positions_exceeded",
                "risk_state": "QUARANTINE",
            },
        )
    )
    await asyncio.sleep(0.05)

    # Check dead letter
    status = pipeline.get_status()
    assert status["active_count"] == 0
    assert status["signals_blocked"] == 1
    assert status["dead_letter_count"] == 1
    assert len(status["recent_dead_letters"]) == 1

    dead = status["recent_dead_letters"][0]
    assert dead["current_stage"] == "risk_blocked"

    await pipeline.stop()
    await bus.stop()


async def test_order_rejected():
    """Test order rejected by exchange."""
    bus = EventBus()
    await bus.start()

    pipeline = SignalPipelineManager(bus=bus, max_active=100, stage_timeout_sec=5.0)
    await pipeline.start()

    # Publish signal
    signal = Signal(
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="sell",
        entry_price=50000.0,
    )
    await bus.publish(Event(event_type=EventType.SIGNAL, data={"signal": signal}))
    await asyncio.sleep(0.05)

    # Risk approves
    await bus.publish(
        Event(
            event_type=EventType.ORDER_REQUEST,
            data={"strategy_id": "test_strategy", "symbol": "BTCUSDT", "side": "sell"},
        )
    )
    await asyncio.sleep(0.05)

    # Order placed
    await bus.publish(
        Event(
            event_type=EventType.ORDER_PLACED,
            data={
                "strategy_id": "test_strategy",
                "symbol": "BTCUSDT",
                "side": "sell",
                "order_id": "ord_789",
            },
        )
    )
    await asyncio.sleep(0.05)

    # Order rejected
    await bus.publish(
        Event(
            event_type=EventType.ORDER_REJECTED,
            data={"order_id": "ord_789", "reason": "insufficient_margin"},
        )
    )
    await asyncio.sleep(0.05)

    # Check dead letter
    status = pipeline.get_status()
    assert status["active_count"] == 0
    assert status["signals_rejected"] == 1
    assert status["dead_letter_count"] == 1

    await pipeline.stop()
    await bus.stop()


async def test_timeout():
    """Test signal timeout detection."""
    bus = EventBus()
    await bus.start()

    pipeline = SignalPipelineManager(
        bus=bus,
        max_active=100,
        stage_timeout_sec=1.0,  # Short timeout
    )
    await pipeline.start()

    # Publish signal and do nothing else
    signal = Signal(
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
    )
    await bus.publish(Event(event_type=EventType.SIGNAL, data={"signal": signal}))
    await asyncio.sleep(0.05)

    # Initially active
    status = pipeline.get_status()
    assert status["active_count"] == 1

    # Wait for timeout
    await asyncio.sleep(2.5)

    # Should be timed out
    status = pipeline.get_status()
    assert status["active_count"] == 0
    assert status["signals_timed_out"] == 1
    assert status["dead_letter_count"] == 1

    await pipeline.stop()
    await bus.stop()


async def test_eviction():
    """Test max_active signals eviction."""
    bus = EventBus()
    await bus.start()

    pipeline = SignalPipelineManager(
        bus=bus,
        max_active=3,
        stage_timeout_sec=10.0,  # Very small limit
    )
    await pipeline.start()

    # Publish 5 signals
    for i in range(5):
        signal = Signal(
            strategy_id=f"strategy_{i}",
            symbol="BTCUSDT",
            side="buy",
            entry_price=50000.0 + i,
        )
        await bus.publish(Event(event_type=EventType.SIGNAL, data={"signal": signal}))
        await asyncio.sleep(0.02)

    # Should have 3 active and 2 evicted
    status = pipeline.get_status()
    assert status["active_count"] == 3
    assert status["signals_evicted"] == 2
    assert status["dead_letter_count"] == 2

    await pipeline.stop()
    await bus.stop()


async def test_metrics_accuracy():
    """Test fill_rate_pct and avg_latency_ms calculation."""
    bus = EventBus()
    await bus.start()

    pipeline = SignalPipelineManager(bus=bus, max_active=100, stage_timeout_sec=5.0)
    await pipeline.start()

    # Track 3 signals: 2 complete, 1 blocked
    for i in range(3):
        signal = Signal(
            strategy_id=f"strategy_{i}",
            symbol="BTCUSDT",
            side="buy",
            entry_price=50000.0 + i,
        )
        await bus.publish(Event(event_type=EventType.SIGNAL, data={"signal": signal}))
        await asyncio.sleep(0.02)

        if i < 2:
            # Complete these
            await bus.publish(
                Event(
                    event_type=EventType.ORDER_REQUEST,
                    data={
                        "strategy_id": f"strategy_{i}",
                        "symbol": "BTCUSDT",
                        "side": "buy",
                    },
                )
            )
            await asyncio.sleep(0.02)

            await bus.publish(
                Event(
                    event_type=EventType.ORDER_PLACED,
                    data={
                        "strategy_id": f"strategy_{i}",
                        "symbol": "BTCUSDT",
                        "side": "buy",
                        "order_id": f"ord_{i}",
                    },
                )
            )
            await asyncio.sleep(0.02)

            await bus.publish(
                Event(
                    event_type=EventType.ORDER_FILLED,
                    data={"order_id": f"ord_{i}", "fill_price": 50000.0, "fill_qty": 0.1},
                )
            )
            await asyncio.sleep(0.02)
        else:
            # Block this one
            await bus.publish(
                Event(
                    event_type=EventType.RISK_BLOCKED,
                    data={
                        "strategy_id": f"strategy_{i}",
                        "symbol": "BTCUSDT",
                        "side": "buy",
                        "reason": "test",
                    },
                )
            )
            await asyncio.sleep(0.02)

    status = pipeline.get_status()
    assert status["signals_tracked"] == 3
    assert status["signals_completed"] == 2
    assert status["signals_blocked"] == 1
    # Fill rate = 2 completed / 3 tracked = 66.67%
    assert 66.0 <= status["fill_rate_pct"] <= 67.0
    assert status["avg_latency_ms"] > 0

    await pipeline.stop()
    await bus.stop()


async def test_get_trace():
    """Test get_trace() finds active, completed, and dead-letter signals."""
    bus = EventBus()
    await bus.start()

    pipeline = SignalPipelineManager(bus=bus, max_active=100, stage_timeout_sec=5.0)
    await pipeline.start()

    # Signal 1: Active
    signal1 = Signal(strategy_id="strategy_1", symbol="BTCUSDT", side="buy", entry_price=50000.0)
    await bus.publish(Event(event_type=EventType.SIGNAL, data={"signal": signal1}))
    await asyncio.sleep(0.05)

    active_traces = list(pipeline._active.values())
    assert len(active_traces) == 1
    corr_id_1 = active_traces[0].correlation_id

    trace = pipeline.get_trace(corr_id_1)
    assert trace is not None
    assert trace["strategy_id"] == "strategy_1"
    assert trace["current_stage"] == "generated"

    # Signal 2: Complete
    signal2 = Signal(strategy_id="strategy_2", symbol="ETHUSDT", side="sell", entry_price=3000.0)
    await bus.publish(Event(event_type=EventType.SIGNAL, data={"signal": signal2}))
    await asyncio.sleep(0.05)

    corr_id_2 = [
        t.correlation_id for t in pipeline._active.values() if t.strategy_id == "strategy_2"
    ][0]

    # Complete signal 2
    await bus.publish(
        Event(
            event_type=EventType.ORDER_REQUEST,
            data={"strategy_id": "strategy_2", "symbol": "ETHUSDT", "side": "sell"},
        )
    )
    await asyncio.sleep(0.05)
    await bus.publish(
        Event(
            event_type=EventType.ORDER_PLACED,
            data={
                "strategy_id": "strategy_2",
                "symbol": "ETHUSDT",
                "side": "sell",
                "order_id": "ord_2",
            },
        )
    )
    await asyncio.sleep(0.05)
    await bus.publish(
        Event(
            event_type=EventType.ORDER_FILLED,
            data={"order_id": "ord_2", "fill_price": 3005.0, "fill_qty": 1.0},
        )
    )
    await asyncio.sleep(0.05)

    trace = pipeline.get_trace(corr_id_2)
    assert trace is not None
    assert trace["current_stage"] == "order_filled"
    assert trace["is_terminal"]

    # Signal 3: Dead letter
    signal3 = Signal(strategy_id="strategy_3", symbol="SOLUSDT", side="buy", entry_price=100.0)
    await bus.publish(Event(event_type=EventType.SIGNAL, data={"signal": signal3}))
    await asyncio.sleep(0.05)

    corr_id_3 = [
        t.correlation_id for t in pipeline._active.values() if t.strategy_id == "strategy_3"
    ][0]

    # Block signal 3
    await bus.publish(
        Event(
            event_type=EventType.RISK_BLOCKED,
            data={
                "strategy_id": "strategy_3",
                "symbol": "SOLUSDT",
                "side": "buy",
                "reason": "test_block",
            },
        )
    )
    await asyncio.sleep(0.05)

    trace = pipeline.get_trace(corr_id_3)
    assert trace is not None
    assert trace["current_stage"] == "risk_blocked"

    await pipeline.stop()
    await bus.stop()


async def test_get_status():
    """Test get_status() returns correct counts."""
    bus = EventBus()
    await bus.start()

    pipeline = SignalPipelineManager(bus=bus, max_active=100, stage_timeout_sec=5.0)
    await pipeline.start()

    # Initial status
    status = pipeline.get_status()
    assert status["active_count"] == 0
    assert status["signals_tracked"] == 0
    assert status["signals_completed"] == 0

    # Add signal
    signal = Signal(strategy_id="test", symbol="BTCUSDT", side="buy", entry_price=50000.0)
    await bus.publish(Event(event_type=EventType.SIGNAL, data={"signal": signal}))
    await asyncio.sleep(0.05)

    status = pipeline.get_status()
    assert status["active_count"] == 1
    assert status["signals_tracked"] == 1

    await pipeline.stop()
    await bus.stop()
