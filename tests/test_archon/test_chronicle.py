"""Tests for Chronicle audit trail."""

import asyncio

from hean.archon.chronicle import Chronicle
from hean.core.bus import EventBus
from hean.core.types import Event, EventType


async def _publish_and_wait(bus: EventBus, event: Event) -> None:
    """Publish an event and give handlers time to process."""
    await bus.publish(event)
    # For fast-path events, handlers are called synchronously
    # For queued events, give the processing loop time
    await asyncio.sleep(0.05)


async def test_records_signal() -> None:
    """Publish SIGNAL event, verify chronicled."""
    bus = EventBus()
    await bus.start()

    chronicle = Chronicle(bus=bus, max_memory=100)
    await chronicle.start()

    signal_event = Event(
        event_type=EventType.SIGNAL,
        data={
            "signal": type(
                "MockSignal",
                (),
                {
                    "strategy_id": "impulse_engine",
                    "symbol": "BTCUSDT",
                    "side": "buy",
                    "confidence": 0.85,
                    "entry_price": 50000.0,
                },
            )(),
            "strategy_id": "impulse_engine",
            "symbol": "BTCUSDT",
        },
    )
    await _publish_and_wait(bus, signal_event)

    assert chronicle.size >= 1
    entries = chronicle.query(event_type="signal")
    assert len(entries) >= 1
    assert entries[0]["event_type"] == "signal"
    assert entries[0]["strategy_id"] == "impulse_engine"
    assert entries[0]["symbol"] == "BTCUSDT"

    await chronicle.stop()
    await bus.stop()


async def test_records_risk_blocked() -> None:
    """Publish RISK_BLOCKED, verify in chronicle."""
    bus = EventBus()
    await bus.start()

    chronicle = Chronicle(bus=bus, max_memory=100)
    await chronicle.start()

    blocked_event = Event(
        event_type=EventType.RISK_BLOCKED,
        data={
            "strategy_id": "momentum_trader",
            "symbol": "ETHUSDT",
            "reason": "max_drawdown_exceeded",
            "risk_state": "QUARANTINE",
        },
    )
    await _publish_and_wait(bus, blocked_event)

    assert chronicle.size >= 1
    entries = chronicle.query(event_type="risk_blocked")
    assert len(entries) >= 1
    assert entries[0]["strategy_id"] == "momentum_trader"
    assert entries[0]["symbol"] == "ETHUSDT"
    assert entries[0]["details"]["reason"] == "max_drawdown_exceeded"

    await chronicle.stop()
    await bus.stop()


async def test_query_filter() -> None:
    """Filter by event_type, symbol, strategy_id."""
    bus = EventBus()
    await bus.start()

    chronicle = Chronicle(bus=bus, max_memory=100)
    await chronicle.start()

    # Publish multiple events
    events = [
        Event(
            event_type=EventType.SIGNAL,
            data={
                "signal": type(
                    "S", (), {"strategy_id": "s1", "symbol": "BTCUSDT",
                              "side": "buy", "confidence": 0.9, "entry_price": 50000.0}
                )(),
                "strategy_id": "s1",
                "symbol": "BTCUSDT",
            },
        ),
        Event(
            event_type=EventType.SIGNAL,
            data={
                "signal": type(
                    "S", (), {"strategy_id": "s2", "symbol": "ETHUSDT",
                              "side": "sell", "confidence": 0.7, "entry_price": 3000.0}
                )(),
                "strategy_id": "s2",
                "symbol": "ETHUSDT",
            },
        ),
        Event(
            event_type=EventType.RISK_BLOCKED,
            data={
                "strategy_id": "s1",
                "symbol": "BTCUSDT",
                "reason": "drawdown",
            },
        ),
    ]

    for e in events:
        await _publish_and_wait(bus, e)

    # Filter by event_type
    signals = chronicle.query(event_type="signal")
    assert len(signals) == 2

    blocked = chronicle.query(event_type="risk_blocked")
    assert len(blocked) == 1

    # Filter by symbol
    btc = chronicle.query(symbol="BTCUSDT")
    assert len(btc) == 2  # 1 signal + 1 blocked

    eth = chronicle.query(symbol="ETHUSDT")
    assert len(eth) == 1

    # Filter by strategy_id
    s1 = chronicle.query(strategy_id="s1")
    assert len(s1) == 2

    s2 = chronicle.query(strategy_id="s2")
    assert len(s2) == 1

    # Combined filter
    s1_btc_signal = chronicle.query(
        event_type="signal", symbol="BTCUSDT", strategy_id="s1"
    )
    assert len(s1_btc_signal) == 1

    await chronicle.stop()
    await bus.stop()


async def test_max_memory() -> None:
    """Exceed max, verify oldest entries dropped."""
    bus = EventBus()
    await bus.start()

    max_mem = 5
    chronicle = Chronicle(bus=bus, max_memory=max_mem)
    await chronicle.start()

    # Publish more than max_memory events
    for i in range(10):
        event = Event(
            event_type=EventType.RISK_BLOCKED,
            data={
                "strategy_id": f"s_{i}",
                "symbol": "BTCUSDT",
                "reason": f"reason_{i}",
            },
        )
        await _publish_and_wait(bus, event)

    # Should have at most max_memory entries
    assert chronicle.size <= max_mem

    # Oldest entries should have been dropped -- entries 5-9 should remain
    entries = chronicle.query(limit=100)
    strategy_ids = [e["strategy_id"] for e in entries]
    # The most recent entries should be present
    assert "s_9" in strategy_ids

    await chronicle.stop()
    await bus.stop()


async def test_signal_journey() -> None:
    """Publish sequence of events, query by correlation_id."""
    bus = EventBus()
    await bus.start()

    chronicle = Chronicle(bus=bus, max_memory=100)
    await chronicle.start()

    corr_id = "test-correlation-123"

    # Simulate a signal journey through the system
    events = [
        Event(
            event_type=EventType.SIGNAL,
            data={
                "_correlation_id": corr_id,
                "signal": type(
                    "S", (), {"strategy_id": "impulse", "symbol": "BTCUSDT",
                              "side": "buy", "confidence": 0.9, "entry_price": 50000.0}
                )(),
                "strategy_id": "impulse",
                "symbol": "BTCUSDT",
            },
        ),
        Event(
            event_type=EventType.ORDER_REQUEST,
            data={
                "_correlation_id": corr_id,
                "strategy_id": "impulse",
                "symbol": "BTCUSDT",
                "side": "buy",
                "size": 0.001,
            },
        ),
        Event(
            event_type=EventType.ORDER_FILLED,
            data={
                "_correlation_id": corr_id,
                "strategy_id": "impulse",
                "symbol": "BTCUSDT",
                "order_id": "ord_123",
                "fill_price": 50100.0,
                "fill_qty": 0.001,
            },
        ),
    ]

    for e in events:
        await _publish_and_wait(bus, e)

    journey = chronicle.get_signal_journey(corr_id)
    assert len(journey) == 3
    assert journey[0]["event_type"] == "signal"
    assert journey[1]["event_type"] == "order_request"
    assert journey[2]["event_type"] == "order_filled"

    # All entries should share the same correlation_id
    for entry in journey:
        assert entry["correlation_id"] == corr_id

    await chronicle.stop()
    await bus.stop()


async def test_size_property() -> None:
    """Verify size property reflects actual count."""
    bus = EventBus()
    await bus.start()

    chronicle = Chronicle(bus=bus, max_memory=100)
    await chronicle.start()

    assert chronicle.size == 0

    event = Event(
        event_type=EventType.RISK_BLOCKED,
        data={
            "strategy_id": "test",
            "symbol": "BTCUSDT",
            "reason": "test",
        },
    )
    await _publish_and_wait(bus, event)
    assert chronicle.size == 1

    await chronicle.stop()
    await bus.stop()
