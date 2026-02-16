"""Tests for multi-timeframe candle aggregation."""

from datetime import datetime, timedelta

import pytest

from hean.core.bus import EventBus
from hean.core.timeframes import Candle, CandleAggregator
from hean.core.types import Event, EventType, Tick


@pytest.mark.asyncio
async def test_candle_boundaries_and_aggregation() -> None:
    """Validate candle boundaries and OHLCV aggregation for 1m/5m/1h/1d."""
    bus = EventBus()
    await bus.start()

    aggregator = CandleAggregator(bus, timeframes=["1m", "5m", "1h", "1d"])
    await aggregator.start()

    received: list[Event] = []

    async def on_candle(event: Event) -> None:
        received.append(event)

    bus.subscribe(EventType.CANDLE, on_candle)

    # Use a fixed POSIX timestamp at an exact minute boundary.
    # datetime.utcfromtimestamp returns naive UTC datetimes that _floor_time
    # also produces via utcfromtimestamp, so comparisons are in the same space.
    _BASE_TS = 1735689600.0  # 2025-01-01 00:00:00 UTC
    base = datetime.utcfromtimestamp(_BASE_TS)

    # Generate ticks within first minute [00:00:00, 00:01:00)
    ticks = [
        Tick(symbol="BTCUSDT", price=100.0, timestamp=datetime.utcfromtimestamp(_BASE_TS + 0), volume=1.0),
        Tick(symbol="BTCUSDT", price=101.0, timestamp=datetime.utcfromtimestamp(_BASE_TS + 10), volume=2.0),
        Tick(symbol="BTCUSDT", price=99.0, timestamp=datetime.utcfromtimestamp(_BASE_TS + 20), volume=3.0),
        Tick(symbol="BTCUSDT", price=102.0, timestamp=datetime.utcfromtimestamp(_BASE_TS + 59), volume=4.0),
    ]

    for t in ticks:
        await bus.publish(Event(event_type=EventType.TICK, data={"tick": t}))

    # Advance into next minute to force close of 1m candle
    next_minute_tick = Tick(
        symbol="BTCUSDT",
        price=103.0,
        timestamp=datetime.utcfromtimestamp(_BASE_TS + 61),
        volume=5.0,
    )
    await bus.publish(Event(event_type=EventType.TICK, data={"tick": next_minute_tick}))

    # Drain all queues: flush() processes CRITICAL→NORMAL→LOW, but TICK (LOW)
    # handlers may emit CANDLE (NORMAL) events that need another flush pass.
    while await bus.flush() > 0:
        pass

    # Extract emitted candles for 1m timeframe
    m1_candles = [
        e.data["candle"]
        for e in received
        if e.data.get("timeframe") == "1m"
    ]

    # First 1m candle should be closed when tick crosses 00:01
    assert len(m1_candles) >= 1
    c1: Candle = m1_candles[0]

    assert c1.open_time == base
    assert c1.close_time == base + timedelta(minutes=1)
    assert c1.open == 100.0
    assert c1.high == 102.0
    assert c1.low == 99.0
    assert c1.close == 102.0
    assert c1.volume == pytest.approx(1.0 + 2.0 + 3.0 + 4.0)

    # 5m candle should span [00:00, 00:05)
    m5_candles = [
        e.data["candle"]
        for e in received
        if e.data.get("timeframe") == "5m"
    ]
    assert len(m5_candles) == 0  # not closed yet; boundary not crossed

    await aggregator.stop()
    await bus.stop()
