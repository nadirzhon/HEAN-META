"""Tests for multi-timeframe candle aggregation."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from hean.core.bus import EventBus
from hean.core.timeframes import CandleAggregator, Candle
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

    # Use a fixed reference time at exact minute boundary
    base = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Generate ticks within first minute [00:00:00, 00:01:00)
    ticks = [
        Tick(symbol="BTCUSDT", price=100.0, timestamp=base + timedelta(seconds=0), volume=1.0),
        Tick(symbol="BTCUSDT", price=101.0, timestamp=base + timedelta(seconds=10), volume=2.0),
        Tick(symbol="BTCUSDT", price=99.0, timestamp=base + timedelta(seconds=20), volume=3.0),
        Tick(symbol="BTCUSDT", price=102.0, timestamp=base + timedelta(seconds=59), volume=4.0),
    ]

    for t in ticks:
        await bus.publish(Event(event_type=EventType.TICK, data={"tick": t}))

    # Advance into next minute to force close of 1m candle
    next_minute_tick = Tick(
        symbol="BTCUSDT",
        price=103.0,
        timestamp=base + timedelta(minutes=1, seconds=1),
        volume=5.0,
    )
    await bus.publish(Event(event_type=EventType.TICK, data={"tick": next_minute_tick}))

    # Give bus a moment to process
    await asyncio.sleep(0)  # type: ignore[name-defined]

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


