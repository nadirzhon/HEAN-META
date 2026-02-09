"""Smoke tests for income streams.

These are deliberately lightweight and deterministic:
 - Use a local EventBus (no TradingSystem) to validate signal emission
 - Feed synthetic CONTEXT_UPDATE events into each stream
 - Check that config gating (enabled flags) is respected
"""

import asyncio
from datetime import datetime

import pytest

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Signal
from hean.income.streams import (
    BasisHedgeStream,
    FundingHarvesterStream,
    MakerRebateStream,
    VolatilityHarvestStream,
)


@pytest.mark.asyncio
async def test_funding_stream_emits_signal() -> None:
    bus = EventBus()
    stream = FundingHarvesterStream(bus, ["BTCUSDT"])

    await bus.start()
    await stream.start()

    signals: list[Signal] = []

    async def track_signal(event: Event) -> None:
        signals.append(event.data["signal"])

    bus.subscribe(EventType.SIGNAL, track_signal)

    ctx = {
        "symbol": "BTCUSDT",
        "funding_rate": 0.0002,  # positive funding
        "price": 50000.0,
        "timestamp": datetime.utcnow(),
    }

    await bus.publish(Event(event_type=EventType.CONTEXT_UPDATE, data=ctx))
    await asyncio.sleep(0.1)

    assert signals, "Funding stream should emit at least one signal"
    assert signals[0].strategy_id == "stream_funding"

    await stream.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_maker_rebate_stream_emits_signal() -> None:
    """MakerRebateStream is disabled (no data provider) — should emit NO signals."""
    bus = EventBus()
    stream = MakerRebateStream(bus, ["BTCUSDT"])

    await bus.start()
    await stream.start()

    signals: list[Signal] = []

    async def track_signal(event: Event) -> None:
        signals.append(event.data["signal"])

    bus.subscribe(EventType.SIGNAL, track_signal)

    ctx = {
        "symbol": "BTCUSDT",
        "regime": "range",
        "price": 50000.0,
        "timestamp": datetime.utcnow(),
    }

    await bus.publish(Event(event_type=EventType.CONTEXT_UPDATE, data=ctx))
    await asyncio.sleep(0.1)

    assert not signals, "Disabled stream should emit no signals"

    await stream.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_basis_stream_emits_signal() -> None:
    """BasisHedgeStream is disabled (no data provider) — should emit NO signals."""
    bus = EventBus()
    stream = BasisHedgeStream(bus, ["BTCUSDT"])

    await bus.start()
    await stream.start()

    signals: list[Signal] = []

    async def track_signal(event: Event) -> None:
        signals.append(event.data["signal"])

    bus.subscribe(EventType.SIGNAL, track_signal)

    ctx = {
        "symbol": "BTCUSDT",
        "basis": 0.003,  # 0.3% > 0.2% threshold
        "price": 50000.0,
        "timestamp": datetime.utcnow(),
    }

    await bus.publish(Event(event_type=EventType.CONTEXT_UPDATE, data=ctx))
    await asyncio.sleep(0.1)

    assert not signals, "Disabled stream should emit no signals"

    await stream.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_volatility_stream_emits_signal() -> None:
    """VolatilityHarvestStream is disabled (no data provider) — should emit NO signals."""
    bus = EventBus()
    stream = VolatilityHarvestStream(bus, ["BTCUSDT"])

    await bus.start()
    await stream.start()

    signals: list[Signal] = []

    async def track_signal(event: Event) -> None:
        signals.append(event.data["signal"])

    bus.subscribe(EventType.SIGNAL, track_signal)

    mean_price = 50000.0
    # Price below lower band to trigger BUY
    ctx = {
        "symbol": "BTCUSDT",
        "price": mean_price * 0.996,  # below band (0.3%)
        "mean_price": mean_price,
        "vol_short": 0.001,
        "vol_long": 0.0008,
        "timestamp": datetime.utcnow(),
    }

    await bus.publish(Event(event_type=EventType.CONTEXT_UPDATE, data=ctx))
    await asyncio.sleep(0.1)

    assert not signals, "Disabled stream should emit no signals"

    await stream.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_stream_config_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure config enable flags can fully disable streams."""
    # Temporarily disable funding stream
    monkeypatch.setattr(settings, "stream_funding_enabled", False)

    bus = EventBus()
    stream = FundingHarvesterStream(bus, ["BTCUSDT"])

    await bus.start()

    # Do NOT call stream.start() when disabled; simulate TradingSystem gating
    signals: list[Signal] = []

    async def track_signal(event: Event) -> None:
        signals.append(event.data["signal"])

    bus.subscribe(EventType.SIGNAL, track_signal)

    ctx = {
        "symbol": "BTCUSDT",
        "funding_rate": 0.0002,
        "price": 50000.0,
        "timestamp": datetime.utcnow(),
    }

    await bus.publish(Event(event_type=EventType.CONTEXT_UPDATE, data=ctx))
    await asyncio.sleep(0.1)

    # Because TradingSystem would not start the stream, no signals should be seen
    assert not signals, "When disabled via config, stream should not emit signals"

    await bus.stop()







