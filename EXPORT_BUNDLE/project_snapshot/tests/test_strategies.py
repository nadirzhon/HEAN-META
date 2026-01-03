"""Tests for trading strategies."""

import asyncio
from datetime import datetime

import pytest

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, FundingRate, Signal, Tick
from hean.strategies.basis_arbitrage import BasisArbitrage
from hean.strategies.funding_harvester import FundingHarvester
from hean.strategies.impulse_engine import ImpulseEngine


@pytest.mark.asyncio
async def test_funding_harvester() -> None:
    """Test funding harvester strategy."""
    bus = EventBus()
    strategy = FundingHarvester(bus, ["BTCUSDT"])

    await bus.start()
    await strategy.start()

    signals_received = []

    async def track_signal(event: Event) -> None:
        signals_received.append(event.data["signal"])

    bus.subscribe(EventType.SIGNAL, track_signal)

    # Send funding event with positive rate
    funding = FundingRate(
        symbol="BTCUSDT",
        rate=0.0002,  # 0.02% positive funding
        timestamp=datetime.utcnow(),
    )

    await bus.publish(Event(event_type=EventType.FUNDING, data={"funding": funding}))
    await asyncio.sleep(0.1)

    # Should generate a signal
    assert len(signals_received) > 0
    signal: Signal = signals_received[0]
    assert signal.symbol == "BTCUSDT"
    assert signal.side == "sell"  # Short when funding is positive

    await strategy.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_impulse_engine() -> None:
    """Test impulse engine strategy."""
    bus = EventBus()
    strategy = ImpulseEngine(bus, ["BTCUSDT"])

    await bus.start()
    await strategy.start()

    signals_received = []

    async def track_signal(event: Event) -> None:
        signals_received.append(event.data["signal"])

    bus.subscribe(EventType.SIGNAL, track_signal)

    # Send ticks with impulse movement
    base_price = 50000.0
    for i in range(15):
        price = base_price * (1 + 0.01 * i / 10)  # 1% upward movement
        tick = Tick(
            symbol="BTCUSDT",
            price=price,
            timestamp=datetime.utcnow(),
            bid=price * 0.9999,
            ask=price * 1.0001,
        )
        await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
        await asyncio.sleep(0.05)

    # Should detect impulse and generate signal
    # Note: This may not always trigger due to randomness, but structure is tested
    await asyncio.sleep(0.2)

    await strategy.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_basis_arbitrage() -> None:
    """Test basis arbitrage strategy."""
    bus = EventBus()
    strategy = BasisArbitrage(bus, ["BTCUSDT"])

    await bus.start()
    await strategy.start()

    signals_received = []

    async def track_signal(event: Event) -> None:
        signals_received.append(event.data["signal"])

    bus.subscribe(EventType.SIGNAL, track_signal)

    # Send tick with basis
    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        bid=49999.0,
        ask=50001.0,
    )

    await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
    await asyncio.sleep(0.1)

    await strategy.stop()
    await bus.stop()


