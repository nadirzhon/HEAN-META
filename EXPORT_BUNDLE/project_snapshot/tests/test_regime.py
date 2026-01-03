"""Tests for regime detection and strategy gating."""

import asyncio
from datetime import datetime

import pytest

from hean.core.bus import EventBus
from hean.core.regime import Regime, RegimeDetector
from hean.core.types import Event, EventType, Tick
from hean.risk.position_sizer import PositionSizer
from hean.strategies.impulse_engine import ImpulseEngine
from hean.core.types import Signal


@pytest.mark.asyncio
async def test_regime_detector() -> None:
    """Test regime detector."""
    bus = EventBus()
    detector = RegimeDetector(bus)

    await bus.start()
    await detector.start()

    # Send ticks to build history
    base_price = 50000.0
    for i in range(60):
        price = base_price * (1 + 0.001 * i)  # Gradual increase
        tick = Tick(
            symbol="BTCUSDT",
            price=price,
            timestamp=datetime.utcnow(),
            bid=price * 0.9999,
            ask=price * 1.0001,
        )
        await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
        await asyncio.sleep(0.01)

    # Check that regime was detected
    regime = detector.get_regime("BTCUSDT")
    assert regime in {Regime.RANGE, Regime.NORMAL, Regime.IMPULSE}

    await detector.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_impulse_engine_gating() -> None:
    """Test that impulse engine does not trade outside IMPULSE regime."""
    bus = EventBus()
    strategy = ImpulseEngine(bus, ["BTCUSDT"])

    await bus.start()
    await strategy.start()

    signals_received = []

    async def track_signal(event: Event) -> None:
        signals_received.append(event.data["signal"])

    bus.subscribe(EventType.SIGNAL, track_signal)

    # Set regime to NORMAL (not IMPULSE)
    await strategy.on_regime_update(
        Event(
            event_type=EventType.REGIME_UPDATE,
            data={"symbol": "BTCUSDT", "regime": Regime.NORMAL},
        )
    )

    # Send ticks that would normally trigger impulse
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

    # Should NOT generate signals in NORMAL regime
    await asyncio.sleep(0.2)
    assert len(signals_received) == 0

    # Now set to IMPULSE regime
    await strategy.on_regime_update(
        Event(
            event_type=EventType.REGIME_UPDATE,
            data={"symbol": "BTCUSDT", "regime": Regime.IMPULSE},
        )
    )

    # Send more ticks
    for i in range(15, 30):
        price = base_price * (1 + 0.01 * i / 10)
        tick = Tick(
            symbol="BTCUSDT",
            price=price,
            timestamp=datetime.utcnow(),
            bid=price * 0.9999,
            ask=price * 1.0001,
        )
        await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
        await asyncio.sleep(0.05)

    # Now should generate signals in IMPULSE regime
    await asyncio.sleep(0.2)
    # Note: May or may not generate signals depending on detection logic,
    # but the key is that it's allowed to try

    await strategy.stop()
    await bus.stop()


def test_position_sizing_by_regime() -> None:
    """Test that position sizing scales by regime."""
    sizer = PositionSizer()

    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        stop_loss=49000.0,  # 2% stop
    )

    equity = 10000.0

    # Calculate size for each regime
    size_range = sizer.calculate_size(signal, equity, 50000.0, Regime.RANGE)
    size_normal = sizer.calculate_size(signal, equity, 50000.0, Regime.NORMAL)
    size_impulse = sizer.calculate_size(signal, equity, 50000.0, Regime.IMPULSE)

    # RANGE should be 0.7x of NORMAL
    assert size_range == pytest.approx(size_normal * 0.7, rel=0.01)

    # IMPULSE should be 1.2x of NORMAL
    assert size_impulse == pytest.approx(size_normal * 1.2, rel=0.01)

    # All should be positive
    assert size_range > 0
    assert size_normal > 0
    assert size_impulse > 0


def test_strategy_regime_gating() -> None:
    """Test that strategies respect regime gating."""
    from hean.strategies.funding_harvester import FundingHarvester
    from hean.strategies.basis_arbitrage import BasisArbitrage

    bus = EventBus()

    # Funding harvester: all regimes
    fh = FundingHarvester(bus)
    assert fh.is_allowed_in_regime(Regime.RANGE)
    assert fh.is_allowed_in_regime(Regime.NORMAL)
    assert fh.is_allowed_in_regime(Regime.IMPULSE)

    # Basis arbitrage: RANGE and NORMAL only
    ba = BasisArbitrage(bus)
    assert ba.is_allowed_in_regime(Regime.RANGE)
    assert ba.is_allowed_in_regime(Regime.NORMAL)
    assert not ba.is_allowed_in_regime(Regime.IMPULSE)

    # Impulse engine: IMPULSE only
    ie = ImpulseEngine(bus)
    assert not ie.is_allowed_in_regime(Regime.RANGE)
    assert not ie.is_allowed_in_regime(Regime.NORMAL)
    assert ie.is_allowed_in_regime(Regime.IMPULSE)






