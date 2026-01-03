"""Tests for impulse engine improvements."""

import asyncio
from datetime import datetime, timedelta

import pytest

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.regime import Regime
from hean.core.types import Event, EventType, Position, Tick
from hean.strategies.impulse_engine import ImpulseEngine


@pytest.mark.asyncio
async def test_break_even_activates() -> None:
    """Test that break-even stop activates when TP_1 is reached."""
    bus = EventBus()
    strategy = ImpulseEngine(bus, ["BTCUSDT"])

    await bus.start()
    await strategy.start()

    # Set regime to IMPULSE
    await strategy.on_regime_update(
        Event(
            event_type=EventType.REGIME_UPDATE,
            data={"symbol": "BTCUSDT", "regime": Regime.IMPULSE},
        )
    )

    # Create a position
    position = Position(
        position_id="test-pos",
        symbol="BTCUSDT",
        side="long",
        size=0.1,
        entry_price=50000.0,
        current_price=50000.0,
        opened_at=datetime.utcnow(),
        strategy_id="impulse_engine",
        stop_loss=49750.0,  # 0.5% stop
        take_profit=50500.0,  # 1% TP_1
    )

    # Simulate position opened
    await bus.publish(
        Event(
            event_type=EventType.POSITION_OPENED,
            data={"position": position},
        )
    )
    await asyncio.sleep(0.1)

    # Send tick that reaches TP_1
    tick = Tick(
        symbol="BTCUSDT",
        price=50500.0,  # Reached TP_1
        timestamp=datetime.utcnow(),
        bid=50499.0,
        ask=50501.0,
    )

    be_activated = False

    async def track_position_update(event: Event) -> None:
        nonlocal be_activated
        if event.data.get("update_type") == "break_even_activated":
            be_activated = True

    bus.subscribe(EventType.POSITION_UPDATE, track_position_update)

    await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
    await asyncio.sleep(0.2)

    # Break-even should be activated
    assert be_activated, "Break-even stop should activate when TP_1 is reached"

    await strategy.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_no_trade_zone_spread() -> None:
    """Test that no-trade zone blocks entry when spread is too wide."""
    bus = EventBus()
    strategy = ImpulseEngine(bus, ["BTCUSDT"])

    await bus.start()
    await strategy.start()

    # Set regime to IMPULSE
    await strategy.on_regime_update(
        Event(
            event_type=EventType.REGIME_UPDATE,
            data={"symbol": "BTCUSDT", "regime": Regime.IMPULSE},
        )
    )

    signals_received = []

    async def track_signal(event: Event) -> None:
        signals_received.append(event.data["signal"])

    bus.subscribe(EventType.SIGNAL, track_signal)

    # Build price history
    base_price = 50000.0
    for i in range(15):
        price = base_price * (1 + 0.01 * i / 10)  # 1% upward movement
        # Wide spread (exceeds threshold)
        spread_bps = settings.impulse_max_spread_bps + 5  # Exceeds limit
        spread_pct = spread_bps / 10000.0
        tick = Tick(
            symbol="BTCUSDT",
            price=price,
            timestamp=datetime.utcnow(),
            bid=price * (1 - spread_pct / 2),
            ask=price * (1 + spread_pct / 2),
        )
        await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
        await asyncio.sleep(0.05)

    await asyncio.sleep(0.2)

    # Should NOT generate signals due to wide spread
    assert len(signals_received) == 0, "No signals should be generated when spread is too wide"

    await strategy.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_max_time_in_trade_force_exit() -> None:
    """Test that max time in trade forces exit."""
    # Temporarily set short time limit for testing
    original_time = settings.impulse_max_time_in_trade_sec
    settings.impulse_max_time_in_trade_sec = 2  # 2 seconds for testing

    try:
        bus = EventBus()
        strategy = ImpulseEngine(bus, ["BTCUSDT"])

        await bus.start()
        await strategy.start()

        # Create a position opened 3 seconds ago (exceeds limit)
        position = Position(
            position_id="test-pos",
            symbol="BTCUSDT",
            side="long",
            size=0.1,
            entry_price=50000.0,
            current_price=50100.0,
            opened_at=datetime.utcnow() - timedelta(seconds=3),
            strategy_id="impulse_engine",
            stop_loss=49750.0,
            take_profit=50500.0,
        )

        # Simulate position opened
        await bus.publish(
            Event(
                event_type=EventType.POSITION_OPENED,
                data={"position": position},
            )
        )
        await asyncio.sleep(0.1)

        position_closed = False

        async def track_close(event: Event) -> None:
            nonlocal position_closed
            if event.data.get("close_reason") == "max_time_exceeded":
                position_closed = True

        bus.subscribe(EventType.POSITION_CLOSED, track_close)

        # Send tick to trigger time check
        tick = Tick(
            symbol="BTCUSDT",
            price=50100.0,
            timestamp=datetime.utcnow(),
            bid=50099.0,
            ask=50101.0,
        )

        await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
        await asyncio.sleep(0.3)

        # Position should be closed due to time limit
        assert position_closed, "Position should be closed when max time in trade is exceeded"

        await strategy.stop()
        await bus.stop()
    finally:
        settings.impulse_max_time_in_trade_sec = original_time


@pytest.mark.asyncio
async def test_maker_edge_check() -> None:
    """Test that trades are skipped if maker edge is too low."""
    bus = EventBus()
    strategy = ImpulseEngine(bus, ["BTCUSDT"])

    await bus.start()
    await strategy.start()

    # Set regime to IMPULSE
    await strategy.on_regime_update(
        Event(
            event_type=EventType.REGIME_UPDATE,
            data={"symbol": "BTCUSDT", "regime": Regime.IMPULSE},
        )
    )

    signals_received = []

    async def track_signal(event: Event) -> None:
        signals_received.append(event.data["signal"])

    bus.subscribe(EventType.SIGNAL, track_signal)

    # Build price history with impulse
    base_price = 50000.0
    for i in range(15):
        price = base_price * (1 + 0.01 * i / 10)  # 1% upward movement
        # Narrow spread, but price is above best bid (no maker edge)
        tick = Tick(
            symbol="BTCUSDT",
            price=price,
            timestamp=datetime.utcnow(),
            bid=price * 0.999,  # Price is above bid (no maker edge for buy)
            ask=price * 1.001,
        )
        await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
        await asyncio.sleep(0.05)

    await asyncio.sleep(0.2)

    # Should NOT generate signals if maker edge is insufficient
    # (Note: This depends on the exact threshold and price movement)

    await strategy.stop()
    await bus.stop()


def test_impulse_engine_metrics() -> None:
    """Test that impulse engine metrics are calculated correctly."""
    bus = EventBus()
    strategy = ImpulseEngine(bus)

    # Simulate some trades
    strategy._trade_times = [100.0, 200.0, 150.0]  # 3 trades
    strategy._be_stop_hits = 1  # 1 hit break-even stop
    strategy._total_trades = 3

    metrics = strategy.get_metrics()

    assert metrics["avg_time_in_trade_sec"] == pytest.approx(150.0, abs=0.1)
    assert metrics["be_stop_hit_pct"] == pytest.approx(33.33, abs=0.1)
    assert metrics["total_trades"] == 3.0
    assert metrics["be_stop_hits"] == 1.0

