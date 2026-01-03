"""Tests for paper broker."""

import asyncio
from datetime import datetime

import pytest

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Order, OrderStatus, Tick
from hean.execution.paper_broker import PaperBroker


@pytest.mark.asyncio
async def test_paper_broker_fill() -> None:
    """Test paper broker order filling."""
    bus = EventBus()
    broker = PaperBroker(bus)

    await bus.start()
    await broker.start()

    # Create a tick to set price
    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        bid=49999.0,
        ask=50001.0,
    )
    await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))

    # Wait for price to be set
    await asyncio.sleep(0.1)

    # Create and submit order
    order = Order(
        order_id="test-order-1",
        strategy_id="test-strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        order_type="market",
        status=OrderStatus.PENDING,
        timestamp=datetime.utcnow(),
    )

    await broker.submit_order(order)

    # Wait for fill
    await asyncio.sleep(0.5)

    # Check order was filled
    assert order.status == OrderStatus.FILLED
    assert order.filled_size > 0
    assert order.avg_fill_price is not None

    await broker.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_paper_broker_fees() -> None:
    """Test that paper broker applies fees."""
    bus = EventBus()
    broker = PaperBroker(bus)

    await bus.start()
    await broker.start()

    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        bid=49999.0,
        ask=50001.0,
    )
    await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
    await asyncio.sleep(0.1)

    order = Order(
        order_id="test-order-2",
        strategy_id="test-strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        order_type="market",
        status=OrderStatus.PENDING,
        timestamp=datetime.utcnow(),
    )

    # Track fill events
    fees_collected = []

    async def track_fill(event: Event) -> None:
        if "fee" in event.data:
            fees_collected.append(event.data["fee"])

    bus.subscribe(EventType.ORDER_FILLED, track_fill)

    await broker.submit_order(order)
    await asyncio.sleep(0.5)

    # Check fee was applied
    assert len(fees_collected) > 0
    assert fees_collected[0] > 0

    await broker.stop()
    await bus.stop()


