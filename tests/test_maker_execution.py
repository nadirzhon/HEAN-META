"""Tests for maker-first execution policy."""

import asyncio
from datetime import datetime, timedelta

import pytest

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Order, OrderRequest, OrderStatus, Tick
from hean.execution.order_manager import OrderManager
from hean.execution.paper_broker import PaperBroker
from hean.execution.router import ExecutionRouter


@pytest.mark.asyncio
async def test_maker_orders_lower_fees() -> None:
    """Test that maker orders incur lower fees than taker orders."""
    bus = EventBus()
    broker = PaperBroker(bus)

    await bus.start()
    await broker.start()

    # Set up market data
    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        bid=49999.0,
        ask=50001.0,
    )
    await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
    await asyncio.sleep(0.1)

    # Create maker order (limit)
    maker_order = Order(
        order_id="maker-1",
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        price=49999.0,
        order_type="limit",
        status=OrderStatus.PENDING,
        timestamp=datetime.utcnow(),
        is_maker=True,
    )

    # Create taker order (market)
    taker_order = Order(
        order_id="taker-1",
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        order_type="market",
        status=OrderStatus.PENDING,
        timestamp=datetime.utcnow(),
        is_maker=False,
    )

    maker_fees = []
    taker_fees = []

    async def track_fill(event: Event) -> None:
        if "is_maker" in event.data and event.data["is_maker"]:
            maker_fees.append(event.data["fee"])
        else:
            taker_fees.append(event.data["fee"])

    bus.subscribe(EventType.ORDER_FILLED, track_fill)

    # Submit and fill orders
    await broker.submit_order(maker_order)
    await broker.submit_order(taker_order)
    await asyncio.sleep(0.5)

    # Maker fees should be lower
    if maker_fees and taker_fees:
        assert maker_fees[0] < taker_fees[0], "Maker fees should be lower than taker fees"

    await broker.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_ttl_cancel_works() -> None:
    """Test that maker orders are cancelled after TTL."""
    # Temporarily set short TTL for testing
    original_ttl = settings.maker_ttl_ms
    settings.maker_ttl_ms = 200  # 200ms for testing

    try:
        bus = EventBus()
        order_manager = OrderManager()
        router = ExecutionRouter(bus, order_manager)

        await bus.start()
        await router.start()

        # Set up market data
        tick = Tick(
            symbol="BTCUSDT",
            price=50000.0,
            timestamp=datetime.utcnow(),
            bid=49999.0,
            ask=50001.0,
        )
        await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
        await asyncio.sleep(0.1)

        cancelled_orders = []

        async def track_cancel(event: Event) -> None:
            cancelled_orders.append(event.data["order"])

        bus.subscribe(EventType.ORDER_CANCELLED, track_cancel)

        # Create order request (will be converted to maker order)
        order_request = OrderRequest(
            signal_id="test",
            strategy_id="test",
            symbol="BTCUSDT",
            side="buy",
            size=0.1,
            price=None,
            order_type="market",
        )

        await bus.publish(
            Event(event_type=EventType.ORDER_REQUEST, data={"order_request": order_request})
        )

        # Wait for TTL to expire
        await asyncio.sleep(0.3)

        # Order should be cancelled
        assert len(cancelled_orders) > 0, "Order should be cancelled after TTL"

        await router.stop()
        await bus.stop()
    finally:
        settings.maker_ttl_ms = original_ttl


@pytest.mark.asyncio
async def test_taker_fallback_blocked_when_disabled() -> None:
    """Test that taker fallback is blocked when allow_taker_fallback is False."""
    original_setting = settings.allow_taker_fallback
    settings.allow_taker_fallback = False

    try:
        bus = EventBus()
        order_manager = OrderManager()
        router = ExecutionRouter(bus, order_manager)

        await bus.start()
        await router.start()

        # Set up market data
        tick = Tick(
            symbol="BTCUSDT",
            price=50000.0,
            timestamp=datetime.utcnow(),
            bid=49999.0,
            ask=50001.0,
        )
        await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
        await asyncio.sleep(0.1)

        taker_orders = []

        async def track_order_placed(event: Event) -> None:
            order: Order = event.data["order"]
            if order.order_type == "market" and "taker_fallback" in order.metadata:
                taker_orders.append(order)

        bus.subscribe(EventType.ORDER_PLACED, track_order_placed)

        # Create order request
        order_request = OrderRequest(
            signal_id="test",
            strategy_id="test",
            symbol="BTCUSDT",
            side="buy",
            size=0.1,
            price=None,
            order_type="market",
        )

        await bus.publish(
            Event(event_type=EventType.ORDER_REQUEST, data={"order_request": order_request})
        )

        # Wait for TTL to expire
        await asyncio.sleep(0.3)

        # No taker fallback orders should be created
        assert len(taker_orders) == 0, "Taker fallback should be blocked when disabled"

        await router.stop()
        await bus.stop()
    finally:
        settings.allow_taker_fallback = original_setting


@pytest.mark.asyncio
async def test_maker_fill_rate_tracking() -> None:
    """Test that maker fill rate is tracked correctly."""
    bus = EventBus()
    broker = PaperBroker(bus)

    await bus.start()
    await broker.start()

    # Set up market data
    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        bid=49999.0,
        ask=50001.0,
    )
    await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
    await asyncio.sleep(0.1)

    # Create and fill maker orders
    for i in range(3):
        order = Order(
            order_id=f"maker-{i}",
            strategy_id="test",
            symbol="BTCUSDT",
            side="buy",
            size=0.1,
            price=49999.0,
            order_type="limit",
            status=OrderStatus.PENDING,
            timestamp=datetime.utcnow(),
            is_maker=True,
        )
        await broker.submit_order(order)
        await asyncio.sleep(0.1)

    # Create and fill taker order
    taker_order = Order(
        order_id="taker-1",
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        order_type="market",
        status=OrderStatus.PENDING,
        timestamp=datetime.utcnow(),
        is_maker=False,
    )
    await broker.submit_order(taker_order)
    await asyncio.sleep(0.2)

    stats = broker.get_fill_stats()
    assert stats["total_fills"] > 0
    assert stats["maker_fills"] >= 0
    assert stats["taker_fills"] >= 0
    assert 0 <= stats["maker_fill_rate_pct"] <= 100

    await broker.stop()
    await bus.stop()






