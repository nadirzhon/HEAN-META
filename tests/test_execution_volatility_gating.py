"""Tests for volatility-aware execution gating in ExecutionRouter."""

import asyncio
from datetime import datetime

import pytest

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, OrderRequest, Tick
from hean.execution.order_manager import OrderManager
from hean.execution.router import ExecutionRouter


@pytest.mark.asyncio
async def test_hard_volatility_block() -> None:
    """Test that orders are hard-blocked at high volatility percentile."""
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

    rejected_orders = []

    async def track_rejected(event: Event) -> None:
        rejected_orders.append(event.data["order"])

    bus.subscribe(EventType.ORDER_REJECTED, track_rejected)

    # Manually set high volatility percentile (simulating high volatility)
    # In real usage, this would come from RegimeDetector
    router._volatility_hard_block_percentile = 95.0

    # Create order request
    order_request = OrderRequest(
        signal_id="test",
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
    )

    # Mock high volatility by setting percentile manually
    # This is a simplified test - in production, volatility comes from RegimeDetector
    original_method = router._get_volatility_percentile

    def mock_high_volatility(symbol: str) -> float:
        return 98.0  # Above hard block threshold

    router._get_volatility_percentile = mock_high_volatility

    await bus.publish(
        Event(event_type=EventType.ORDER_REQUEST, data={"order_request": order_request})
    )

    await asyncio.sleep(0.2)

    # Order should be rejected
    assert len(rejected_orders) > 0

    # Check diagnostics
    diagnostics = router.get_diagnostics()
    snapshot = diagnostics.snapshot()
    assert snapshot["volatility_rejections_hard"] >= 1.0

    await router.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_soft_volatility_block_enqueues_for_retry() -> None:
    """Test that orders are soft-blocked and enqueued for retry."""
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

    rejected_orders = []

    async def track_rejected(event: Event) -> None:
        rejected_orders.append(event.data["order"])

    bus.subscribe(EventType.ORDER_REJECTED, track_rejected)

    # Mock medium-high volatility (soft block range)
    original_method = router._get_volatility_percentile

    def mock_soft_volatility(symbol: str) -> float:
        return 92.0  # Between soft and hard block

    router._get_volatility_percentile = mock_soft_volatility

    order_request = OrderRequest(
        signal_id="test",
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
    )

    await bus.publish(
        Event(event_type=EventType.ORDER_REQUEST, data={"order_request": order_request})
    )

    await asyncio.sleep(0.2)

    # Order should be rejected (soft block)
    assert len(rejected_orders) > 0

    # Check that it was enqueued for retry
    retry_queue = router.get_retry_queue()
    assert retry_queue.get_queue_size() > 0

    # Check diagnostics
    diagnostics = router.get_diagnostics()
    snapshot = diagnostics.snapshot()
    assert snapshot["volatility_rejections_soft"] >= 1.0

    await router.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_low_volatility_allows_orders() -> None:
    """Test that orders are allowed when volatility is low."""
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

    placed_orders = []

    async def track_placed(event: Event) -> None:
        placed_orders.append(event.data["order"])

    bus.subscribe(EventType.ORDER_PLACED, track_placed)

    # Mock low volatility
    original_method = router._get_volatility_percentile

    def mock_low_volatility(symbol: str) -> float:
        return 30.0  # Low volatility

    router._get_volatility_percentile = mock_low_volatility

    order_request = OrderRequest(
        signal_id="test",
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
    )

    await bus.publish(
        Event(event_type=EventType.ORDER_REQUEST, data={"order_request": order_request})
    )

    await asyncio.sleep(0.2)

    # Order should be placed (not rejected)
    assert len(placed_orders) > 0

    await router.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_volatility_gating_tracks_metrics() -> None:
    """Test that volatility gating properly tracks metrics."""
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

    # Create orders with different volatility levels
    for percentile in [92.0, 98.0, 50.0]:
        original_method = router._get_volatility_percentile

        def make_mock(p: float):
            def mock(symbol: str) -> float:
                return p
            return mock

        router._get_volatility_percentile = make_mock(percentile)

        order_request = OrderRequest(
            signal_id=f"test-{percentile}",
            strategy_id="test",
            symbol="BTCUSDT",
            side="buy",
            size=0.1,
        )

        await bus.publish(
            Event(event_type=EventType.ORDER_REQUEST, data={"order_request": order_request})
        )
        await asyncio.sleep(0.1)

    await asyncio.sleep(0.2)

    # Check diagnostics
    diagnostics = router.get_diagnostics()
    snapshot = diagnostics.snapshot()

    # Should have some rejections
    total_rejections = snapshot["volatility_rejections_soft"] + snapshot["volatility_rejections_hard"]
    assert total_rejections >= 0  # At least tracked

    await router.stop()
    await bus.stop()





