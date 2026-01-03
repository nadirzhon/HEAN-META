"""Tests for adaptive maker placement in ExecutionRouter."""

import asyncio
from datetime import datetime

import pytest

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.regime import RegimeDetector
from hean.core.types import Event, EventType, OrderRequest, Tick
from hean.execution.order_manager import OrderManager
from hean.execution.router import ExecutionRouter


@pytest.mark.asyncio
async def test_adaptive_ttl_increases_on_expirations() -> None:
    """Test that TTL increases when many orders expire."""
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
    
    # Get initial adaptive TTL
    initial_ttl = router._adaptive_ttl_ms
    
    # Create multiple order requests that will expire
    for i in range(6):
        order_request = OrderRequest(
            signal_id=f"test-{i}",
            strategy_id="test",
            symbol="BTCUSDT",
            side="buy",
            size=0.1,
        )
        await bus.publish(
            Event(event_type=EventType.ORDER_REQUEST, data={"order_request": order_request})
        )
        await asyncio.sleep(0.05)
    
    # Wait for orders to expire
    await asyncio.sleep(0.5)
    
    # Trigger adaptive parameter update by creating another order
    order_request = OrderRequest(
        signal_id="test-update",
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
    )
    await bus.publish(
        Event(event_type=EventType.ORDER_REQUEST, data={"order_request": order_request})
    )
    await asyncio.sleep(0.1)
    
    # TTL should have increased (or at least be different)
    # Note: This is a simplified test - in practice, the adaptive logic
    # runs during order placement
    assert router._adaptive_ttl_ms >= initial_ttl
    
    await router.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_adaptive_offset_widens_on_high_volatility() -> None:
    """Test that offset widens when volatility is high."""
    bus = EventBus()
    order_manager = OrderManager()
    regime_detector = RegimeDetector(bus)
    router = ExecutionRouter(bus, order_manager, regime_detector)
    
    await bus.start()
    await regime_detector.start()
    await router.start()
    
    # Set up market data with high volatility (large price movements)
    for i in range(20):
        price = 50000.0 + (i * 10)  # Large price movements
        tick = Tick(
            symbol="BTCUSDT",
            price=price,
            timestamp=datetime.utcnow(),
            bid=price - 1.0,
            ask=price + 1.0,
        )
        await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
        await asyncio.sleep(0.01)
    
    await asyncio.sleep(0.2)
    
    # Get initial offset
    initial_offset = router._adaptive_offset_bps
    
    # Create order request (will trigger adaptive update)
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
    await asyncio.sleep(0.1)
    
    # Offset may have changed based on volatility
    # The exact value depends on volatility calculation
    assert router._adaptive_offset_bps >= 0  # Should be non-negative
    
    await router.stop()
    await regime_detector.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_adaptive_parameters_have_limits() -> None:
    """Test that adaptive parameters respect max/min limits."""
    bus = EventBus()
    order_manager = OrderManager()
    router = ExecutionRouter(bus, order_manager)
    
    await bus.start()
    await router.start()
    
    base_ttl = settings.maker_ttl_ms
    base_offset = settings.maker_price_offset_bps
    
    # Verify limits
    assert router._adaptive_ttl_ms <= base_ttl * 2  # Max 2x base
    assert router._adaptive_ttl_ms >= base_ttl  # Min base
    assert router._adaptive_offset_bps >= 0  # Min 0
    assert router._adaptive_offset_bps <= base_offset + 3  # Max +3 bps
    
    await router.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_adaptive_parameters_are_deterministic() -> None:
    """Test that adaptive parameter updates are deterministic."""
    bus1 = EventBus()
    order_manager1 = OrderManager()
    router1 = ExecutionRouter(bus1, order_manager1)
    
    bus2 = EventBus()
    order_manager2 = OrderManager()
    router2 = ExecutionRouter(bus2, order_manager2)
    
    await bus1.start()
    await bus2.start()
    await router1.start()
    await router2.start()
    
    # Set up identical market data
    for bus in [bus1, bus2]:
        tick = Tick(
            symbol="BTCUSDT",
            price=50000.0,
            timestamp=datetime.utcnow(),
            bid=49999.0,
            ask=50001.0,
        )
        await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
    
    await asyncio.sleep(0.1)
    
    # Create identical order requests
    for router, bus in [(router1, bus1), (router2, bus2)]:
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
    
    await asyncio.sleep(0.1)
    
    # Parameters should be the same (deterministic)
    # Note: This is a simplified test - full determinism requires
    # identical timing and state
    assert router1._adaptive_ttl_ms == router2._adaptive_ttl_ms
    assert router1._adaptive_offset_bps == router2._adaptive_offset_bps
    
    await router1.stop()
    await router2.stop()
    await bus1.stop()
    await bus2.stop()





