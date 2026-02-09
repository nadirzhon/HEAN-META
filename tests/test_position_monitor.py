"""Tests for PositionMonitor."""

import asyncio
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

# Set test environment before importing config
os.environ["TRADING_MODE"] = "live"
os.environ["DRY_RUN"] = "true"
os.environ["LIVE_CONFIRM"] = "NO"

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Position
from hean.execution.position_monitor import PositionMonitor
from hean.portfolio.accounting import PortfolioAccounting


@pytest.fixture
def bus():
    """Create event bus."""
    return EventBus()


@pytest.fixture
def accounting():
    """Create portfolio accounting."""
    return PortfolioAccounting(initial_capital=1000.0)


@pytest.fixture
def position_monitor(bus, accounting):
    """Create position monitor."""
    return PositionMonitor(bus, accounting)


@pytest.mark.asyncio
async def test_position_monitor_initialization(position_monitor):
    """Test position monitor initializes correctly."""
    assert position_monitor._running is False
    assert position_monitor._force_close_enabled is True
    assert position_monitor._positions_force_closed == 0


@pytest.mark.asyncio
async def test_position_monitor_start_stop(position_monitor):
    """Test position monitor can start and stop."""
    await position_monitor.start()
    assert position_monitor._running is True

    await position_monitor.stop()
    assert position_monitor._running is False


@pytest.mark.asyncio
async def test_force_close_stale_position(bus, accounting, position_monitor):
    """Test position monitor force-closes stale positions."""
    # Set short max_hold_seconds for testing
    position_monitor._max_hold_seconds = 2  # 2 seconds
    position_monitor._check_interval_seconds = 1  # Check every 1 second

    # Create a stale position (old timestamp)
    old_timestamp = datetime.utcnow() - timedelta(seconds=5)  # 5 seconds ago
    position = Position(
        position_id="test_pos_1",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        entry_price=50000.0,
        current_price=50000.0,
        timestamp=old_timestamp,
        opened_at=old_timestamp,  # Add opened_at field
    )

    # Add position to accounting
    accounting._positions["test_pos_1"] = position

    # Track events published to bus
    events_published = []
    force_closes = 0

    async def event_handler(event: Event):
        nonlocal force_closes
        events_published.append(event)
        # Simulate position removal after close order is sent
        # In real system, this would happen when the order fills
        if "force_close" in event.data["order_request"].metadata:
            force_closes += 1
            accounting._positions.pop("test_pos_1", None)

    bus.subscribe(EventType.ORDER_REQUEST, event_handler)

    # Start position monitor
    await position_monitor.start()

    # Wait for monitor to check positions (should detect and close)
    await asyncio.sleep(2)

    # Stop monitor
    await position_monitor.stop()

    # Verify force-close was triggered (using handler count to avoid double-count)
    assert force_closes == 1, f"Expected 1 force close, got {force_closes}"

    # Verify order request was published
    assert len(events_published) > 0
    order_request_event = events_published[0]
    assert order_request_event.event_type == EventType.ORDER_REQUEST

    order_request = order_request_event.data["order_request"]
    assert order_request.symbol == "BTCUSDT"
    assert order_request.side == "sell"  # Opposite of position side
    assert order_request.size == 0.1
    assert order_request.order_type == "market"
    assert order_request.metadata["force_close"] is True


@pytest.mark.asyncio
async def test_no_force_close_for_fresh_positions(bus, accounting, position_monitor):
    """Test position monitor does not force-close fresh positions."""
    # Set max_hold_seconds
    position_monitor._max_hold_seconds = 10  # 10 seconds
    position_monitor._check_interval_seconds = 1  # Check every 1 second

    # Create a fresh position (recent timestamp)
    fresh_timestamp = datetime.utcnow() - timedelta(seconds=2)  # 2 seconds ago
    position = Position(
        position_id="test_pos_2",
        strategy_id="test_strategy",
        symbol="ETHUSDT",
        side="sell",
        size=0.5,
        entry_price=3000.0,
        current_price=3000.0,
        timestamp=fresh_timestamp,
        opened_at=fresh_timestamp,  # Add opened_at field
    )

    # Add position to accounting
    accounting._positions["test_pos_2"] = position

    # Track events published to bus
    events_published = []
    force_closes = 0

    async def event_handler(event: Event):
        nonlocal force_closes
        events_published.append(event)
        # Simulate position removal if force close
        if "force_close" in event.data["order_request"].metadata:
            force_closes += 1
            accounting._positions.pop("test_pos_2", None)

    bus.subscribe(EventType.ORDER_REQUEST, event_handler)

    # Start position monitor
    await position_monitor.start()

    # Wait for monitor to check positions
    await asyncio.sleep(2)

    # Stop monitor
    await position_monitor.stop()

    # Verify NO force-close was triggered
    assert position_monitor._positions_force_closed == 0

    # Verify NO order request was published
    assert len(events_published) == 0


@pytest.mark.asyncio
async def test_disable_force_close(bus, accounting, position_monitor):
    """Test disabling force-close functionality."""
    # Disable force-close
    position_monitor.enable_force_close(False)
    assert position_monitor._force_close_enabled is False

    # Create a stale position
    old_timestamp = datetime.utcnow() - timedelta(seconds=100)
    position = Position(
        position_id="test_pos_3",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        entry_price=50000.0,
        current_price=50000.0,
        timestamp=old_timestamp,
        opened_at=old_timestamp,  # Add opened_at field
    )

    accounting._positions["test_pos_3"] = position

    # Check positions (should do nothing)
    await position_monitor._check_positions()

    # Verify NO force-close
    assert position_monitor._positions_force_closed == 0


def test_get_statistics(position_monitor):
    """Test getting position monitor statistics."""
    stats = position_monitor.get_statistics()

    assert "positions_force_closed" in stats
    assert "force_close_enabled" in stats
    assert "max_hold_seconds" in stats
    assert "check_interval_seconds" in stats
    assert "recent_force_closes" in stats

    assert stats["positions_force_closed"] == 0
    assert stats["force_close_enabled"] is True


def test_set_max_hold_seconds(position_monitor):
    """Test updating max_hold_seconds."""
    position_monitor.set_max_hold_seconds(600)
    assert position_monitor._max_hold_seconds == 600

    # Test minimum value (1 minute)
    position_monitor.set_max_hold_seconds(30)
    assert position_monitor._max_hold_seconds == 60  # Enforced minimum
