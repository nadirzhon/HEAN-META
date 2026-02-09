"""Tests for smart execution algorithms (TWAP, VWAP, Iceberg Detection)."""

import asyncio
from datetime import datetime, timedelta

import pytest

from hean.core.bus import EventBus
from hean.core.types import EventType, OrderRequest
from hean.execution.smart_execution import (
    IcebergDetector,
    TWAPExecutor,
    VWAPExecutor,
)


@pytest.fixture
def event_bus() -> EventBus:
    """Create event bus for testing."""
    return EventBus()


@pytest.fixture
def mock_execution_router():
    """Create mock execution router."""

    class MockRouter:
        def __init__(self):
            self.submitted_orders = []

        async def submit_order(self, order_request: OrderRequest):
            self.submitted_orders.append(order_request)

    return MockRouter()


@pytest.fixture
def twap_executor(event_bus, mock_execution_router):
    """Create TWAP executor."""
    return TWAPExecutor(event_bus, mock_execution_router)


@pytest.fixture
def vwap_executor(event_bus, mock_execution_router):
    """Create VWAP executor."""
    return VWAPExecutor(event_bus, mock_execution_router)


@pytest.fixture
def iceberg_detector():
    """Create iceberg detector."""
    return IcebergDetector()


@pytest.fixture
def sample_order_request():
    """Create sample order request."""
    return OrderRequest(
        signal_id="test_signal",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=1.0,
        price=50000.0,
        order_type="limit",
    )


class TestTWAPExecutor:
    """Test TWAP execution."""

    async def test_twap_executor_initialization(self, twap_executor):
        """Test TWAP executor initializes correctly."""
        assert twap_executor._running is False
        assert len(twap_executor._active_orders) == 0

        await twap_executor.start()
        assert twap_executor._running is True

        await twap_executor.stop()
        assert twap_executor._running is False

    async def test_twap_creates_slices(self, twap_executor, sample_order_request):
        """Test TWAP creates correct number of slices."""
        await twap_executor.start()

        parent_order_id = await twap_executor.execute_twap(
            sample_order_request, num_slices=5, interval_seconds=10
        )

        assert parent_order_id in twap_executor._active_orders
        slices = twap_executor._active_orders[parent_order_id]
        assert len(slices) == 5

        # Check slice sizes
        expected_slice_size = sample_order_request.size / 5
        for slice_obj in slices:
            assert slice_obj.size == pytest.approx(expected_slice_size, rel=1e-6)
            assert slice_obj.symbol == sample_order_request.symbol
            assert slice_obj.side == sample_order_request.side

        await twap_executor.stop()

    async def test_twap_validates_parameters(self, twap_executor, sample_order_request):
        """Test TWAP validates input parameters."""
        await twap_executor.start()

        with pytest.raises(ValueError, match="num_slices must be >= 1"):
            await twap_executor.execute_twap(
                sample_order_request, num_slices=0, interval_seconds=60
            )

        with pytest.raises(ValueError, match="interval_seconds must be >= 1"):
            await twap_executor.execute_twap(
                sample_order_request, num_slices=5, interval_seconds=0
            )

        await twap_executor.stop()

    async def test_twap_get_order_status(self, twap_executor, sample_order_request):
        """Test TWAP order status retrieval."""
        await twap_executor.start()

        parent_order_id = await twap_executor.execute_twap(
            sample_order_request, num_slices=3, interval_seconds=5
        )

        status = twap_executor.get_order_status(parent_order_id)

        assert status["parent_order_id"] == parent_order_id
        assert status["total_slices"] == 3
        assert status["pending_slices"] == 3
        assert status["executed_slices"] == 0
        assert status["progress_pct"] == 0.0

        await twap_executor.stop()

    async def test_twap_execution_publishes_events(
        self, twap_executor, sample_order_request, event_bus
    ):
        """Test TWAP publishes events to event bus."""
        events_received = []

        async def event_handler(event):
            events_received.append(event)

        event_bus.subscribe(EventType.ORDER_PLACED, event_handler)
        await event_bus.start()
        await twap_executor.start()

        await twap_executor.execute_twap(
            sample_order_request, num_slices=2, interval_seconds=1
        )

        # Give time for events to propagate
        await asyncio.sleep(0.2)

        # Should have received start event
        assert len(events_received) > 0
        assert events_received[0].event_type == EventType.ORDER_PLACED
        assert events_received[0].data["execution_type"] == "TWAP"

        await twap_executor.stop()
        await event_bus.stop()


class TestVWAPExecutor:
    """Test VWAP execution."""

    async def test_vwap_executor_initialization(self, vwap_executor):
        """Test VWAP executor initializes correctly."""
        assert vwap_executor._running is False
        assert len(vwap_executor._volume_history) == 0

        await vwap_executor.start()
        assert vwap_executor._running is True

        await vwap_executor.stop()
        assert vwap_executor._running is False

    async def test_vwap_volume_tracking(self, vwap_executor):
        """Test VWAP tracks volume correctly."""
        await vwap_executor.start()

        symbol = "BTCUSDT"
        vwap_executor.update_volume(symbol, 100.0)
        vwap_executor.update_volume(symbol, 150.0)
        vwap_executor.update_volume(symbol, 200.0)

        assert symbol in vwap_executor._volume_history
        assert len(vwap_executor._volume_history[symbol]) == 3

        volume_rate = vwap_executor._get_recent_volume_rate(symbol)
        assert volume_rate > 0

        await vwap_executor.stop()

    async def test_vwap_validates_participation_rate(
        self, vwap_executor, sample_order_request
    ):
        """Test VWAP validates participation rate."""
        await vwap_executor.start()

        with pytest.raises(ValueError, match="target_participation must be in"):
            await vwap_executor.execute_vwap(
                sample_order_request, target_participation=0.0
            )

        with pytest.raises(ValueError, match="target_participation must be in"):
            await vwap_executor.execute_vwap(
                sample_order_request, target_participation=1.5
            )

        await vwap_executor.stop()

    async def test_vwap_clamps_participation_rate(
        self, vwap_executor, sample_order_request
    ):
        """Test VWAP clamps participation to safe bounds."""
        await vwap_executor.start()

        # Should clamp to max
        parent_order_id = await vwap_executor.execute_vwap(
            sample_order_request, target_participation=0.5  # Above max 0.25
        )

        order_data = vwap_executor._active_orders[parent_order_id]
        assert order_data["target_participation"] == 0.25  # Clamped to max

        await vwap_executor.stop()

    async def test_vwap_get_order_status(self, vwap_executor, sample_order_request):
        """Test VWAP order status retrieval."""
        await vwap_executor.start()

        # Add some volume data
        vwap_executor.update_volume(sample_order_request.symbol, 1000.0)

        parent_order_id = await vwap_executor.execute_vwap(
            sample_order_request, target_participation=0.1
        )

        status = vwap_executor.get_order_status(parent_order_id)

        assert status["parent_order_id"] == parent_order_id
        assert status["symbol"] == sample_order_request.symbol
        assert status["total_size"] == sample_order_request.size
        assert status["target_participation"] == 0.1

        await vwap_executor.stop()


class TestIcebergDetector:
    """Test iceberg order detection."""

    def test_iceberg_detector_initialization(self, iceberg_detector):
        """Test iceberg detector initializes correctly."""
        assert len(iceberg_detector._detected_levels) == 0
        assert iceberg_detector._refresh_threshold == 3

    def test_iceberg_orderbook_update(self, iceberg_detector):
        """Test iceberg detector processes orderbook updates."""
        symbol = "BTCUSDT"
        bids = [(50000.0, 1.0), (49999.0, 2.0)]
        asks = [(50001.0, 1.5), (50002.0, 2.5)]

        iceberg_detector.update_orderbook(symbol, bids, asks)

        assert symbol in iceberg_detector._detected_levels
        # Should have 4 levels detected (2 bids + 2 asks)
        assert len(iceberg_detector._detected_levels[symbol]) == 4

    def test_iceberg_refresh_detection(self, iceberg_detector):
        """Test iceberg detector identifies level refreshes."""
        symbol = "BTCUSDT"
        price = 50000.0
        size = 1.0

        bids = [(price, size)]
        asks = []

        # First update
        iceberg_detector.update_orderbook(symbol, bids, asks)

        # Wait to simulate refresh
        import time

        time.sleep(0.1)

        # Second update (refresh)
        iceberg_detector.update_orderbook(symbol, bids, asks)

        # Check refresh count
        level = iceberg_detector._detected_levels[symbol][price]
        assert level.refresh_count >= 1

    def test_iceberg_detect_requires_threshold(self, iceberg_detector):
        """Test iceberg detection requires minimum refreshes."""
        symbol = "BTCUSDT"
        bids = [(50000.0, 1.0)]
        asks = []

        # Single update
        iceberg_detector.update_orderbook(symbol, bids, asks)

        # Should not return iceberg (below threshold)
        icebergs = iceberg_detector.detect_iceberg(symbol, "buy")
        assert len(icebergs) == 0

    def test_iceberg_get_all_icebergs(self, iceberg_detector):
        """Test getting all detected icebergs."""
        # Add multiple symbols
        iceberg_detector.update_orderbook("BTCUSDT", [(50000.0, 1.0)], [])
        iceberg_detector.update_orderbook("ETHUSDT", [(3000.0, 2.0)], [])

        all_icebergs = iceberg_detector.get_all_icebergs(min_confidence=0.0)

        # Should have symbols registered
        assert "BTCUSDT" in iceberg_detector._detected_levels
        assert "ETHUSDT" in iceberg_detector._detected_levels

    def test_iceberg_clear_old_levels(self, iceberg_detector):
        """Test clearing old iceberg levels."""
        symbol = "BTCUSDT"
        bids = [(50000.0, 1.0)]
        asks = []

        iceberg_detector.update_orderbook(symbol, bids, asks)

        # Clear with very short max age
        cleared = iceberg_detector.clear_old_levels(max_age_seconds=0)

        # Should have cleared the level
        assert cleared >= 1


class TestSmartExecutionIntegration:
    """Integration tests for smart execution components."""

    async def test_twap_vwap_coordination(
        self, twap_executor, vwap_executor, sample_order_request
    ):
        """Test TWAP and VWAP can run concurrently."""
        await twap_executor.start()
        await vwap_executor.start()

        # Submit both types of orders
        twap_id = await twap_executor.execute_twap(
            sample_order_request, num_slices=3, interval_seconds=5
        )

        vwap_id = await vwap_executor.execute_vwap(
            sample_order_request, target_participation=0.1
        )

        # Both should be tracked
        assert twap_id in twap_executor._active_orders
        assert vwap_id in vwap_executor._active_orders

        await twap_executor.stop()
        await vwap_executor.stop()

    async def test_iceberg_detection_with_execution(
        self, iceberg_detector, twap_executor, sample_order_request
    ):
        """Test iceberg detection alongside order execution."""
        await twap_executor.start()

        # Detect icebergs
        symbol = sample_order_request.symbol
        iceberg_detector.update_orderbook(
            symbol, [(50000.0, 10.0), (49999.0, 5.0)], [(50001.0, 8.0)]
        )

        detected = iceberg_detector.detect_iceberg(symbol, "buy")

        # Execute TWAP order
        twap_id = await twap_executor.execute_twap(
            sample_order_request, num_slices=5, interval_seconds=2
        )

        # Both should work independently
        assert twap_id in twap_executor._active_orders
        assert symbol in iceberg_detector._detected_levels

        await twap_executor.stop()
