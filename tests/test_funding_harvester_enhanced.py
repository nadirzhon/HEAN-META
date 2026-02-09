"""Tests for enhanced FundingHarvester strategy."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, FundingRate, Tick
from hean.strategies.funding_harvester import FundingHarvester


@pytest.fixture
def event_bus() -> EventBus:
    """Create event bus for testing."""
    return EventBus()


@pytest.fixture
def mock_http_client() -> MagicMock:
    """Create mock HTTP client."""
    client = MagicMock()
    client.get_funding_rate = AsyncMock()
    return client


@pytest.fixture
async def strategy(event_bus: EventBus, mock_http_client: MagicMock) -> FundingHarvester:
    """Create FundingHarvester strategy."""
    # Start event bus first
    await event_bus.start()

    strategy = FundingHarvester(
        bus=event_bus,
        symbols=["BTCUSDT", "ETHUSDT"],
        http_client=mock_http_client,
    )
    await strategy.start()
    return strategy


@pytest.mark.asyncio
async def test_historical_funding_initialization(strategy: FundingHarvester) -> None:
    """Test that historical funding tracking is initialized."""
    assert hasattr(strategy, "_historical_funding")
    assert "BTCUSDT" in strategy._historical_funding
    assert "ETHUSDT" in strategy._historical_funding
    assert len(strategy._historical_funding["BTCUSDT"]) == 0


@pytest.mark.asyncio
async def test_on_tick_stores_data(strategy: FundingHarvester, event_bus: EventBus) -> None:
    """Test that tick events are stored."""
    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=1.0,
        bid=49995.0,
        ask=50005.0,
    )

    await event_bus.publish(Event(EventType.TICK, data={"tick": tick}))
    await asyncio.sleep(0.01)  # Let event propagate

    assert "BTCUSDT" in strategy._last_tick
    assert strategy._last_tick["BTCUSDT"].price == 50000.0


@pytest.mark.asyncio
async def test_on_funding_stores_historical_data(
    strategy: FundingHarvester, event_bus: EventBus
) -> None:
    """Test that funding events are stored in history."""
    funding = FundingRate(
        symbol="BTCUSDT",
        rate=0.0001,
        timestamp=datetime.utcnow(),
        next_funding_time=datetime.utcnow() + timedelta(hours=8),
    )

    await event_bus.publish(Event(EventType.FUNDING, data={"funding": funding}))
    await asyncio.sleep(0.01)  # Let event propagate

    assert "BTCUSDT" in strategy._historical_funding
    assert len(strategy._historical_funding["BTCUSDT"]) == 1
    assert strategy._historical_funding["BTCUSDT"][0]["rate"] == 0.0001


@pytest.mark.asyncio
async def test_predict_next_funding_with_insufficient_data(
    strategy: FundingHarvester,
) -> None:
    """Test prediction with insufficient historical data."""
    # No history yet
    predicted = strategy.predict_next_funding("BTCUSDT")
    assert predicted == 0.0

    # Add current funding
    strategy._last_funding["BTCUSDT"] = FundingRate(
        symbol="BTCUSDT",
        rate=0.0005,
        timestamp=datetime.utcnow(),
    )

    predicted = strategy.predict_next_funding("BTCUSDT")
    assert predicted == 0.0005


@pytest.mark.asyncio
async def test_predict_next_funding_with_history(strategy: FundingHarvester) -> None:
    """Test prediction with historical data."""
    # Add historical funding data
    base_time = datetime.utcnow()
    rates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]

    for i, rate in enumerate(rates):
        strategy._historical_funding["BTCUSDT"].append({
            "rate": rate,
            "timestamp": base_time + timedelta(hours=i * 8),
        })

    predicted = strategy.predict_next_funding("BTCUSDT")

    # Should predict something based on weighted average and momentum
    assert isinstance(predicted, float)
    # Recent trend is upward, so prediction should be positive
    assert predicted > 0.0001


@pytest.mark.asyncio
async def test_signal_generation_with_timing_optimization(
    strategy: FundingHarvester, event_bus: EventBus
) -> None:
    """Test that signals are only generated in optimal timing window."""
    # Add tick data
    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=1.0,
        bid=49995.0,
        ask=50005.0,
    )
    await event_bus.publish(Event(EventType.TICK, data={"tick": tick}))
    await asyncio.sleep(0.01)  # Let event propagate

    signals = []

    async def capture_signal(event: Event) -> None:
        if event.event_type == EventType.SIGNAL:
            signals.append(event.data["signal"])

    event_bus.subscribe(EventType.SIGNAL, capture_signal)

    # Test 1: Funding too far away (5 hours = 18000s)
    funding_far = FundingRate(
        symbol="BTCUSDT",
        rate=0.0005,
        timestamp=datetime.utcnow(),
        next_funding_time=datetime.utcnow() + timedelta(hours=5),
    )
    await event_bus.publish(Event(EventType.FUNDING, data={"funding": funding_far}))
    await asyncio.sleep(0.01)  # Let event propagate
    assert len(signals) == 0, "Should not signal when funding is too far"

    # Test 2: Funding in optimal window (1.5 hours = 5400s)
    funding_optimal = FundingRate(
        symbol="BTCUSDT",
        rate=0.0005,
        timestamp=datetime.utcnow(),
        next_funding_time=datetime.utcnow() + timedelta(hours=1.5),
    )
    await event_bus.publish(Event(EventType.FUNDING, data={"funding": funding_optimal}))
    await asyncio.sleep(0.01)  # Let event propagate
    assert len(signals) == 1, "Should signal when funding is in optimal window"


@pytest.mark.asyncio
async def test_signal_metadata_includes_predictions(
    strategy: FundingHarvester, event_bus: EventBus
) -> None:
    """Test that signals include prediction metadata."""
    # Setup: add tick and historical funding
    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=1.0,
        bid=49995.0,
        ask=50005.0,
    )
    await event_bus.publish(Event(EventType.TICK, data={"tick": tick}))
    await asyncio.sleep(0.01)  # Let event propagate

    # Add some history
    for i in range(5):
        strategy._historical_funding["BTCUSDT"].append({
            "rate": 0.0001 * (i + 1),
            "timestamp": datetime.utcnow() + timedelta(hours=i * 8),
        })

    signals = []

    async def capture_signal(event: Event) -> None:
        if event.event_type == EventType.SIGNAL:
            signals.append(event.data["signal"])

    event_bus.subscribe(EventType.SIGNAL, capture_signal)

    # Trigger signal with optimal timing
    funding = FundingRate(
        symbol="BTCUSDT",
        rate=0.0005,
        timestamp=datetime.utcnow(),
        next_funding_time=datetime.utcnow() + timedelta(hours=2),
    )
    await event_bus.publish(Event(EventType.FUNDING, data={"funding": funding}))
    await asyncio.sleep(0.01)  # Let event propagate

    assert len(signals) == 1
    signal = signals[0]

    # Check metadata
    assert "predicted_funding" in signal.metadata
    assert "confidence" in signal.metadata
    assert "time_to_funding_hrs" in signal.metadata
    assert isinstance(signal.metadata["predicted_funding"], float)
    assert isinstance(signal.metadata["confidence"], float)


@pytest.mark.asyncio
async def test_fetch_funding_rates(
    strategy: FundingHarvester, mock_http_client: MagicMock
) -> None:
    """Test fetching funding rates from API."""
    # Setup mock response
    mock_http_client.get_funding_rate.return_value = {
        "fundingRate": "0.0001",
        "nextFundingTime": str(int((datetime.utcnow() + timedelta(hours=8)).timestamp() * 1000)),
    }

    # Fetch funding rates
    await strategy.fetch_funding_rates()

    # Verify API was called for all symbols
    assert mock_http_client.get_funding_rate.call_count == 2  # BTCUSDT, ETHUSDT


@pytest.mark.asyncio
async def test_funding_threshold_filtering(
    strategy: FundingHarvester, event_bus: EventBus
) -> None:
    """Test that signals are filtered by funding rate threshold."""
    # Add tick
    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=1.0,
        bid=49995.0,
        ask=50005.0,
    )
    await event_bus.publish(Event(EventType.TICK, data={"tick": tick}))
    await asyncio.sleep(0.01)  # Let event propagate

    signals = []

    async def capture_signal(event: Event) -> None:
        if event.event_type == EventType.SIGNAL:
            signals.append(event.data["signal"])

    event_bus.subscribe(EventType.SIGNAL, capture_signal)

    # Test with very low funding (below threshold)
    low_funding = FundingRate(
        symbol="BTCUSDT",
        rate=0.00005,  # 0.005% - below 0.01% threshold
        timestamp=datetime.utcnow(),
        next_funding_time=datetime.utcnow() + timedelta(hours=2),
    )
    await event_bus.publish(Event(EventType.FUNDING, data={"funding": low_funding}))
    await asyncio.sleep(0.01)  # Let event propagate

    assert len(signals) == 0, "Should not signal when funding is below threshold"


@pytest.mark.asyncio
async def test_prediction_alignment_confidence(strategy: FundingHarvester) -> None:
    """Test that confidence reflects prediction alignment."""
    # Add consistent upward trend
    for i in range(10):
        strategy._historical_funding["BTCUSDT"].append({
            "rate": 0.0001 * (i + 1),
            "timestamp": datetime.utcnow() + timedelta(hours=i * 8),
        })

    # Current and predicted should align
    strategy._last_funding["BTCUSDT"] = FundingRate(
        symbol="BTCUSDT",
        rate=0.0010,
        timestamp=datetime.utcnow(),
    )

    predicted = strategy.predict_next_funding("BTCUSDT")

    # Both should be positive (aligned)
    assert predicted > 0


@pytest.mark.asyncio
async def test_no_signal_without_tick_data(
    strategy: FundingHarvester, event_bus: EventBus
) -> None:
    """Test that no signal is generated without tick data."""
    signals = []

    async def capture_signal(event: Event) -> None:
        if event.event_type == EventType.SIGNAL:
            signals.append(event.data["signal"])

    event_bus.subscribe(EventType.SIGNAL, capture_signal)

    # Trigger funding without tick data
    funding = FundingRate(
        symbol="BTCUSDT",
        rate=0.0005,
        timestamp=datetime.utcnow(),
        next_funding_time=datetime.utcnow() + timedelta(hours=2),
    )
    await event_bus.publish(Event(EventType.FUNDING, data={"funding": funding}))
    await asyncio.sleep(0.01)  # Let event propagate

    assert len(signals) == 0, "Should not signal without tick data"


@pytest.mark.asyncio
async def test_side_selection_based_on_funding(
    strategy: FundingHarvester, event_bus: EventBus
) -> None:
    """Test that side is correctly selected based on funding rate sign."""
    # Add tick
    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=1.0,
        bid=49995.0,
        ask=50005.0,
    )
    await event_bus.publish(Event(EventType.TICK, data={"tick": tick}))
    await asyncio.sleep(0.01)  # Let event propagate

    signals = []

    async def capture_signal(event: Event) -> None:
        if event.event_type == EventType.SIGNAL:
            signals.append(event.data["signal"])

    event_bus.subscribe(EventType.SIGNAL, capture_signal)

    # Test positive funding -> short
    positive_funding = FundingRate(
        symbol="BTCUSDT",
        rate=0.0005,  # Positive
        timestamp=datetime.utcnow(),
        next_funding_time=datetime.utcnow() + timedelta(hours=2),
    )
    await event_bus.publish(Event(EventType.FUNDING, data={"funding": positive_funding}))
    await asyncio.sleep(0.01)  # Let event propagate

    assert len(signals) == 1
    assert signals[0].side == "sell", "Positive funding should trigger short"

    # Clear signals
    signals.clear()
    strategy._positions.clear()

    # Test negative funding -> long
    negative_funding = FundingRate(
        symbol="ETHUSDT",  # Different symbol
        rate=-0.0005,  # Negative
        timestamp=datetime.utcnow(),
        next_funding_time=datetime.utcnow() + timedelta(hours=2),
    )

    # Add tick for ETH
    eth_tick = Tick(
        symbol="ETHUSDT",
        price=3000.0,
        timestamp=datetime.utcnow(),
        volume=1.0,
        bid=2995.0,
        ask=3005.0,
    )
    await event_bus.publish(Event(EventType.TICK, data={"tick": eth_tick}))
    await asyncio.sleep(0.01)  # Let event propagate
    await event_bus.publish(Event(EventType.FUNDING, data={"funding": negative_funding}))
    await asyncio.sleep(0.01)  # Let event propagate

    assert len(signals) == 1
    assert signals[0].side == "buy", "Negative funding should trigger long"
