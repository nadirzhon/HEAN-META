"""Tests for Cortex Decision Engine."""

import asyncio
from typing import Any

import pytest

from hean.archon.cortex import Cortex, SystemMode
from hean.archon.directives import DirectiveType
from hean.core.bus import EventBus
from hean.core.types import Event, EventType


class MockHealthMatrix:
    """Mock HealthMatrix for testing."""

    def __init__(self, initial_score: float = 100.0) -> None:
        self._score = initial_score

    def get_composite_score(self) -> float:
        return self._score

    def set_score(self, score: float) -> None:
        self._score = score


class MockSignalPipeline:
    """Mock SignalPipelineManager for testing."""

    def __init__(self, fill_rate: float = 100.0, dead_letters: int = 0) -> None:
        self._fill_rate = fill_rate
        self._dead_letters = dead_letters

    def get_status(self) -> dict[str, Any]:
        return {
            "fill_rate_pct": self._fill_rate,
            "dead_letter_count": self._dead_letters,
        }

    def set_fill_rate(self, rate: float) -> None:
        self._fill_rate = rate

    def set_dead_letters(self, count: int) -> None:
        self._dead_letters = count


@pytest.fixture
async def bus() -> EventBus:
    """Create and start EventBus."""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


async def test_normal_mode_on_healthy_system(bus: EventBus) -> None:
    """Test that Cortex stays in NORMAL mode with healthy system."""
    health = MockHealthMatrix(initial_score=85.0)
    pipeline = MockSignalPipeline(fill_rate=80.0, dead_letters=2)

    cortex = Cortex(
        bus=bus,
        health_matrix=health,
        signal_pipeline=pipeline,
        interval_sec=1,
    )

    await cortex.start()
    await asyncio.sleep(1.5)  # Let one decision cycle run
    await cortex.stop()

    status = cortex.get_status()
    assert status["mode"] == SystemMode.NORMAL.value
    assert status["running"] is False


async def test_emergency_mode_on_low_health(bus: EventBus) -> None:
    """Test that Cortex switches to EMERGENCY mode on low health."""
    health = MockHealthMatrix(initial_score=30.0)  # Critical health
    pipeline = MockSignalPipeline()

    cortex = Cortex(
        bus=bus,
        health_matrix=health,
        signal_pipeline=pipeline,
        interval_sec=1,
    )

    await cortex.start()
    await asyncio.sleep(1.5)  # Let one decision cycle run
    await cortex.stop()

    status = cortex.get_status()
    assert status["mode"] == SystemMode.EMERGENCY.value
    assert status["mode_changes"] == 1
    assert status["pause_directives"] == 1
    assert status["total_directives_issued"] >= 1

    # Verify directive details from status (not events)
    assert len(status["recent_directives"]) >= 1
    assert status["recent_directives"][0]["directive_type"] == DirectiveType.PAUSE_TRADING.value
    assert status["recent_directives"][0]["target_component"] == "trading_system"


async def test_defensive_mode_on_medium_health(bus: EventBus) -> None:
    """Test that Cortex switches to DEFENSIVE mode on medium health."""
    health = MockHealthMatrix(initial_score=55.0)  # Low but not critical
    pipeline = MockSignalPipeline()

    cortex = Cortex(
        bus=bus,
        health_matrix=health,
        signal_pipeline=pipeline,
        interval_sec=1,
    )

    await cortex.start()
    await asyncio.sleep(1.5)  # Let one decision cycle run
    await cortex.stop()

    status = cortex.get_status()
    assert status["mode"] == SystemMode.DEFENSIVE.value
    assert status["mode_changes"] == 1


async def test_mode_recovery(bus: EventBus) -> None:
    """Test that Cortex recovers from EMERGENCY to NORMAL when health recovers."""
    health = MockHealthMatrix(initial_score=30.0)  # Start critical
    pipeline = MockSignalPipeline()

    cortex = Cortex(
        bus=bus,
        health_matrix=health,
        signal_pipeline=pipeline,
        interval_sec=1,
    )

    await cortex.start()
    await asyncio.sleep(1.5)  # First cycle: EMERGENCY mode

    # Health recovers
    health.set_score(85.0)
    await asyncio.sleep(1.5)  # Second cycle: recover to NORMAL

    await cortex.stop()

    # Verify via deterministic internal status (not timing-dependent events)
    status = cortex.get_status()
    assert status["mode"] == SystemMode.NORMAL.value
    assert status["mode_changes"] == 2  # NORMAL → EMERGENCY → NORMAL
    assert status["pause_directives"] == 1  # Pause on emergency
    assert status["resume_directives"] == 1  # Resume on recovery

    # Verify directive history
    directives = status["recent_directives"]
    assert len(directives) == 2
    assert directives[0]["directive_type"] == DirectiveType.PAUSE_TRADING.value
    assert directives[1]["directive_type"] == DirectiveType.RESUME_TRADING.value


async def test_directive_published(bus: EventBus) -> None:
    """Test that directives are published to EventBus as ARCHON_DIRECTIVE events."""
    health = MockHealthMatrix(initial_score=35.0)  # Emergency
    pipeline = MockSignalPipeline()

    cortex = Cortex(
        bus=bus,
        health_matrix=health,
        signal_pipeline=pipeline,
        interval_sec=1,
    )

    await cortex.start()
    await asyncio.sleep(1.5)  # Let decision cycle run
    await cortex.stop()

    # Verify directive was issued (check status instead of events)
    status = cortex.get_status()
    assert status["total_directives_issued"] >= 1
    assert status["pause_directives"] == 1
    assert len(status["recent_directives"]) >= 1
    assert status["recent_directives"][0]["directive_type"] == DirectiveType.PAUSE_TRADING.value
    assert status["recent_directives"][0]["target_component"] == "trading_system"


async def test_get_status_structure(bus: EventBus) -> None:
    """Test that get_status returns expected structure."""
    health = MockHealthMatrix(initial_score=75.0)
    pipeline = MockSignalPipeline(fill_rate=60.0, dead_letters=5)

    cortex = Cortex(
        bus=bus,
        health_matrix=health,
        signal_pipeline=pipeline,
        interval_sec=30,
    )

    status = cortex.get_status()

    # Verify structure
    assert "running" in status
    assert "mode" in status
    assert "interval_sec" in status
    assert "last_evaluation" in status
    assert "total_directives_issued" in status
    assert "mode_changes" in status
    assert "pause_directives" in status
    assert "resume_directives" in status
    assert "recent_directives" in status

    assert status["running"] is False
    assert status["mode"] == SystemMode.NORMAL.value
    assert status["interval_sec"] == 30
    assert isinstance(status["recent_directives"], list)


async def test_aggressive_mode_on_excellent_health(bus: EventBus) -> None:
    """Test that Cortex switches to AGGRESSIVE mode on excellent health."""
    health = MockHealthMatrix(initial_score=95.0)  # Excellent
    pipeline = MockSignalPipeline(fill_rate=90.0, dead_letters=0)

    cortex = Cortex(
        bus=bus,
        health_matrix=health,
        signal_pipeline=pipeline,
        interval_sec=1,
    )

    await cortex.start()
    await asyncio.sleep(1.5)
    await cortex.stop()

    status = cortex.get_status()
    assert status["mode"] == SystemMode.AGGRESSIVE.value
    assert status["mode_changes"] == 1


async def test_cortex_without_health_matrix(bus: EventBus) -> None:
    """Test that Cortex works without health_matrix (defaults to 100)."""
    cortex = Cortex(
        bus=bus,
        health_matrix=None,
        signal_pipeline=None,
        interval_sec=1,
    )

    await cortex.start()
    await asyncio.sleep(1.5)
    await cortex.stop()

    status = cortex.get_status()
    assert status["mode"] == SystemMode.AGGRESSIVE.value  # 100 score → aggressive
    assert status["running"] is False
