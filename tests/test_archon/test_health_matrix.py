"""Tests for HealthMatrix."""

import asyncio
from unittest.mock import MagicMock

from hean.archon.health_matrix import HealthMatrix
from hean.archon.heartbeat import HeartbeatRegistry
from hean.core.bus import BusHealthStatus, EventBus


def _make_bus_with_health(
    is_healthy: bool = True,
    is_degraded: bool = False,
    is_circuit_open: bool = False,
    handler_errors: int = 0,
    events_processed: int = 100,
) -> EventBus:
    """Create an EventBus mock with controllable health status."""
    bus = EventBus()
    health = BusHealthStatus(
        is_healthy=is_healthy,
        is_degraded=is_degraded,
        is_circuit_open=is_circuit_open,
    )
    bus.get_health = MagicMock(return_value=health)  # type: ignore[method-assign]
    bus.get_metrics = MagicMock(  # type: ignore[method-assign]
        return_value={
            "events_processed": events_processed,
            "handler_errors": handler_errors,
        }
    )
    return bus


async def test_composite_score_all_healthy() -> None:
    """All sources good -> score near 100."""
    bus = _make_bus_with_health(
        is_healthy=True,
        is_degraded=False,
        is_circuit_open=False,
        handler_errors=0,
        events_processed=100,
    )
    heartbeat = HeartbeatRegistry(default_interval=5.0, failure_threshold=3)
    heartbeat.register("comp_a")
    heartbeat.beat("comp_a")
    heartbeat.register("comp_b")
    heartbeat.beat("comp_b")

    matrix = HealthMatrix(
        bus=bus, heartbeat=heartbeat, signal_pipeline=None
    )

    # Manually refresh score (don't rely on background loop)
    matrix._refresh_score()
    score = matrix.get_composite_score()

    # All healthy: bus=1.0*40 + heartbeat=1.0*30 + pipeline=1.0*20 + error=1.0*10 = 100
    assert score == 100.0


async def test_composite_score_bus_degraded() -> None:
    """Bus degraded -> score drops."""
    bus = _make_bus_with_health(
        is_healthy=False,
        is_degraded=True,
        is_circuit_open=False,
        handler_errors=0,
        events_processed=100,
    )
    heartbeat = HeartbeatRegistry(default_interval=5.0, failure_threshold=3)
    heartbeat.register("comp_a")
    heartbeat.beat("comp_a")

    matrix = HealthMatrix(
        bus=bus, heartbeat=heartbeat, signal_pipeline=None
    )
    matrix._refresh_score()
    score = matrix.get_composite_score()

    # bus=0.5*40 + heartbeat=1.0*30 + pipeline=1.0*20 + error=1.0*10 = 80
    assert score == 80.0


async def test_composite_score_bus_circuit_open() -> None:
    """Bus circuit open -> score drops significantly."""
    bus = _make_bus_with_health(
        is_healthy=False,
        is_degraded=False,
        is_circuit_open=True,
        handler_errors=0,
        events_processed=100,
    )
    heartbeat = HeartbeatRegistry(default_interval=5.0, failure_threshold=3)
    heartbeat.register("comp_a")
    heartbeat.beat("comp_a")

    matrix = HealthMatrix(
        bus=bus, heartbeat=heartbeat, signal_pipeline=None
    )
    matrix._refresh_score()
    score = matrix.get_composite_score()

    # bus=0.0*40 + heartbeat=1.0*30 + pipeline=1.0*20 + error=1.0*10 = 60
    assert score == 60.0


async def test_composite_score_unhealthy_components() -> None:
    """Dead components -> score drops via heartbeat weight."""
    bus = _make_bus_with_health(
        is_healthy=True,
        is_degraded=False,
        is_circuit_open=False,
        handler_errors=0,
        events_processed=100,
    )
    heartbeat = HeartbeatRegistry(
        default_interval=0.01, failure_threshold=3
    )
    heartbeat.register("alive", interval=100.0)  # long interval, stays healthy
    heartbeat.beat("alive")

    heartbeat.register("dead_a", interval=0.01)
    heartbeat.register("dead_b", interval=0.01)
    # Force both dead by setting last_beat far in the past
    import time

    heartbeat._last_beat["dead_a"] = time.time() - 10.0
    heartbeat._last_beat["dead_b"] = time.time() - 10.0

    matrix = HealthMatrix(
        bus=bus, heartbeat=heartbeat, signal_pipeline=None
    )
    matrix._refresh_score()
    score = matrix.get_composite_score()

    # bus=1.0*40 + heartbeat=(1/3)*30 + pipeline=1.0*20 + error=1.0*10
    # = 40 + 10 + 20 + 10 = 80
    assert score == 80.0


async def test_composite_score_with_handler_errors() -> None:
    """Handler errors reduce the error-rate component."""
    bus = _make_bus_with_health(
        is_healthy=True,
        is_degraded=False,
        is_circuit_open=False,
        handler_errors=50,
        events_processed=100,
    )
    heartbeat = HeartbeatRegistry(default_interval=5.0, failure_threshold=3)

    matrix = HealthMatrix(
        bus=bus, heartbeat=heartbeat, signal_pipeline=None
    )
    matrix._refresh_score()
    score = matrix.get_composite_score()

    # bus=1.0*40 + heartbeat=1.0*30 + pipeline=1.0*20 + error=0.5*10
    # = 40 + 30 + 20 + 5 = 95
    assert score == 95.0


async def test_composite_score_with_pipeline() -> None:
    """Signal pipeline fill rate affects pipeline component."""
    bus = _make_bus_with_health(
        is_healthy=True,
        is_degraded=False,
        is_circuit_open=False,
        handler_errors=0,
        events_processed=100,
    )
    heartbeat = HeartbeatRegistry(default_interval=5.0, failure_threshold=3)

    # Mock signal pipeline
    pipeline = MagicMock()
    pipeline.get_status.return_value = {
        "fill_rate_pct": 50.0,
    }

    matrix = HealthMatrix(
        bus=bus, heartbeat=heartbeat, signal_pipeline=pipeline
    )
    matrix._refresh_score()
    score = matrix.get_composite_score()

    # bus=1.0*40 + heartbeat=1.0*30 + pipeline=0.5*20 + error=1.0*10
    # = 40 + 30 + 10 + 10 = 90
    assert score == 90.0


async def test_full_status_structure() -> None:
    """Verify returned dict has all expected keys."""
    bus = _make_bus_with_health()
    heartbeat = HeartbeatRegistry(default_interval=5.0, failure_threshold=3)
    heartbeat.register("comp_a")
    heartbeat.beat("comp_a")

    matrix = HealthMatrix(
        bus=bus, heartbeat=heartbeat, signal_pipeline=None
    )
    matrix._refresh_score()
    status = await matrix.get_full_status()

    assert "composite_score" in status
    assert "bus_health" in status
    assert "heartbeats" in status
    assert "pipeline_metrics" in status
    assert "breakdown" in status

    # Check bus_health sub-keys
    bh = status["bus_health"]
    assert "is_healthy" in bh
    assert "is_degraded" in bh
    assert "is_circuit_open" in bh
    assert "queue_utilization_pct" in bh

    # Check breakdown sub-keys
    bd = status["breakdown"]
    assert "bus_score" in bd
    assert "heartbeat_score" in bd
    assert "pipeline_score" in bd
    assert "error_score" in bd
    assert "composite" in bd


async def test_start_stop() -> None:
    """Start and stop the health matrix without errors."""
    bus = _make_bus_with_health()
    heartbeat = HeartbeatRegistry(default_interval=5.0, failure_threshold=3)

    matrix = HealthMatrix(
        bus=bus, heartbeat=heartbeat, signal_pipeline=None
    )
    await matrix.start()
    # Give the background loop a moment to start
    await asyncio.sleep(0.05)
    assert matrix._running is True

    await matrix.stop()
    assert matrix._running is False
