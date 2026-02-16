"""Tests for HeartbeatRegistry."""

import time

from hean.archon.heartbeat import HeartbeatRegistry


def test_register_and_beat() -> None:
    """Register component, beat, verify healthy."""
    registry = HeartbeatRegistry(default_interval=1.0, failure_threshold=3)
    registry.register("strategy_impulse", interval=1.0)
    registry.beat("strategy_impulse")

    status = registry.get_status()
    assert "strategy_impulse" in status
    info = status["strategy_impulse"]
    assert info["healthy"] is True
    assert info["missed_beats"] == 0
    assert info["interval_sec"] == 1.0


def test_missed_beats() -> None:
    """Register, don't beat, verify unhealthy after threshold."""
    registry = HeartbeatRegistry(default_interval=0.01, failure_threshold=3)
    registry.register("slow_component", interval=0.01)

    # Artificially set last_beat to far in the past
    registry._last_beat["slow_component"] = time.time() - 0.1

    status = registry.get_status()
    info = status["slow_component"]
    # With interval=0.01s and 0.1s elapsed, missed_beats = int(0.1/0.01) = 10
    assert info["missed_beats"] >= 3
    assert info["healthy"] is False


def test_get_unhealthy() -> None:
    """Multiple components, some healthy some not."""
    registry = HeartbeatRegistry(default_interval=1.0, failure_threshold=3)

    registry.register("healthy_a", interval=1.0)
    registry.beat("healthy_a")

    registry.register("healthy_b", interval=1.0)
    registry.beat("healthy_b")

    registry.register("dead_c", interval=0.01)
    # Force dead_c to be far in the past
    registry._last_beat["dead_c"] = time.time() - 1.0

    unhealthy = registry.get_unhealthy()
    assert "dead_c" in unhealthy
    assert "healthy_a" not in unhealthy
    assert "healthy_b" not in unhealthy


def test_metadata() -> None:
    """Beat with metadata, verify in status."""
    registry = HeartbeatRegistry(default_interval=5.0, failure_threshold=3)
    registry.register("risk_governor")
    registry.beat("risk_governor", metadata={"state": "NORMAL", "level": 0})

    status = registry.get_status()
    info = status["risk_governor"]
    assert info["metadata"]["state"] == "NORMAL"
    assert info["metadata"]["level"] == 0


def test_unregister() -> None:
    """Unregister removes the component from tracking."""
    registry = HeartbeatRegistry(default_interval=5.0, failure_threshold=3)
    registry.register("temp_component")
    assert "temp_component" in registry.get_status()

    registry.unregister("temp_component")
    assert "temp_component" not in registry.get_status()


def test_auto_register_on_beat() -> None:
    """Beating an unregistered component auto-registers it."""
    registry = HeartbeatRegistry(default_interval=2.0, failure_threshold=3)
    registry.beat("auto_registered")

    status = registry.get_status()
    assert "auto_registered" in status
    assert status["auto_registered"]["healthy"] is True


def test_default_interval() -> None:
    """Components use default_interval when no explicit interval given."""
    registry = HeartbeatRegistry(default_interval=7.5, failure_threshold=3)
    registry.register("default_interval_component")

    status = registry.get_status()
    assert status["default_interval_component"]["interval_sec"] == 7.5
