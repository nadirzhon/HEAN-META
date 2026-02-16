"""Tests for Archon integration."""

from hean.config import settings
from hean.core.bus import EventBus


async def test_archon_start_stop():
    """Test Archon starts and stops cleanly."""
    from hean.archon.archon import Archon

    bus = EventBus()
    await bus.start()

    archon = Archon(bus=bus, settings=settings)
    await archon.start()

    assert archon._running is True

    await archon.stop()
    assert archon._running is False

    await bus.stop()


async def test_archon_with_disabled_subsystems():
    """Test Archon with some subsystems disabled."""
    from hean.archon.archon import Archon
    from hean.config import HEANSettings

    bus = EventBus()
    await bus.start()

    # Create settings with some subsystems disabled
    custom_settings = HEANSettings(
        archon_enabled=True,
        archon_signal_pipeline_enabled=False,
        archon_reconciliation_enabled=False,
        archon_cortex_enabled=False,
        archon_chronicle_enabled=False,
    )

    archon = Archon(bus=bus, settings=custom_settings)
    await archon.start()

    # Should still start successfully
    assert archon._running is True
    assert archon.signal_pipeline is None
    assert archon.reconciler is None
    assert archon.cortex is None
    assert archon.chronicle is None

    await archon.stop()

    await bus.stop()


async def test_archon_get_status():
    """Test Archon status dict structure."""
    from hean.archon.archon import Archon

    bus = EventBus()
    await bus.start()

    archon = Archon(bus=bus, settings=settings)
    await archon.start()

    status = archon.get_status()
    assert "running" in status
    assert "signal_pipeline" in status
    assert "health" in status
    assert "heartbeats" in status
    assert "cortex" in status
    assert "reconciler_active" in status
    assert "chronicle_active" in status

    await archon.stop()

    await bus.stop()


async def test_archon_graceful_failure():
    """Test Archon handles subsystem failures gracefully."""
    from hean.archon.archon import Archon

    bus = EventBus()
    await bus.start()

    archon = Archon(bus=bus, settings=settings)

    # Start without required components for reconciler
    await archon.start()  # Should not crash

    # Check status still works
    status = archon.get_status()
    assert status["running"] is True

    await archon.stop()

    await bus.stop()
