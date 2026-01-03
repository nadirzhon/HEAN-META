"""Tests for execution smoke test."""

import pytest

from hean.config import settings
from hean.process_factory.integrations.bybit_actions import NotEnabledError
from hean.process_factory.integrations.smoke_test import run_smoke_test


@pytest.mark.asyncio
async def test_smoke_test_requires_flags(monkeypatch):
    """Test that smoke test requires proper flags."""
    # Save original values
    original_enabled = settings.process_factory_enabled
    original_allow_actions = settings.process_factory_allow_actions
    original_dry_run = settings.dry_run

    try:
        # Test: process_factory_enabled must be true
        monkeypatch.setattr(settings, "process_factory_enabled", False)
        with pytest.raises(ValueError, match="PROCESS_FACTORY_ENABLED"):
            await run_smoke_test()

        # Test: process_factory_allow_actions must be true
        monkeypatch.setattr(settings, "process_factory_enabled", True)
        monkeypatch.setattr(settings, "process_factory_allow_actions", False)
        with pytest.raises(ValueError, match="PROCESS_FACTORY_ALLOW_ACTIONS"):
            await run_smoke_test()

        # Test: dry_run must be false
        monkeypatch.setattr(settings, "process_factory_allow_actions", True)
        monkeypatch.setattr(settings, "dry_run", True)
        with pytest.raises(ValueError, match="DRY_RUN"):
            await run_smoke_test()

    finally:
        # Restore original values
        monkeypatch.setattr(settings, "process_factory_enabled", original_enabled)
        monkeypatch.setattr(settings, "process_factory_allow_actions", original_allow_actions)
        monkeypatch.setattr(settings, "dry_run", original_dry_run)


def test_bybit_actions_not_enabled_error():
    """Test that DefaultBybitActions raises NotEnabledError with clear message."""
    from hean.process_factory.integrations.bybit_actions import DefaultBybitActions

    adapter = DefaultBybitActions()

    # Test place_limit_postonly
    import asyncio

    async def test_place():
        with pytest.raises(NotEnabledError) as exc_info:
            await adapter.place_limit_postonly("BTCUSDT", "BUY", 0.001, 50000.0)
        assert "PROCESS_FACTORY_ALLOW_ACTIONS" in str(exc_info.value)
        assert "DRY_RUN" in str(exc_info.value)

    asyncio.run(test_place())

