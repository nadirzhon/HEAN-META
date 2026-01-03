"""Tests for configuration."""

import os
import pytest

from hean.config import HEANSettings


def test_default_settings() -> None:
    """Test default settings."""
    settings = HEANSettings()
    assert settings.trading_mode == "paper"
    assert settings.initial_capital == 10000.0
    assert settings.max_daily_drawdown_pct == 5.0


def test_live_trading_requires_confirm() -> None:
    """Test that live trading requires LIVE_CONFIRM=YES."""
    # Clear environment
    if "LIVE_CONFIRM" in os.environ:
        del os.environ["LIVE_CONFIRM"]

    settings = HEANSettings()
    assert settings.trading_mode == "paper"
    assert not settings.is_live

    # Set LIVE_CONFIRM=YES
    os.environ["LIVE_CONFIRM"] = "YES"
    settings = HEANSettings()
    assert settings.trading_mode == "live"
    assert settings.is_live

    # Cleanup
    del os.environ["LIVE_CONFIRM"]


def test_config_validation() -> None:
    """Test configuration validation."""
    with pytest.raises(ValueError):
        HEANSettings(initial_capital=-1000.0)

    with pytest.raises(ValueError):
        HEANSettings(max_daily_drawdown_pct=150.0)  # > 100%

    with pytest.raises(ValueError):
        HEANSettings(reinvest_rate=1.5)  # > 1.0


