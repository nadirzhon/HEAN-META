"""Tests for configuration."""

import os

import pytest

from hean.config import HEANSettings


def test_default_settings() -> None:
    """Test default settings.

    Note: HEAN uses live trading mode only (on Bybit testnet).
    There is no paper trading mode.
    """
    settings = HEANSettings()
    # Always live mode (on testnet)
    assert settings.trading_mode == "live"
    # Capital should be positive (may be overridden by .env)
    assert settings.initial_capital > 0
    # Max drawdown should be between 0 and 100
    assert 0 < settings.max_daily_drawdown_pct <= 100


def test_live_trading_mode() -> None:
    """Test that trading mode is always live (testnet only).

    HEAN removed paper trading mode. All trading happens on Bybit testnet.
    """
    settings = HEANSettings()
    # Always live (but on testnet)
    assert settings.trading_mode == "live"
    assert settings.bybit_testnet is True  # Always testnet


def test_config_validation() -> None:
    """Test configuration validation."""
    with pytest.raises(ValueError):
        HEANSettings(initial_capital=-1000.0)

    with pytest.raises(ValueError):
        HEANSettings(max_daily_drawdown_pct=150.0)  # > 100%

    with pytest.raises(ValueError):
        HEANSettings(reinvest_rate=1.5)  # > 1.0
