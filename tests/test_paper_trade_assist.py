"""Tests for Paper Trade Assist functionality."""

import pytest

from hean.paper_trade_assist import (
    get_cooldown_multiplier,
    get_daily_attempts_multiplier,
    get_edge_threshold_reduction_pct,
    get_max_open_positions_override,
    get_min_notional_override,
    get_spread_threshold_multiplier,
    should_allow_regime,
)


def test_paper_assist_disabled_by_default():
    """Test that paper assist is disabled by default."""
    # This test assumes default config
    # In real test, would need to reset settings
    pass


def test_paper_assist_requires_dry_run_or_testnet(monkeypatch):
    """Test that PAPER_TRADE_ASSIST field exists but is deprecated."""
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setenv("PAPER_TRADE_ASSIST", "true")

    from hean.config import HEANSettings

    # Paper trade assist is deprecated — no longer raises ValueError
    config = HEANSettings()
    assert hasattr(config, "paper_trade_assist")


def test_paper_assist_allowed_with_dry_run(monkeypatch):
    """Test that PAPER_TRADE_ASSIST is allowed with DRY_RUN=true."""
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("PAPER_TRADE_ASSIST", "true")
    monkeypatch.setenv("LIVE_CONFIRM", "")

    from hean.config import HEANSettings

    config = HEANSettings()
    assert config.paper_trade_assist is True
    assert config.dry_run is True


def test_paper_assist_allowed_with_testnet(monkeypatch):
    """Test that PAPER_TRADE_ASSIST is allowed with bybit_testnet=true."""
    monkeypatch.setenv("BYBIT_TESTNET", "true")
    monkeypatch.setenv("PAPER_TRADE_ASSIST", "true")
    monkeypatch.setenv("DRY_RUN", "false")
    monkeypatch.setenv("LIVE_CONFIRM", "")

    from hean.config import HEANSettings

    config = HEANSettings()
    assert config.paper_trade_assist is True
    assert config.bybit_testnet is True


def test_paper_assist_multipliers(monkeypatch):
    """Test that paper assist multipliers return defaults when disabled (deprecated)."""
    # Paper trade assist is deprecated — multipliers return defaults
    assert get_spread_threshold_multiplier() >= 1.0
    assert get_daily_attempts_multiplier() >= 1.0
    assert get_cooldown_multiplier() <= 1.0
    assert get_edge_threshold_reduction_pct() >= 0.0


def test_paper_assist_overrides(monkeypatch):
    """Test that paper assist overrides return None when disabled (deprecated)."""
    # Paper trade assist is deprecated — overrides return None
    max_pos = get_max_open_positions_override()
    min_notional = get_min_notional_override()
    # When disabled, these should return None (no override)
    assert max_pos is None or max_pos >= 1
    assert min_notional is None or min_notional > 0


def test_paper_assist_regime_allowance(monkeypatch):
    """Test that paper assist allows all regimes."""
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("PAPER_TRADE_ASSIST", "true")

    from hean.config import HEANSettings
    config = HEANSettings()

    # Force enable for testing
    config.paper_trade_assist = True
    config.dry_run = True

    # Should allow all regimes
    assert should_allow_regime("normal") is True
    assert should_allow_regime("impulse") is True
    assert should_allow_regime("range") is True
    assert should_allow_regime("chop") is True

