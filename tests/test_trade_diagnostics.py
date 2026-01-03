"""Tests for trade diagnostics."""

from hean.process_factory.integrations.trade_diagnostics import (
    check_dry_run,
    check_live_enabled,
    check_process_factory_actions,
    log_trade_blocked,
)


def test_log_trade_blocked(caplog):
    """Test that trade_blocked events are logged."""
    log_trade_blocked(
        symbol="BTCUSDT",
        strategy_id="test_strategy",
        reasons=["dry_run", "min_notional"],
        suggested_fix=["Set DRY_RUN=false", "Increase order size"],
    )

    # Check that log contains the event
    assert "event=trade_blocked" in caplog.text
    assert "BTCUSDT" in caplog.text
    assert "test_strategy" in caplog.text
    assert "dry_run" in caplog.text or "min_notional" in caplog.text


def test_check_dry_run():
    """Test dry run check."""
    from hean.config import settings
    import pytest

    # Save original
    original_dry_run = settings.dry_run

    try:
        # Test with dry_run=True
        settings._dry_run = True  # type: ignore
        allowed, reasons, fixes = check_dry_run()
        assert not allowed
        assert "dry_run" in reasons
        assert len(fixes) > 0

        # Test with dry_run=False
        settings._dry_run = False  # type: ignore
        allowed, reasons, fixes = check_dry_run()
        assert allowed
        assert len(reasons) == 0
    finally:
        settings._dry_run = original_dry_run  # type: ignore


def test_check_process_factory_actions():
    """Test process factory actions check."""
    from hean.config import settings
    import pytest

    # Save original
    original_allow_actions = settings.process_factory_allow_actions

    try:
        # Test with allow_actions=False
        settings._process_factory_allow_actions = False  # type: ignore
        allowed, reasons, fixes = check_process_factory_actions()
        assert not allowed
        assert "process_factory_allow_actions_false" in reasons
        assert len(fixes) > 0

        # Test with allow_actions=True
        settings._process_factory_allow_actions = True  # type: ignore
        allowed, reasons, fixes = check_process_factory_actions()
        assert allowed
        assert len(reasons) == 0
    finally:
        settings._process_factory_allow_actions = original_allow_actions  # type: ignore

