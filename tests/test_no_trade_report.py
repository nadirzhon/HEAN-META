"""Tests for NoTradeReport helper."""

from hean.observability.no_trade_report import NoTradeReport, no_trade_report


def test_no_trade_report_increment_and_summary() -> None:
    """Counters increment correctly and summary shape is as expected."""
    report = NoTradeReport()
    report.reset()

    report.increment("risk_limits_reject", "BTCUSDT", "impulse_engine")
    report.increment("risk_limits_reject", "BTCUSDT", "impulse_engine")
    report.increment("cooldown_reject", "ETHUSDT", "impulse_engine")
    report.increment("filter_reject", "ETHUSDT", "other_strategy")

    summary = report.get_summary()

    # Totals per reason
    assert summary.totals["risk_limits_reject"] == 2
    assert summary.totals["cooldown_reject"] == 1
    assert summary.totals["filter_reject"] == 1

    # Per-strategy breakdown
    assert summary.per_strategy["impulse_engine"]["risk_limits_reject"] == 2
    assert summary.per_strategy["impulse_engine"]["cooldown_reject"] == 1
    assert summary.per_strategy["other_strategy"]["filter_reject"] == 1

    # Per-symbol breakdown
    assert summary.per_symbol["BTCUSDT"]["risk_limits_reject"] == 2
    assert summary.per_symbol["ETHUSDT"]["cooldown_reject"] == 1
    assert summary.per_symbol["ETHUSDT"]["filter_reject"] == 1


def test_no_trade_report_global_singleton_reset() -> None:
    """Global singleton can be reset between runs."""
    no_trade_report.reset()
    no_trade_report.increment("risk_limits_reject", "BTCUSDT", "impulse_engine")

    summary1 = no_trade_report.get_summary()
    assert summary1.totals["risk_limits_reject"] == 1

    # After reset, counters should be cleared
    no_trade_report.reset()
    summary2 = no_trade_report.get_summary()
    assert summary2.totals == {}
    assert summary2.per_strategy == {}
    assert summary2.per_symbol == {}






