"""Tests for no-trade report counters."""

import pytest

from hean.observability.no_trade_report import no_trade_report


def test_signals_emitted_counter() -> None:
    """Test signals_emitted counter increments."""
    no_trade_report.reset()
    
    no_trade_report.increment_pipeline("signals_emitted", "impulse_engine")
    no_trade_report.increment_pipeline("signals_emitted", "impulse_engine")
    no_trade_report.increment_pipeline("signals_emitted", "funding_harvester")
    
    summary = no_trade_report.get_summary()
    assert summary.pipeline_counters["signals_emitted"] == 3
    assert summary.pipeline_per_strategy["impulse_engine"]["signals_emitted"] == 2
    assert summary.pipeline_per_strategy["funding_harvester"]["signals_emitted"] == 1


def test_signals_rejected_counters() -> None:
    """Test signal rejection counters."""
    no_trade_report.reset()
    
    no_trade_report.increment_pipeline("signals_rejected_risk", "impulse_engine")
    no_trade_report.increment_pipeline("signals_rejected_daily_attempts", "impulse_engine")
    no_trade_report.increment_pipeline("signals_rejected_cooldown", "funding_harvester")
    no_trade_report.increment_pipeline("signals_blocked_decision_memory", "impulse_engine")
    
    summary = no_trade_report.get_summary()
    assert summary.pipeline_counters["signals_rejected_risk"] == 1
    assert summary.pipeline_counters["signals_rejected_daily_attempts"] == 1
    assert summary.pipeline_counters["signals_rejected_cooldown"] == 1
    assert summary.pipeline_counters["signals_blocked_decision_memory"] == 1
    assert summary.pipeline_per_strategy["impulse_engine"]["signals_rejected_risk"] == 1


def test_execution_counters() -> None:
    """Test execution-level counters."""
    no_trade_report.reset()
    
    no_trade_report.increment_pipeline("execution_soft_vol_blocks", "impulse_engine")
    no_trade_report.increment_pipeline("execution_hard_vol_blocks", "impulse_engine")
    no_trade_report.increment_pipeline("execution_soft_vol_blocks", "funding_harvester")
    
    summary = no_trade_report.get_summary()
    assert summary.pipeline_counters["execution_soft_vol_blocks"] == 2
    assert summary.pipeline_counters["execution_hard_vol_blocks"] == 1


def test_order_counters() -> None:
    """Test order creation and placement counters."""
    no_trade_report.reset()
    
    no_trade_report.increment_pipeline("orders_created", "impulse_engine")
    no_trade_report.increment_pipeline("orders_created", "impulse_engine")
    no_trade_report.increment_pipeline("maker_orders_placed", "impulse_engine")
    no_trade_report.increment_pipeline("maker_orders_cancelled_ttl", "impulse_engine")
    no_trade_report.increment_pipeline("maker_orders_filled", "impulse_engine")
    
    summary = no_trade_report.get_summary()
    assert summary.pipeline_counters["orders_created"] == 2
    assert summary.pipeline_counters["maker_orders_placed"] == 1
    assert summary.pipeline_counters["maker_orders_cancelled_ttl"] == 1
    assert summary.pipeline_counters["maker_orders_filled"] == 1
    assert summary.pipeline_per_strategy["impulse_engine"]["orders_created"] == 2


def test_position_counters() -> None:
    """Test position open/close counters."""
    no_trade_report.reset()
    
    no_trade_report.increment_pipeline("positions_opened", "impulse_engine")
    no_trade_report.increment_pipeline("positions_opened", "funding_harvester")
    no_trade_report.increment_pipeline("positions_closed", "impulse_engine")
    
    summary = no_trade_report.get_summary()
    assert summary.pipeline_counters["positions_opened"] == 2
    assert summary.pipeline_counters["positions_closed"] == 1
    assert summary.pipeline_per_strategy["impulse_engine"]["positions_opened"] == 1
    assert summary.pipeline_per_strategy["impulse_engine"]["positions_closed"] == 1


def test_full_pipeline_trace() -> None:
    """Test a complete pipeline trace from signal to fill."""
    no_trade_report.reset()
    
    # Simulate a successful trade pipeline
    no_trade_report.increment_pipeline("signals_emitted", "impulse_engine")
    no_trade_report.increment_pipeline("orders_created", "impulse_engine")
    no_trade_report.increment_pipeline("maker_orders_placed", "impulse_engine")
    no_trade_report.increment_pipeline("maker_orders_filled", "impulse_engine")
    no_trade_report.increment_pipeline("positions_opened", "impulse_engine")
    no_trade_report.increment_pipeline("positions_closed", "impulse_engine")
    
    summary = no_trade_report.get_summary()
    assert summary.pipeline_counters["signals_emitted"] == 1
    assert summary.pipeline_counters["orders_created"] == 1
    assert summary.pipeline_counters["maker_orders_placed"] == 1
    assert summary.pipeline_counters["maker_orders_filled"] == 1
    assert summary.pipeline_counters["positions_opened"] == 1
    assert summary.pipeline_counters["positions_closed"] == 1


def test_starvation_diagnosis() -> None:
    """Test counters help diagnose starvation (signals but no fills)."""
    no_trade_report.reset()
    
    # Simulate starvation: many signals, but all blocked
    for _ in range(10):
        no_trade_report.increment_pipeline("signals_emitted", "impulse_engine")
        no_trade_report.increment_pipeline("signals_rejected_risk", "impulse_engine")
    
    for _ in range(5):
        no_trade_report.increment_pipeline("orders_created", "impulse_engine")
        no_trade_report.increment_pipeline("execution_hard_vol_blocks", "impulse_engine")
    
    summary = no_trade_report.get_summary()
    assert summary.pipeline_counters["signals_emitted"] == 10
    assert summary.pipeline_counters["signals_rejected_risk"] == 10
    assert summary.pipeline_counters["orders_created"] == 5
    assert summary.pipeline_counters["execution_hard_vol_blocks"] == 5
    assert summary.pipeline_counters.get("maker_orders_filled", 0) == 0
    assert summary.pipeline_counters.get("positions_opened", 0) == 0

