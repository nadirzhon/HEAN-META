"""Tests for execution diagnostics."""

from datetime import datetime, timedelta

import pytest

from hean.core.types import Order, OrderStatus
from hean.execution.execution_diagnostics import ExecutionDiagnostics


def test_execution_diagnostics_initialization() -> None:
    """Test initialization of execution diagnostics."""
    diagnostics = ExecutionDiagnostics()
    assert diagnostics.get_maker_fill_rate() == 0.0
    assert diagnostics.get_avg_time_to_fill_ms() == 0.0
    assert diagnostics.get_volatility_rejection_rate() == 0.0


def test_record_maker_attempt() -> None:
    """Test recording maker attempt."""
    diagnostics = ExecutionDiagnostics()
    order = Order(
        order_id="test-1",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.01,
        order_type="limit",
        status=OrderStatus.PENDING,
        timestamp=datetime.utcnow(),
        is_maker=True,
        placed_at=datetime.utcnow(),
    )
    
    diagnostics.record_maker_attempt(order)
    snapshot = diagnostics.snapshot()
    assert snapshot["maker_attempted"] == 1.0
    assert snapshot["maker_filled"] == 0.0


def test_record_maker_fill() -> None:
    """Test recording maker fill."""
    diagnostics = ExecutionDiagnostics()
    order = Order(
        order_id="test-1",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.01,
        order_type="limit",
        status=OrderStatus.PENDING,
        timestamp=datetime.utcnow(),
        is_maker=True,
        placed_at=datetime.utcnow() - timedelta(milliseconds=100),
    )
    
    diagnostics.record_maker_attempt(order)
    diagnostics.record_maker_fill(order)
    
    snapshot = diagnostics.snapshot()
    assert snapshot["maker_attempted"] == 1.0
    assert snapshot["maker_filled"] == 1.0
    assert snapshot["maker_fill_rate"] == 100.0
    assert snapshot["avg_time_to_fill_ms"] > 0


def test_record_maker_expired() -> None:
    """Test recording maker expiration."""
    diagnostics = ExecutionDiagnostics()
    order = Order(
        order_id="test-1",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.01,
        order_type="limit",
        status=OrderStatus.PLACED,
        timestamp=datetime.utcnow(),
        is_maker=True,
        placed_at=datetime.utcnow(),
    )
    
    diagnostics.record_maker_attempt(order)
    diagnostics.record_maker_expired(order)
    
    snapshot = diagnostics.snapshot()
    assert snapshot["maker_attempted"] == 1.0
    assert snapshot["maker_expired"] == 1.0
    assert snapshot["maker_fill_rate"] == 0.0


def test_record_volatility_rejections() -> None:
    """Test recording volatility rejections."""
    diagnostics = ExecutionDiagnostics()
    order = Order(
        order_id="test-1",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.01,
        order_type="limit",
        status=OrderStatus.REJECTED,
        timestamp=datetime.utcnow(),
    )
    
    diagnostics.record_volatility_rejection_soft(order)
    diagnostics.record_volatility_rejection_hard(order)
    
    snapshot = diagnostics.snapshot()
    assert snapshot["volatility_rejections_soft"] == 1.0
    assert snapshot["volatility_rejections_hard"] == 1.0


def test_maker_fill_rate_calculation() -> None:
    """Test maker fill rate calculation."""
    diagnostics = ExecutionDiagnostics()
    
    # Create multiple orders
    for i in range(10):
        order = Order(
            order_id=f"test-{i}",
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            side="buy",
            size=0.01,
            order_type="limit",
            status=OrderStatus.PENDING,
            timestamp=datetime.utcnow(),
            is_maker=True,
            placed_at=datetime.utcnow(),
        )
        diagnostics.record_maker_attempt(order)
        
        # Fill 7 out of 10
        if i < 7:
            diagnostics.record_maker_fill(order)
    
    assert diagnostics.get_maker_fill_rate() == 70.0


def test_avg_time_to_fill() -> None:
    """Test average time to fill calculation."""
    diagnostics = ExecutionDiagnostics()
    
    # Create orders with different fill times
    for i, delay_ms in enumerate([100, 200, 300]):
        order = Order(
            order_id=f"test-{i}",
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            side="buy",
            size=0.01,
            order_type="limit",
            status=OrderStatus.PENDING,
            timestamp=datetime.utcnow(),
            is_maker=True,
            placed_at=datetime.utcnow() - timedelta(milliseconds=delay_ms),
        )
        diagnostics.record_maker_attempt(order)
        diagnostics.record_maker_fill(order)
    
    avg_time = diagnostics.get_avg_time_to_fill_ms()
    assert avg_time == pytest.approx(200.0, abs=10.0)


def test_volatility_rejection_rate() -> None:
    """Test volatility rejection rate calculation."""
    diagnostics = ExecutionDiagnostics()
    
    # Create orders
    for i in range(10):
        order = Order(
            order_id=f"test-{i}",
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            side="buy",
            size=0.01,
            order_type="limit",
            status=OrderStatus.PENDING,
            timestamp=datetime.utcnow(),
            is_maker=True,
            placed_at=datetime.utcnow(),
        )
        diagnostics.record_maker_attempt(order)
        
        # Reject 3 by volatility
        if i < 3:
            diagnostics.record_volatility_rejection_soft(order)
    
    # Total attempts = 10, rejections = 3
    # Rate = 3 / (10 + 3) * 100 = ~23%
    rate = diagnostics.get_volatility_rejection_rate()
    assert rate == pytest.approx(23.0, abs=1.0)


def test_recent_expired_count() -> None:
    """Test recent expired count."""
    diagnostics = ExecutionDiagnostics()
    
    # Create expired orders
    now = datetime.utcnow()
    for i in range(5):
        order = Order(
            order_id=f"test-{i}",
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            side="buy",
            size=0.01,
            order_type="limit",
            status=OrderStatus.PLACED,
            timestamp=now,
            is_maker=True,
            placed_at=now,
        )
        diagnostics.record_maker_attempt(order)
        diagnostics.record_maker_expired(order)
        diagnostics.finalize_record(order.order_id)
    
    # All should be recent (within 60 seconds)
    count = diagnostics.get_recent_expired_count(lookback_seconds=60)
    assert count == 5


def test_snapshot() -> None:
    """Test snapshot method."""
    diagnostics = ExecutionDiagnostics()
    
    order = Order(
        order_id="test-1",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.01,
        order_type="limit",
        status=OrderStatus.PENDING,
        timestamp=datetime.utcnow(),
        is_maker=True,
        placed_at=datetime.utcnow(),
    )
    
    diagnostics.record_maker_attempt(order)
    diagnostics.record_maker_fill(order)
    
    snapshot = diagnostics.snapshot()
    assert "maker_fill_rate" in snapshot
    assert "avg_time_to_fill_ms" in snapshot
    assert "volatility_rejection_rate" in snapshot
    assert "maker_attempted" in snapshot
    assert "maker_filled" in snapshot
    assert "maker_expired" in snapshot
    assert "volatility_rejections_soft" in snapshot
    assert "volatility_rejections_hard" in snapshot


def test_reset() -> None:
    """Test reset method."""
    diagnostics = ExecutionDiagnostics()
    
    order = Order(
        order_id="test-1",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.01,
        order_type="limit",
        status=OrderStatus.PENDING,
        timestamp=datetime.utcnow(),
        is_maker=True,
        placed_at=datetime.utcnow(),
    )
    
    diagnostics.record_maker_attempt(order)
    diagnostics.record_maker_fill(order)
    
    assert diagnostics.get_maker_fill_rate() > 0
    
    diagnostics.reset()
    
    assert diagnostics.get_maker_fill_rate() == 0.0
    assert diagnostics.get_avg_time_to_fill_ms() == 0.0





