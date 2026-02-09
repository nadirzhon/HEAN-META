"""Tests for execution metrics in BacktestMetrics."""

from datetime import datetime

from hean.backtest.metrics import BacktestMetrics
from hean.core.bus import EventBus
from hean.core.types import Order, OrderStatus
from hean.execution.order_manager import OrderManager
from hean.execution.paper_broker import PaperBroker
from hean.execution.router import ExecutionRouter


def test_backtest_metrics_includes_execution_diagnostics() -> None:
    """Test that BacktestMetrics includes execution diagnostics."""
    bus = EventBus()
    order_manager = OrderManager()
    router = ExecutionRouter(bus, order_manager)
    paper_broker = PaperBroker(bus)

    metrics = BacktestMetrics(
        accounting=None,
        paper_broker=paper_broker,
        execution_router=router,
    )

    # Record some execution data
    diagnostics = router.get_diagnostics()
    order = Order(
        order_id="test-1",
        strategy_id="test",
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

    # Calculate metrics
    result = metrics.calculate()

    # Check execution section exists
    assert "execution" in result
    exec_metrics = result["execution"]

    # Check execution diagnostics fields
    assert "maker_fill_rate" in exec_metrics
    assert "avg_time_to_fill_ms" in exec_metrics
    assert "volatility_rejection_rate" in exec_metrics
    assert "volatility_rejections_soft" in exec_metrics
    assert "volatility_rejections_hard" in exec_metrics
    assert "maker_attempted" in exec_metrics
    assert "maker_filled" in exec_metrics
    assert "maker_expired" in exec_metrics


def test_backtest_metrics_includes_retry_queue_metrics() -> None:
    """Test that BacktestMetrics includes retry queue metrics."""
    bus = EventBus()
    order_manager = OrderManager()
    router = ExecutionRouter(bus, order_manager)

    metrics = BacktestMetrics(
        accounting=None,
        paper_broker=None,
        execution_router=router,
    )

    # Calculate metrics
    result = metrics.calculate()

    # Check execution section exists
    assert "execution" in result
    exec_metrics = result["execution"]

    # Check retry queue fields
    assert "retry_success_rate" in exec_metrics
    assert "retry_queue_size" in exec_metrics


def test_backtest_metrics_json_export() -> None:
    """Test that execution metrics are included in JSON export."""
    import json
    import os
    import tempfile

    bus = EventBus()
    order_manager = OrderManager()
    router = ExecutionRouter(bus, order_manager)
    paper_broker = PaperBroker(bus)

    metrics = BacktestMetrics(
        accounting=None,
        paper_broker=paper_broker,
        execution_router=router,
    )

    # Record some data
    diagnostics = router.get_diagnostics()
    order = Order(
        order_id="test-1",
        strategy_id="test",
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

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        metrics.save_json(temp_path)

        # Read back and verify
        with open(temp_path) as f:
            data = json.load(f)

        assert "execution" in data
        exec_metrics = data["execution"]
        assert "maker_fill_rate" in exec_metrics
        assert "retry_success_rate" in exec_metrics
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_backtest_metrics_print_report_includes_execution() -> None:
    """Test that print_report includes execution metrics."""
    import io
    import sys

    bus = EventBus()
    order_manager = OrderManager()
    router = ExecutionRouter(bus, order_manager)
    paper_broker = PaperBroker(bus)

    metrics = BacktestMetrics(
        accounting=None,
        paper_broker=paper_broker,
        execution_router=router,
    )

    # Record some data
    diagnostics = router.get_diagnostics()
    order = Order(
        order_id="test-1",
        strategy_id="test",
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

    # Capture print output
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        metrics.print_report()
        output = captured_output.getvalue()

        # Check that execution section is printed
        assert "Execution:" in output
        assert "Maker Fill Rate" in output
        assert "Avg Time To Fill" in output
        assert "Soft Volatility Blocks" in output
        assert "Hard Volatility Blocks" in output
        assert "Retry Success Rate" in output
    finally:
        sys.stdout = sys.__stdout__


def test_backtest_metrics_without_router() -> None:
    """Test that BacktestMetrics works without execution router."""
    paper_broker = PaperBroker(EventBus())

    metrics = BacktestMetrics(
        accounting=None,
        paper_broker=paper_broker,
        execution_router=None,
    )

    # Should not raise error
    result = metrics.calculate()

    # Execution section may or may not exist depending on paper_broker
    # But should not crash





