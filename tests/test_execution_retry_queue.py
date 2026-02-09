"""Tests for execution retry queue."""

from datetime import datetime, timedelta

from hean.core.types import Order, OrderRequest, OrderStatus
from hean.execution.maker_retry_queue import MakerRetryQueue


def test_hard_block_leads_to_enqueue() -> None:
    """Test that hard volatility block leads to enqueue."""
    queue = MakerRetryQueue(max_retries=2)

    order = Order(
        order_id="test_order_1",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        price=50000.0,
        order_type="limit",
        status=OrderStatus.REJECTED,
        timestamp=datetime.utcnow(),
    )

    request = OrderRequest(
        signal_id="test_signal",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        price=50000.0,
    )

    # Enqueue for retry
    result = queue.enqueue_for_retry(order, request, reason="volatility_hard_block")
    assert result is True
    assert queue.get_queue_size() == 1


def test_improved_volatility_allows_retry() -> None:
    """Test that improved volatility allows retry and order placement."""
    queue = MakerRetryQueue(max_retries=2, min_retry_delay_seconds=0)  # No delay for test

    order = Order(
        order_id="test_order_2",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        price=50000.0,
        order_type="limit",
        status=OrderStatus.REJECTED,
        timestamp=datetime.utcnow(),
    )

    request = OrderRequest(
        signal_id="test_signal",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        price=50000.0,
    )

    # Enqueue
    queue.enqueue_for_retry(order, request, reason="volatility_hard_block")

    # Simulate volatility improvement: current < previous * 0.9
    previous_vol = 0.01  # 1% volatility when blocked
    current_vol = 0.008  # 0.8% volatility now (20% improvement)

    # Get ready retries
    ready = queue.get_ready_retries(
        current_volatility=current_vol,
        previous_volatility=previous_vol,
        regime_changed=False,
        drawdown_worsened=False,
        capital_preservation_active=False,
    )

    assert len(ready) == 1
    assert ready[0].symbol == "BTCUSDT"
    assert queue.get_queue_size() == 0  # Removed after being ready


def test_retries_stop_at_max() -> None:
    """Test that retries stop at max_retries."""
    queue = MakerRetryQueue(max_retries=2, min_retry_delay_seconds=0)

    order = Order(
        order_id="test_order_3",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        price=50000.0,
        order_type="limit",
        status=OrderStatus.REJECTED,
        timestamp=datetime.utcnow(),
    )

    request = OrderRequest(
        signal_id="test_signal",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        price=50000.0,
    )

    # First enqueue
    queue.enqueue_for_retry(order, request, reason="volatility_hard_block")

    # Simulate retry attempt (volatility didn't improve)
    previous_vol = 0.01
    current_vol = 0.0095  # Only 5% improvement, not enough

    ready = queue.get_ready_retries(
        current_volatility=current_vol,
        previous_volatility=previous_vol,
        regime_changed=False,
        drawdown_worsened=False,
        capital_preservation_active=False,
    )

    # Should not be ready (volatility not improved enough)
    assert len(ready) == 0

    # Manually increment retry count to simulate multiple attempts
    entry = queue._queue[0]
    entry.retry_count = 3  # Exceed max (max is 2, so 3 > 2)

    # Try again - should exceed max and be removed
    ready = queue.get_ready_retries(
        current_volatility=current_vol,
        previous_volatility=previous_vol,
        regime_changed=False,
        drawdown_worsened=False,
        capital_preservation_active=False,
    )

    # Should be removed (exceeded max retries)
    assert queue.get_queue_size() == 0


def test_retry_respects_delay() -> None:
    """Test that retry respects minimum delay."""
    queue = MakerRetryQueue(max_retries=2, min_retry_delay_seconds=5)

    order = Order(
        order_id="test_order_4",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        price=50000.0,
        order_type="limit",
        status=OrderStatus.REJECTED,
        timestamp=datetime.utcnow() - timedelta(seconds=2),  # 2 seconds ago
    )

    request = OrderRequest(
        signal_id="test_signal",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        price=50000.0,
    )

    queue.enqueue_for_retry(order, request, reason="volatility_hard_block")

    # Update last_attempt_at to recent time
    entry = queue._queue[0]
    entry.last_attempt_at = datetime.utcnow() - timedelta(seconds=2)

    # Volatility improved, but not enough time passed
    ready = queue.get_ready_retries(
        current_volatility=0.005,
        previous_volatility=0.01,
        regime_changed=False,
        drawdown_worsened=False,
        capital_preservation_active=False,
    )

    # Should not be ready (delay not met)
    assert len(ready) == 0


def test_retry_respects_capital_preservation() -> None:
    """Test that retry is blocked when capital preservation is active."""
    queue = MakerRetryQueue(max_retries=2, min_retry_delay_seconds=0)

    order = Order(
        order_id="test_order_5",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        price=50000.0,
        order_type="limit",
        status=OrderStatus.REJECTED,
        timestamp=datetime.utcnow(),
    )

    request = OrderRequest(
        signal_id="test_signal",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        price=50000.0,
    )

    queue.enqueue_for_retry(order, request, reason="volatility_hard_block")

    # Capital preservation active
    ready = queue.get_ready_retries(
        current_volatility=0.005,
        previous_volatility=0.01,
        regime_changed=False,
        drawdown_worsened=False,
        capital_preservation_active=True,  # Active
    )

    # Should be empty (capital preservation blocks retries)
    assert len(ready) == 0

