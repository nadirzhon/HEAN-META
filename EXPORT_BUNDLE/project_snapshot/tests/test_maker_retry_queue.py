"""Tests for maker retry queue."""

from datetime import datetime, timedelta

import pytest

from hean.core.types import Order, OrderRequest, OrderStatus
from hean.execution.maker_retry_queue import MakerRetryQueue


def test_retry_queue_initialization() -> None:
    """Test initialization of retry queue."""
    queue = MakerRetryQueue()
    assert queue.get_queue_size() == 0
    assert queue.get_retry_success_rate() == 0.0


def test_enqueue_for_retry() -> None:
    """Test enqueueing an order for retry."""
    queue = MakerRetryQueue(max_retries=2)
    
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
    )
    
    request = OrderRequest(
        signal_id="signal-1",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.01,
    )
    
    result = queue.enqueue_for_retry(order, request, reason="volatility_expired")
    assert result is True
    assert queue.get_queue_size() == 1


def test_enqueue_max_retries() -> None:
    """Test that max retries are enforced."""
    queue = MakerRetryQueue(max_retries=2)
    
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
    )
    
    request = OrderRequest(
        signal_id="signal-1",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.01,
    )
    
    # Enqueue 3 times (should fail on 3rd)
    assert queue.enqueue_for_retry(order, request) is True
    assert queue.enqueue_for_retry(order, request) is True
    assert queue.enqueue_for_retry(order, request) is False  # Max retries exceeded


def test_get_ready_retries_volatility_improved() -> None:
    """Test getting ready retries when volatility improved."""
    queue = MakerRetryQueue(max_retries=2, min_retry_delay_seconds=1)
    
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
    )
    
    request = OrderRequest(
        signal_id="signal-1",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.01,
    )
    
    queue.enqueue_for_retry(order, request)
    
    # Wait for min delay
    import time
    time.sleep(1.1)
    
    # Volatility improved (current < previous * 0.9)
    ready = queue.get_ready_retries(
        current_volatility=0.005,  # 0.5%
        previous_volatility=0.01,  # 1.0% (higher)
        regime_changed=False,
        drawdown_worsened=False,
        capital_preservation_active=False,
    )
    
    assert len(ready) == 1
    assert ready[0].symbol == "BTCUSDT"


def test_get_ready_retries_capital_preservation() -> None:
    """Test that retries are blocked when capital preservation is active."""
    queue = MakerRetryQueue()
    
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
    )
    
    request = OrderRequest(
        signal_id="signal-1",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.01,
    )
    
    queue.enqueue_for_retry(order, request)
    
    ready = queue.get_ready_retries(
        current_volatility=0.005,
        previous_volatility=0.01,
        regime_changed=False,
        drawdown_worsened=False,
        capital_preservation_active=True,  # Active
    )
    
    assert len(ready) == 0


def test_get_ready_retries_regime_changed() -> None:
    """Test that retry queue is cleared when regime changes."""
    queue = MakerRetryQueue()
    
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
    )
    
    request = OrderRequest(
        signal_id="signal-1",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.01,
    )
    
    queue.enqueue_for_retry(order, request)
    assert queue.get_queue_size() == 1
    
    ready = queue.get_ready_retries(
        current_volatility=0.005,
        previous_volatility=0.01,
        regime_changed=True,  # Regime changed
        drawdown_worsened=False,
        capital_preservation_active=False,
    )
    
    assert len(ready) == 0
    assert queue.get_queue_size() == 0  # Queue cleared


def test_get_ready_retries_drawdown_worsened() -> None:
    """Test that retries are blocked when drawdown worsened."""
    queue = MakerRetryQueue()
    
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
    )
    
    request = OrderRequest(
        signal_id="signal-1",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.01,
    )
    
    queue.enqueue_for_retry(order, request)
    
    ready = queue.get_ready_retries(
        current_volatility=0.005,
        previous_volatility=0.01,
        regime_changed=False,
        drawdown_worsened=True,  # Drawdown worsened
        capital_preservation_active=False,
    )
    
    assert len(ready) == 0


def test_remove_order() -> None:
    """Test removing an order from retry queue."""
    queue = MakerRetryQueue()
    
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
    )
    
    request = OrderRequest(
        signal_id="signal-1",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.01,
    )
    
    queue.enqueue_for_retry(order, request)
    assert queue.get_queue_size() == 1
    
    queue.remove_order("test-1")
    assert queue.get_queue_size() == 0


def test_retry_success_rate() -> None:
    """Test retry success rate calculation."""
    queue = MakerRetryQueue(max_retries=1)
    
    # Create orders that will succeed and fail
    for i in range(5):
        order = Order(
            order_id=f"test-{i}",
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            side="buy",
            size=0.01,
            order_type="limit",
            status=OrderStatus.PLACED,
            timestamp=datetime.utcnow(),
            is_maker=True,
        )
        
        request = OrderRequest(
            signal_id=f"signal-{i}",
            strategy_id="test_strategy",
            symbol="BTCUSDT",
            side="buy",
            size=0.01,
        )
        
        queue.enqueue_for_retry(order, request)
    
    # Simulate some successes and failures
    # This is simplified - in real usage, success/failure is determined by fill
    # For testing, we'll manually track via get_ready_retries
    
    # The success rate starts at 0 (no retries processed yet)
    assert queue.get_retry_success_rate() == 0.0


def test_clear() -> None:
    """Test clearing the retry queue."""
    queue = MakerRetryQueue()
    
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
    )
    
    request = OrderRequest(
        signal_id="signal-1",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.01,
    )
    
    queue.enqueue_for_retry(order, request)
    assert queue.get_queue_size() == 1
    
    queue.clear()
    assert queue.get_queue_size() == 0


def test_reset_metrics() -> None:
    """Test resetting metrics."""
    queue = MakerRetryQueue()
    
    # Metrics are internal, but we can verify reset doesn't break
    queue.reset_metrics()
    assert queue.get_retry_success_rate() == 0.0





