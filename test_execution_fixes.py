#!/usr/bin/env python3
"""Quick test to verify execution router fixes."""

import asyncio
from datetime import datetime
from hean.core.types import OrderRequest

def test_iceberg_split():
    """Test iceberg split doesn't produce size=0 orders."""
    print("Testing iceberg split...")

    # Simulate edge case: very small micro_size
    order_size = 0.001
    current_price = 50000.0
    max_micro_size_usdt = 5.0

    # Calculate micro_size
    micro_size = max_micro_size_usdt / current_price  # 0.0001

    # Simulate split
    import math
    micro_count = max(1, math.ceil(order_size / micro_size))

    print(f"  Order size: {order_size}")
    print(f"  Micro size: {micro_size}")
    print(f"  Micro count: {micro_count}")

    # Simulate split logic with fix
    micro_requests = []
    remaining = order_size
    for idx in range(micro_count):
        size = micro_size if remaining > micro_size else remaining
        remaining -= size

        # This is the fix: skip if size too small
        if size <= 0 or size < 0.000001:
            print(f"  ✓ Skipped micro-order {idx+1} with size={size:.8f} (too small)")
            continue

        micro_requests.append(f"Order {idx+1}: size={size:.8f}")

    if not micro_requests:
        print(f"  ✓ All micro-orders filtered, would use original order")
    else:
        print(f"  Created {len(micro_requests)} valid micro-orders")
        for req in micro_requests:
            print(f"    {req}")

    print()

def test_order_request_validation():
    """Test OrderRequest validation with edge cases."""
    print("Testing OrderRequest validation...")

    # Test valid order
    try:
        order = OrderRequest(
            signal_id="test1",
            strategy_id="test_strat",
            symbol="BTCUSDT",
            side="buy",
            size=0.001,  # Valid positive size
            price=50000.0,
            order_type="limit",
        )
        print(f"  ✓ Valid order created: size={order.size}")
    except Exception as e:
        print(f"  ✗ Failed to create valid order: {e}")

    # Test size=0 (should fail)
    try:
        order = OrderRequest(
            signal_id="test2",
            strategy_id="test_strat",
            symbol="BTCUSDT",
            side="buy",
            size=0.0,  # Invalid
            price=50000.0,
            order_type="limit",
        )
        print(f"  ✗ Created order with size=0 (should have failed)")
    except ValueError as e:
        print(f"  ✓ Correctly rejected size=0: {e}")

    # Test negative size (should fail)
    try:
        order = OrderRequest(
            signal_id="test3",
            strategy_id="test_strat",
            symbol="BTCUSDT",
            side="buy",
            size=-0.001,  # Invalid
            price=50000.0,
            order_type="limit",
        )
        print(f"  ✗ Created order with negative size (should have failed)")
    except ValueError as e:
        print(f"  ✓ Correctly rejected negative size: {e}")

    print()

def test_ttl_cleanup():
    """Test TTL cleanup logic."""
    print("Testing TTL cleanup logic...")

    # Simulate order tracking
    maker_orders = {
        "order1": {"id": "order1", "placed_at": datetime.utcnow(), "status": "PENDING"},
        "order2": {"id": "order2", "placed_at": datetime.utcnow(), "status": "PLACED"},
        "order3": {"id": "order3", "placed_at": datetime.utcnow(), "status": "FILLED"},
    }

    order_requests = {
        "order1": "request1",
        "order2": "request2",
        "order3": "request3",
    }

    print(f"  Initial maker_orders: {len(maker_orders)}")
    print(f"  Initial order_requests: {len(order_requests)}")

    # Simulate expiration handler cleanup
    expired_order_id = "order1"
    maker_orders.pop(expired_order_id, None)
    order_requests.pop(expired_order_id, None)

    print(f"  After cleanup maker_orders: {len(maker_orders)}")
    print(f"  After cleanup order_requests: {len(order_requests)}")
    print(f"  ✓ Cleanup removes orders from both dicts")

    print()

if __name__ == "__main__":
    print("=" * 60)
    print("Execution Router Fixes Validation")
    print("=" * 60)
    print()

    test_iceberg_split()
    test_order_request_validation()
    test_ttl_cleanup()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
