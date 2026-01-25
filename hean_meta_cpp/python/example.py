#!/usr/bin/env python3
"""
Example usage of HEAN-META C++ Order Engine from Python.

This demonstrates the Python bindings for the high-performance C++ order engine.
"""

import time
import hean_meta_cpp as hmc


def example_order_placement():
    """Example: Placing orders"""
    print("=" * 60)
    print("Example 1: Order Placement")
    print("=" * 60)

    engine = hmc.OrderEngine()

    # Place market order
    print("\n📝 Placing market order...")
    result = engine.place_market_order("BTCUSDT", hmc.Side.BUY, 0.1)

    print(f"  Success: {result.success}")
    print(f"  Order ID: {result.order_id}")
    print(f"  Latency: {result.latency_us}μs")

    # Place limit order
    print("\n📝 Placing limit order...")
    result = engine.place_limit_order("ETHUSDT", hmc.Side.SELL, 1.0, 3000.0)

    print(f"  Success: {result.success}")
    print(f"  Order ID: {result.order_id}")
    print(f"  Latency: {result.latency_us}μs")

    # Get order details
    print("\n🔍 Getting order details...")
    order = engine.get_order(result.order_id)
    if order:
        print(f"  Symbol: {order.symbol}")
        print(f"  Side: {order.side}")
        print(f"  Type: {order.type}")
        print(f"  Quantity: {order.quantity}")
        print(f"  Price: {order.price}")
        print(f"  Status: {order.status}")


def example_position_management():
    """Example: Position management"""
    print("\n" + "=" * 60)
    print("Example 2: Position Management")
    print("=" * 60)

    pm = hmc.PositionManager()

    # Open long position
    print("\n📈 Opening long position...")
    pm.update_position("BTCUSDT", 0.5, 45000.0)

    pos = pm.get_position("BTCUSDT")
    if pos:
        print(f"  Symbol: {pos.symbol}")
        print(f"  Quantity: {pos.quantity}")
        print(f"  Entry Price: ${pos.entry_price:,.2f}")
        print(f"  Side: {'LONG' if pos.is_long() else 'SHORT'}")

    # Add to position
    print("\n📈 Adding to position...")
    pm.update_position("BTCUSDT", 0.3, 46000.0)

    pos = pm.get_position("BTCUSDT")
    if pos:
        print(f"  New Quantity: {pos.quantity}")
        print(f"  New Entry Price: ${pos.entry_price:,.2f}")

    # Update unrealized PnL
    current_price = 48000.0
    print(f"\n💰 Updating PnL (current price: ${current_price:,.2f})...")
    pm.update_unrealized_pnl("BTCUSDT", current_price)

    pos = pm.get_position("BTCUSDT")
    if pos:
        print(f"  Unrealized PnL: ${pos.unrealized_pnl:,.2f}")
        print(f"  Realized PnL: ${pos.realized_pnl:,.2f}")

    # Close position
    print("\n🔒 Closing position...")
    pm.close_position("BTCUSDT", current_price)

    pos = pm.get_position("BTCUSDT")
    if pos:
        print(f"  Quantity: {pos.quantity} (flat: {pos.is_flat()})")
        print(f"  Total Realized PnL: ${pos.realized_pnl:,.2f}")


def example_performance_test():
    """Example: Performance testing"""
    print("\n" + "=" * 60)
    print("Example 3: Performance Test")
    print("=" * 60)

    engine = hmc.OrderEngine()
    num_orders = 10000

    print(f"\n⏱️  Placing {num_orders:,} orders...")

    start = time.perf_counter()

    for i in range(num_orders):
        engine.place_market_order("BTCUSDT", hmc.Side.BUY, 0.001)

    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000
    throughput = num_orders / (end - start)
    avg_latency_us = engine.get_avg_latency_us()

    print(f"\n📊 Performance Results:")
    print(f"  Total time: {elapsed_ms:.2f}ms")
    print(f"  Throughput: {throughput:,.0f} orders/sec")
    print(f"  Average latency: {avg_latency_us}μs")
    print(f"  Total orders: {engine.get_total_orders():,}")
    print(f"  Active orders: {engine.get_active_order_count():,}")

    # Check performance target
    if avg_latency_us < 100:
        print(f"\n  ✅ Performance target met (<100μs)")
    else:
        print(f"\n  ⚠️  Performance target not met (>100μs)")


def example_order_lifecycle():
    """Example: Complete order lifecycle"""
    print("\n" + "=" * 60)
    print("Example 4: Order Lifecycle")
    print("=" * 60)

    engine = hmc.OrderEngine()

    # 1. Place order
    print("\n1️⃣  Placing limit order...")
    result = engine.place_limit_order("BTCUSDT", hmc.Side.BUY, 0.5, 45000.0)
    order_id = result.order_id
    print(f"    Order placed: {order_id}")

    # 2. Check order status
    print("\n2️⃣  Checking order status...")
    order = engine.get_order(order_id)
    if order:
        print(f"    Status: {order.status}")
        print(f"    Is active: {order.is_active()}")

    # 3. Update order price
    print("\n3️⃣  Updating order price...")
    updated = engine.update_order(order_id, 46000.0)
    print(f"    Update {'successful' if updated else 'failed'}")

    order = engine.get_order(order_id)
    if order:
        print(f"    New price: ${order.price:,.2f}")

    # 4. Simulate fill (for testing)
    print("\n4️⃣  Simulating order fill...")
    engine.simulate_fill(order_id, 46000.0)

    order = engine.get_order(order_id)
    if order:
        print(f"    Status: {order.status}")
        print(f"    Filled quantity: {order.filled_quantity}")
        print(f"    Average fill price: ${order.avg_fill_price:,.2f}")
        print(f"    Is complete: {order.is_complete()}")


def example_multi_symbol():
    """Example: Multi-symbol trading"""
    print("\n" + "=" * 60)
    print("Example 5: Multi-Symbol Trading")
    print("=" * 60)

    engine = hmc.OrderEngine()
    pm = hmc.PositionManager()

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    prices = {
        "BTCUSDT": 45000.0,
        "ETHUSDT": 3000.0,
        "SOLUSDT": 100.0,
    }

    # Place orders for multiple symbols
    print("\n📝 Placing orders for multiple symbols...")
    for symbol in symbols:
        result = engine.place_market_order(symbol, hmc.Side.BUY, 0.1)
        print(f"  {symbol}: {result.order_id} ({result.latency_us}μs)")

        # Simulate position
        pm.update_position(symbol, 0.1, prices[symbol])

    # Get all active orders
    print("\n📋 Active orders:")
    active_orders = engine.get_active_orders()
    print(f"  Total: {len(active_orders)}")

    # Get all positions
    print("\n📊 Open positions:")
    positions = pm.get_open_positions()
    for pos in positions:
        print(f"  {pos.symbol}: {pos.quantity} @ ${pos.entry_price:,.2f}")

    # Calculate total PnL
    current_prices = {
        "BTCUSDT": 46000.0,
        "ETHUSDT": 3100.0,
        "SOLUSDT": 105.0,
    }

    total_unrealized = pm.get_total_unrealized_pnl(current_prices)
    total_realized = pm.get_total_realized_pnl()

    print(f"\n💰 Portfolio PnL:")
    print(f"  Unrealized: ${total_unrealized:,.2f}")
    print(f"  Realized: ${total_realized:,.2f}")
    print(f"  Total: ${total_unrealized + total_realized:,.2f}")


def main():
    print("\n" + "=" * 60)
    print("HEAN-META C++ Order Engine - Python Examples")
    print("=" * 60)

    example_order_placement()
    example_position_management()
    example_performance_test()
    example_order_lifecycle()
    example_multi_symbol()

    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
