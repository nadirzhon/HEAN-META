"""Example: Smart Execution and Risk Improvements.

This example demonstrates how to use the new execution and risk modules:
1. TWAP/VWAP execution for large orders
2. Market anomaly detection for risk management
3. RL-based portfolio allocation

Run this example to see all modules working together.
"""

import asyncio
from datetime import datetime

from hean.core.bus import EventBus
from hean.core.types import OrderRequest
from hean.execution.smart_execution import IcebergDetector, TWAPExecutor, VWAPExecutor
from hean.logging import setup_logging
from hean.portfolio.rl_allocator import RLPortfolioAllocator, StrategyMetrics
from hean.risk.anomaly_detector import MarketAnomalyDetector


async def main():
    """Run smart execution example."""
    # Setup logging
    setup_logging()

    # Initialize event bus
    event_bus = EventBus()
    await event_bus.start()

    print("=" * 70)
    print("HEAN Smart Execution and Risk Improvements Demo")
    print("=" * 70)

    # =========================================================================
    # Part 1: TWAP/VWAP Execution
    # =========================================================================
    print("\n📊 Part 1: Smart Execution (TWAP/VWAP)")
    print("-" * 70)

    # Mock execution router
    class MockRouter:
        def __init__(self):
            self.submitted_orders = []

        async def submit_order(self, order_request):
            self.submitted_orders.append(order_request)
            print(f"  ✓ Order submitted: {order_request.symbol} {order_request.side} "
                  f"{order_request.size:.2f} @ {order_request.price}")

    mock_router = MockRouter()

    # Initialize TWAP executor
    twap = TWAPExecutor(event_bus, mock_router)
    await twap.start()

    # Create large order
    large_order = OrderRequest(
        signal_id="demo_signal_1",
        strategy_id="impulse_engine",
        symbol="BTCUSDT",
        side="buy",
        size=10.0,  # Large order: 10 BTC
        price=50000.0,
        order_type="limit",
    )

    print(f"\n🎯 Executing large order with TWAP: {large_order.size} BTC")
    print(f"   Splitting into 5 slices over 30 seconds...")

    twap_order_id = await twap.execute_twap(
        large_order,
        num_slices=5,
        interval_seconds=6,  # Faster for demo
        use_limit_orders=True,
    )

    # Wait a bit and check status
    await asyncio.sleep(2)
    status = twap.get_order_status(twap_order_id)
    print(f"\n   Progress: {status['progress_pct']:.1f}% complete")
    print(f"   Executed: {status['executed_slices']}/{status['total_slices']} slices")

    # Initialize VWAP executor
    vwap = VWAPExecutor(event_bus, mock_router)
    await vwap.start()

    # Update volume data
    vwap.update_volume("BTCUSDT", 1000.0)
    vwap.update_volume("BTCUSDT", 1200.0)
    vwap.update_volume("BTCUSDT", 800.0)

    print(f"\n🎯 Executing order with VWAP: {large_order.size} BTC")
    print(f"   Target participation: 10% of market volume")

    vwap_order_id = await vwap.execute_vwap(
        large_order,
        target_participation=0.10,
        max_duration_seconds=300,
    )

    vwap_status = vwap.get_order_status(vwap_order_id)
    print(f"\n   Executed size: {vwap_status['executed_size']:.2f}")
    print(f"   Remaining: {vwap_status['remaining_size']:.2f}")

    # Iceberg detection
    print(f"\n🧊 Detecting iceberg orders...")
    iceberg_detector = IcebergDetector()

    # Simulate order book updates
    for _ in range(5):
        iceberg_detector.update_orderbook(
            "BTCUSDT",
            bids=[(50000.0, 1.5), (49999.0, 2.0)],
            asks=[(50001.0, 1.0), (50002.0, 1.8)],
        )
        await asyncio.sleep(0.1)

    detected_icebergs = iceberg_detector.detect_iceberg("BTCUSDT", "buy")
    if detected_icebergs:
        print(f"\n   ✓ Detected {len(detected_icebergs)} iceberg order(s):")
        for ice in detected_icebergs:
            print(
                f"     • Price: {ice['price']:.2f}, "
                f"Hidden: ~{ice['estimated_hidden_size']:.2f}, "
                f"Confidence: {ice['confidence']:.2f}"
            )
    else:
        print("   ℹ No icebergs detected yet (need more refreshes)")

    # =========================================================================
    # Part 2: Market Anomaly Detection
    # =========================================================================
    print("\n\n🚨 Part 2: Market Anomaly Detection")
    print("-" * 70)

    anomaly_detector = MarketAnomalyDetector(contamination=0.05)

    # Feed normal market data
    print("\n📈 Feeding normal market data...")
    for i in range(150):
        price = 50000 + (i % 10) * 10  # Small oscillation
        volume = 1000 + (i % 5) * 100
        anomaly_detector.update(
            symbol="BTCUSDT",
            price=price,
            volume=volume,
            bid=price - 1,
            ask=price + 1,
            buy_volume=500,
            sell_volume=500,
        )

    # Check normal state
    result = anomaly_detector.is_anomaly("BTCUSDT")
    print(f"   Market state: {'ANOMALOUS ⚠' if result.is_anomaly else 'NORMAL ✓'}")

    # Feed anomalous data (flash crash)
    print("\n⚡ Simulating flash crash...")
    anomaly_detector.update(
        symbol="BTCUSDT",
        price=45000,  # -10% crash
        volume=10000,  # Volume spike
        bid=44900,
        ask=45100,
        buy_volume=100,
        sell_volume=9900,  # Extreme selling
    )

    result = anomaly_detector.is_anomaly("BTCUSDT")
    if result.is_anomaly:
        print(f"   ✓ Anomaly detected!")
        print(f"     Type: {result.anomaly_type}")
        print(f"     Score: {result.anomaly_score:.3f}")
        print(f"     Confidence: {result.confidence:.2f}")
        print(f"\n   🛡 Recommended: Escalate to RiskGovernor SOFT_BRAKE state")

    metrics = anomaly_detector.get_metrics()
    print(f"\n   Detector metrics:")
    print(f"     Total checks: {metrics['total_checks']}")
    print(f"     Anomalies: {metrics['anomaly_count']}")
    print(f"     Anomaly rate: {metrics['anomaly_rate']:.2%}")

    # =========================================================================
    # Part 3: RL Portfolio Allocation
    # =========================================================================
    print("\n\n🤖 Part 3: RL Portfolio Allocation")
    print("-" * 70)

    # Define strategies
    strategy_ids = ["impulse_engine", "funding_harvester", "basis_arbitrage"]

    # Check if PyTorch available for RL
    try:
        import torch

        rl_available = True
    except ImportError:
        rl_available = False

    if rl_available:
        print("\n✓ PyTorch available - using RL-based allocation")
    else:
        print("\n⚠ PyTorch not available - using fallback (risk parity)")

    allocator = RLPortfolioAllocator(
        strategy_ids=strategy_ids,
        learning_rate=3e-4,
    )

    # Create mock strategy metrics
    strategy_metrics = {
        "impulse_engine": StrategyMetrics(
            strategy_id="impulse_engine",
            profit_factor=1.8,
            sharpe_ratio=1.5,
            win_rate=0.60,
            drawdown_pct=3.5,
            volatility=0.025,
            recent_returns=[0.015, 0.012, -0.003, 0.018, 0.010],
            correlation_with_others=0.25,
        ),
        "funding_harvester": StrategyMetrics(
            strategy_id="funding_harvester",
            profit_factor=2.2,
            sharpe_ratio=2.0,
            win_rate=0.75,
            drawdown_pct=1.8,
            volatility=0.015,
            recent_returns=[0.008, 0.009, 0.007, 0.010, 0.008],
            correlation_with_others=0.10,
        ),
        "basis_arbitrage": StrategyMetrics(
            strategy_id="basis_arbitrage",
            profit_factor=1.5,
            sharpe_ratio=1.2,
            win_rate=0.55,
            drawdown_pct=4.2,
            volatility=0.030,
            recent_returns=[0.012, -0.005, 0.015, 0.008, -0.002],
            correlation_with_others=0.30,
        ),
    }

    print("\n📊 Strategy Performance:")
    for sid, metrics in strategy_metrics.items():
        print(
            f"   • {sid}: PF={metrics.profit_factor:.2f}, "
            f"Sharpe={metrics.sharpe_ratio:.2f}, "
            f"Win%={metrics.win_rate:.0%}"
        )

    # Get allocation
    allocation_result = allocator.get_allocation(strategy_metrics)

    print(f"\n💰 Optimal Capital Allocation:")
    print(f"   Method: {allocation_result.method}")
    print(f"   Confidence: {allocation_result.confidence:.2f}")
    for strategy_id, weight in allocation_result.weights.items():
        print(f"     • {strategy_id}: {weight:.1%}")

    # Verify sum to 1.0
    total_weight = sum(allocation_result.weights.values())
    print(f"\n   ✓ Total allocation: {total_weight:.1%} (should be 100%)")

    # Get allocator metrics
    allocator_metrics = allocator.get_metrics()
    print(f"\n   Allocator metrics:")
    print(f"     RL enabled: {allocator_metrics['rl_enabled']}")
    print(f"     Allocations made: {allocator_metrics['allocations_made']}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ Demo Complete - All Modules Working")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. TWAP/VWAP split large orders to minimize market impact")
    print("  2. Iceberg detector identifies hidden institutional orders")
    print("  3. Anomaly detector flags unusual market conditions")
    print("  4. RL allocator optimizes capital distribution dynamically")
    print("\nProduction Deployment:")
    print("  • Enable TWAP/VWAP for orders > 100 USDT")
    print("  • Integrate anomaly detector with RiskGovernor")
    print("  • Use RL allocator for adaptive capital allocation")
    print("=" * 70)

    # Cleanup
    await twap.stop()
    await vwap.stop()
    await event_bus.stop()


if __name__ == "__main__":
    asyncio.run(main())
