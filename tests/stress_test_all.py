"""Stress test suite for 1000 orders/sec simulation.

Tests memory leaks in C++/Python bridge and overall system stability.
"""

import asyncio
import gc
import tracemalloc
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import pytest

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, OrderRequest, OrderStatus
from hean.execution.order_manager import OrderManager
from hean.execution.router import ExecutionRouter
from hean.logging import get_logger
from hean.portfolio.accounting import PortfolioAccounting

logger = get_logger(__name__)


class StressTestResult:
    """Results from stress test."""
    
    def __init__(self) -> None:
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.memory_snapshots: list[tuple[float, int]] = []
        self.latency_samples: list[float] = []
        self.errors: list[str] = []
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        duration = self.end_time - self.start_time
        orders_per_sec = self.total_orders / duration if duration > 0 else 0
        avg_latency = sum(self.latency_samples) / len(self.latency_samples) if self.latency_samples else 0
        max_latency = max(self.latency_samples) if self.latency_samples else 0
        
        memory_growth = 0
        if len(self.memory_snapshots) >= 2:
            initial_memory = self.memory_snapshots[0][1]
            final_memory = self.memory_snapshots[-1][1]
            memory_growth = final_memory - initial_memory
        
        return {
            "total_orders": self.total_orders,
            "successful_orders": self.successful_orders,
            "failed_orders": self.failed_orders,
            "duration_seconds": duration,
            "orders_per_second": orders_per_sec,
            "avg_latency_ms": avg_latency * 1000,
            "max_latency_ms": max_latency * 1000,
            "memory_growth_bytes": memory_growth,
            "memory_growth_mb": memory_growth / (1024 * 1024),
            "error_count": len(self.errors),
            "errors": self.errors[:10],  # First 10 errors
        }


class StressTest:
    """Stress test for system at 1000 orders/sec."""
    
    def __init__(
        self,
        orders_per_second: int = 1000,
        duration_seconds: int = 10,
    ) -> None:
        """Initialize stress test.
        
        Args:
            orders_per_second: Target orders per second
            duration_seconds: Test duration in seconds
        """
        self._orders_per_second = orders_per_second
        self._duration_seconds = duration_seconds
        self._bus = EventBus()
        self._order_manager = OrderManager()
        self._accounting = PortfolioAccounting(initial_capital=100000.0)
        self._execution_router: ExecutionRouter | None = None
        self._result = StressTestResult()
        self._running = False
        
    async def run(self) -> StressTestResult:
        """Run stress test.
        
        Returns:
            StressTestResult with test results
        """
        logger.info(
            f"Starting stress test: {self._orders_per_second} orders/sec "
            f"for {self._duration_seconds} seconds"
        )
        
        # Start memory tracking
        tracemalloc.start()
        
        # Start event bus
        await self._bus.start()
        
        # Initialize execution router
        from hean.core.regime import RegimeDetector
        regime_detector = RegimeDetector(self._bus)
        self._execution_router = ExecutionRouter(
            self._bus,
            self._order_manager,
            regime_detector,
        )
        
        # Track order events
        order_fills: dict[str, float] = {}
        
        async def on_order_filled(event: Event) -> None:
            """Track order fills."""
            order_id = event.data.get("order_id")
            if order_id and order_id in order_fills:
                fill_time = time.time()
                latency = fill_time - order_fills[order_id]
                self._result.latency_samples.append(latency)
                self._result.successful_orders += 1
                del order_fills[order_id]
        
        async def on_order_failed(event: Event) -> None:
            """Track order failures."""
            order_id = event.data.get("order_id")
            if order_id and order_id in order_fills:
                self._result.failed_orders += 1
                error = event.data.get("error", "Unknown error")
                self._result.errors.append(f"Order {order_id}: {error}")
                del order_fills[order_id]
        
        self._bus.subscribe(EventType.ORDER_FILLED, on_order_filled)
        self._bus.subscribe(EventType.ERROR, on_order_failed)
        
        # Start order generation
        self._running = True
        self._result.start_time = time.time()
        
        # Create order generation tasks
        order_task = asyncio.create_task(self._generate_orders())
        memory_task = asyncio.create_task(self._monitor_memory())
        
        # Wait for duration
        await asyncio.sleep(self._duration_seconds)
        
        # Stop
        self._running = False
        self._result.end_time = time.time()
        
        # Wait for tasks to complete
        await order_task
        await memory_task
        
        # Stop memory tracking
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Cleanup
        await self._bus.stop()
        
        logger.info("Stress test completed")
        logger.info(f"Results: {self._result.to_dict()}")
        
        return self._result
    
    async def _generate_orders(self) -> None:
        """Generate orders at target rate."""
        interval = 1.0 / self._orders_per_second
        order_id_counter = 0
        
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
        
        while self._running:
            order_start = time.time()
            
            # Generate order
            symbol = symbols[order_id_counter % len(symbols)]
            order_id = f"stress_{order_id_counter}_{int(time.time() * 1000)}"
            order_id_counter += 1
            
            order_request = OrderRequest(
                symbol=symbol,
                side="BUY" if order_id_counter % 2 == 0 else "SELL",
                quantity=0.01,
                order_type="MARKET",
                strategy_id="stress_test",
                timestamp=datetime.now(timezone.utc),
            )
            
            # Track order
            self._result.total_orders += 1
            request_time = time.time()
            
            # Submit order
            try:
                await self._bus.publish(Event(
                    event_type=EventType.ORDER_REQUEST,
                    data={
                        "order_request": order_request,
                        "order_id": order_id,
                    },
                ))
                
                # Track for latency measurement
                # We'll track it in the event handlers
            
            except Exception as e:
                self._result.failed_orders += 1
                self._result.errors.append(f"Failed to submit order {order_id}: {e}")
            
            # Rate limiting
            elapsed = time.time() - order_start
            sleep_time = max(0, interval - elapsed)
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                # Overload warning
                overload_ms = abs(sleep_time) * 1000
                if overload_ms > 10:
                    logger.warning(
                        f"Order generation overload: {overload_ms:.2f}ms behind schedule"
                    )
    
    async def _monitor_memory(self) -> None:
        """Monitor memory usage during test."""
        snapshot_interval = 1.0  # Snapshot every second
        
        while self._running:
            current, peak = tracemalloc.get_traced_memory()
            timestamp = time.time() - self._result.start_time
            self._result.memory_snapshots.append((timestamp, current))
            
            # Force garbage collection
            gc.collect()
            
            await asyncio.sleep(snapshot_interval)


@pytest.mark.asyncio
async def test_stress_1000_orders_per_second():
    """Stress test: 1000 orders/sec for 10 seconds."""
    stress_test = StressTest(orders_per_second=1000, duration_seconds=10)
    result = await stress_test.run()
    
    # Assertions
    assert result.total_orders > 0, "No orders were generated"
    
    # Check memory leak (should not grow more than 100MB in 10 seconds)
    memory_growth_mb = result.memory_growth_mb
    assert memory_growth_mb < 100, f"Memory leak detected: {memory_growth_mb:.2f}MB growth"
    
    # Check latency (average should be < 100ms)
    if result.latency_samples:
        avg_latency_ms = sum(result.latency_samples) / len(result.latency_samples) * 1000
        assert avg_latency_ms < 100, f"High latency: {avg_latency_ms:.2f}ms average"
    
    # Check success rate (should be > 90%)
    if result.total_orders > 0:
        success_rate = result.successful_orders / result.total_orders
        assert success_rate > 0.9, f"Low success rate: {success_rate:.2%}"
    
    logger.info("Stress test passed")
    logger.info(f"Results: {result.to_dict()}")


@pytest.mark.asyncio
async def test_stress_memory_leak_detection():
    """Test specifically for memory leaks in C++/Python bridge."""
    import gc
    
    # Test with C++ GraphEngine if available
    try:
        from hean.core.intelligence.graph_engine import GraphEngine
        
        tracemalloc.start()
        initial_snapshot = tracemalloc.take_snapshot()
        
        # Create many GraphEngine instances and updates
        for i in range(1000):
            engine = GraphEngine(window_size=100)
            engine.add_asset("BTCUSDT")
            engine.update_price("BTCUSDT", 50000.0 + i * 0.1)
            engine.recalculate()
            
            if i % 100 == 0:
                gc.collect()
        
        # Force garbage collection
        gc.collect()
        
        final_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Compare snapshots
        top_stats = final_snapshot.compare_to(initial_snapshot, "lineno")
        
        total_growth = sum(stat.size_diff for stat in top_stats[:10])
        growth_mb = total_growth / (1024 * 1024)
        
        logger.info(f"Memory growth in C++ bridge test: {growth_mb:.2f}MB")
        assert growth_mb < 50, f"Memory leak in C++ bridge: {growth_mb:.2f}MB growth"
        
    except ImportError:
        logger.warning("GraphEngine not available, skipping C++ bridge test")
        pytest.skip("GraphEngine not available")


if __name__ == "__main__":
    # Run stress test directly
    async def main():
        stress_test = StressTest(orders_per_second=1000, duration_seconds=10)
        result = await stress_test.run()
        
        print("\n" + "="*80)
        print("STRESS TEST RESULTS")
        print("="*80)
        print(f"Total orders: {result.total_orders}")
        print(f"Successful: {result.successful_orders}")
        print(f"Failed: {result.failed_orders}")
        print(f"Duration: {result.end_time - result.start_time:.2f}s")
        print(f"Orders/sec: {result.total_orders / (result.end_time - result.start_time):.2f}")
        if result.latency_samples:
            avg_latency = sum(result.latency_samples) / len(result.latency_samples)
            print(f"Avg latency: {avg_latency * 1000:.2f}ms")
            print(f"Max latency: {max(result.latency_samples) * 1000:.2f}ms")
        print(f"Memory growth: {result.memory_growth_mb:.2f}MB")
        print(f"Errors: {len(result.errors)}")
        if result.errors:
            print("\nFirst 5 errors:")
            for error in result.errors[:5]:
                print(f"  - {error}")
        print("="*80)
    
    asyncio.run(main())
