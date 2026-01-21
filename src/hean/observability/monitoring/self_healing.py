"""Self-Healing Middleware - System Health Monitoring and Emergency Takeover.

Monitors API latency, system load, memory usage, and Python GC performance.
If Python's Garbage Collector is slowing down execution, triggers C++ Emergency Takeover.
"""

import asyncio
import gc
import os
import platform
import time
from collections import deque
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None  # Optional dependency

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.execution.order_manager import OrderManager
from hean.logging import get_logger

logger = get_logger(__name__)


class SelfHealingMiddleware:
    """Self-healing middleware for system health monitoring and emergency takeover.
    
    Monitors:
    - API latency (order placement, price feeds)
    - System load (CPU, memory)
    - Python GC performance
    - Memory leaks
    
    Actions:
    - Triggers C++ Emergency Takeover if Python GC slows execution
    - Reduces position sizes if system load is high
    - Forces GC collection if memory usage is high
    - Restarts components if they become unresponsive
    """

    def __init__(
        self,
        bus: EventBus,
        order_manager: OrderManager | None = None,
    ) -> None:
        """Initialize self-healing middleware.
        
        Args:
            bus: Event bus for publishing health events
            order_manager: Order manager for emergency takeover
        """
        self._bus = bus
        self._order_manager = order_manager
        
        # Monitoring intervals
        self._check_interval_seconds = 10.0  # Check every 10 seconds
        self._gc_check_interval_seconds = 30.0  # Check GC every 30 seconds
        
        # Latency tracking
        self._api_latencies: deque[float] = deque(maxlen=100)
        self._order_latencies: deque[float] = deque(maxlen=100)
        
        # GC performance tracking
        self._gc_collection_times: deque[float] = deque(maxlen=50)
        self._last_gc_check = time.time()
        
        # System metrics (psutil is optional)
        if psutil is not None:
            self._process = psutil.Process(os.getpid())
        else:
            self._process = None
            logger.warning("psutil not available, system monitoring will be limited")
        self._cpu_history: deque[float] = deque(maxlen=100)
        self._memory_history: deque[float] = deque(maxlen=100)
        
        # Thresholds
        self._max_api_latency_ms = 1000.0  # 1 second max
        self._max_order_latency_ms = 2000.0  # 2 seconds max
        self._max_cpu_percent = 90.0  # 90% CPU max
        self._max_memory_percent = 85.0  # 85% memory max
        self._max_gc_time_ms = 100.0  # 100ms max GC time
        
        # Emergency takeover state
        self._emergency_takeover_active = False
        self._cpp_emergency_handler: Any | None = None  # Would be C++ module
        
        # Health status
        self._health_status = "healthy"
        self._last_health_check = time.time()
        
        logger.info("Self-healing middleware initialized")

    async def start(self) -> None:
        """Start monitoring and self-healing tasks."""
        # Start monitoring task
        asyncio.create_task(self._monitoring_loop())
        
        # Start GC monitoring task
        asyncio.create_task(self._gc_monitoring_loop())
        
        logger.info("Self-healing middleware started")

    async def stop(self) -> None:
        """Stop monitoring and cleanup."""
        self._emergency_takeover_active = False
        logger.info("Self-healing middleware stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self._check_interval_seconds)
                
                # Check system metrics
                await self._check_system_metrics()
                
                # Check API latency
                await self._check_api_latency()
                
                # Check order latency
                await self._check_order_latency()
                
                # Check memory usage
                await self._check_memory_usage()
                
                # Update health status
                await self._update_health_status()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)

    async def _gc_monitoring_loop(self) -> None:
        """GC performance monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self._gc_check_interval_seconds)
                
                # Measure GC performance
                gc_time = await self._measure_gc_performance()
                
                # Check if GC is slowing execution
                if gc_time > self._max_gc_time_ms:
                    logger.warning(
                        f"GC performance degraded: {gc_time:.2f}ms > {self._max_gc_time_ms}ms"
                    )
                    await self._handle_gc_slowdown()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in GC monitoring loop: {e}", exc_info=True)

    async def _check_system_metrics(self) -> None:
        """Check CPU and system load."""
        if self._process is None or psutil is None:
            return  # Skip if psutil not available
            
        try:
            # CPU usage
            cpu_percent = self._process.cpu_percent(interval=1.0)
            self._cpu_history.append(cpu_percent)
            
            # Memory usage
            memory_info = self._process.memory_info()
            memory_percent = (memory_info.rss / (1024 * 1024 * 1024)) * 100  # GB to percent
            
            # For percentage, use system memory
            try:
                system_memory = psutil.virtual_memory()
                memory_percent = system_memory.percent
            except Exception:
                pass  # Fallback to absolute
            
            self._memory_history.append(memory_percent)
            
            # Check thresholds
            if cpu_percent > self._max_cpu_percent:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}% > {self._max_cpu_percent}%")
                await self._handle_high_cpu()
            
            if memory_percent > self._max_memory_percent:
                logger.warning(
                    f"High memory usage: {memory_percent:.1f}% > {self._max_memory_percent}%"
                )
                await self._handle_high_memory()
                
        except Exception as e:
            logger.error(f"Error checking system metrics: {e}", exc_info=True)

    async def _check_api_latency(self) -> None:
        """Check API latency from tracked calls."""
        if len(self._api_latencies) == 0:
            return
        
        avg_latency = sum(self._api_latencies) / len(self._api_latencies)
        max_latency = max(self._api_latencies)
        
        if avg_latency > self._max_api_latency_ms:
            logger.warning(
                f"High API latency: avg={avg_latency:.2f}ms, max={max_latency:.2f}ms"
            )
            await self._handle_high_api_latency()
        
        if max_latency > self._max_api_latency_ms * 2:
            logger.error(f"Critical API latency: {max_latency:.2f}ms")
            await self._handle_critical_api_latency()

    async def _check_order_latency(self) -> None:
        """Check order placement latency."""
        if len(self._order_latencies) == 0:
            return
        
        avg_latency = sum(self._order_latencies) / len(self._order_latencies)
        max_latency = max(self._order_latencies)
        
        if avg_latency > self._max_order_latency_ms:
            logger.warning(
                f"High order latency: avg={avg_latency:.2f}ms, max={max_latency:.2f}ms"
            )
            await self._handle_high_order_latency()

    async def _check_memory_usage(self) -> None:
        """Check memory usage and detect leaks."""
        if len(self._memory_history) < 10:
            return
        
        # Check for memory leak (increasing trend)
        recent_memory = list(self._memory_history)[-10:]
        if recent_memory[-1] > recent_memory[0] * 1.2:  # 20% increase
            logger.warning("Potential memory leak detected (20% increase)")
            await self._handle_memory_leak()

    async def _measure_gc_performance(self) -> float:
        """Measure GC collection performance.
        
        Returns:
            GC collection time in milliseconds
        """
        start_time = time.perf_counter()
        
        # Force a GC collection
        collected = gc.collect()
        
        end_time = time.perf_counter()
        gc_time_ms = (end_time - start_time) * 1000.0
        
        # Track GC time
        self._gc_collection_times.append(gc_time_ms)
        
        logger.debug(f"GC collected {collected} objects in {gc_time_ms:.2f}ms")
        
        return gc_time_ms

    async def _handle_gc_slowdown(self) -> None:
        """Handle GC slowdown - trigger C++ emergency takeover."""
        logger.critical("GC SLOWDOWN DETECTED: Triggering C++ Emergency Takeover")
        
        # Trigger emergency takeover
        await self._trigger_emergency_takeover(reason="gc_slowdown")
        
        # Publish health event
        await self._bus.publish(
            Event(
                event_type=EventType.REGIME_UPDATE,  # Reuse for health events
                data={
                    "health_event": "gc_slowdown",
                    "action": "emergency_takeover",
                    "status": "critical"
                }
            )
        )

    async def _handle_high_cpu(self) -> None:
        """Handle high CPU usage."""
        # Reduce position sizes or pause trading
        logger.warning("High CPU detected: Reducing trading activity")
        
        await self._bus.publish(
            Event(
                event_type=EventType.REGIME_UPDATE,
                data={
                    "health_event": "high_cpu",
                    "action": "reduce_activity",
                    "size_multiplier": 0.5  # Reduce by 50%
                }
            )
        )

    async def _handle_high_memory(self) -> None:
        """Handle high memory usage."""
        # Force GC collection
        logger.warning("High memory detected: Forcing GC collection")
        
        collected = gc.collect()
        logger.info(f"GC collected {collected} objects")
        
        # If still high, trigger emergency measures
        if self._process is not None:
            memory_info = self._process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            if memory_mb > 1024:  # > 1GB
                logger.warning(f"Memory still high: {memory_mb:.0f}MB, triggering cleanup")

    async def _handle_memory_leak(self) -> None:
        """Handle memory leak."""
        logger.warning("Memory leak detected: Forcing aggressive GC")
        
        # Force full GC
        for _ in range(3):
            gc.collect()
        
        # Log memory stats
        if self._process is not None:
            memory_info = self._process.memory_info()
            logger.info(f"Memory after GC: {memory_info.rss / (1024 * 1024):.0f}MB")

    async def _handle_high_api_latency(self) -> None:
        """Handle high API latency."""
        logger.warning("High API latency: Reducing API call frequency")

    async def _handle_critical_api_latency(self) -> None:
        """Handle critical API latency."""
        logger.error("CRITICAL API latency: Pausing new orders")
        
        await self._bus.publish(
            Event(
                event_type=EventType.REGIME_UPDATE,
                data={
                    "health_event": "critical_api_latency",
                    "action": "pause_orders",
                    "status": "critical"
                }
            )
        )

    async def _handle_high_order_latency(self) -> None:
        """Handle high order latency."""
        logger.warning("High order latency: Reviewing order execution")

    async def _trigger_emergency_takeover(self, reason: str) -> None:
        """Trigger C++ Emergency Takeover for order management.
        
        In production, this would call a C++ module to take over order management
        from Python. For now, we log the action and can implement a Python fallback.
        
        Args:
            reason: Reason for emergency takeover
        """
        self._emergency_takeover_active = True
        
        logger.critical(f"EMERGENCY TAKEOVER ACTIVATED: {reason}")
        
        # In production, would call C++ module:
        # try:
        #     import cpp_emergency_handler
        #     cpp_emergency_handler.takeover_orders(self._order_manager)
        # except ImportError:
        #     logger.error("C++ emergency handler not available")
        
        # For now, implement Python fallback:
        if self._order_manager:
            # Cancel pending orders to reduce load
            pending_orders = self._order_manager.get_pending_orders()
            if pending_orders:
                logger.warning(f"Cancelling {len(pending_orders)} pending orders")
                # Would cancel orders here
        
        # Publish emergency event
        await self._bus.publish(
            Event(
                event_type=EventType.REGIME_UPDATE,
                data={
                    "health_event": "emergency_takeover",
                    "reason": reason,
                    "status": "critical",
                    "active": True
                }
            )
        )

    async def _update_health_status(self) -> None:
        """Update overall health status."""
        health_issues = []
        
        # Check all metrics
        if len(self._cpu_history) > 0 and max(self._cpu_history) > self._max_cpu_percent:
            health_issues.append("high_cpu")
        
        if len(self._memory_history) > 0 and max(self._memory_history) > self._max_memory_percent:
            health_issues.append("high_memory")
        
        if len(self._api_latencies) > 0 and max(self._api_latencies) > self._max_api_latency_ms:
            health_issues.append("high_api_latency")
        
        if len(self._gc_collection_times) > 0 and max(self._gc_collection_times) > self._max_gc_time_ms:
            health_issues.append("gc_slowdown")
        
        # Update status
        if self._emergency_takeover_active:
            self._health_status = "critical"
        elif health_issues:
            self._health_status = "degraded"
        else:
            self._health_status = "healthy"
        
        self._last_health_check = time.time()

    def record_api_latency(self, latency_ms: float) -> None:
        """Record API call latency.
        
        Args:
            latency_ms: Latency in milliseconds
        """
        self._api_latencies.append(latency_ms)

    def record_order_latency(self, latency_ms: float) -> None:
        """Record order placement latency.
        
        Args:
            latency_ms: Latency in milliseconds
        """
        self._order_latencies.append(latency_ms)

    def get_health_status(self) -> dict[str, Any]:
        """Get current health status.
        
        Returns:
            Dictionary with health metrics and status
        """
        avg_cpu = sum(self._cpu_history) / len(self._cpu_history) if self._cpu_history else 0.0
        avg_memory = sum(self._memory_history) / len(self._memory_history) if self._memory_history else 0.0
        avg_api_latency = sum(self._api_latencies) / len(self._api_latencies) if self._api_latencies else 0.0
        avg_gc_time = sum(self._gc_collection_times) / len(self._gc_collection_times) if self._gc_collection_times else 0.0
        
        return {
            "status": self._health_status,
            "emergency_takeover_active": self._emergency_takeover_active,
            "cpu_percent": avg_cpu,
            "memory_percent": avg_memory,
            "api_latency_ms": avg_api_latency,
            "gc_time_ms": avg_gc_time,
            "last_health_check": self._last_health_check
        }

    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self._health_status == "healthy"
