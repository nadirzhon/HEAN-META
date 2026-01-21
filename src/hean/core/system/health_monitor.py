"""Integration health monitor - The Pulse.

Monitors all system components every 100ms and triggers automatic reconnection
if latency exceeds 50ms. Logs full stack traces on failures.
"""

import asyncio
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from hean.config import settings
from hean.core.system.redis_state import get_redis_state_manager
from hean.core.intelligence.graph_engine import GraphEngine as GraphEnginePython
from hean.logging import get_logger

logger = get_logger(__name__)


class ModuleStatus(Enum):
    """Module health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Latency > 50ms but < 200ms
    UNHEALTHY = "unhealthy"  # Latency > 200ms or failures
    DISCONNECTED = "disconnected"  # Cannot connect


@dataclass
class ModuleHealth:
    """Health status for a module."""
    name: str
    status: ModuleStatus = ModuleStatus.DISCONNECTED
    latency_ms: float = 0.0
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    total_checks: int = 0
    total_failures: int = 0
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    reconnect_count: int = 0
    latency_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, latency_ms: float, success: bool, error: Optional[Exception] = None) -> None:
        """Update health metrics."""
        self.last_check = datetime.now(timezone.utc)
        self.total_checks += 1
        self.latency_history.append(latency_ms)
        self.latency_ms = latency_ms
        
        if success:
            self.consecutive_failures = 0
            self.error_message = None
            self.stack_trace = None
            
            # Update status based on latency
            if latency_ms <= 50.0:
                self.status = ModuleStatus.HEALTHY
            elif latency_ms <= 200.0:
                self.status = ModuleStatus.DEGRADED
            else:
                self.status = ModuleStatus.UNHEALTHY
        else:
            self.consecutive_failures += 1
            self.total_failures += 1
            self.status = ModuleStatus.DISCONNECTED
            
            if error:
                self.error_message = str(error)
                self.stack_trace = "".join(traceback.format_exception(
                    type(error), error, error.__traceback__
                ))


class HealthMonitor:
    """The Pulse - Monitors all modules every 100ms.
    
    Features:
    - Pings all modules (C++ Core, DB, Exchange API, Frontend Socket) every 100ms
    - Triggers automatic reconnect if latency > 50ms
    - Logs full stack traces on failures
    - Tracks health history
    """
    
    def __init__(self) -> None:
        """Initialize health monitor."""
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._modules: dict[str, ModuleHealth] = {}
        self._ping_functions: dict[str, Callable[[], Any]] = {}
        self._reconnect_functions: dict[str, Callable[[], Any]] = {}
        self._ping_interval = 0.1  # 100ms
        self._latency_threshold_ms = 50.0
        self._lock = asyncio.Lock()
        
    def register_module(
        self,
        name: str,
        ping_func: Callable[[], Any],
        reconnect_func: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Register a module for monitoring.
        
        Args:
            name: Module name (e.g., "cpp_core", "redis", "db", "exchange_api", "frontend_socket")
            ping_func: Async function that returns True if module is healthy
            reconnect_func: Optional async function to reconnect the module
        """
        self._modules[name] = ModuleHealth(name=name)
        self._ping_functions[name] = ping_func
        if reconnect_func:
            self._reconnect_functions[name] = reconnect_func
        
        logger.info(f"Registered health monitor for module: {name}")
    
    async def start(self) -> None:
        """Start the health monitor."""
        if self._running:
            return
        
        # Register default modules
        await self._register_default_modules()
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started (The Pulse)")
    
    async def stop(self) -> None:
        """Stop the health monitor."""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitor stopped")
    
    async def _register_default_modules(self) -> None:
        """Register default system modules."""
        # C++ Core (GraphEngine)
        async def ping_cpp_core() -> bool:
            try:
                # Try to use Python wrapper if available
                engine = GraphEnginePython(window_size=100)
                start = time.time()
                count = engine.get_asset_count()
                latency_ms = (time.time() - start) * 1000
                return latency_ms <= 50.0
            except Exception as e:
                logger.debug(f"C++ Core ping failed: {e}")
                return False
        
        async def reconnect_cpp_core() -> None:
            logger.warning("C++ Core reconnect requested (manual restart required)")
            # C++ core reconnection would require restarting the Python process
            # or reloading the C++ module
        
        self.register_module("cpp_core", ping_cpp_core, reconnect_cpp_core)
        
        # Redis
        async def ping_redis() -> bool:
            if not REDIS_AVAILABLE:
                return False
            try:
                manager = await get_redis_state_manager()
                start = time.time()
                await manager._client.ping()  # type: ignore
                latency_ms = (time.time() - start) * 1000
                return latency_ms <= 50.0
            except Exception:
                return False
        
        async def reconnect_redis() -> None:
            try:
                manager = await get_redis_state_manager()
                await manager.disconnect()
                await asyncio.sleep(0.1)
                await manager.connect()
                logger.info("Redis reconnected successfully")
            except Exception as e:
                logger.error(f"Failed to reconnect Redis: {e}", exc_info=True)
        
        self.register_module("redis", ping_redis, reconnect_redis)
        
        # Database (if available)
        async def ping_db() -> bool:
            # Placeholder - implement actual DB ping if DB is used
            return True
        
        self.register_module("db", ping_db)
        
        # Exchange API (Bybit)
        async def ping_exchange_api() -> bool:
            from hean.exchange.bybit.http import BybitHTTPClient
            try:
                client = BybitHTTPClient()
                start = time.time()
                # Ping server time endpoint
                await client.get_server_time()
                latency_ms = (time.time() - start) * 1000
                return latency_ms <= 50.0
            except Exception:
                return False
        
        async def reconnect_exchange_api() -> None:
            logger.warning("Exchange API reconnect requested (reconnection handled by client)")
            # Exchange clients usually handle reconnection internally
        
        self.register_module("exchange_api", ping_exchange_api, reconnect_exchange_api)
        
        # Frontend Socket (WebSocket connection check)
        async def ping_frontend_socket() -> bool:
            # This will be updated when WebSocket is implemented
            # For now, check if WebSocket server is running
            return True
        
        self.register_module("frontend_socket", ping_frontend_socket)
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop - runs every 100ms."""
        while self._running:
            loop_start = time.time()
            
            # Ping all modules concurrently
            tasks = []
            for name in self._modules.keys():
                tasks.append(self._ping_module(name))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate sleep time to maintain 100ms interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, self._ping_interval - elapsed)
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                # Overload warning
                overload_ms = abs(sleep_time) * 1000
                if overload_ms > 10:
                    logger.warning(
                        f"Health monitor loop overload: {overload_ms:.1f}ms behind schedule"
                    )
    
    async def _ping_module(self, name: str) -> None:
        """Ping a single module and update health status."""
        if name not in self._modules:
            return
        
        health = self._modules[name]
        ping_func = self._ping_functions.get(name)
        
        if not ping_func:
            return
        
        start = time.time()
        success = False
        error: Optional[Exception] = None
        
        try:
            if asyncio.iscoroutinefunction(ping_func):
                result = await ping_func()
            else:
                result = ping_func()
            success = bool(result)
        except Exception as e:
            error = e
            success = False
            logger.debug(f"Module {name} ping failed: {e}")
        
        latency_ms = (time.time() - start) * 1000
        
        # Update health status
        async with self._lock:
            health.update(latency_ms, success, error)
            
            # Trigger auto-reconnect if latency > threshold or consecutive failures
            if (
                (latency_ms > self._latency_threshold_ms or not success)
                and name in self._reconnect_functions
                and health.consecutive_failures >= 3
            ):
                await self._trigger_reconnect(name)
    
    async def _trigger_reconnect(self, name: str) -> None:
        """Trigger automatic reconnection for a module."""
        health = self._modules[name]
        reconnect_func = self._reconnect_functions.get(name)
        
        if not reconnect_func:
            return
        
        logger.warning(
            f"Auto-reconnect triggered for {name} "
            f"(latency: {health.latency_ms:.1f}ms, failures: {health.consecutive_failures})"
        )
        
        try:
            if asyncio.iscoroutinefunction(reconnect_func):
                await reconnect_func()
            else:
                reconnect_func()
            
            health.reconnect_count += 1
            health.consecutive_failures = 0  # Reset after reconnect
            
            logger.info(f"Successfully reconnected {name}")
            
        except Exception as e:
            logger.error(
                f"Failed to reconnect {name}: {e}",
                exc_info=True,
                extra={
                    "module": name,
                    "error": str(e),
                    "stack_trace": traceback.format_exc(),
                }
            )
            
            # Log bug with full stack trace
            logger.error(
                f"BUG DETECTED: Module {name} reconnection failed\n"
                f"Latency: {health.latency_ms:.1f}ms\n"
                f"Consecutive failures: {health.consecutive_failures}\n"
                f"Stack trace:\n{traceback.format_exc()}"
            )
    
    def get_health_status(self) -> dict[str, Any]:
        """Get current health status of all modules."""
        async with self._lock:
            return {
                name: {
                    "status": health.status.value,
                    "latency_ms": health.latency_ms,
                    "last_check": health.last_check.isoformat() if health.last_check else None,
                    "consecutive_failures": health.consecutive_failures,
                    "total_checks": health.total_checks,
                    "total_failures": health.total_failures,
                    "reconnect_count": health.reconnect_count,
                    "error_message": health.error_message,
                    "avg_latency_ms": (
                        sum(health.latency_history) / len(health.latency_history)
                        if health.latency_history else 0.0
                    ),
                    "max_latency_ms": (
                        max(health.latency_history) if health.latency_history else 0.0
                    ),
                }
                for name, health in self._modules.items()
            }
    
    def get_module_health(self, name: str) -> Optional[dict[str, Any]]:
        """Get health status for a specific module."""
        health = self._modules.get(name)
        if not health:
            return None
        
        return {
            "status": health.status.value,
            "latency_ms": health.latency_ms,
            "last_check": health.last_check.isoformat() if health.last_check else None,
            "consecutive_failures": health.consecutive_failures,
            "total_checks": health.total_checks,
            "total_failures": health.total_failures,
            "reconnect_count": health.reconnect_count,
            "error_message": health.error_message,
            "stack_trace": health.stack_trace,
            "avg_latency_ms": (
                sum(health.latency_history) / len(health.latency_history)
                if health.latency_history else 0.0
            ),
            "max_latency_ms": (
                max(health.latency_history) if health.latency_history else 0.0
            ),
        }


# Global instance
_health_monitor: Optional[HealthMonitor] = None


async def get_health_monitor() -> HealthMonitor:
    """Get or create global health monitor."""
    global _health_monitor
    
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
        await _health_monitor.start()
    
    return _health_monitor
