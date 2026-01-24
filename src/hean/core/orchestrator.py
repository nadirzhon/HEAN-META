"""Trading System Orchestrator - Central Coordinator for all modules."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class ComponentState(str, Enum):
    """Component health states."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    STARTING = "starting"
    STOPPED = "stopped"


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ComponentHealth:
    """Health status of a component."""

    name: str
    state: ComponentState
    last_check: float
    failure_count: int = 0
    success_count: int = 0
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    name: str
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0

    def record_success(self) -> None:
        """Record a successful operation."""
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} CLOSED (recovered)")

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(f"Circuit breaker {self.name} OPENED (too many failures)")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.success_count = 0
            logger.warning(f"Circuit breaker {self.name} reopened (test failed)")

    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Try to recover after timeout
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} HALF_OPEN (testing recovery)")
                return True
            return False

        # HALF_OPEN: allow limited requests for testing
        return True


class TradingOrchestrator:
    """Central orchestrator for all trading system components.

    Responsibilities:
    - Component lifecycle management (initialization, shutdown)
    - Health monitoring and fault detection
    - Circuit breakers for fault tolerance
    - Event coordination via EventBus
    - Graceful degradation and recovery
    """

    def __init__(self, bus: Optional[EventBus] = None) -> None:
        """Initialize orchestrator.

        Args:
            bus: Optional EventBus instance. If None, creates a new one.
        """
        self.bus = bus or EventBus()
        self._components: dict[str, Any] = {}
        self._health: dict[str, ComponentHealth] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._running = False
        self._start_time = 0.0
        self._health_check_task: Optional[asyncio.Task[None]] = None

    def register_component(
        self,
        name: str,
        component: Any,
        circuit_breaker: bool = True,
        failure_threshold: int = 5,
        timeout: float = 60.0,
    ) -> None:
        """Register a component for orchestration.

        Args:
            name: Component name (unique identifier)
            component: Component instance
            circuit_breaker: Whether to enable circuit breaker
            failure_threshold: Failures before opening circuit
            timeout: Seconds before attempting recovery
        """
        self._components[name] = component
        self._health[name] = ComponentHealth(
            name=name, state=ComponentState.STOPPED, last_check=time.time()
        )

        if circuit_breaker:
            self._circuit_breakers[name] = CircuitBreaker(
                name=name, failure_threshold=failure_threshold, timeout=timeout
            )

        logger.info(f"Registered component: {name}")

    async def start(self) -> None:
        """Start all components in dependency order."""
        if self._running:
            logger.warning("Orchestrator already running")
            return

        logger.info("Starting Trading Orchestrator...")
        self._running = True
        self._start_time = time.time()

        # Start EventBus first
        await self.bus.start()
        logger.info("EventBus started")

        # Subscribe to errors for monitoring
        self.bus.subscribe(EventType.ERROR, self._handle_error_event)

        # Initialize components
        for name, component in self._components.items():
            try:
                self._health[name].state = ComponentState.STARTING
                logger.info(f"Starting component: {name}")

                # Call start() if available
                if hasattr(component, "start"):
                    start_method = component.start
                    if asyncio.iscoroutinefunction(start_method):
                        await start_method()
                    else:
                        start_method()

                self._health[name].state = ComponentState.HEALTHY
                self._health[name].last_check = time.time()
                logger.info(f"Component {name} started successfully")

            except Exception as e:
                logger.error(f"Failed to start component {name}: {e}", exc_info=True)
                self._health[name].state = ComponentState.FAILED
                self._health[name].error_message = str(e)

                # Notify via event bus
                await self.bus.publish(
                    Event(
                        event_type=EventType.ERROR,
                        data={"component": name, "error": str(e), "stage": "startup"},
                    )
                )

        # Start periodic health checks
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("Trading Orchestrator started successfully")

    async def stop(self) -> None:
        """Stop all components gracefully."""
        if not self._running:
            logger.warning("Orchestrator not running")
            return

        logger.info("Stopping Trading Orchestrator...")
        self._running = False

        # Stop health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Stop components in reverse order
        for name, component in reversed(list(self._components.items())):
            try:
                logger.info(f"Stopping component: {name}")
                self._health[name].state = ComponentState.STOPPED

                # Call stop() if available
                if hasattr(component, "stop"):
                    stop_method = component.stop
                    if asyncio.iscoroutinefunction(stop_method):
                        await stop_method()
                    else:
                        stop_method()

                logger.info(f"Component {name} stopped")

            except Exception as e:
                logger.error(f"Error stopping component {name}: {e}", exc_info=True)

        # Stop EventBus last
        await self.bus.stop()
        logger.info("EventBus stopped")

        logger.info("Trading Orchestrator stopped successfully")

    async def execute_with_circuit_breaker(
        self, component_name: str, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute a function with circuit breaker protection.

        Args:
            component_name: Name of the component
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            RuntimeError: If circuit is open
        """
        breaker = self._circuit_breakers.get(component_name)
        if not breaker:
            # No circuit breaker, execute directly
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

        # Check if circuit allows execution
        if not breaker.can_execute():
            error_msg = f"Circuit breaker {component_name} is OPEN, rejecting request"
            logger.warning(error_msg)
            raise RuntimeError(error_msg)

        # Execute with circuit breaker
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Record success
            breaker.record_success()
            self._health[component_name].failure_count = 0
            return result

        except Exception as e:
            # Record failure
            breaker.record_failure()
            self._health[component_name].failure_count += 1
            self._health[component_name].error_message = str(e)

            # Update component state
            if breaker.state == CircuitState.OPEN:
                self._health[component_name].state = ComponentState.FAILED
            else:
                self._health[component_name].state = ComponentState.DEGRADED

            logger.error(f"Component {component_name} failed: {e}", exc_info=True)
            raise

    async def health_check(self) -> dict[str, Any]:
        """Get health status of all components.

        Returns:
            Health status dict
        """
        uptime = time.time() - self._start_time if self._running else 0

        components_health = {}
        for name, health in self._health.items():
            breaker = self._circuit_breakers.get(name)
            components_health[name] = {
                "state": health.state.value,
                "last_check": health.last_check,
                "failure_count": health.failure_count,
                "error_message": health.error_message,
                "circuit_breaker": breaker.state.value if breaker else "disabled",
                "metadata": health.metadata,
            }

        overall_state = "healthy"
        if any(h.state == ComponentState.FAILED for h in self._health.values()):
            overall_state = "failed"
        elif any(h.state == ComponentState.DEGRADED for h in self._health.values()):
            overall_state = "degraded"

        return {
            "orchestrator": {
                "running": self._running,
                "uptime_seconds": uptime,
                "start_time": datetime.fromtimestamp(self._start_time).isoformat()
                if self._start_time
                else None,
                "overall_state": overall_state,
            },
            "components": components_health,
            "event_bus": {
                "queue_size": self.bus._queue.qsize(),
                "max_queue_size": self.bus._queue.maxsize,
            },
        }

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                for name, component in self._components.items():
                    try:
                        # Call health_check() if available
                        if hasattr(component, "health_check"):
                            health_method = component.health_check
                            if asyncio.iscoroutinefunction(health_method):
                                result = await health_method()
                            else:
                                result = health_method()

                            # Update health status
                            if result:
                                self._health[name].last_check = time.time()
                                if self._health[name].state == ComponentState.DEGRADED:
                                    # Component recovered
                                    self._health[name].state = ComponentState.HEALTHY
                                    logger.info(f"Component {name} recovered")

                    except Exception as e:
                        logger.error(
                            f"Health check failed for {name}: {e}", exc_info=True
                        )
                        self._health[name].state = ComponentState.DEGRADED
                        self._health[name].error_message = str(e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}", exc_info=True)

    async def _handle_error_event(self, event: Event) -> None:
        """Handle error events from components."""
        data = event.data
        component = data.get("component", "unknown")
        error = data.get("error", "")

        logger.error(f"Error event from {component}: {error}")

        # Update component health
        if component in self._health:
            self._health[component].failure_count += 1
            self._health[component].error_message = error

            # Check if component should be marked as degraded/failed
            if self._health[component].failure_count >= 3:
                self._health[component].state = ComponentState.DEGRADED
                logger.warning(f"Component {component} marked as DEGRADED")

    async def get_metrics(self) -> dict[str, Any]:
        """Get orchestrator metrics for monitoring.

        Returns:
            Metrics dict for Prometheus/Grafana
        """
        health = await self.health_check()

        # Calculate metrics
        total_components = len(self._components)
        healthy_components = sum(
            1 for h in self._health.values() if h.state == ComponentState.HEALTHY
        )
        degraded_components = sum(
            1 for h in self._health.values() if h.state == ComponentState.DEGRADED
        )
        failed_components = sum(
            1 for h in self._health.values() if h.state == ComponentState.FAILED
        )

        open_circuits = sum(
            1
            for cb in self._circuit_breakers.values()
            if cb.state == CircuitState.OPEN
        )

        return {
            "orchestrator_uptime_seconds": health["orchestrator"]["uptime_seconds"],
            "components_total": total_components,
            "components_healthy": healthy_components,
            "components_degraded": degraded_components,
            "components_failed": failed_components,
            "circuit_breakers_open": open_circuits,
            "event_bus_queue_size": health["event_bus"]["queue_size"],
            "event_bus_queue_max": health["event_bus"]["max_queue_size"],
        }
