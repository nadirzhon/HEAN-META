"""Async event bus for event-driven architecture.

Enhanced with:
- Multi-priority queues (CRITICAL, NORMAL, LOW)
- Health monitoring and circuit breaker
- Adaptive backpressure
- Metrics for observability
"""

import asyncio
import concurrent.futures
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)

# Fast-path events that bypass the queue for minimal latency
FAST_PATH_EVENTS = {EventType.SIGNAL, EventType.ORDER_REQUEST, EventType.ORDER_FILLED}


class EventPriority(Enum):
    """Event priority levels for queue routing."""
    CRITICAL = 0  # ORDER_*, SIGNAL, POSITION_* - never drop
    NORMAL = 1    # PNL_UPDATE, REGIME, STATUS - wait briefly
    LOW = 2       # TICK, HEARTBEAT - drop under pressure


# Map event types to priorities
EVENT_PRIORITY_MAP: dict[EventType, EventPriority] = {
    # Critical - trading operations
    EventType.SIGNAL: EventPriority.CRITICAL,
    EventType.ORDER_REQUEST: EventPriority.CRITICAL,
    EventType.ORDER_FILLED: EventPriority.CRITICAL,
    EventType.ORDER_CANCELLED: EventPriority.CRITICAL,
    EventType.ORDER_REJECTED: EventPriority.CRITICAL,
    EventType.POSITION_OPENED: EventPriority.CRITICAL,
    EventType.POSITION_CLOSED: EventPriority.CRITICAL,
    EventType.RISK_ALERT: EventPriority.CRITICAL,
    # Normal - important but can wait
    EventType.PNL_UPDATE: EventPriority.NORMAL,
    EventType.REGIME_UPDATE: EventPriority.NORMAL,
    EventType.STATUS: EventPriority.NORMAL,
    EventType.FUNDING_UPDATE: EventPriority.NORMAL,
    EventType.CANDLE: EventPriority.NORMAL,
    # Low - high volume, can drop
    EventType.TICK: EventPriority.LOW,
    EventType.HEARTBEAT: EventPriority.LOW,
    EventType.COUNCIL_REVIEW: EventPriority.LOW,
    EventType.COUNCIL_RECOMMENDATION: EventPriority.LOW,
}


@dataclass
class BusHealthStatus:
    """Health status of the EventBus."""
    is_healthy: bool = True
    is_degraded: bool = False
    is_circuit_open: bool = False
    queue_utilization_pct: float = 0.0
    events_per_second: float = 0.0
    drop_rate_pct: float = 0.0
    avg_processing_time_ms: float = 0.0
    last_error: str | None = None
    last_check_time: float = field(default_factory=time.time)


class EventBus:
    """Async event bus for pub/sub communication.

    Enhanced features:
    - Multi-priority queues for different event types
    - Circuit breaker to prevent cascade failures
    - Health monitoring with degraded/unhealthy states
    - Adaptive backpressure based on queue utilization
    """

    # Circuit breaker thresholds
    CIRCUIT_OPEN_THRESHOLD = 0.95  # Open circuit at 95% queue utilization
    CIRCUIT_CLOSE_THRESHOLD = 0.70  # Close circuit at 70% utilization
    DEGRADED_THRESHOLD = 0.80  # Mark as degraded at 80% utilization

    def __init__(self, max_queue_size: int = 50000) -> None:
        """Initialize the event bus.

        Args:
            max_queue_size: Maximum queue size to prevent memory leaks (default 50000)
        """
        self._subscribers: dict[EventType, list[Callable[[Event], Any]]] = defaultdict(list)

        # Multi-priority queues
        self._critical_queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size // 5)
        self._normal_queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size // 2)
        self._low_queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)

        # Legacy queue for backwards compatibility
        self._queue: asyncio.Queue[Event] = self._normal_queue

        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._priority_tasks: list[asyncio.Task[None]] = []
        self._batch_size = 10  # Process up to 10 events in a batch
        self._batch_timeout = 0.01  # 10ms timeout for batching

        # CRITICAL: Thread pool for sync handlers to prevent blocking event loop
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="EventBus-SyncHandler"
        )

        # Circuit breaker state
        self._circuit_open = False
        self._circuit_open_time: float | None = None

        # Health monitoring
        self._health = BusHealthStatus()
        self._last_health_check = time.time()
        self._health_check_interval = 5.0  # Check every 5 seconds

        # Rate tracking for metrics
        self._events_last_second: list[float] = []
        self._processing_times: list[float] = []
        self._max_processing_samples = 100

        # Metrics for monitoring overflow
        self._metrics = {
            "events_published": 0,
            "events_dropped": 0,
            "events_delayed": 0,
            "events_processed": 0,
            "handler_errors": 0,
            "fast_path_dispatched": 0,
            "queued_dispatched": 0,
            "critical_queue_published": 0,
            "normal_queue_published": 0,
            "low_queue_published": 0,
            "circuit_breaker_trips": 0,
        }

    def _get_queue_for_event(self, event: Event) -> tuple[asyncio.Queue[Event], EventPriority]:
        """Get the appropriate queue for an event based on its priority."""
        priority = EVENT_PRIORITY_MAP.get(event.event_type, EventPriority.NORMAL)

        if priority == EventPriority.CRITICAL:
            return self._critical_queue, priority
        elif priority == EventPriority.LOW:
            return self._low_queue, priority
        else:
            return self._normal_queue, priority

    def _get_total_queue_utilization(self) -> float:
        """Calculate total queue utilization across all priority queues."""
        total_size = (
            self._critical_queue.qsize() +
            self._normal_queue.qsize() +
            self._low_queue.qsize()
        )
        total_capacity = (
            self._critical_queue.maxsize +
            self._normal_queue.maxsize +
            self._low_queue.maxsize
        )
        return total_size / total_capacity if total_capacity > 0 else 0.0

    def _update_health_status(self) -> None:
        """Update health status based on current metrics."""
        now = time.time()
        if now - self._last_health_check < self._health_check_interval:
            return

        self._last_health_check = now
        utilization = self._get_total_queue_utilization()

        # Calculate events per second
        cutoff = now - 1.0
        self._events_last_second = [t for t in self._events_last_second if t > cutoff]
        events_per_second = len(self._events_last_second)

        # Calculate drop rate
        total_attempted = self._metrics["events_published"] + self._metrics["events_dropped"]
        drop_rate = (self._metrics["events_dropped"] / total_attempted * 100) if total_attempted > 0 else 0.0

        # Calculate avg processing time
        avg_processing_ms = 0.0
        if self._processing_times:
            avg_processing_ms = sum(self._processing_times) / len(self._processing_times) * 1000

        # Update health status
        self._health = BusHealthStatus(
            is_healthy=utilization < self.DEGRADED_THRESHOLD and not self._circuit_open,
            is_degraded=self.DEGRADED_THRESHOLD <= utilization < self.CIRCUIT_OPEN_THRESHOLD,
            is_circuit_open=self._circuit_open,
            queue_utilization_pct=utilization * 100,
            events_per_second=events_per_second,
            drop_rate_pct=drop_rate,
            avg_processing_time_ms=avg_processing_ms,
            last_check_time=now,
        )

        # Log warnings for degraded state
        if self._health.is_degraded:
            logger.warning(
                f"[EventBus] DEGRADED: queue utilization {utilization:.1%}, "
                f"drop rate {drop_rate:.1%}, {events_per_second:.0f} events/sec"
            )

        # Check circuit breaker
        if not self._circuit_open and utilization >= self.CIRCUIT_OPEN_THRESHOLD:
            self._circuit_open = True
            self._circuit_open_time = now
            self._metrics["circuit_breaker_trips"] += 1
            logger.error(
                f"[EventBus] CIRCUIT BREAKER OPEN: queue utilization {utilization:.1%}. "
                f"Dropping LOW priority events until utilization drops below {self.CIRCUIT_CLOSE_THRESHOLD:.0%}"
            )
        elif self._circuit_open and utilization <= self.CIRCUIT_CLOSE_THRESHOLD:
            self._circuit_open = False
            duration = now - (self._circuit_open_time or now)
            logger.info(
                f"[EventBus] Circuit breaker closed after {duration:.1f}s. "
                f"Queue utilization now {utilization:.1%}"
            )

    def get_health(self) -> BusHealthStatus:
        """Get current health status."""
        self._update_health_status()
        return self._health

    def subscribe(self, event_type: EventType, handler: Callable[[Event], Any]) -> None:
        """Subscribe a handler to an event type."""
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed handler {handler.__name__} to {event_type}")

    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], Any]) -> None:
        """Unsubscribe a handler from an event type."""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            logger.debug(f"Unsubscribed handler from {event_type}")

    async def _dispatch_fast(self, event: Event) -> None:
        """Dispatch event immediately without queueing (fast-path).

        Used for time-critical events to minimize latency.
        """
        logger.debug(f"Fast-path dispatch: {event.event_type}")
        await self._dispatch(event)
        self._metrics["fast_path_dispatched"] += 1
        self._metrics["events_processed"] += 1

    async def publish(self, event: Event) -> None:
        """Publish an event to the bus with priority-based routing.

        Time-critical events (SIGNAL, ORDER_REQUEST, ORDER_FILLED) bypass the queue
        and are dispatched immediately for minimal latency.

        Priority routing:
        - CRITICAL: Trading events - never dropped, wait if needed
        - NORMAL: Important events - wait briefly, then fail
        - LOW: High-volume events (TICK) - drop under pressure

        Circuit breaker: When queue utilization exceeds 95%, LOW priority events
        are dropped until utilization drops below 70%.
        """
        # Track event rate
        self._events_last_second.append(time.time())

        # Update health status periodically
        self._update_health_status()

        # Fast-path for time-critical events
        if event.event_type in FAST_PATH_EVENTS:
            logger.debug(f"Publishing {event.event_type} event via fast-path")
            await self._dispatch_fast(event)
            return

        # Get appropriate queue based on priority
        queue, priority = self._get_queue_for_event(event)

        logger.debug(f"Publishing {event.event_type} event to {priority.name} queue")

        # Circuit breaker: drop LOW priority when circuit is open
        if self._circuit_open and priority == EventPriority.LOW:
            self._metrics["events_dropped"] += 1
            # Only log every 100th drop to avoid log spam
            if self._metrics["events_dropped"] % 100 == 1:
                logger.warning(
                    f"[EventBus] Circuit open - dropping {event.event_type}. "
                    f"Total dropped: {self._metrics['events_dropped']}"
                )
            return

        try:
            # Try non-blocking first
            queue.put_nowait(event)
            self._metrics["events_published"] += 1
            self._metrics[f"{priority.name.lower()}_queue_published"] += 1
        except asyncio.QueueFull:
            queue_size = queue.qsize()

            # Handle based on priority
            if priority == EventPriority.LOW:
                # Drop low-value events instead of killing the loop
                self._metrics["events_dropped"] += 1
                if self._metrics["events_dropped"] % 100 == 1:
                    logger.warning(
                        f"[EventBus] {priority.name} queue full ({queue_size}/{queue.maxsize}). "
                        f"Dropping {event.event_type}. Total dropped: {self._metrics['events_dropped']}"
                    )
                return

            elif priority == EventPriority.CRITICAL:
                # CRITICAL events: wait indefinitely (up to 5 seconds)
                try:
                    await asyncio.wait_for(queue.put(event), timeout=5.0)
                    self._metrics["events_delayed"] += 1
                    self._metrics["events_published"] += 1
                    self._metrics["critical_queue_published"] += 1
                    logger.warning(
                        f"[EventBus] CRITICAL queue was full. Backpressured {event.event_type}. "
                        f"Delayed: {self._metrics['events_delayed']} total"
                    )
                except TimeoutError as e:
                    # CRITICAL events should never be dropped - this is a system failure
                    self._health.last_error = f"CRITICAL event {event.event_type} could not be published"
                    logger.error(
                        f"[EventBus] FATAL: Cannot publish CRITICAL event {event.event_type} after 5s wait. "
                        f"System is severely overloaded!"
                    )
                    raise RuntimeError(
                        f"EventBus CRITICAL queue full for 5+ seconds. "
                        f"Event processing is severely behind. Event: {event.event_type}"
                    ) from e

            else:  # NORMAL priority
                # Wait briefly before giving up
                try:
                    await asyncio.wait_for(queue.put(event), timeout=1.0)
                    self._metrics["events_delayed"] += 1
                    self._metrics["events_published"] += 1
                    self._metrics["normal_queue_published"] += 1
                    logger.warning(
                        f"[EventBus] NORMAL queue was full. Backpressured {event.event_type}. "
                        f"Delayed: {self._metrics['events_delayed']} total"
                    )
                except TimeoutError:
                    self._metrics["events_dropped"] += 1
                    logger.error(
                        f"[EventBus] NORMAL queue full ({queue_size}/{queue.maxsize}). "
                        f"Dropping {event.event_type} after 1s wait. "
                        f"Dropped: {self._metrics['events_dropped']} total"
                    )

    async def _process_events(self) -> None:
        """Process events from all priority queues.

        Priority order: CRITICAL > NORMAL > LOW
        Critical events are always processed first.
        """
        while self._running:
            try:
                batch: list[Event] = []
                start_time = time.time()

                # Priority 1: Process all critical events first
                while not self._critical_queue.empty() and len(batch) < self._batch_size:
                    try:
                        event = self._critical_queue.get_nowait()
                        batch.append(event)
                    except asyncio.QueueEmpty:
                        break

                # Priority 2: Then normal events
                while not self._normal_queue.empty() and len(batch) < self._batch_size:
                    try:
                        event = self._normal_queue.get_nowait()
                        batch.append(event)
                    except asyncio.QueueEmpty:
                        break

                # Priority 3: Then low priority events (only if queues not too full)
                if not self._circuit_open:
                    while not self._low_queue.empty() and len(batch) < self._batch_size:
                        try:
                            event = self._low_queue.get_nowait()
                            batch.append(event)
                        except asyncio.QueueEmpty:
                            break

                # If no events, wait for any queue
                if not batch:
                    try:
                        # Wait on all queues using asyncio.wait
                        wait_tasks = [
                            asyncio.create_task(self._critical_queue.get()),
                            asyncio.create_task(self._normal_queue.get()),
                        ]
                        if not self._circuit_open:
                            wait_tasks.append(asyncio.create_task(self._low_queue.get()))

                        done, pending = await asyncio.wait(
                            wait_tasks,
                            timeout=1.0,
                            return_when=asyncio.FIRST_COMPLETED
                        )

                        # Cancel pending tasks
                        for task in pending:
                            task.cancel()
                            try:
                                await task
                            except (asyncio.CancelledError, asyncio.QueueEmpty):
                                pass

                        # Get completed events
                        for task in done:
                            try:
                                event = task.result()
                                batch.append(event)
                            except (asyncio.CancelledError, asyncio.QueueEmpty):
                                pass

                    except TimeoutError:
                        continue

                # Dispatch all events in batch concurrently
                if batch:
                    tasks = [self._dispatch(event) for event in batch]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    self._metrics["events_processed"] += len(batch)
                    self._metrics["queued_dispatched"] += len(batch)

                    # Track processing time
                    processing_time = time.time() - start_time
                    self._processing_times.append(processing_time)
                    if len(self._processing_times) > self._max_processing_samples:
                        self._processing_times.pop(0)

            except asyncio.CancelledError:
                # CRITICAL FIX: CancelledError must be re-raised, not caught
                # If we catch it, the task won't actually cancel and the event loop stays alive
                raise
            except Exception as e:
                logger.error(f"Error processing events: {e}", exc_info=True)
                self._health.last_error = str(e)
                # Publish error event
                error_event = Event(
                    event_type=EventType.ERROR,
                    data={"error": str(e), "exception_type": type(e).__name__},
                )
                await self._dispatch(error_event)

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to all subscribers."""
        handlers = self._subscribers.get(event.event_type, [])
        if not handlers:
            logger.warning(f"No subscribers for {event.event_type}")
            if event.event_type in (
                EventType.ORDER_REQUEST,
                EventType.SIGNAL,
                EventType.ORDER_FILLED,
            ):
                logger.error(f"CRITICAL: {event.event_type} has no subscribers!")
            return

        logger.debug(f"Dispatching {event.event_type} to {len(handlers)} handlers")
        # Dispatch to all handlers concurrently
        tasks = [self._safe_call_handler(handler, event) for handler in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any exceptions from handlers
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._metrics["handler_errors"] += 1
                logger.error(
                    f"Handler {handlers[i].__name__} raised exception for {event.event_type}: {result}",
                    exc_info=result,
                )

    async def _safe_call_handler(self, handler: Callable[[Event], Any], event: Event) -> None:
        """Safely call a handler, catching exceptions.

        CRITICAL: Sync handlers are executed in thread pool to prevent blocking event loop.
        """
        try:
            logger.debug(f"Calling handler {handler.__name__} for {event.event_type}")
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                # CRITICAL: Sync handlers must run in thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self._executor, handler, event)
            logger.debug(f"Handler {handler.__name__} completed successfully")
        except Exception as e:
            logger.error(
                f"Handler {handler.__name__} raised exception for {event.event_type}: {e}",
                exc_info=True,
            )

    async def start(self) -> None:
        """Start the event bus."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")

    async def flush(self, max_events: int | None = None) -> int:
        """Process all remaining events from all priority queues.

        Processes in priority order: CRITICAL > NORMAL > LOW

        Args:
            max_events: Maximum number of events to process. If None, process all.

        Returns:
            Number of events processed.
        """
        processed = 0
        queues = [self._critical_queue, self._normal_queue, self._low_queue]

        for queue in queues:
            while self._running:
                if max_events is not None and processed >= max_events:
                    return processed
                try:
                    event = queue.get_nowait()
                    await self._dispatch(event)
                    processed += 1
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    logger.error(f"Error processing event during flush: {e}", exc_info=True)

        return processed

    async def stop(self) -> None:
        """Stop the event bus.

        BUG FIX: Previously, this method set _running=False and awaited the task,
        but the task's while loop checks _running after a timeout, so it would
        continue running until the next timeout. We must CANCEL the task explicitly.

        PRINCIPLE: Async tasks with infinite loops must be cancelled, not just
        awaited, when shutting down. Setting a flag is not sufficient if the
        task is blocked in a timeout operation.
        """
        if not self._running:
            logger.debug("[EventBus] Already stopped")
            return

        logger.info("[EventBus] Stopping...")
        self._running = False

        # Cancel the processing task explicitly
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("[EventBus] Processing task cancelled")
            except Exception as e:
                logger.error(f"[EventBus] Error stopping task: {e}", exc_info=True)

        # Drain any remaining events from ALL priority queues
        total_drained = 0
        for queue_name, queue in [
            ("critical", self._critical_queue),
            ("normal", self._normal_queue),
            ("low", self._low_queue),
        ]:
            drained = 0
            while not queue.empty():
                try:
                    queue.get_nowait()
                    drained += 1
                except asyncio.QueueEmpty:
                    break
            if drained > 0:
                logger.debug(f"[EventBus] Drained {drained} events from {queue_name} queue")
            total_drained += drained

        if total_drained > 0:
            logger.info(f"[EventBus] Drained {total_drained} total events from all queues")

        # Shutdown thread pool executor
        self._executor.shutdown(wait=True)

        # Log final health status
        logger.info(
            f"[EventBus] Stopped. Final metrics: "
            f"published={self._metrics['events_published']}, "
            f"processed={self._metrics['events_processed']}, "
            f"dropped={self._metrics['events_dropped']}, "
            f"circuit_trips={self._metrics['circuit_breaker_trips']}"
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get current EventBus metrics for monitoring.

        Returns:
            Dictionary with metrics:
            - events_published: Total events successfully published
            - events_dropped: Events dropped due to queue overflow
            - events_delayed: Events delayed due to backpressure
            - events_processed: Events processed by handlers
            - handler_errors: Number of handler exceptions
            - queue sizes and utilization per priority
            - circuit breaker state
            - health status
        """
        health = self.get_health()

        return {
            **self._metrics,
            # Legacy compatibility
            "queue_size": self._normal_queue.qsize(),
            "queue_capacity": self._normal_queue.maxsize,
            # Per-priority queue stats
            "critical_queue_size": self._critical_queue.qsize(),
            "critical_queue_capacity": self._critical_queue.maxsize,
            "normal_queue_size": self._normal_queue.qsize(),
            "normal_queue_capacity": self._normal_queue.maxsize,
            "low_queue_size": self._low_queue.qsize(),
            "low_queue_capacity": self._low_queue.maxsize,
            # Overall utilization
            "total_queue_utilization_pct": self._get_total_queue_utilization() * 100,
            # Circuit breaker
            "circuit_breaker_open": self._circuit_open,
            # Health
            "is_healthy": health.is_healthy,
            "is_degraded": health.is_degraded,
            "events_per_second": health.events_per_second,
            "drop_rate_pct": health.drop_rate_pct,
            "avg_processing_time_ms": health.avg_processing_time_ms,
        }

    async def events(self, event_type: EventType) -> AsyncIterator[Event]:
        """Create an async iterator for events of a specific type."""
        queue: asyncio.Queue[Event] = asyncio.Queue()

        async def handler(event: Event) -> None:
            await queue.put(event)

        self.subscribe(event_type, handler)
        try:
            while self._running:
                event = await queue.get()
                yield event
        finally:
            self.unsubscribe(event_type, handler)
