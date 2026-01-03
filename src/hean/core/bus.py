"""Async event bus for event-driven architecture."""

import asyncio
import concurrent.futures
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from typing import Any

from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class EventBus:
    """Async event bus for pub/sub communication."""

    def __init__(self, max_queue_size: int = 10000) -> None:
        """Initialize the event bus.
        
        Args:
            max_queue_size: Maximum queue size to prevent memory leaks (default 10000)
        """
        self._subscribers: dict[EventType, list[Callable[[Event], Any]]] = defaultdict(list)
        # CRITICAL: Limit queue size to prevent memory leaks
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._batch_size = 10  # Process up to 10 events in a batch
        self._batch_timeout = 0.01  # 10ms timeout for batching
        # CRITICAL: Thread pool for sync handlers to prevent blocking event loop
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="EventBus-SyncHandler"
        )

    def subscribe(self, event_type: EventType, handler: Callable[[Event], Any]) -> None:
        """Subscribe a handler to an event type."""
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed handler {handler.__name__} to {event_type}")

    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], Any]) -> None:
        """Unsubscribe a handler from an event type."""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            logger.debug(f"Unsubscribed handler from {event_type}")

    async def publish(self, event: Event) -> None:
        """Publish an event to the bus.
        
        CRITICAL: Uses put_nowait with timeout to prevent blocking while still protecting against memory leaks.
        If queue is full, waits briefly then raises error.
        """
        logger.debug(f"Publishing {event.event_type} event to queue")
        try:
            # Try non-blocking first
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Queue is full - this is a critical condition
            queue_size = self._queue.qsize()
            logger.error(
                f"EventBus queue is full ({queue_size}/{self._queue.maxsize} events). "
                f"Cannot publish {event.event_type} event. Event processing is severely behind."
            )
            # Raise error to prevent silent drops (critical for trading system)
            raise RuntimeError(
                f"EventBus queue full ({queue_size}/{self._queue.maxsize}). "
                "Event processing is falling behind. This indicates a serious performance issue. "
                "Check handler performance and consider increasing max_queue_size."
            )

    async def _process_events(self) -> None:
        """Process events from the queue with batching for performance."""
        while self._running:
            try:
                # Collect events in a batch
                batch: list[Event] = []

                # Get first event
                try:
                    first_event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                    batch.append(first_event)
                except TimeoutError:
                    continue

                # Collect more events up to batch_size or timeout
                batch_start = asyncio.get_event_loop().time()
                while len(batch) < self._batch_size and self._running:
                    try:
                        elapsed = asyncio.get_event_loop().time() - batch_start
                        if elapsed >= self._batch_timeout:
                            break

                        remaining_time = self._batch_timeout - elapsed
                        event = await asyncio.wait_for(self._queue.get(), timeout=remaining_time)
                        batch.append(event)
                    except TimeoutError:
                        break

                # Dispatch all events in batch concurrently
                if batch:
                    tasks = [self._dispatch(event) for event in batch]
                    await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.CancelledError:
                # CRITICAL FIX: CancelledError must be re-raised, not caught
                # If we catch it, the task won't actually cancel and the event loop stays alive
                raise
            except Exception as e:
                logger.error(f"Error processing events: {e}", exc_info=True)
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
        """Process all remaining events in the queue.

        Args:
            max_events: Maximum number of events to process. If None, process all.

        Returns:
            Number of events processed.
        """
        processed = 0
        while self._running:
            if max_events is not None and processed >= max_events:
                break
            try:
                event = self._queue.get_nowait()
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

        # Drain any remaining events in the queue to prevent memory leaks
        drained = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break

        if drained > 0:
            logger.debug(f"[EventBus] Drained {drained} remaining events from queue")

        # Shutdown thread pool executor
        self._executor.shutdown(wait=True, timeout=5.0)

        logger.info("[EventBus] Stopped")

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
