"""SSE event stream service."""

import asyncio
import json
from collections import deque
from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class EventStreamService:
    """Service for streaming events via SSE."""

    def __init__(self, bus: EventBus | None = None) -> None:
        """Initialize event stream service."""
        self._bus = bus
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._event_buffer: deque[dict[str, Any]] = deque(maxlen=5000)  # Ring buffer
        self._running = False
        self._task: asyncio.Task[None] | None = None

    def set_bus(self, bus: EventBus) -> None:
        """Set event bus."""
        self._bus = bus

    async def start(self) -> None:
        """Start event stream service."""
        if self._running:
            return
        self._running = True
        if self._bus:
            # Subscribe to all event types
            self._bus.subscribe(EventType.TICK, self._handle_event)
            self._bus.subscribe(EventType.SIGNAL, self._handle_event)
            self._bus.subscribe(EventType.ORDER_REQUEST, self._handle_event)
            self._bus.subscribe(EventType.ORDER_PLACED, self._handle_event)
            self._bus.subscribe(EventType.ORDER_FILLED, self._handle_event)
            self._bus.subscribe(EventType.ORDER_CANCELLED, self._handle_event)
            self._bus.subscribe(EventType.POSITION_OPENED, self._handle_event)
            self._bus.subscribe(EventType.POSITION_CLOSED, self._handle_event)
            self._bus.subscribe(EventType.RISK_BLOCKED, self._handle_event)
            self._bus.subscribe(EventType.KILLSWITCH_TRIGGERED, self._handle_event)
            self._bus.subscribe(EventType.ERROR, self._handle_event)
        logger.info("Event stream service started")

    async def stop(self) -> None:
        """Stop event stream service."""
        if not self._running:
            return
        self._running = False
        # Unsubscribe from bus
        if self._bus:
            self._bus.unsubscribe(EventType.TICK, self._handle_event)
            self._bus.unsubscribe(EventType.SIGNAL, self._handle_event)
            self._bus.unsubscribe(EventType.ORDER_REQUEST, self._handle_event)
            self._bus.unsubscribe(EventType.ORDER_PLACED, self._handle_event)
            self._bus.unsubscribe(EventType.ORDER_FILLED, self._handle_event)
            self._bus.unsubscribe(EventType.POSITION_OPENED, self._handle_event)
            self._bus.unsubscribe(EventType.POSITION_CLOSED, self._handle_event)
            self._bus.unsubscribe(EventType.RISK_BLOCKED, self._handle_event)
            self._bus.unsubscribe(EventType.KILLSWITCH_TRIGGERED, self._handle_event)
            self._bus.unsubscribe(EventType.ERROR, self._handle_event)
        # Close all subscriber queues
        for queue in list(self._subscribers):
            await queue.put(None)  # Signal close
        self._subscribers.clear()
        logger.info("Event stream service stopped")

    async def _handle_event(self, event: Event) -> None:
        """Handle event from bus."""
        if not self._running:
            return

        event_data = {
            "event": event.event_type.value,
            "data": event.data,
            "timestamp": event.timestamp.isoformat(),
        }

        # Add to buffer
        self._event_buffer.append(event_data)

        # Send to all subscribers
        dead_queues = []
        for queue in self._subscribers:
            try:
                await queue.put(event_data)
            except Exception as e:
                logger.warning(f"Failed to send event to subscriber: {e}")
                dead_queues.append(queue)

        # Remove dead queues
        for queue in dead_queues:
            self._subscribers.discard(queue)

    async def stream(self, request: Request) -> StreamingResponse:
        """Create SSE stream for client."""
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def generate() -> Any:
            """Generate SSE messages."""
            # Send initial buffer (last 100 events)
            buffer_events = list(self._event_buffer)[-100:]
            for event_data in buffer_events:
                yield f"data: {json.dumps(event_data)}\n\n"

            # Add to subscribers
            self._subscribers.add(queue)

            try:
                while self._running:
                    # Check if client disconnected
                    if await request.is_disconnected():
                        break

                    try:
                        # Wait for event with timeout
                        event_data = await asyncio.wait_for(queue.get(), timeout=30.0)
                        if event_data is None:  # Close signal
                            break

                        yield f"data: {json.dumps(event_data)}\n\n"
                    except TimeoutError:
                        # Send heartbeat
                        yield ": heartbeat\n\n"
                    except Exception as e:
                        logger.error(f"Error in event stream: {e}")
                        break
            finally:
                self._subscribers.discard(queue)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


# Global instance
event_stream_service = EventStreamService()

