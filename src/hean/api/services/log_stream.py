"""SSE log stream service."""

import asyncio
import json
import logging
from collections import deque
from datetime import datetime
from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse

from hean.logging import get_logger

logger = get_logger(__name__)


class LogStreamHandler(logging.Handler):
    """Custom logging handler for streaming logs via SSE."""

    def __init__(self, max_buffer_size: int = 2000) -> None:
        """Initialize log stream handler."""
        super().__init__()
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._log_buffer: deque[dict[str, Any]] = deque(maxlen=max_buffer_size)
        self._lock = asyncio.Lock()

    async def emit(self, record: logging.LogRecord) -> None:
        """Emit log record to subscribers."""
        log_data = {
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "module": record.module,
            "request_id": getattr(record, "request_id", None),
        }

        # Add to buffer
        self._log_buffer.append(log_data)

        # Send to all subscribers
        async with self._lock:
            dead_queues = []
            for queue in self._subscribers:
                try:
                    await queue.put(log_data)
                except Exception as e:
                    logger.warning(f"Failed to send log to subscriber: {e}")
                    dead_queues.append(queue)

            # Remove dead queues
            for queue in dead_queues:
                self._subscribers.discard(queue)

    def add_subscriber(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Add subscriber queue."""
        self._subscribers.add(queue)

    def remove_subscriber(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove subscriber queue."""
        self._subscribers.discard(queue)

    def get_recent_logs(self, count: int = 100) -> list[dict[str, Any]]:
        """Get recent logs from buffer."""
        return list(self._log_buffer)[-count:]


class LogStreamService:
    """Service for streaming logs via SSE."""

    def __init__(self) -> None:
        """Initialize log stream service."""
        self._handler = LogStreamHandler()
        self._handler.setLevel(logging.INFO)

    def setup(self) -> None:
        """Setup log stream handler."""
        root_logger = logging.getLogger()
        root_logger.addHandler(self._handler)

    async def stream(self, request: Request) -> StreamingResponse:
        """Create SSE stream for logs."""
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def generate() -> Any:
            """Generate SSE messages."""
            # Send initial buffer (last 100 logs)
            recent_logs = self._handler.get_recent_logs(100)
            for log_data in recent_logs:
                yield f"data: {json.dumps(log_data)}\n\n"

            # Add to subscribers
            self._handler.add_subscriber(queue)

            try:
                while True:
                    # Check if client disconnected
                    if await request.is_disconnected():
                        break

                    try:
                        # Wait for log with timeout
                        log_data = await asyncio.wait_for(queue.get(), timeout=30.0)
                        if log_data is None:  # Close signal
                            break

                        yield f"data: {json.dumps(log_data)}\n\n"
                    except TimeoutError:
                        # Send heartbeat
                        yield ": heartbeat\n\n"
                    except Exception as e:
                        logger.error(f"Error in log stream: {e}")
                        break
            finally:
                self._handler.remove_subscriber(queue)

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
log_stream_service = LogStreamService()

