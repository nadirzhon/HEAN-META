"""Structured logging setup with request_id/trace_id support."""

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

from hean.config import settings

# Context variables for request_id and trace_id
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)


class RequestIDFilter(logging.Filter):
    """Logging filter to add request_id and trace_id to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request_id and trace_id to log record."""
        record.request_id = request_id_var.get() or "N/A"
        record.trace_id = trace_id_var.get() or "N/A"
        return True


def setup_logging() -> None:
    """Configure structured logging for the application."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Create formatter with request_id and trace_id
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | [request_id=%(request_id)s] [trace_id=%(trace_id)s] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestIDFilter())

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Suppress noisy loggers
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def set_request_id(request_id: str | None) -> None:
    """Set request_id in context."""
    request_id_var.set(request_id)


def set_trace_id(trace_id: str | None) -> None:
    """Set trace_id in context."""
    trace_id_var.set(trace_id)


def get_request_id() -> str | None:
    """Get current request_id from context."""
    return request_id_var.get()


def get_trace_id() -> str | None:
    """Get current trace_id from context."""
    return trace_id_var.get()


def generate_request_id() -> str:
    """Generate a new request_id."""
    return str(uuid.uuid4())


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(name)


def log_exception(
    logger: logging.Logger, exc: Exception, context: dict[str, Any] | None = None
) -> None:
    """Log an exception with context."""
    context_str = f" | Context: {context}" if context else ""
    logger.error(f"Exception: {exc}{context_str}", exc_info=True)
