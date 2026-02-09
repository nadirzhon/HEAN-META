"""Retry utilities with exponential backoff."""

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


async def retry_with_backoff(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> T:
    """
    Retry async function with exponential backoff.

    Args:
        func: Async function to retry
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exceptions to catch and retry
        **kwargs: Keyword arguments for func

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        except exceptions as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(
                    f"All {max_retries} retry attempts failed for {func.__name__}",
                    exc_info=True,
                    extra={
                        "function": func.__name__,
                        "attempts": attempt + 1,
                        "error": str(e),
                    },
                )
                raise

            # Calculate delay with exponential backoff
            delay = min(
                initial_delay * (exponential_base**attempt),
                max_delay,
            )

            logger.warning(
                f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                f"after {delay:.2f}s delay: {e}",
                extra={
                    "function": func.__name__,
                    "attempt": attempt + 1,
                    "delay": delay,
                    "error": str(e),
                },
            )

            await asyncio.sleep(delay)

    # Should never reach here, but for type safety
    raise last_exception or RuntimeError("Retry failed without exception")


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
):
    """
    Decorator for adding retry logic to async functions.

    Example:
        @with_retry(max_retries=5, initial_delay=2.0)
        async def fetch_data():
            response = await http_client.get("/api/data")
            return response.json()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_with_backoff(
                func,
                *args,
                max_retries=max_retries,
                initial_delay=initial_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                exceptions=exceptions,
                **kwargs,
            )

        return wrapper

    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold exceeded, requests fail fast
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection."""
        import time

        if self.state == "OPEN":
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time > self.recovery_timeout
            ):
                # Try to recover
                self.state = "HALF_OPEN"
                logger.info(
                    f"Circuit breaker HALF_OPEN for {func.__name__}, testing recovery"
                )
            else:
                raise RuntimeError(
                    f"Circuit breaker OPEN for {func.__name__}, "
                    f"failing fast (failures: {self.failure_count})"
                )

        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

            # Success - reset circuit breaker
            if self.state == "HALF_OPEN":
                logger.info(f"Circuit breaker CLOSED for {func.__name__}, service recovered")
            self.state = "CLOSED"
            self.failure_count = 0
            self.last_failure_time = None

            return result

        except self.expected_exception:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(
                    f"Circuit breaker OPEN for {func.__name__} "
                    f"after {self.failure_count} failures",
                    extra={
                        "function": func.__name__,
                        "failures": self.failure_count,
                        "threshold": self.failure_threshold,
                    },
                )

            raise
