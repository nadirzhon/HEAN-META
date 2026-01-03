"""System metrics and counters."""

from collections import defaultdict
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)


class SystemMetrics:
    """Tracks system-level metrics and counters."""

    def __init__(self) -> None:
        """Initialize system metrics."""
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)

    def increment(self, metric: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self._counters[metric] += value

    def set_gauge(self, metric: str, value: float) -> None:
        """Set a gauge metric."""
        self._gauges[metric] = value

    def record_histogram(self, metric: str, value: float) -> None:
        """Record a histogram value."""
        self._histograms[metric].append(value)

    def get_counters(self) -> dict[str, int]:
        """Get all counter metrics."""
        return dict(self._counters)

    def get_gauges(self) -> dict[str, float]:
        """Get all gauge metrics."""
        return dict(self._gauges)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all metrics."""
        return {
            "counters": self.get_counters(),
            "gauges": self.get_gauges(),
            "histogram_counts": {k: len(v) for k, v in self._histograms.items()},
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


# Global metrics instance
metrics = SystemMetrics()
