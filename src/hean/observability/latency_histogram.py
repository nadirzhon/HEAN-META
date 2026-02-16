"""Latency Histogram with P99.9 percentile tracking and alerting.

Provides high-precision latency tracking for critical paths:
- API response times
- Order execution latency
- WebSocket message delivery
- Signal processing time
- Event bus latency
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)


class LatencyAlertLevel(str, Enum):
    """Latency alert severity levels."""

    OK = "ok"  # Within normal range
    WARNING = "warning"  # P99.9 > warning threshold
    CRITICAL = "critical"  # P99.9 > critical threshold


@dataclass
class LatencyAlert:
    """A latency alert event."""

    histogram_name: str
    level: LatencyAlertLevel
    p999_ms: float
    threshold_ms: float
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "histogram_name": self.histogram_name,
            "level": self.level.value,
            "p999_ms": round(self.p999_ms, 3),
            "threshold_ms": self.threshold_ms,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class LatencyStats:
    """Computed latency statistics."""

    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    p999_ms: float
    window_seconds: float
    alert_level: LatencyAlertLevel


class LatencyHistogram:
    """High-precision latency histogram with percentile tracking.

    Uses a rolling window to track recent latency samples and
    compute percentiles efficiently. Supports alerting when
    P99.9 exceeds configured thresholds.
    """

    # Prometheus histogram buckets (in milliseconds)
    DEFAULT_BUCKETS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

    def __init__(
        self,
        name: str,
        window_seconds: int = 300,  # 5 minute window
        max_samples: int = 10000,
        p999_warning_ms: float = 500.0,
        p999_critical_ms: float = 1000.0,
        buckets: list[float] | None = None,
    ) -> None:
        """Initialize latency histogram.

        Args:
            name: Histogram name (e.g., "api_latency", "order_execution")
            window_seconds: Rolling window size in seconds
            max_samples: Maximum samples to keep
            p999_warning_ms: P99.9 warning threshold in ms
            p999_critical_ms: P99.9 critical threshold in ms
            buckets: Custom bucket boundaries (ms) for Prometheus
        """
        self.name = name
        self.window_seconds = window_seconds
        self.max_samples = max_samples
        self.p999_warning_ms = p999_warning_ms
        self.p999_critical_ms = p999_critical_ms
        self.buckets = buckets or self.DEFAULT_BUCKETS

        # Rolling window: (timestamp_ns, latency_ms)
        self._samples: deque[tuple[int, float]] = deque(maxlen=max_samples)

        # Prometheus-style bucket counts
        self._bucket_counts: dict[float, int] = dict.fromkeys(self.buckets, 0)
        self._bucket_counts[float("inf")] = 0  # +Inf bucket

        # Running stats
        self._total_count = 0
        self._total_sum_ms = 0.0

        # Alert state
        self._current_alert_level = LatencyAlertLevel.OK
        self._last_alert_time: float = 0
        self._alert_cooldown_seconds = 60  # Don't spam alerts

        # Recent alerts
        self._recent_alerts: deque[LatencyAlert] = deque(maxlen=100)

    def record(self, latency_ms: float) -> None:
        """Record a latency sample.

        Args:
            latency_ms: Latency in milliseconds
        """
        now_ns = time.time_ns()
        self._samples.append((now_ns, latency_ms))

        # Update bucket counts
        for bucket in self.buckets:
            if latency_ms <= bucket:
                self._bucket_counts[bucket] += 1
                break
        else:
            self._bucket_counts[float("inf")] += 1

        # Update running totals
        self._total_count += 1
        self._total_sum_ms += latency_ms

        # Check for alerts (periodically)
        if self._total_count % 100 == 0:
            self._check_alert()

    def record_timing(self, start_time_ns: int) -> float:
        """Record timing from a start timestamp.

        Args:
            start_time_ns: Start time in nanoseconds

        Returns:
            Latency in milliseconds
        """
        latency_ms = (time.time_ns() - start_time_ns) / 1_000_000
        self.record(latency_ms)
        return latency_ms

    def _prune_old_samples(self) -> None:
        """Remove samples outside the window."""
        if not self._samples:
            return

        cutoff_ns = time.time_ns() - (self.window_seconds * 1_000_000_000)
        while self._samples and self._samples[0][0] < cutoff_ns:
            self._samples.popleft()

    def _get_window_samples(self) -> list[float]:
        """Get latency values within the window."""
        self._prune_old_samples()
        return [sample[1] for sample in self._samples]

    def percentile(self, p: float) -> float:
        """Calculate percentile from recent samples.

        Args:
            p: Percentile (0-100)

        Returns:
            Latency value at percentile (ms), or 0 if no samples
        """
        samples = self._get_window_samples()
        if not samples:
            return 0.0

        sorted_samples = sorted(samples)
        k = (len(sorted_samples) - 1) * p / 100.0
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_samples) else f

        if f == c:
            return sorted_samples[f]

        # Linear interpolation
        return sorted_samples[f] + (k - f) * (sorted_samples[c] - sorted_samples[f])

    def get_stats(self) -> LatencyStats:
        """Get comprehensive latency statistics.

        Returns:
            LatencyStats with all percentiles
        """
        samples = self._get_window_samples()

        if not samples:
            return LatencyStats(
                count=0,
                min_ms=0.0,
                max_ms=0.0,
                mean_ms=0.0,
                p50_ms=0.0,
                p90_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                p999_ms=0.0,
                window_seconds=self.window_seconds,
                alert_level=LatencyAlertLevel.OK,
            )

        sorted_samples = sorted(samples)
        total = sum(samples)

        return LatencyStats(
            count=len(samples),
            min_ms=sorted_samples[0],
            max_ms=sorted_samples[-1],
            mean_ms=total / len(samples),
            p50_ms=self._percentile_from_sorted(sorted_samples, 50),
            p90_ms=self._percentile_from_sorted(sorted_samples, 90),
            p95_ms=self._percentile_from_sorted(sorted_samples, 95),
            p99_ms=self._percentile_from_sorted(sorted_samples, 99),
            p999_ms=self._percentile_from_sorted(sorted_samples, 99.9),
            window_seconds=self.window_seconds,
            alert_level=self._current_alert_level,
        )

    def _percentile_from_sorted(self, sorted_samples: list[float], p: float) -> float:
        """Calculate percentile from pre-sorted samples."""
        if not sorted_samples:
            return 0.0

        k = (len(sorted_samples) - 1) * p / 100.0
        f = int(k)
        c = min(f + 1, len(sorted_samples) - 1)

        if f == c:
            return sorted_samples[f]

        return sorted_samples[f] + (k - f) * (sorted_samples[c] - sorted_samples[f])

    def _check_alert(self) -> None:
        """Check if P99.9 exceeds thresholds and emit alerts."""
        p999 = self.percentile(99.9)

        if p999 == 0:
            return

        now = time.time()
        new_level = LatencyAlertLevel.OK

        if p999 >= self.p999_critical_ms:
            new_level = LatencyAlertLevel.CRITICAL
        elif p999 >= self.p999_warning_ms:
            new_level = LatencyAlertLevel.WARNING

        # Check if we should emit an alert
        if new_level != LatencyAlertLevel.OK:
            if new_level != self._current_alert_level or (
                now - self._last_alert_time > self._alert_cooldown_seconds
            ):
                alert = LatencyAlert(
                    histogram_name=self.name,
                    level=new_level,
                    p999_ms=p999,
                    threshold_ms=(
                        self.p999_critical_ms
                        if new_level == LatencyAlertLevel.CRITICAL
                        else self.p999_warning_ms
                    ),
                    message=f"{self.name} P99.9 latency {p999:.1f}ms exceeds threshold",
                )
                self._recent_alerts.append(alert)
                self._last_alert_time = now

                log_func = logger.critical if new_level == LatencyAlertLevel.CRITICAL else logger.warning
                log_func(
                    "Latency alert: %s P99.9=%.1fms (threshold=%.1fms)",
                    self.name,
                    p999,
                    alert.threshold_ms,
                )

        self._current_alert_level = new_level

    def get_recent_alerts(self, limit: int = 10) -> list[LatencyAlert]:
        """Get recent latency alerts.

        Args:
            limit: Maximum alerts to return

        Returns:
            List of recent alerts
        """
        return list(self._recent_alerts)[-limit:]

    def get_prometheus_metrics(self) -> dict[str, Any]:
        """Get metrics in Prometheus-compatible format.

        Returns:
            Dict with histogram data for Prometheus
        """
        stats = self.get_stats()

        # Cumulative bucket counts for Prometheus
        cumulative_counts: dict[str, int] = {}
        running_total = 0
        for bucket in sorted(self.buckets):
            running_total += self._bucket_counts[bucket]
            cumulative_counts[f"le_{bucket}"] = running_total
        cumulative_counts["le_inf"] = self._total_count

        return {
            "name": self.name,
            "type": "histogram",
            "buckets": cumulative_counts,
            "sum": self._total_sum_ms,
            "count": self._total_count,
            "percentiles": {
                "p50": round(stats.p50_ms, 3),
                "p90": round(stats.p90_ms, 3),
                "p95": round(stats.p95_ms, 3),
                "p99": round(stats.p99_ms, 3),
                "p999": round(stats.p999_ms, 3),
            },
            "alert_level": self._current_alert_level.value,
        }

    def to_prometheus_text(self) -> str:
        """Export histogram in Prometheus text format.

        Returns:
            Prometheus exposition format string
        """
        lines = [
            f"# HELP hean_{self.name}_latency_ms Latency histogram for {self.name}",
            f"# TYPE hean_{self.name}_latency_ms histogram",
        ]

        # Cumulative bucket counts
        running_total = 0
        for bucket in sorted(self.buckets):
            running_total += self._bucket_counts[bucket]
            lines.append(f'hean_{self.name}_latency_ms_bucket{{le="{bucket}"}} {running_total}')
        lines.append(f'hean_{self.name}_latency_ms_bucket{{le="+Inf"}} {self._total_count}')
        lines.append(f"hean_{self.name}_latency_ms_sum {self._total_sum_ms}")
        lines.append(f"hean_{self.name}_latency_ms_count {self._total_count}")

        # Add summary percentiles as separate gauge
        stats = self.get_stats()
        lines.extend([
            f"# HELP hean_{self.name}_p999_ms P99.9 latency for {self.name}",
            f"# TYPE hean_{self.name}_p999_ms gauge",
            f"hean_{self.name}_p999_ms {round(stats.p999_ms, 3)}",
        ])

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Reset all histogram data."""
        self._samples.clear()
        self._bucket_counts = dict.fromkeys(self.buckets, 0)
        self._bucket_counts[float("inf")] = 0
        self._total_count = 0
        self._total_sum_ms = 0.0
        self._current_alert_level = LatencyAlertLevel.OK


class LatencyHistogramRegistry:
    """Registry for multiple latency histograms."""

    def __init__(self) -> None:
        """Initialize registry."""
        self._histograms: dict[str, LatencyHistogram] = {}

    def register(
        self,
        name: str,
        window_seconds: int = 300,
        p999_warning_ms: float = 500.0,
        p999_critical_ms: float = 1000.0,
    ) -> LatencyHistogram:
        """Register a new histogram or return existing.

        Args:
            name: Histogram name
            window_seconds: Rolling window size
            p999_warning_ms: P99.9 warning threshold
            p999_critical_ms: P99.9 critical threshold

        Returns:
            LatencyHistogram instance
        """
        if name not in self._histograms:
            self._histograms[name] = LatencyHistogram(
                name=name,
                window_seconds=window_seconds,
                p999_warning_ms=p999_warning_ms,
                p999_critical_ms=p999_critical_ms,
            )
        return self._histograms[name]

    def get(self, name: str) -> LatencyHistogram | None:
        """Get histogram by name."""
        return self._histograms.get(name)

    def record(self, name: str, latency_ms: float) -> None:
        """Record latency to a histogram, creating if needed.

        Args:
            name: Histogram name
            latency_ms: Latency in milliseconds
        """
        if name not in self._histograms:
            self._histograms[name] = LatencyHistogram(name=name)
        self._histograms[name].record(latency_ms)

    def get_all_stats(self) -> dict[str, LatencyStats]:
        """Get stats from all histograms.

        Returns:
            Dict mapping histogram name to stats
        """
        return {name: hist.get_stats() for name, hist in self._histograms.items()}

    def get_all_alerts(self, limit_per_histogram: int = 5) -> list[LatencyAlert]:
        """Get recent alerts from all histograms.

        Args:
            limit_per_histogram: Max alerts per histogram

        Returns:
            List of all recent alerts, sorted by timestamp
        """
        alerts = []
        for hist in self._histograms.values():
            alerts.extend(hist.get_recent_alerts(limit_per_histogram))
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all histograms for API response.

        Returns:
            Summary dict
        """
        histogram_summaries = {}
        for name, hist in self._histograms.items():
            stats = hist.get_stats()
            histogram_summaries[name] = {
                "count": stats.count,
                "mean_ms": round(stats.mean_ms, 3),
                "p50_ms": round(stats.p50_ms, 3),
                "p90_ms": round(stats.p90_ms, 3),
                "p95_ms": round(stats.p95_ms, 3),
                "p99_ms": round(stats.p99_ms, 3),
                "p999_ms": round(stats.p999_ms, 3),
                "min_ms": round(stats.min_ms, 3),
                "max_ms": round(stats.max_ms, 3),
                "alert_level": stats.alert_level.value,
            }

        # Check for any active alerts
        has_critical = any(
            h.get_stats().alert_level == LatencyAlertLevel.CRITICAL
            for h in self._histograms.values()
        )
        has_warning = any(
            h.get_stats().alert_level == LatencyAlertLevel.WARNING
            for h in self._histograms.values()
        )

        return {
            "histograms": histogram_summaries,
            "histogram_count": len(self._histograms),
            "overall_status": (
                "critical" if has_critical else "warning" if has_warning else "ok"
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def to_prometheus_text(self) -> str:
        """Export all histograms in Prometheus text format.

        Returns:
            Prometheus exposition format string
        """
        lines = []
        for hist in self._histograms.values():
            lines.append(hist.to_prometheus_text())
        return "\n".join(lines)


# Global registry instance
latency_histograms = LatencyHistogramRegistry()

# Pre-register common histograms with appropriate thresholds
latency_histograms.register(
    "api_response",
    p999_warning_ms=200.0,
    p999_critical_ms=500.0,
)
latency_histograms.register(
    "order_execution",
    p999_warning_ms=100.0,
    p999_critical_ms=300.0,
)
latency_histograms.register(
    "websocket_message",
    p999_warning_ms=50.0,
    p999_critical_ms=100.0,
)
latency_histograms.register(
    "signal_processing",
    p999_warning_ms=10.0,
    p999_critical_ms=50.0,
)
latency_histograms.register(
    "event_bus",
    p999_warning_ms=5.0,
    p999_critical_ms=20.0,
)
