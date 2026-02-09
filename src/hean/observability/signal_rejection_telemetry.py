"""Signal Rejection Telemetry - Detailed tracking of why signals are rejected.

Provides comprehensive telemetry for:
- Real-time rejection rate monitoring
- Rejection reason breakdown
- Time-series analysis of rejections
- Per-strategy and per-symbol rejection patterns
- Prometheus metrics integration
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)


class RejectionCategory(str, Enum):
    """Categories of signal rejections."""

    RISK = "risk"  # Risk limits, drawdown, Kelly, etc.
    REGIME = "regime"  # Regime gating
    EXECUTION = "execution"  # Maker edge, spread, volatility
    COOLDOWN = "cooldown"  # Cooldown periods
    FILTER = "filter"  # Strategy filters
    ANOMALY = "anomaly"  # Price anomaly detection
    ORACLE = "oracle"  # Oracle/prediction blocks
    OFI = "ofi"  # Order flow imbalance blocks
    MULTI_FACTOR = "multi_factor"  # Multi-factor confirmation blocks
    OTHER = "other"


@dataclass
class RejectionEvent:
    """A single rejection event with full context."""

    timestamp: datetime
    reason: str
    category: RejectionCategory
    symbol: str | None
    strategy_id: str | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RejectionStats:
    """Statistics for a time window."""

    total_rejections: int
    total_signals: int
    rejection_rate: float  # 0.0 to 1.0
    by_category: dict[str, int]
    by_reason: dict[str, int]
    by_symbol: dict[str, int]
    by_strategy: dict[str, int]
    time_window_minutes: int


class SignalRejectionTelemetry:
    """Comprehensive signal rejection tracking.

    Features:
    - Rolling window statistics (1m, 5m, 1h, 24h)
    - Per-reason, per-symbol, per-strategy breakdown
    - Recent rejection history for debugging
    - Prometheus metrics integration
    """

    def __init__(self, history_size: int = 1000) -> None:
        """Initialize the telemetry service.

        Args:
            history_size: Maximum number of recent rejections to keep
        """
        self._history_size = history_size
        self._recent_rejections: deque[RejectionEvent] = deque(maxlen=history_size)

        # Counters
        self._total_signals: int = 0
        self._total_rejections: int = 0

        # Per-category counters
        self._by_category: defaultdict[str, int] = defaultdict(int)
        self._by_reason: defaultdict[str, int] = defaultdict(int)
        self._by_symbol: defaultdict[str, int] = defaultdict(int)
        self._by_strategy: defaultdict[str, int] = defaultdict(int)

        # Time-windowed counters (for rate calculation)
        self._window_signals: deque[datetime] = deque()
        self._window_rejections: deque[datetime] = deque()

        # Reason to category mapping
        self._reason_categories: dict[str, RejectionCategory] = {
            # Risk category
            "risk_limits_reject": RejectionCategory.RISK,
            "daily_attempts_reject": RejectionCategory.RISK,
            "kelly_reject": RejectionCategory.RISK,
            "position_size_reject": RejectionCategory.RISK,
            "drawdown_reject": RejectionCategory.RISK,
            "quarantine_reject": RejectionCategory.RISK,
            # Regime category
            "regime_reject": RejectionCategory.REGIME,
            "regime_gating_reject": RejectionCategory.REGIME,
            # Execution category
            "maker_edge_reject": RejectionCategory.EXECUTION,
            "spread_reject": RejectionCategory.EXECUTION,
            "edge_reject": RejectionCategory.EXECUTION,
            # Cooldown category
            "cooldown_reject": RejectionCategory.COOLDOWN,
            # Filter category
            "filter_reject": RejectionCategory.FILTER,
            "volatility_soft_penalty": RejectionCategory.FILTER,
            "volatility_hard_reject": RejectionCategory.FILTER,
            "volatility_penalty": RejectionCategory.FILTER,
            "volatility_breakout_snipe": RejectionCategory.FILTER,
            # Anomaly category
            "price_anomaly_block": RejectionCategory.ANOMALY,
            "stale_price": RejectionCategory.ANOMALY,
            # Oracle category
            "oracle_reversal_block": RejectionCategory.ORACLE,
            "oracle_price_prediction_block": RejectionCategory.ORACLE,
            # OFI category
            "ofi_pressure_block": RejectionCategory.OFI,
            "ofi_prediction_block": RejectionCategory.OFI,
            # Multi-factor category
            "multi_factor_reject": RejectionCategory.MULTI_FACTOR,
            # Other
            "decision_memory_block": RejectionCategory.OTHER,
            "protection_block": RejectionCategory.OTHER,
        }

    def record_signal(self) -> None:
        """Record that a signal was generated (for rate calculation)."""
        self._total_signals += 1
        now = datetime.utcnow()
        self._window_signals.append(now)
        self._cleanup_old_entries()

    def record_rejection(
        self,
        reason: str,
        symbol: str | None = None,
        strategy_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a signal rejection.

        Args:
            reason: Reason for rejection (canonical string)
            symbol: Trading symbol
            strategy_id: Strategy that generated the signal
            details: Additional context about the rejection
        """
        now = datetime.utcnow()
        category = self._reason_categories.get(reason, RejectionCategory.OTHER)

        event = RejectionEvent(
            timestamp=now,
            reason=reason,
            category=category,
            symbol=symbol,
            strategy_id=strategy_id,
            details=details or {},
        )

        # Store in history
        self._recent_rejections.append(event)

        # Update counters
        self._total_rejections += 1
        self._by_category[category.value] += 1
        self._by_reason[reason] += 1

        if symbol:
            self._by_symbol[symbol] += 1
        if strategy_id:
            self._by_strategy[strategy_id] += 1

        # Update time window
        self._window_rejections.append(now)
        self._cleanup_old_entries()

        # Log significant rejections
        if category in (RejectionCategory.RISK, RejectionCategory.ANOMALY):
            logger.warning(
                f"[REJECTION TELEMETRY] {category.value}/{reason}: "
                f"symbol={symbol}, strategy={strategy_id}, details={details}"
            )
        else:
            logger.debug(
                f"[REJECTION TELEMETRY] {category.value}/{reason}: "
                f"symbol={symbol}, strategy={strategy_id}"
            )

    def _cleanup_old_entries(self, max_age_hours: int = 24) -> None:
        """Remove entries older than max_age_hours."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

        while self._window_signals and self._window_signals[0] < cutoff:
            self._window_signals.popleft()

        while self._window_rejections and self._window_rejections[0] < cutoff:
            self._window_rejections.popleft()

    def get_rejection_rate(self, minutes: int = 60) -> float:
        """Get rejection rate for the last N minutes.

        Args:
            minutes: Time window in minutes

        Returns:
            Rejection rate (0.0 to 1.0)
        """
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)

        signals_in_window = sum(1 for ts in self._window_signals if ts >= cutoff)
        rejections_in_window = sum(1 for ts in self._window_rejections if ts >= cutoff)

        if signals_in_window == 0:
            return 0.0

        return rejections_in_window / signals_in_window

    def get_stats(self, minutes: int = 60) -> RejectionStats:
        """Get comprehensive statistics for a time window.

        Args:
            minutes: Time window in minutes

        Returns:
            RejectionStats with full breakdown
        """
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)

        # Count within window
        window_rejections = [r for r in self._recent_rejections if r.timestamp >= cutoff]
        signals_in_window = sum(1 for ts in self._window_signals if ts >= cutoff)

        # Build breakdown
        by_category: dict[str, int] = defaultdict(int)
        by_reason: dict[str, int] = defaultdict(int)
        by_symbol: dict[str, int] = defaultdict(int)
        by_strategy: dict[str, int] = defaultdict(int)

        for r in window_rejections:
            by_category[r.category.value] += 1
            by_reason[r.reason] += 1
            if r.symbol:
                by_symbol[r.symbol] += 1
            if r.strategy_id:
                by_strategy[r.strategy_id] += 1

        rejection_rate = (
            len(window_rejections) / signals_in_window if signals_in_window > 0 else 0.0
        )

        return RejectionStats(
            total_rejections=len(window_rejections),
            total_signals=signals_in_window,
            rejection_rate=rejection_rate,
            by_category=dict(by_category),
            by_reason=dict(by_reason),
            by_symbol=dict(by_symbol),
            by_strategy=dict(by_strategy),
            time_window_minutes=minutes,
        )

    def get_recent_rejections(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent rejection events for debugging.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of rejection events as dicts
        """
        recent = list(self._recent_rejections)[-limit:]
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "reason": r.reason,
                "category": r.category.value,
                "symbol": r.symbol,
                "strategy_id": r.strategy_id,
                "details": r.details,
            }
            for r in reversed(recent)
        ]

    def get_summary(self) -> dict[str, Any]:
        """Get full telemetry summary.

        Returns:
            Dict with all telemetry data
        """
        return {
            "total_signals": self._total_signals,
            "total_rejections": self._total_rejections,
            "overall_rejection_rate": (
                self._total_rejections / self._total_signals
                if self._total_signals > 0
                else 0.0
            ),
            "by_category": dict(self._by_category),
            "by_reason": dict(self._by_reason),
            "by_symbol": dict(self._by_symbol),
            "by_strategy": dict(self._by_strategy),
            "rates": {
                "1m": self.get_rejection_rate(1),
                "5m": self.get_rejection_rate(5),
                "15m": self.get_rejection_rate(15),
                "1h": self.get_rejection_rate(60),
            },
            "history_size": len(self._recent_rejections),
        }

    def get_prometheus_metrics(self) -> dict[str, float]:
        """Get metrics suitable for Prometheus export.

        Returns:
            Dict of metric_name -> value
        """
        stats = self.get_stats(minutes=60)

        metrics = {
            "signal_rejection_total": float(self._total_rejections),
            "signal_rejection_rate_1m": self.get_rejection_rate(1),
            "signal_rejection_rate_5m": self.get_rejection_rate(5),
            "signal_rejection_rate_1h": self.get_rejection_rate(60),
        }

        # Per-category metrics
        for category, count in stats.by_category.items():
            metrics[f"signal_rejection_by_category_{category}"] = float(count)

        return metrics

    def reset(self) -> None:
        """Reset all counters and history."""
        self._recent_rejections.clear()
        self._total_signals = 0
        self._total_rejections = 0
        self._by_category.clear()
        self._by_reason.clear()
        self._by_symbol.clear()
        self._by_strategy.clear()
        self._window_signals.clear()
        self._window_rejections.clear()


# Global singleton
signal_rejection_telemetry = SignalRejectionTelemetry()
