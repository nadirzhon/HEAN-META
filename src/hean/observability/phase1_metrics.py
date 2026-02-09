"""Phase 1 Performance Tracking Metrics.

Tracks Kelly Criterion, Adaptive Execution, and Regime-Aware Sizing metrics
for measuring the impact of Phase 1 improvements.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Phase1Metrics:
    """Tracks Phase 1 enhancement metrics."""

    # Kelly Criterion metrics
    kelly_fractions: dict[str, float] = field(default_factory=dict)  # strategy_id -> kelly_fraction
    kelly_adjustments: int = 0  # Total adaptive adjustments
    confidence_boosts: int = 0  # Times confidence increased size
    streak_penalties: int = 0  # Times streak reduced size

    # Adaptive execution metrics
    adaptive_ttl_ms: float = 0.0  # Current adaptive TTL
    adaptive_offset_bps: float = 0.0  # Current adaptive offset
    ttl_adjustments: int = 0  # Total TTL changes
    maker_fills: int = 0  # Maker fills
    maker_expirations: int = 0  # Maker expirations
    fill_rate_window: deque = field(default_factory=lambda: deque(maxlen=100))  # Recent fill outcomes

    # Orderbook imbalance metrics
    imbalance_signals: int = 0  # Orderbook imbalance trades
    imbalance_edge_bps_total: float = 0.0  # Total edge captured from imbalance

    # Regime-aware sizing metrics
    regime_boosts: int = 0  # IMPULSE regime size boosts
    regime_reductions: int = 0  # RANGE regime size reductions
    current_regime: str | None = None
    current_size_multiplier: float = 1.0

    # Performance tracking
    last_updated: datetime | None = None
    update_count: int = 0

    def record_kelly_calculation(self, strategy_id: str, kelly_fraction: float) -> None:
        """Record Kelly fraction calculation.

        Args:
            strategy_id: Strategy identifier
            kelly_fraction: Calculated Kelly fraction
        """
        old_fraction = self.kelly_fractions.get(strategy_id)
        self.kelly_fractions[strategy_id] = kelly_fraction

        if old_fraction is not None and abs(kelly_fraction - old_fraction) > 0.01:
            self.kelly_adjustments += 1
            logger.debug(
                f"[Phase1Metrics] Kelly adjusted for {strategy_id}: "
                f"{old_fraction:.4f} → {kelly_fraction:.4f}"
            )

        self.last_updated = datetime.utcnow()
        self.update_count += 1

    def record_confidence_scaling(self, boosted: bool) -> None:
        """Record confidence scaling event.

        Args:
            boosted: True if confidence increased size
        """
        if boosted:
            self.confidence_boosts += 1

    def record_streak_penalty(self) -> None:
        """Record streak penalty application."""
        self.streak_penalties += 1

    def record_ttl_adjustment(self, new_ttl_ms: float) -> None:
        """Record adaptive TTL adjustment.

        Args:
            new_ttl_ms: New TTL in milliseconds
        """
        old_ttl = self.adaptive_ttl_ms
        if old_ttl > 0 and abs(new_ttl_ms - old_ttl) > 10:
            self.ttl_adjustments += 1
            logger.debug(
                f"[Phase1Metrics] TTL adjusted: {old_ttl:.0f}ms → {new_ttl_ms:.0f}ms "
                f"(change: {((new_ttl_ms / old_ttl) - 1) * 100:+.1f}%)"
            )

        self.adaptive_ttl_ms = new_ttl_ms
        self.last_updated = datetime.utcnow()

    def record_offset_adjustment(self, new_offset_bps: float) -> None:
        """Record adaptive offset adjustment.

        Args:
            new_offset_bps: New offset in basis points
        """
        self.adaptive_offset_bps = new_offset_bps
        self.last_updated = datetime.utcnow()

    def record_maker_fill(self) -> None:
        """Record successful maker fill."""
        self.maker_fills += 1
        self.fill_rate_window.append(True)
        self.last_updated = datetime.utcnow()

    def record_maker_expiration(self) -> None:
        """Record maker order expiration."""
        self.maker_expirations += 1
        self.fill_rate_window.append(False)
        self.last_updated = datetime.utcnow()

    def record_imbalance_signal(self, edge_bps: float) -> None:
        """Record orderbook imbalance signal.

        Args:
            edge_bps: Edge captured in basis points
        """
        self.imbalance_signals += 1
        self.imbalance_edge_bps_total += edge_bps
        logger.info(
            f"[Phase1Metrics] Imbalance signal #{self.imbalance_signals}: "
            f"edge={edge_bps:.2f} bps"
        )
        self.last_updated = datetime.utcnow()

    def record_regime_sizing(
        self, regime: str, size_multiplier: float, is_boost: bool
    ) -> None:
        """Record regime-aware sizing adjustment.

        Args:
            regime: Current regime
            size_multiplier: Applied size multiplier
            is_boost: True if size was increased
        """
        self.current_regime = regime
        self.current_size_multiplier = size_multiplier

        if is_boost:
            self.regime_boosts += 1
        else:
            self.regime_reductions += 1

        self.last_updated = datetime.utcnow()

    def get_fill_rate_pct(self) -> float:
        """Get maker fill rate percentage.

        Returns:
            Fill rate as percentage (0-100)
        """
        if len(self.fill_rate_window) == 0:
            return 0.0

        fills = sum(1 for outcome in self.fill_rate_window if outcome)
        return (fills / len(self.fill_rate_window)) * 100

    def get_average_imbalance_edge_bps(self) -> float:
        """Get average edge from imbalance signals.

        Returns:
            Average edge in basis points
        """
        if self.imbalance_signals == 0:
            return 0.0

        return self.imbalance_edge_bps_total / self.imbalance_signals

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary for export.

        Returns:
            Dictionary of Phase 1 metrics
        """
        return {
            # Kelly metrics
            "kelly_strategies_tracked": len(self.kelly_fractions),
            "kelly_avg_fraction": (
                sum(self.kelly_fractions.values()) / len(self.kelly_fractions)
                if self.kelly_fractions
                else 0.0
            ),
            "kelly_adjustments_total": self.kelly_adjustments,
            "kelly_confidence_boosts": self.confidence_boosts,
            "kelly_streak_penalties": self.streak_penalties,
            # Execution metrics
            "adaptive_ttl_ms": self.adaptive_ttl_ms,
            "adaptive_offset_bps": self.adaptive_offset_bps,
            "ttl_adjustments_total": self.ttl_adjustments,
            "maker_fills": self.maker_fills,
            "maker_expirations": self.maker_expirations,
            "maker_fill_rate_pct": self.get_fill_rate_pct(),
            # Imbalance metrics
            "imbalance_signals_total": self.imbalance_signals,
            "imbalance_avg_edge_bps": self.get_average_imbalance_edge_bps(),
            # Regime metrics
            "regime_boosts_total": self.regime_boosts,
            "regime_reductions_total": self.regime_reductions,
            "current_regime": self.current_regime,
            "current_size_multiplier": self.current_size_multiplier,
            # Meta
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "update_count": self.update_count,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.kelly_fractions.clear()
        self.kelly_adjustments = 0
        self.confidence_boosts = 0
        self.streak_penalties = 0

        self.adaptive_ttl_ms = 0.0
        self.adaptive_offset_bps = 0.0
        self.ttl_adjustments = 0
        self.maker_fills = 0
        self.maker_expirations = 0
        self.fill_rate_window.clear()

        self.imbalance_signals = 0
        self.imbalance_edge_bps_total = 0.0

        self.regime_boosts = 0
        self.regime_reductions = 0
        self.current_regime = None
        self.current_size_multiplier = 1.0

        self.last_updated = None
        self.update_count = 0


# Global Phase 1 metrics instance
phase1_metrics = Phase1Metrics()
