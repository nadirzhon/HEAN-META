"""Canary Tester - Monitors canary performance and triggers promotion/rollback."""

from datetime import datetime
from typing import Any

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class CanaryMetrics:
    """Canary performance metrics."""

    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.max_dd_pct = 0.0
        self.sharpe = 0.0
        self.profit_factor = 0.0
        self.returns: list[float] = []
        self.started_at = datetime.utcnow()
        self.last_updated_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.wins / self.trades if self.trades > 0 else 0.0,
            "total_pnl": self.total_pnl,
            "max_dd_pct": self.max_dd_pct,
            "sharpe": self.sharpe,
            "profit_factor": self.profit_factor,
            "duration_hours": (datetime.utcnow() - self.started_at).total_seconds() / 3600,
            "started_at": self.started_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat(),
        }


class CanaryTester:
    """Canary tester for monitoring and auto-promoting/rolling back strategies.

    Monitors canary performance for 48h and compares against baseline.
    Auto-promotes if better, auto-rollbacks if worse.
    """

    def __init__(self, bus: EventBus, baseline_sharpe: float = 1.5) -> None:
        """Initialize canary tester.

        Args:
            bus: Event bus for publishing results
            baseline_sharpe: Baseline Sharpe ratio to beat
        """
        self._bus = bus
        self._enabled = getattr(settings, "ai_factory_enabled", False)
        self._baseline_sharpe = baseline_sharpe
        self._canary_period_hours = 48  # 48-hour canary period
        self._canary_metrics: dict[str, CanaryMetrics] = {}

        logger.info(
            f"Canary Tester initialized: enabled={self._enabled}, "
            f"baseline_sharpe={baseline_sharpe:.2f}, period={self._canary_period_hours}h"
        )

    async def start_canary(self, strategy_id: str) -> None:
        """Start canary monitoring for a strategy.

        Args:
            strategy_id: Strategy ID to monitor
        """
        if strategy_id not in self._canary_metrics:
            self._canary_metrics[strategy_id] = CanaryMetrics(strategy_id)
            logger.info(f"Started canary monitoring for {strategy_id}")

    async def update_metrics(
        self,
        strategy_id: str,
        trade_pnl: float,
        win: bool,
    ) -> None:
        """Update canary metrics with new trade.

        Args:
            strategy_id: Strategy ID
            trade_pnl: Trade PnL
            win: True if winning trade
        """
        if strategy_id not in self._canary_metrics:
            await self.start_canary(strategy_id)

        metrics = self._canary_metrics[strategy_id]
        metrics.trades += 1
        if win:
            metrics.wins += 1
        else:
            metrics.losses += 1
        metrics.total_pnl += trade_pnl
        metrics.returns.append(trade_pnl)
        metrics.last_updated_at = datetime.utcnow()

        # Recalculate metrics (simplified)
        metrics.profit_factor = (
            (metrics.wins / metrics.losses) if metrics.losses > 0 else metrics.wins
        )
        # Calculate Sharpe from actual returns
        if hasattr(metrics, 'returns') and len(metrics.returns) >= 2:
            returns = metrics.returns
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_return = variance ** 0.5
            metrics.sharpe = (mean_return / std_return) * (252 ** 0.5) if std_return > 0 else 0.0
        else:
            metrics.sharpe = 0.0  # Not enough data for Sharpe calculation

    async def check_quality_gate(
        self,
        strategy_id: str,
    ) -> dict[str, Any]:
        """Check if canary passes quality gate.

        Quality gate criteria:
        - Canary period >= 48h
        - Min 30 trades
        - Sharpe >= baseline * 1.1 (10% improvement)
        - Max DD < 15%

        Args:
            strategy_id: Strategy ID to check

        Returns:
            Quality gate result
        """
        if strategy_id not in self._canary_metrics:
            return {
                "passed": False,
                "reason": "Canary metrics not found",
                "decision": "WAIT",
            }

        metrics = self._canary_metrics[strategy_id]
        duration_hours = (datetime.utcnow() - metrics.started_at).total_seconds() / 3600

        # Check criteria
        criteria = {
            "duration_ok": duration_hours >= self._canary_period_hours,
            "min_trades_ok": metrics.trades >= 30,
            "sharpe_ok": metrics.sharpe >= self._baseline_sharpe * 1.1,
            "max_dd_ok": metrics.max_dd_pct < 15.0,
        }

        all_passed = all(criteria.values())

        decision = "PROMOTE" if all_passed else "WAIT"
        if duration_hours >= self._canary_period_hours and not all_passed:
            decision = "ROLLBACK"  # Failed after full canary period

        result = {
            "passed": all_passed,
            "strategy_id": strategy_id,
            "criteria": criteria,
            "metrics": metrics.to_dict(),
            "baseline_sharpe": self._baseline_sharpe,
            "decision": decision,
            "checked_at": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"Quality gate for {strategy_id}: {decision} "
            f"(Sharpe={metrics.sharpe:.2f} vs baseline={self._baseline_sharpe:.2f})"
        )

        # Publish experiment result
        await self._bus.publish(Event(
            event_type=EventType.STRATEGY_UPDATED,
            data={
                "type": "EXPERIMENT_RESULT",
                "experiment_id": f"canary_{strategy_id}",
                "strategy": strategy_id,
                "phase": "canary",
                "metrics": metrics.to_dict(),
                "comparison_vs_baseline": {
                    "sharpe_delta": metrics.sharpe - self._baseline_sharpe,
                    "better": metrics.sharpe >= self._baseline_sharpe * 1.1,
                },
                "decision": decision,
                "timestamp": datetime.utcnow().isoformat(),
            }
        ))

        return result

    def get_results(self, strategy_id: str) -> CanaryMetrics | None:
        """Get canary results.

        Args:
            strategy_id: Strategy ID

        Returns:
            Canary metrics or None
        """
        return self._canary_metrics.get(strategy_id)

    def get_all_canaries(self) -> list[dict[str, Any]]:
        """Get all active canaries.

        Returns:
            List of canary metrics
        """
        return [metrics.to_dict() for metrics in self._canary_metrics.values()]
