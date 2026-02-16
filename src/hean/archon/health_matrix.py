"""HealthMatrix -- aggregates health from multiple sources into a composite score."""

import asyncio
from typing import Any

from hean.archon.heartbeat import HeartbeatRegistry
from hean.archon.signal_pipeline_manager import SignalPipelineManager
from hean.core.bus import EventBus
from hean.logging import get_logger

logger = get_logger(__name__)


class HealthMatrix:
    """Aggregates health from EventBus, HeartbeatRegistry, and
    SignalPipelineManager into a composite score 0-100.

    Score formula:
        40%  EventBus health (not degraded=1.0, degraded=0.5, circuit_open=0.0)
        30%  Heartbeat health (fraction of healthy components)
        20%  Signal Pipeline fill rate (fill_rate_pct / 100, capped at 1.0)
        10%  Error rate (1.0 - handler_errors / total_events, capped at 0.0)
    """

    def __init__(
        self,
        bus: EventBus,
        heartbeat: HeartbeatRegistry | None = None,
        signal_pipeline: SignalPipelineManager | None = None,
    ) -> None:
        self._bus = bus
        self._heartbeat = heartbeat
        self._signal_pipeline = signal_pipeline

        self._running = False
        self._task: asyncio.Task[None] | None = None

        # Cache the latest composite score
        self._cached_score: float = 100.0

    async def start(self) -> None:
        """Start periodic health assessment loop."""
        self._running = True
        self._task = asyncio.create_task(self._assessment_loop())
        logger.info("[HealthMatrix] Started health assessment")

    async def stop(self) -> None:
        """Stop the health assessment loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[HealthMatrix] Stopped")

    def get_composite_score(self) -> float:
        """Return the composite health score (0-100).

        This is a synchronous method that returns the cached score,
        updated every assessment cycle.
        """
        return self._cached_score

    async def get_full_status(self) -> dict[str, Any]:
        """Return detailed health breakdown.

        Returns:
            Dict containing composite_score, bus_health, heartbeats,
            pipeline_metrics, and per-category breakdown.
        """
        bus_health = self._bus.get_health()
        heartbeats = self._heartbeat.get_status() if self._heartbeat else {}
        pipeline_metrics = self._signal_pipeline.get_status() if self._signal_pipeline else None

        breakdown = self._compute_breakdown(bus_health, heartbeats, pipeline_metrics)

        return {
            "composite_score": round(self._cached_score, 2),
            "bus_health": {
                "is_healthy": bus_health.is_healthy,
                "is_degraded": bus_health.is_degraded,
                "is_circuit_open": bus_health.is_circuit_open,
                "queue_utilization_pct": round(bus_health.queue_utilization_pct, 2),
                "events_per_second": bus_health.events_per_second,
                "drop_rate_pct": round(bus_health.drop_rate_pct, 2),
            },
            "heartbeats": heartbeats,
            "pipeline_metrics": pipeline_metrics,
            "breakdown": breakdown,
        }

    # -- Internal --------------------------------------------------------

    def _compute_breakdown(
        self,
        bus_health: Any,
        heartbeats: dict[str, dict[str, Any]],
        pipeline_metrics: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Compute per-category health scores and the composite."""
        # 1. EventBus health (40%)
        if bus_health.is_circuit_open:
            bus_score = 0.0
        elif bus_health.is_degraded:
            bus_score = 0.5
        else:
            bus_score = 1.0

        # 2. Heartbeat health (30%) -- fraction of healthy components
        total_components = len(heartbeats)
        if total_components > 0:
            healthy_count = sum(1 for info in heartbeats.values() if info.get("healthy", False))
            heartbeat_score = healthy_count / total_components
        else:
            # No components registered yet -- assume OK
            heartbeat_score = 1.0

        # 3. Signal Pipeline fill rate (20%)
        if pipeline_metrics:
            fill_rate = pipeline_metrics.get("fill_rate_pct", 0.0)
            pipeline_score = min(fill_rate / 100.0, 1.0)
        else:
            # Pipeline not enabled -- neutral score
            pipeline_score = 1.0

        # 4. Error rate (10%)
        bus_metrics = self._bus.get_metrics()
        total_events = bus_metrics.get("events_processed", 0)
        handler_errors = bus_metrics.get("handler_errors", 0)
        if total_events > 0:
            error_ratio = handler_errors / total_events
            error_score = max(1.0 - error_ratio, 0.0)
        else:
            error_score = 1.0

        # Weighted composite
        composite = (
            bus_score * 40.0 + heartbeat_score * 30.0 + pipeline_score * 20.0 + error_score * 10.0
        )

        return {
            "bus_score": round(bus_score * 100, 2),
            "bus_weight": 0.4,
            "heartbeat_score": round(heartbeat_score * 100, 2),
            "heartbeat_weight": 0.3,
            "pipeline_score": round(pipeline_score * 100, 2),
            "pipeline_weight": 0.2,
            "error_score": round(error_score * 100, 2),
            "error_weight": 0.1,
            "composite": round(composite, 2),
        }

    def _refresh_score(self) -> None:
        """Refresh the cached composite score."""
        bus_health = self._bus.get_health()
        heartbeats = self._heartbeat.get_status() if self._heartbeat else {}
        pipeline_metrics = self._signal_pipeline.get_status() if self._signal_pipeline else None

        breakdown = self._compute_breakdown(bus_health, heartbeats, pipeline_metrics)
        self._cached_score = breakdown["composite"]

    async def _assessment_loop(self) -> None:
        """Periodically refresh composite health score."""
        while self._running:
            try:
                await asyncio.sleep(5.0)
                self._refresh_score()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "[HealthMatrix] Assessment loop error: %s",
                    e,
                    exc_info=True,
                )
