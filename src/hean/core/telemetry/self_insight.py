"""Self-analytics collector for Brain and Council."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import datetime
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class SelfInsightCollector:
    """Publishes periodic SELF_ANALYTICS snapshots from decision, position, and failure events."""

    def __init__(self, bus: EventBus, publish_interval: int = 60) -> None:
        self._bus = bus
        self._publish_interval = publish_interval
        self._running = False
        self._task: asyncio.Task | None = None

        self._decisions: deque[dict[str, Any]] = deque(maxlen=200)
        self._positions: deque[dict[str, Any]] = deque(maxlen=200)
        self._failures: deque[dict[str, Any]] = deque(maxlen=50)
        self._symbiont_updates: deque[dict[str, Any]] = deque(maxlen=50)
        self._physics_states: deque[dict[str, Any]] = deque(maxlen=100)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._bus.subscribe(EventType.ORDER_DECISION, self._handle_order_decision)
        self._bus.subscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        self._bus.subscribe(EventType.ERROR, self._handle_error)
        self._bus.subscribe(EventType.STRATEGY_PARAMS_UPDATED, self._handle_symbiont_update)
        self._bus.subscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        self._task = asyncio.create_task(self._publish_loop())
        logger.info("SelfInsightCollector started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._bus.unsubscribe(EventType.ORDER_DECISION, self._handle_order_decision)
        self._bus.unsubscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        self._bus.unsubscribe(EventType.ERROR, self._handle_error)
        self._bus.unsubscribe(EventType.STRATEGY_PARAMS_UPDATED, self._handle_symbiont_update)
        self._bus.unsubscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        logger.info("SelfInsightCollector stopped")

    async def _publish_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._publish_interval)
            await self._publish_snapshot()

    async def _publish_snapshot(self) -> None:
        if not self._running:
            return
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision_history": list(self._decisions),
            "position_summary": self._summarize_positions(),
            "failures": list(self._failures),
            "symbiont_updates": list(self._symbiont_updates),
            "recent_physics": list(self._physics_states)[-5:],
        }
        await self._bus.publish(Event(event_type=EventType.SELF_ANALYTICS, data=snapshot))

    def _summarize_positions(self) -> dict[str, Any]:
        summary: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "total_pnl": 0.0})
        for entry in self._positions:
            strat = entry["strategy_id"]
            summary[strat]["count"] += 1
            summary[strat]["total_pnl"] += entry["pnl"]
        return {
            strat: {
                "trades": data["count"],
                "avg_pnl": data["total_pnl"] / data["count"] if data["count"] else 0.0,
            }
            for strat, data in summary.items()
        }

    async def _handle_order_decision(self, event: Event) -> None:
        decision = event.data.get("decision")
        strategy_id = event.data.get("strategy_id")
        if not strategy_id:
            return
        self._decisions.append({
            "timestamp": event.timestamp.isoformat(),
            "strategy_id": strategy_id,
            "decision": decision,
            "reason_codes": event.data.get("reason_codes", []),
            "gating_flags": event.data.get("gating_flags", {}),
        })

    async def _handle_position_closed(self, event: Event) -> None:
        position = event.data.get("position")
        if not position:
            return
        self._positions.append({
            "timestamp": event.timestamp.isoformat(),
            "strategy_id": position.strategy_id,
            "symbol": position.symbol,
            "pnl": position.realized_pnl,
            "duration_sec": (position.closed_at.timestamp() - position.opened_at.timestamp()) if position.opened_at and position.closed_at else 0.0,
            "phase": position.metadata.get("phase"),
        })

    async def _handle_error(self, event: Event) -> None:
        self._failures.append({
            "timestamp": event.timestamp.isoformat(),
            "message": event.data.get("message", "unknown"),
            "context": event.data.get("context"),
        })

    async def _handle_symbiont_update(self, event: Event) -> None:
        self._symbiont_updates.append({
            "timestamp": event.timestamp.isoformat(),
            "strategy_id": event.data.get("strategy_id"),
            "params": event.data.get("params"),
            "fitness": event.data.get("fitness"),
        })

    async def _handle_physics_update(self, event: Event) -> None:
        physics = event.data.get("physics")
        if not physics:
            return
        self._physics_states.append({
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": physics.get("symbol"),
            "phase": physics.get("phase"),
            "temperature": physics.get("temperature"),
            "entropy": physics.get("entropy"),
        })
