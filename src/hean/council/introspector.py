"""Collects system state for AI Council review sessions."""

import inspect
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType

logger = logging.getLogger(__name__)


class Introspector:
    """Collects comprehensive system state for council review."""

    def __init__(
        self,
        bus: EventBus,
        accounting: Any | None = None,
        strategies: dict[str, Any] | None = None,
        killswitch: Any | None = None,
    ) -> None:
        self._bus = bus
        self._accounting = accounting
        self._strategies = strategies or {}
        self._killswitch = killswitch

        self._recent_risk_alerts: deque[dict[str, Any]] = deque(maxlen=100)
        self._recent_order_events: deque[dict[str, Any]] = deque(maxlen=200)
        self._recent_pnl_updates: deque[dict[str, Any]] = deque(maxlen=100)
        self._error_log: deque[dict[str, Any]] = deque(maxlen=50)
        self._killswitch_events: deque[dict[str, Any]] = deque(maxlen=20)
        self._self_insights: dict[str, Any] | None = None

    async def start(self) -> None:
        """Subscribe to events for data collection."""
        self._bus.subscribe(EventType.SELF_ANALYTICS, self._handle_self_insight)
        self._bus.subscribe(EventType.RISK_ALERT, self._handle_risk_alert)
        self._bus.subscribe(EventType.RISK_BLOCKED, self._handle_risk_blocked)
        self._bus.subscribe(EventType.ORDER_FILLED, self._handle_order_event)
        self._bus.subscribe(EventType.ORDER_REJECTED, self._handle_order_event)
        self._bus.subscribe(EventType.PNL_UPDATE, self._handle_pnl_update)
        self._bus.subscribe(EventType.ERROR, self._handle_error)
        self._bus.subscribe(EventType.KILLSWITCH_TRIGGERED, self._handle_killswitch)
        logger.info("Council Introspector subscribed to events")

    async def stop(self) -> None:
        """Clear buffers."""
        self._bus.unsubscribe(EventType.SELF_ANALYTICS, self._handle_self_insight)
        self._recent_risk_alerts.clear()
        self._recent_order_events.clear()
        self._recent_pnl_updates.clear()
        self._error_log.clear()
        self._killswitch_events.clear()

    def collect_snapshot(self) -> dict[str, Any]:
        """Collect comprehensive system snapshot for council review."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "trading_metrics": self._collect_trading_metrics(),
            "strategy_performance": self._collect_strategy_performance(),
            "risk_state": self._collect_risk_state(),
            "system_health": self._collect_system_health(),
            "error_summary": self._collect_error_summary(),
            "code_structure": self._collect_code_structure(),
            "recent_events": self._collect_recent_events(),
            "self_insights": self._self_insights,
        }

    # -- Event handlers --

    async def _handle_risk_alert(self, event: Event) -> None:
        self._recent_risk_alerts.append({
            "timestamp": event.timestamp.isoformat(),
            "type": "risk_alert",
            **event.data,
        })

    async def _handle_risk_blocked(self, event: Event) -> None:
        self._recent_risk_alerts.append({
            "timestamp": event.timestamp.isoformat(),
            "type": "risk_blocked",
            **event.data,
        })

    async def _handle_order_event(self, event: Event) -> None:
        self._recent_order_events.append({
            "timestamp": event.timestamp.isoformat(),
            "type": event.event_type.value,
            **event.data,
        })

    async def _handle_pnl_update(self, event: Event) -> None:
        self._recent_pnl_updates.append({
            "timestamp": event.timestamp.isoformat(),
            **event.data,
        })

    async def _handle_error(self, event: Event) -> None:
        self._error_log.append({
            "timestamp": event.timestamp.isoformat(),
            **event.data,
        })

    async def _handle_killswitch(self, event: Event) -> None:
        self._killswitch_events.append({
            "timestamp": event.timestamp.isoformat(),
            **event.data,
        })

    async def _handle_self_insight(self, event: Event) -> None:
        self._self_insights = event.data

    # -- Collectors --

    def _collect_trading_metrics(self) -> dict[str, Any]:
        if not self._accounting:
            return {"status": "no_accounting"}
        try:
            equity = self._accounting.get_equity()
            drawdown, drawdown_pct = self._accounting.get_drawdown(equity)
            return {
                "equity": round(equity, 2),
                "initial_capital": self._accounting._initial_capital,
                "daily_pnl": round(self._accounting.get_daily_pnl(equity), 2),
                "realized_pnl": round(self._accounting.get_realized_pnl_total(), 2),
                "unrealized_pnl": round(equity - self._accounting._initial_capital - self._accounting.get_realized_pnl_total(), 2),
                "drawdown": round(drawdown, 2),
                "drawdown_pct": round(drawdown_pct, 2),
                "total_fees": round(self._accounting.get_total_fees(), 2),
                "open_positions": len(self._accounting.get_positions()),
            }
        except Exception as e:
            logger.warning(f"Failed to collect trading metrics: {e}")
            return {"status": "error", "error": str(e)}

    def _collect_strategy_performance(self) -> dict[str, Any]:
        if not self._accounting:
            return {}
        try:
            return self._accounting.get_strategy_metrics()
        except Exception as e:
            logger.warning(f"Failed to collect strategy metrics: {e}")
            return {"error": str(e)}

    def _collect_risk_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "killswitch_triggered": False,
            "killswitch_reason": "",
            "killswitch_events": list(self._killswitch_events),
            "recent_risk_alerts": list(self._recent_risk_alerts)[-10:],
        }
        if self._killswitch:
            state["killswitch_triggered"] = self._killswitch._triggered
            state["killswitch_reason"] = getattr(self._killswitch, "_trigger_reason", "")
        return state

    def _collect_system_health(self) -> dict[str, Any]:
        try:
            return self._bus.get_metrics()
        except Exception as e:
            logger.warning(f"Failed to collect bus metrics: {e}")
            return {"error": str(e)}

    def _collect_error_summary(self) -> dict[str, Any]:
        return {
            "total_errors": len(self._error_log),
            "recent_errors": list(self._error_log)[-10:],
        }

    def _collect_code_structure(self) -> dict[str, Any]:
        structure: dict[str, Any] = {"strategies": {}}
        for strategy_id, strategy_obj in self._strategies.items():
            try:
                source_file = inspect.getfile(type(strategy_obj))
                source = Path(source_file).read_text(encoding="utf-8")[:3000]
                structure["strategies"][strategy_id] = {
                    "class_name": type(strategy_obj).__name__,
                    "source_file": source_file,
                    "source_excerpt": source,
                }
            except Exception:
                structure["strategies"][strategy_id] = {
                    "class_name": type(strategy_obj).__name__,
                }
        return structure

    def _collect_recent_events(self) -> dict[str, Any]:
        return {
            "recent_fills": list(self._recent_order_events)[-20:],
            "recent_pnl": list(self._recent_pnl_updates)[-10:],
        }
