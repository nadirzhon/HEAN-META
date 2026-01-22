"""Agent registry for AI Catalyst observability."""

import asyncio
from datetime import datetime
from typing import Any, Literal

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class AgentRegistry:
    """Minimal agent registry for tracking active agents and their status.

    Tracks:
    - Active agents (name, role, status, current_task, last_heartbeat)
    - Agent events (AGENT_STEP, AGENT_STATUS)
    - Today's changelog (manual + git log integration)
    """

    def __init__(self, bus: EventBus) -> None:
        """Initialize agent registry.

        Args:
            bus: Event bus for publishing AI catalyst events
        """
        self._bus = bus
        self._agents: dict[str, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._max_events = 200  # Keep last 200 events

    def register_agent(
        self,
        name: str,
        role: str,
        status: Literal["idle", "working", "paused", "error"] = "idle",
        current_task: str | None = None,
    ) -> None:
        """Register or update an agent.

        Args:
            name: Agent name/ID
            role: Agent role (e.g., "signal_generator", "risk_monitor", "optimizer")
            status: Current status
            current_task: Current task description
        """
        self._agents[name] = {
            "name": name,
            "role": role,
            "status": status,
            "current_task": current_task,
            "last_heartbeat": datetime.utcnow().isoformat(),
        }

        # Publish agent status event
        asyncio.create_task(
            self._bus.publish(
                Event(
                    event_type=EventType.SIGNAL,  # Reuse existing event type
                    data={
                        "type": "AGENT_STATUS",
                        "agent_name": name,
                        "role": role,
                        "status": status,
                        "current_task": current_task,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
            )
        )

    async def emit_agent_step(
        self,
        agent_name: str,
        step_description: str,
        result: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit an agent step event.

        Args:
            agent_name: Agent name
            step_description: Description of the step
            result: Result of the step
            metadata: Additional metadata
        """
        event_data = {
            "type": "AGENT_STEP",
            "agent_name": agent_name,
            "step_description": step_description,
            "result": result,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Store event
        self._events.append(event_data)
        self._events = self._events[-self._max_events:]  # Keep last N events

        # Publish to bus
        await self._bus.publish(
            Event(
                event_type=EventType.SIGNAL,  # Reuse existing event type
                data=event_data,
            )
        )

    def get_agents(self) -> list[dict[str, Any]]:
        """Get list of all registered agents."""
        return list(self._agents.values())

    def get_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent agent events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of recent events
        """
        return self._events[-limit:]

    def get_agent(self, name: str) -> dict[str, Any] | None:
        """Get specific agent by name."""
        return self._agents.get(name)
