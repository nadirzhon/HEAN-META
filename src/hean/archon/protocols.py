"""Protocols and types for ARCHON components."""

from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ComponentState(str, Enum):
    """Unified component lifecycle state."""

    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    FAILED = "failed"


@runtime_checkable
class ArchonComponent(Protocol):
    """Protocol that all ARCHON-managed components must satisfy."""

    @property
    def component_id(self) -> str: ...

    @property
    def component_state(self) -> ComponentState: ...

    async def health_check(self) -> dict[str, Any]: ...

    async def get_metrics(self) -> dict[str, Any]: ...
