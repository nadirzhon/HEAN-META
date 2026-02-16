"""HeartbeatRegistry -- tracks component liveness via periodic heartbeats."""

import time
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)


class HeartbeatRegistry:
    """Tracks component liveness via periodic heartbeats.

    Components register themselves with an expected heartbeat interval.
    When they miss more than ``failure_threshold`` consecutive beats
    they are considered unhealthy.
    """

    def __init__(
        self,
        default_interval: float = 5.0,
        failure_threshold: int = 3,
    ) -> None:
        self._default_interval = default_interval
        self._failure_threshold = failure_threshold

        # component_id -> last heartbeat timestamp (time.time())
        self._last_beat: dict[str, float] = {}
        # component_id -> expected interval in seconds
        self._intervals: dict[str, float] = {}
        # component_id -> optional metadata dict
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        component_id: str,
        interval: float | None = None,
    ) -> None:
        """Register a component for heartbeat tracking.

        Args:
            component_id: Unique identifier for the component.
            interval: Expected heartbeat interval in seconds.
                      Defaults to ``default_interval``.
        """
        self._intervals[component_id] = interval if interval is not None else self._default_interval
        self._last_beat[component_id] = time.time()
        self._metadata[component_id] = {}
        logger.info(
            "[Heartbeat] Registered component %s (interval=%.1fs)",
            component_id,
            self._intervals[component_id],
        )

    def beat(
        self,
        component_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a heartbeat for a component.

        Args:
            component_id: The component sending the heartbeat.
            metadata: Optional metadata to attach to this heartbeat.
        """
        if component_id not in self._intervals:
            # Auto-register on first beat if not already registered
            self.register(component_id)
        self._last_beat[component_id] = time.time()
        if metadata is not None:
            self._metadata[component_id] = metadata

    def get_status(self) -> dict[str, dict[str, Any]]:
        """Get per-component heartbeat status.

        Returns:
            Dict keyed by component_id, each value containing:
            - last_beat_ago_sec: seconds since last heartbeat
            - missed_beats: estimated number of missed intervals
            - healthy: True if missed_beats < failure_threshold
            - interval_sec: expected interval
            - metadata: latest metadata from the component
        """
        now = time.time()
        result: dict[str, dict[str, Any]] = {}

        for cid in self._intervals:
            interval = self._intervals[cid]
            last_beat = self._last_beat.get(cid, 0.0)
            elapsed = now - last_beat
            missed = int(elapsed / interval) if interval > 0 else 0
            healthy = missed < self._failure_threshold

            result[cid] = {
                "last_beat_ago_sec": round(elapsed, 2),
                "missed_beats": missed,
                "healthy": healthy,
                "interval_sec": interval,
                "metadata": self._metadata.get(cid, {}),
            }

        return result

    def get_unhealthy(self) -> list[str]:
        """Return list of component IDs that are unhealthy.

        A component is unhealthy when it has missed more than
        ``failure_threshold`` consecutive heartbeat intervals.
        """
        now = time.time()
        unhealthy: list[str] = []

        for cid in self._intervals:
            interval = self._intervals[cid]
            last_beat = self._last_beat.get(cid, 0.0)
            elapsed = now - last_beat
            missed = int(elapsed / interval) if interval > 0 else 0
            if missed >= self._failure_threshold:
                unhealthy.append(cid)

        return unhealthy

    def unregister(self, component_id: str) -> None:
        """Remove a component from heartbeat tracking.

        Args:
            component_id: The component to unregister.
        """
        self._intervals.pop(component_id, None)
        self._last_beat.pop(component_id, None)
        self._metadata.pop(component_id, None)
        logger.info("[Heartbeat] Unregistered component %s", component_id)
