"""Lightweight telemetry service for unified event envelopes and heartbeat."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Deque, Optional

from hean.api.telemetry.events import EventEnvelope, make_event
from hean.logging import get_logger

logger = get_logger(__name__)

BroadcastFn = Callable[[str, dict[str, Any]], Awaitable[None]]


class TelemetryService:
    """Tracks telemetry stats, emits heartbeats, and standardizes events."""

    def __init__(self, window_seconds: int = 60) -> None:
        self._window_seconds = window_seconds
        self._events: Deque[float] = deque()
        self._events_total = 0
        self._last_event: EventEnvelope | None = None
        self._last_heartbeat: EventEnvelope | None = None
        self._start_time = time.time()
        self._engine_state: str = "STOPPED"
        self._broadcast: BroadcastFn | None = None
        self._lock = asyncio.Lock()
        self._seq_counter: int = 0
        self._recent_events: Deque[EventEnvelope] = deque(maxlen=500)

    def set_broadcast(self, broadcast: BroadcastFn) -> None:
        """Inject async broadcast function (e.g., WebSocket topic fan-out)."""
        self._broadcast = broadcast

    def set_engine_state(self, state: str) -> None:
        """Record engine lifecycle state for heartbeat."""
        self._engine_state = state.upper()

    def get_engine_state(self) -> str:
        return self._engine_state

    async def record_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        severity: str = "INFO",
        source: str = "engine",
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
        publish_ws: bool = False,
        topic: str | None = None,
        count_only: bool = False,
    ) -> EventEnvelope:
        """Store envelope, update counters, optionally broadcast."""
        async with self._lock:
            if count_only:
                # Fast path: only increment counters without mutating history/seq
                now = datetime.now(timezone.utc).timestamp()
                self._events.append(now)
                self._events_total += 1
                cutoff = now - self._window_seconds
                while self._events and self._events[0] < cutoff:
                    self._events.popleft()
                return make_event(
                    type=event_type,
                    payload=payload or {},
                    severity=severity,
                    source=source,
                    correlation_id=correlation_id,
                    context=context,
                    seq=self._seq_counter,
                )

            seq = self._next_seq()
            envelope = make_event(
                type=event_type,
                payload=payload or {},
                severity=severity,
                source=source,
                correlation_id=correlation_id,
                context=context,
                seq=seq,
            )
            now = envelope.ts.timestamp()
            self._events.append(now)
            self._events_total += 1
            self._last_event = envelope
            self._recent_events.append(envelope)
            cutoff = now - self._window_seconds
            while self._events and self._events[0] < cutoff:
                self._events.popleft()

        if publish_ws and self._broadcast and not count_only:
            try:
                await self._broadcast(topic or event_type, envelope.as_dict())
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"Telemetry broadcast failed for {event_type}: {exc}")

        return envelope

    def _next_seq(self) -> int:
        """Generate next monotonic sequence number."""
        self._seq_counter += 1
        return self._seq_counter

    def events_per_sec(self) -> float:
        """Sliding-window events per second."""
        now = time.time()
        cutoff = now - self._window_seconds
        # Fast path without lock; slight race acceptable for telemetry
        while self._events and self._events[0] < cutoff:
            self._events.popleft()
        if not self._events:
            return 0.0
        span = max(now - self._events[0], 1e-6)
        return round(len(self._events) / span, 3)

    def uptime_seconds(self) -> float:
        return max(time.time() - self._start_time, 0.0)

    def last_event_ts_iso(self) -> Optional[str]:
        return self._last_event.ts.isoformat() if self._last_event else None

    def last_heartbeat(self) -> Optional[dict[str, Any]]:
        return self._last_heartbeat.as_dict() if self._last_heartbeat else None

    def last_seq(self) -> int:
        """Return last assigned sequence id."""
        return self._seq_counter

    def history(self, limit: int = 200) -> list[EventEnvelope]:
        """Return recent envelopes (oldest first) up to limit."""
        if limit <= 0:
            return []
        events = list(self._recent_events)[-limit:]
        return events

    async def emit_heartbeat(
        self,
        *,
        engine_state: str,
        mode: str,
        ws_clients: int,
        events_per_sec: float | None = None,
        last_event_ts: str | None = None,
        source: str = "engine",
    ) -> EventEnvelope:
        """Build and broadcast heartbeat telemetry."""
        payload = {
            "engine_state": engine_state,
            "uptime_sec": round(self.uptime_seconds(), 2),
            "mode": mode,
            "ws_clients": ws_clients,
            "events_per_sec": events_per_sec if events_per_sec is not None else self.events_per_sec(),
            "last_event_ts": last_event_ts or self.last_event_ts_iso(),
        }

        envelope = await self.record_event(
            "HEARTBEAT",
            payload=payload,
            severity="INFO",
            source=source,
            context={"topic": "system_heartbeat", "mode": mode},
            publish_ws=True,
            topic="system_heartbeat",
        )
        self._last_heartbeat = envelope
        return envelope

    def summary(self, *, ws_clients: int, mode: str) -> dict[str, Any]:
        """Return snapshot for /telemetry/summary."""
        return {
            "engine_state": self._engine_state,
            "uptime_sec": round(self.uptime_seconds(), 2),
            "events_per_sec": self.events_per_sec(),
            "last_event_ts": self.last_event_ts_iso(),
            "last_seq": self._seq_counter,
            "ws_clients": ws_clients,
            "mode": mode,
            "events_total": self._events_total,
        }


# Global instance reused across the API modules
telemetry_service = TelemetryService()
