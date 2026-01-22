"""Unified telemetry event envelope and helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

ALLOWED_SEVERITIES = {"INFO", "WARN", "ERROR"}


class EventEnvelope(BaseModel):
    """Canonical telemetry envelope for all realtime and control events."""

    seq: int = Field(default=0, description="Monotonic sequence number for lossless replay")
    ts: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Event timestamp (UTC). Accepts ISO string or unix ms.",
    )
    type: str = Field(..., description="Logical event type, e.g., HEARTBEAT, CONTROL_RESULT")
    severity: str = Field(default="INFO", description="INFO | WARN | ERROR")
    source: str = Field(default="engine", description="Producer of the event")
    correlation_id: str | None = Field(default=None, description="Request/trace correlation id")
    payload: dict[str, Any] = Field(default_factory=dict, description="Event payload")
    context: dict[str, Any] = Field(default_factory=dict, description="Contextual tags (symbol, mode, etc.)")

    @field_validator("ts", mode="before")
    @classmethod
    def _coerce_ts(cls, value: Any) -> datetime:
        """Accept unix ms or ISO strings in addition to datetime."""
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value / 1000.0, tz=UTC)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except Exception:
                # Fallback: assume seconds
                return datetime.fromtimestamp(float(value), tz=UTC)
        return value

    @field_validator("severity")
    @classmethod
    def _validate_severity(cls, value: str) -> str:
        sev = value.upper()
        if sev not in ALLOWED_SEVERITIES:
            raise ValueError(f"Invalid severity '{value}'. Allowed: {', '.join(sorted(ALLOWED_SEVERITIES))}")
        return sev

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-friendly dict with ISO timestamp."""
        data = self.model_dump()
        data["ts"] = self.ts.isoformat()
        return data


def make_event(
    type: str,
    payload: dict[str, Any] | None,
    severity: str = "INFO",
    source: str = "engine",
    correlation_id: str | None = None,
    context: dict[str, Any] | None = None,
    seq: int = 0,
) -> EventEnvelope:
    """Helper to build EventEnvelope with sane defaults."""
    return EventEnvelope(
        type=type,
        payload=payload or {},
        severity=severity.upper(),
        source=source,
        correlation_id=correlation_id,
        context=context or {},
        seq=seq,
    )
