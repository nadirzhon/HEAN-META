"""Signal lifecycle tracking types."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SignalStage(str, Enum):
    """Stages in the signal lifecycle."""

    GENERATED = "generated"
    FILTERED = "filtered"
    RISK_CHECKING = "risk_checking"
    RISK_APPROVED = "risk_approved"
    RISK_BLOCKED = "risk_blocked"
    ORDER_CREATING = "order_creating"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_TIMEOUT = "order_timeout"
    POSITION_OPENED = "position_opened"
    DEAD_LETTER = "dead_letter"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STAGES


_TERMINAL_STAGES = {
    SignalStage.RISK_BLOCKED,
    SignalStage.ORDER_REJECTED,
    SignalStage.ORDER_CANCELLED,
    SignalStage.ORDER_TIMEOUT,
    SignalStage.DEAD_LETTER,
    SignalStage.ORDER_FILLED,
    SignalStage.POSITION_OPENED,
}


@dataclass
class StageRecord:
    """Record of a signal passing through a stage."""

    stage: SignalStage
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalTrace:
    """Complete lifecycle trace of one signal."""

    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str = ""
    symbol: str = ""
    side: str = ""
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    stages: list[StageRecord] = field(default_factory=list)
    current_stage: SignalStage = SignalStage.GENERATED
    order_id: str | None = None
    position_id: str | None = None

    def advance(self, stage: SignalStage, details: dict[str, Any] | None = None) -> None:
        """Advance signal to next stage."""
        now = datetime.utcnow()
        self.current_stage = stage
        self.stages.append(
            StageRecord(
                stage=stage,
                timestamp=now,
                details=details or {},
            )
        )

    @property
    def is_terminal(self) -> bool:
        return self.current_stage.is_terminal

    @property
    def latency_ms(self) -> float:
        """Total latency from creation to last stage."""
        if not self.stages:
            return 0.0
        last = self.stages[-1].timestamp
        return (last - self.created_at).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "correlation_id": self.correlation_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "side": self.side,
            "confidence": self.confidence,
            "current_stage": self.current_stage.value,
            "is_terminal": self.is_terminal,
            "latency_ms": round(self.latency_ms, 2),
            "created_at": self.created_at.isoformat(),
            "order_id": self.order_id,
            "stages": [
                {
                    "stage": s.stage.value,
                    "timestamp": s.timestamp.isoformat(),
                    "details": s.details,
                }
                for s in self.stages
            ],
        }
