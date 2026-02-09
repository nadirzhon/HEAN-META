"""Brain analysis models - matches iOS BrainThought/BrainAnalysis models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BrainThought(BaseModel):
    """Single thought in the brain's analysis process."""

    id: str
    timestamp: str
    stage: str  # anomaly, physics, xray, decision
    content: str
    confidence: float | None = None


class Force(BaseModel):
    """Market force identified by the brain."""

    name: str
    direction: str  # bullish, bearish, neutral
    magnitude: float = 0.0


class TradingSignal(BaseModel):
    """Trading signal produced by brain analysis."""

    symbol: str
    action: str  # BUY, SELL, HOLD, NEUTRAL
    confidence: float
    reason: str


class BrainAnalysis(BaseModel):
    """Complete brain analysis result."""

    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    thoughts: list[BrainThought] = []
    forces: list[Force] = []
    signal: TradingSignal | None = None
    summary: str = ""
    market_regime: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
