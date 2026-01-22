"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EngineStatus(str, Enum):
    """Engine status enumeration."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReasonCode(BaseModel):
    """Reason code for blocked signals/orders."""

    code: str = Field(..., description="Reason code identifier")
    message: str = Field(..., description="Human-readable message")
    measured: dict[str, Any] = Field(default_factory=dict, description="Measured values")
    thresholds: dict[str, Any] = Field(default_factory=dict, description="Threshold values")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str | None = Field(default=None, description="Trading symbol")


class EngineStartRequest(BaseModel):
    """Request model for starting engine."""

    confirm_phrase: str | None = Field(
        default=None, description="Confirmation phrase for live trading"
    )


class EngineStopRequest(BaseModel):
    """Request model for stopping engine."""

    pass


class EnginePauseRequest(BaseModel):
    """Request model for pausing engine."""

    pass


class TestOrderRequest(BaseModel):
    """Request model for placing test order."""

    symbol: str = Field(default="BTCUSDT", description="Trading symbol")
    side: str = Field(default="buy", description="Order side: buy or sell")
    size: float = Field(default=0.001, description="Order size")
    price: float | None = Field(default=None, description="Limit price (optional)")


class TestRoundtripRequest(BaseModel):
    """Request model for end-to-end paper roundtrip test."""

    symbol: str = Field(default="BTCUSDT", description="Trading symbol")
    side: str = Field(default="buy", description="Order side: buy or sell")
    size: float = Field(default=0.001, description="Order size")
    take_profit_pct: float = Field(default=0.3, description="TP distance in percent")
    stop_loss_pct: float = Field(default=0.3, description="SL distance in percent")
    hold_timeout_sec: int = Field(default=10, description="TTL seconds for forced exit")


class ClosePositionRequest(BaseModel):
    """Request model for closing position."""

    position_id: str = Field(..., description="Position ID to close")
    confirm_phrase: str | None = Field(
        default=None, description="Confirmation phrase for live trading"
    )


class CancelAllOrdersRequest(BaseModel):
    """Request model for cancelling all orders."""

    confirm_phrase: str | None = Field(
        default=None, description="Confirmation phrase for live trading"
    )


class StrategyEnableRequest(BaseModel):
    """Request model for enabling strategy."""

    enabled: bool = Field(..., description="Enable or disable strategy")


class StrategyParamsRequest(BaseModel):
    """Request model for updating strategy parameters."""

    params: dict[str, Any] = Field(..., description="Strategy parameters")


class RiskLimitsRequest(BaseModel):
    """Request model for updating risk limits."""

    max_open_positions: int | None = None
    max_daily_attempts: int | None = None
    max_exposure_usd: float | None = None
    min_notional_usd: float | None = None
    cooldown_seconds: int | None = None


class BacktestRequest(BaseModel):
    """Request model for running backtest."""

    symbol: str = Field(default="BTCUSDT", description="Trading symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000.0, description="Initial capital")
    strategy_id: str | None = None


class EvaluateRequest(BaseModel):
    """Request model for running evaluation."""

    symbol: str = Field(default="BTCUSDT", description="Trading symbol")
    days: int = Field(default=7, description="Number of days to evaluate")


class EventStreamMessage(BaseModel):
    """SSE message for event stream."""

    event: str = Field(..., description="Event type")
    data: dict[str, Any] = Field(..., description="Event data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LogStreamMessage(BaseModel):
    """SSE message for log stream."""

    level: str = Field(..., description="Log level")
    message: str = Field(..., description="Log message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    module: str | None = None
    request_id: str | None = None


class JobResponse(BaseModel):
    """Job response model."""

    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)


class AnalyticsSummary(BaseModel):
    """Analytics summary response."""

    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_trade_duration_sec: float = 0.0
    trades_per_day: float = 0.0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0


class BlockedSignalsAnalytics(BaseModel):
    """Blocked signals analytics response."""

    total_blocks: int = 0
    top_reasons: list[dict[str, Any]] = Field(default_factory=list)
    blocks_by_hour: dict[str, int] = Field(default_factory=dict)
    recent_blocks: list[ReasonCode] = Field(default_factory=list)
