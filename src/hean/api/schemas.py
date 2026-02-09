"""Pydantic schemas for API requests and responses."""

import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Symbol validation pattern (e.g., BTCUSDT, ETHUSDT)
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{2,10}USDT?$")


def validate_symbol(v: str) -> str:
    """Validate trading symbol format."""
    if not SYMBOL_PATTERN.match(v):
        raise ValueError(
            "Symbol must be in format like BTCUSDT, ETHUSDT (2-10 uppercase alphanumeric + USDT)"
        )
    return v


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
    size: float = Field(default=0.001, description="Order size", gt=0)
    price: float | None = Field(default=None, description="Limit price (optional)", gt=0)

    _validate_symbol = field_validator("symbol")(validate_symbol)

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: str) -> str:
        """Validate order side."""
        if v.lower() not in ("buy", "sell"):
            raise ValueError("Side must be 'buy' or 'sell'")
        return v.lower()


class TestRoundtripRequest(BaseModel):
    """Request model for end-to-end paper roundtrip test."""

    symbol: str = Field(default="BTCUSDT", description="Trading symbol")
    side: str = Field(default="buy", description="Order side: buy or sell")
    size: float = Field(default=0.001, description="Order size", gt=0)
    take_profit_pct: float = Field(default=0.3, description="TP distance in percent", gt=0)
    stop_loss_pct: float = Field(default=0.3, description="SL distance in percent", gt=0)
    hold_timeout_sec: int = Field(default=10, description="TTL seconds for forced exit", gt=0)

    _validate_symbol = field_validator("symbol")(validate_symbol)

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: str) -> str:
        """Validate order side."""
        if v.lower() not in ("buy", "sell"):
            raise ValueError("Side must be 'buy' or 'sell'")
        return v.lower()


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

    max_open_positions: int | None = Field(None, gt=0, le=100)
    max_daily_attempts: int | None = Field(None, gt=0, le=1000)
    max_exposure_usd: float | None = Field(None, gt=0)
    min_notional_usd: float | None = Field(None, gt=0)
    cooldown_seconds: int | None = Field(None, ge=0, le=86400)


class BacktestRequest(BaseModel):
    """Request model for running backtest."""

    symbol: str = Field(default="BTCUSDT", description="Trading symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000.0, description="Initial capital", gt=0)
    strategy_id: str | None = None

    _validate_symbol = field_validator("symbol")(validate_symbol)

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        """Validate date format YYYY-MM-DD."""
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("Date must be in format YYYY-MM-DD")
        return v


class EvaluateRequest(BaseModel):
    """Request model for running evaluation."""

    symbol: str = Field(default="BTCUSDT", description="Trading symbol")
    days: int = Field(default=7, description="Number of days to evaluate", gt=0, le=365)

    _validate_symbol = field_validator("symbol")(validate_symbol)


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


# WebSocket Message Schemas


class WebSocketActionType(str, Enum):
    """Valid WebSocket action types."""

    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"


class WebSocketTopicType(str, Enum):
    """Valid WebSocket subscription topics."""

    SYSTEM_STATUS = "system_status"
    TELEMETRY = "telemetry"
    MARKET_DATA = "market_data"
    TRADING_SIGNALS = "trading_signals"
    POSITIONS = "positions"
    ORDERS = "orders"
    PERFORMANCE = "performance"
    RISK = "risk"
    LOGS = "logs"
    SNAPSHOT = "snapshot"
    SYSTEM_HEARTBEAT = "system_heartbeat"  # UI compatibility
    ORDER_DECISIONS = "order_decisions"  # UI compatibility
    RISK_EVENTS = "risk_events"  # UI compatibility
    STRATEGY_EVENTS = "strategy_events"  # UI compatibility
    MARKET_TICKS = "market_ticks"  # UI compatibility
    TRADING_METRICS = "trading_metrics"  # UI compatibility


class WebSocketMessage(BaseModel):
    """Schema for incoming WebSocket messages."""

    action: WebSocketActionType = Field(
        ..., description="Action to perform (subscribe, unsubscribe, ping)"
    )
    topic: WebSocketTopicType | None = Field(
        None,
        description="Topic to subscribe/unsubscribe from (required for subscribe/unsubscribe)",
    )
    data: dict[str, Any] | None = Field(
        None, description="Optional data payload for the action"
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate that topic is provided for subscribe/unsubscribe actions."""
        if self.action in (
            WebSocketActionType.SUBSCRIBE,
            WebSocketActionType.UNSUBSCRIBE,
        ):
            if self.topic is None:
                raise ValueError(f"Topic is required for {self.action.value} action")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "action": "subscribe",
                    "topic": "system_status",
                },
                {
                    "action": "unsubscribe",
                    "topic": "telemetry",
                },
                {
                    "action": "ping",
                },
            ]
        }
    }
