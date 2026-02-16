"""Event types and DTOs for the event bus."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class EventType(str, Enum):
    """Event type enumeration."""

    # Market events
    TICK = "tick"
    FUNDING = "funding"
    FUNDING_UPDATE = "funding_update"
    ORDER_BOOK_UPDATE = "order_book_update"
    REGIME_UPDATE = "regime_update"

    # Strategy events
    SIGNAL = "signal"
    STRATEGY_PARAMS_UPDATED = "strategy_params_updated"

    # Risk events
    ORDER_REQUEST = "order_request"
    RISK_BLOCKED = "risk_blocked"
    RISK_ALERT = "risk_alert"

    # Execution events
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"

    # Portfolio events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATE = "position_update"
    POSITION_CLOSE_REQUEST = "position_close_request"
    EQUITY_UPDATE = "equity_update"
    PNL_UPDATE = "pnl_update"
    ORDER_DECISION = "order_decision"
    ORDER_EXIT_DECISION = "order_exit_decision"

    # System events
    STOP_TRADING = "stop_trading"
    KILLSWITCH_TRIGGERED = "killswitch_triggered"
    KILLSWITCH_RESET = "killswitch_reset"
    ERROR = "error"
    STATUS = "status"
    HEARTBEAT = "heartbeat"

    # Market structure / context events
    CANDLE = "candle"
    CONTEXT_UPDATE = "context_update"

    # Meta-learning events
    META_LEARNING_PATCH = "meta_learning_patch"

    # Brain/AI analysis events
    BRAIN_ANALYSIS = "brain_analysis"

    # Integration events (ContextAggregator)
    CONTEXT_READY = "context_ready"
    PHYSICS_UPDATE = "physics_update"
    ORACLE_PREDICTION = "oracle_prediction"
    OFI_UPDATE = "ofi_update"
    CAUSAL_SIGNAL = "causal_signal"

    # Self-analysis telemetry
    SELF_ANALYTICS = "self_analytics"

    # Council events
    COUNCIL_REVIEW = "council_review"
    COUNCIL_RECOMMENDATION = "council_recommendation"

    # Digital Organism events
    MARKET_GENOME_UPDATE = "market_genome_update"
    RISK_SIMULATION_RESULT = "risk_simulation_result"
    META_STRATEGY_UPDATE = "meta_strategy_update"

    # Archon orchestration events
    ARCHON_DIRECTIVE = "archon_directive"
    ARCHON_HEARTBEAT = "archon_heartbeat"
    SIGNAL_PIPELINE_UPDATE = "signal_pipeline_update"
    RECONCILIATION_ALERT = "reconciliation_alert"


@dataclass
class Event:
    """Base event class."""

    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: dict[str, Any] = field(default_factory=dict)


class Tick(BaseModel):
    """Market tick data."""

    symbol: str
    price: float
    timestamp: datetime
    volume: float = 0.0
    bid: float | None = None
    ask: float | None = None


class FundingRate(BaseModel):
    """Funding rate data."""

    symbol: str
    rate: float  # Funding rate as decimal (e.g., 0.0001 = 0.01%)
    timestamp: datetime
    next_funding_time: datetime | None = None


class Signal(BaseModel):
    """Trading signal from a strategy.

    Required fields for proper risk management:
    - confidence: Signal quality (0.0-1.0), used for Kelly sizing
    - urgency: Time sensitivity (0.0-1.0), used for execution routing
    - stop_loss: Required for position sizing, if None will be rejected by risk layer
    """

    strategy_id: str
    symbol: str
    side: str  # "buy" or "sell"
    entry_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    take_profit_1: float | None = None  # First take profit level (for break-even)
    size: float | None = None  # If None, risk layer will size
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)  # Signal quality for Kelly sizing
    urgency: float = Field(default=0.5, ge=0.0, le=1.0)  # Time sensitivity for execution
    metadata: dict[str, Any] = Field(default_factory=dict)
    prefer_maker: bool = False  # Prefer maker orders
    min_maker_edge_bps: int | None = None  # Minimum maker edge in bps


class OrderRequest(BaseModel):
    """Order request from risk layer."""

    signal_id: str
    strategy_id: str
    symbol: str
    side: str  # "buy" or "sell"
    size: float
    price: float | None = None  # None for market orders
    order_type: str = "market"  # "market" or "limit"
    stop_loss: float | None = None
    take_profit: float | None = None
    reduce_only: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: str) -> str:
        """Validate side is 'buy' or 'sell'."""
        if v not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {v}, must be 'buy' or 'sell'")
        return v

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: float) -> float:
        """Validate size is positive."""
        if v <= 0:
            raise ValueError(f"Invalid size: {v}, must be > 0")
        return v

    @field_validator("order_type")
    @classmethod
    def validate_order_type(cls, v: str) -> str:
        """Validate order_type is 'market' or 'limit'."""
        if v not in ("market", "limit"):
            raise ValueError(f"Invalid order_type: {v}, must be 'market' or 'limit'")
        return v

    @model_validator(mode="after")
    def validate_price_for_limit(self) -> "OrderRequest":
        """Validate price is provided for limit orders."""
        if self.order_type == "limit" and self.price is None:
            raise ValueError("Limit orders require a price")
        return self


class OrderStatus(str, Enum):
    """Order status enumeration."""

    PENDING = "pending"
    PLACED = "placed"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Order(BaseModel):
    """Order representation."""

    order_id: str
    strategy_id: str
    symbol: str
    side: str
    size: float
    filled_size: float = 0.0
    price: float | None = None
    avg_fill_price: float | None = None
    order_type: str
    status: OrderStatus
    stop_loss: float | None = None
    take_profit: float | None = None
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_maker: bool = False  # True if this is a maker order
    placed_at: datetime | None = None  # When order was placed


class Position(BaseModel):
    """Position representation."""

    position_id: str
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime | None = None
    strategy_id: str
    stop_loss: float | None = None
    take_profit: float | None = None
    take_profit_1: float | None = None  # First TP level (for break-even)
    break_even_activated: bool = False  # Whether break-even stop is active
    max_time_sec: int | None = None  # Maximum time in trade (seconds)
    metadata: dict[str, Any] = Field(default_factory=dict)  # Optional strategy-specific context


class EquitySnapshot(BaseModel):
    """Equity snapshot at a point in time."""

    timestamp: datetime
    equity: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    drawdown: float
    drawdown_pct: float
