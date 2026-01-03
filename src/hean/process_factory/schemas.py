"""Pydantic schemas for Process Factory."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProcessType(str, Enum):
    """Type of process."""

    TRADING = "TRADING"
    EARN = "EARN"
    CAMPAIGN = "CAMPAIGN"
    BONUS = "BONUS"
    DATA = "DATA"
    ACCESS = "ACCESS"
    OTHER = "OTHER"


class ActionStepKind(str, Enum):
    """Type of action step."""

    API_CALL = "API_CALL"
    HUMAN_TASK = "HUMAN_TASK"
    COMPUTE = "COMPUTE"
    WAIT = "WAIT"


class ActionStep(BaseModel):
    """A single step in a process."""

    step_id: str = Field(..., description="Unique step identifier")
    kind: ActionStepKind = Field(..., description="Type of action step")
    params: dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    timeout: int = Field(default=300, ge=1, description="Timeout in seconds")
    retries: int = Field(default=0, ge=0, description="Number of retries on failure")
    depends_on: list[str] = Field(
        default_factory=list, description="List of step_ids this step depends on"
    )
    description: str = Field(default="", description="Human-readable description")


class SafetyPolicy(BaseModel):
    """Safety constraints for a process."""

    max_capital_usd: float = Field(
        default=1000.0, gt=0, description="Maximum capital allocation in USD"
    )
    require_manual_approval: bool = Field(
        default=True, description="Require manual approval before execution"
    )
    max_risk_factor: float = Field(
        default=3.0, gt=0, description="Maximum risk factor (1.0 = low, 5.0 = high)"
    )
    allowed_timeframes: list[str] = Field(
        default_factory=list, description="Allowed timeframes for execution"
    )
    restrictions: dict[str, Any] = Field(
        default_factory=dict, description="Additional safety restrictions"
    )


class KillCondition(BaseModel):
    """Condition that triggers process termination."""

    metric: str = Field(..., description="Metric to monitor (e.g., 'fail_rate', 'pnl_sum')")
    threshold: float = Field(..., description="Threshold value")
    comparison: str = Field(
        default=">", description="Comparison operator: >, >=, <, <=, ==, !="
    )
    window_runs: int = Field(
        default=10, ge=1, description="Number of runs to evaluate over"
    )


class ScaleRule(BaseModel):
    """Rule for scaling process allocation."""

    metric: str = Field(..., description="Metric to monitor")
    threshold: float = Field(..., description="Threshold value")
    comparison: str = Field(
        default=">", description="Comparison operator: >, >=, <, <=, ==, !="
    )
    scale_multiplier: float = Field(
        default=1.5, gt=0, description="Multiplier to apply when condition met"
    )
    window_runs: int = Field(default=5, ge=1, description="Number of runs to evaluate over")
    max_allocation_usd: float = Field(
        default=10000.0, gt=0, description="Maximum allocation after scaling"
    )


class MeasurementSpec(BaseModel):
    """Specification for process measurement."""

    metrics: list[str] = Field(
        default_factory=lambda: [
            "capital_delta",
            "time_hours",
            "drawdown",
            "fail_rate",
            "roi",
            "volatility_exposure",
            "fee_drag",
        ],
        description="List of metrics to track",
    )
    attribution_rule: str = Field(
        default="direct",
        description="How to attribute PnL/reward to process (direct, proportional, time_weighted)",
    )


class ProcessDefinition(BaseModel):
    """Definition of a process."""

    id: str = Field(..., description="Unique process identifier")
    name: str = Field(..., description="Human-readable process name")
    type: ProcessType = Field(..., description="Process type")
    description: str = Field(..., description="Process description")
    requirements: dict[str, Any] = Field(
        default_factory=dict,
        description="Requirements dict (e.g., needs_bybit=True, needs_ui=False)",
    )
    inputs_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON schema for process inputs"
    )
    actions: list[ActionStep] = Field(
        default_factory=list, description="List of action steps"
    )
    expected_outputs: list[str] = Field(
        default_factory=list, description="Expected output keys"
    )
    safety: SafetyPolicy = Field(
        default_factory=SafetyPolicy, description="Safety policy"
    )
    measurement: MeasurementSpec = Field(
        default_factory=MeasurementSpec, description="Measurement specification"
    )
    kill_conditions: list[KillCondition] = Field(
        default_factory=list, description="Conditions that trigger termination"
    )
    scale_rules: list[ScaleRule] = Field(
        default_factory=list, description="Rules for scaling allocation"
    )
    version: str = Field(default="1.0.0", description="Process version")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class ProcessRunStatus(str, Enum):
    """Status of a process run."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    KILLED = "KILLED"
    PAUSED = "PAUSED"


class ProcessRun(BaseModel):
    """A single execution of a process."""

    run_id: str = Field(..., description="Unique run identifier")
    process_id: str = Field(..., description="Process definition ID")
    started_at: datetime = Field(..., description="Start timestamp")
    finished_at: datetime | None = Field(default=None, description="Finish timestamp")
    status: ProcessRunStatus = Field(default=ProcessRunStatus.PENDING, description="Run status")
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Measured metrics"
    )
    logs_ref: str | None = Field(default=None, description="Reference to log file/stream")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Run inputs")
    outputs: dict[str, Any] = Field(default_factory=dict, description="Run outputs")
    error: str | None = Field(default=None, description="Error message if failed")
    capital_allocated_usd: float = Field(
        default=0.0, ge=0, description="Capital allocated to this run"
    )


class ProcessPortfolioState(str, Enum):
    """State of a process in the portfolio."""

    NEW = "NEW"
    TESTING = "TESTING"
    CORE = "CORE"
    PAUSED = "PAUSED"
    KILLED = "KILLED"


class ProcessPortfolioEntry(BaseModel):
    """Entry in the process portfolio."""

    process_id: str = Field(..., description="Process ID")
    state: ProcessPortfolioState = Field(..., description="Current state")
    weight: float = Field(default=0.0, ge=0, le=1, description="Allocation weight (0-1)")
    runs_count: int = Field(default=0, ge=0, description="Total number of runs")
    wins: int = Field(default=0, ge=0, description="Number of successful runs")
    losses: int = Field(default=0, ge=0, description="Number of failed runs")
    pnl_sum: float = Field(default=0.0, description="Cumulative PnL in USD")
    max_dd: float = Field(default=0.0, ge=0, description="Maximum drawdown")
    avg_roi: float = Field(default=0.0, description="Average ROI")
    fail_rate: float = Field(default=0.0, ge=0, le=1, description="Failure rate (0-1)")
    time_efficiency: float = Field(
        default=0.0, ge=0, description="Time efficiency (profit per hour)"
    )
    last_run_at: datetime | None = Field(default=None, description="Last run timestamp")


class OpportunitySource(str, Enum):
    """Source of an opportunity."""

    TRADING = "TRADING"
    EARN = "EARN"
    CAMPAIGN = "CAMPAIGN"
    BONUS = "BONUS"
    DATA = "DATA"


class Opportunity(BaseModel):
    """An opportunity for capital allocation."""

    id: str = Field(..., description="Unique opportunity identifier")
    source: OpportunitySource = Field(..., description="Opportunity source")
    expected_profit_usd: float = Field(..., description="Expected profit in USD")
    time_hours: float = Field(..., ge=0, description="Expected time in hours")
    risk_factor: float = Field(default=1.0, ge=0.1, le=5.0, description="Risk factor (0.1-5.0)")
    complexity: int = Field(default=3, ge=1, le=5, description="Complexity (1-5)")
    confidence: float = Field(default=0.5, ge=0, le=1, description="Confidence (0-1)")
    process_id: str | None = Field(
        default=None, description="Associated process ID if known"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class DailyCapitalPlan(BaseModel):
    """Daily capital allocation plan."""

    date: datetime = Field(..., description="Plan date")
    reserve_usd: float = Field(default=0.0, ge=0, description="Reserve allocation in USD")
    active_usd: float = Field(default=0.0, ge=0, description="Active allocation in USD")
    experimental_usd: float = Field(
        default=0.0, ge=0, description="Experimental allocation in USD"
    )
    allocations: dict[str, float] = Field(
        default_factory=dict, description="Process ID -> USD allocation mapping"
    )
    total_capital_usd: float = Field(..., ge=0, description="Total available capital")


class BybitEnvironmentSnapshot(BaseModel):
    """Snapshot of Bybit environment state."""

    timestamp: datetime = Field(..., description="Snapshot timestamp")
    snapshot_id: str = Field(
        default_factory=lambda: f"snapshot_{datetime.now().isoformat()}",
        description="Unique snapshot identifier",
    )
    balances: dict[str, float] = Field(
        default_factory=dict, description="Asset -> balance mapping"
    )
    positions: list[dict[str, Any]] = Field(
        default_factory=list, description="Open positions"
    )
    open_orders: list[dict[str, Any]] = Field(
        default_factory=list, description="Open orders"
    )
    funding_rates: dict[str, float] = Field(
        default_factory=dict, description="Symbol -> funding rate mapping"
    )
    fees: dict[str, Any] = Field(default_factory=dict, description="Fee information")
    earn_availability: dict[str, Any] = Field(
        default_factory=dict, description="Earn product availability (UNKNOWN if not accessible)"
    )
    campaign_availability: dict[str, Any] = Field(
        default_factory=dict, description="Campaign availability (UNKNOWN if not accessible)"
    )
    source_flags: dict[str, str] = Field(
        default_factory=dict,
        description="Source flags: API | MANUAL | UNKNOWN for each data point",
    )
    # Enhanced fields
    fee_tier: str | None = Field(
        default=None, description="Fee tier (VIP0, VIP1, etc.) if accessible"
    )
    maker_fee_bps: float | None = Field(
        default=None, ge=0, description="Maker fee in basis points (if accessible)"
    )
    taker_fee_bps: float | None = Field(
        default=None, ge=0, description="Taker fee in basis points (if accessible)"
    )
    instrument_constraints: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Instrument constraints (min_order_size, leverage_limits, etc.)",
    )
    staleness_hours: float | None = Field(
        default=None,
        ge=0,
        description="Hours since snapshot was created (computed on load)",
    )

    def model_post_init(self, __context: Any) -> None:
        """Compute staleness on initialization."""
        if self.staleness_hours is None:
            age = (datetime.now() - self.timestamp).total_seconds() / 3600
            object.__setattr__(self, "staleness_hours", age)

    def is_stale(self, max_age_hours: float = 24.0) -> bool:
        """Check if snapshot is stale.

        Args:
            max_age_hours: Maximum age in hours before considered stale (default 24)

        Returns:
            True if snapshot is stale
        """
        if self.staleness_hours is None:
            age = (datetime.now() - self.timestamp).total_seconds() / 3600
            return age > max_age_hours
        return self.staleness_hours > max_age_hours

