# ARCHON — Technical Implementation Prompt

> **Autonomous Runtime Controller & Holistic Orchestration Nexus**
> Центральный Мозг-Оркестратор торговой системы HEAN.

---

## ROLE

You are a senior Python backend engineer implementing ARCHON — the central orchestration brain for the HEAN event-driven crypto trading system. You write production-grade async Python 3.11+ code. You follow existing HEAN patterns exactly. You do NOT break existing functionality.

---

## CONTEXT: What is HEAN

HEAN is an event-driven crypto trading system for Bybit Testnet. All components communicate via `EventBus` (async pub/sub with priority queues). The signal chain is:

```
TICK → Strategy → filter cascade → SIGNAL → RiskGovernor → ORDER_REQUEST → ExecutionRouter → Bybit HTTP → ORDER_FILLED → Position update
```

### Existing Architecture (DO NOT MODIFY these files unless explicitly stated)

| Component | File | Purpose |
|-----------|------|---------|
| EventBus | `src/hean/core/bus.py` | Async pub/sub, 3 priority queues (CRITICAL/NORMAL/LOW), fast-path for SIGNAL/ORDER_REQUEST/ORDER_FILLED, circuit breaker |
| Event/EventType | `src/hean/core/types.py` | Event dataclass (`event_type`, `timestamp`, `data` dict), EventType enum |
| Signal | `src/hean/core/types.py` | Pydantic model: `strategy_id`, `symbol`, `side`, `entry_price`, `stop_loss`, `confidence`, `urgency`, `metadata` |
| OrderRequest | `src/hean/core/types.py` | Pydantic model: `signal_id`, `strategy_id`, `symbol`, `side`, `size`, `price`, `order_type`, `stop_loss`, `take_profit`, `metadata` |
| TradingSystem | `src/hean/main.py` | God-object (~3800 lines), creates and wires ALL components in `start()`, main loop in `run()` |
| EngineFacade | `src/hean/api/engine_facade.py` | Unified API interface to TradingSystem. `get_facade()` returns global instance |
| FastAPI app | `src/hean/api/main.py` | Registers routers under `/api/v1/`, middleware injects `engine_facade` into `request.state` |
| BaseStrategy | `src/hean/strategies/base.py` | ABC with `strategy_id`, `_bus`, `start()/stop()`, subscribes to TICK/FUNDING/REGIME_UPDATE/CONTEXT_READY/PHYSICS_UPDATE |
| RiskGovernor | `src/hean/risk/risk_governor.py` | State machine: `NORMAL → SOFT_BRAKE → QUARANTINE → HARD_STOP`. Uses high-water-mark drawdown |
| KillSwitch | `src/hean/risk/killswitch.py` | Emergency halt at >20% drawdown |
| ComponentRegistry | `src/hean/core/system/component_registry.py` | Lifecycle manager for optional components (RL Risk, Oracle, Strategy Allocator) |
| HealthMonitor | `src/hean/core/system/health_monitor.py` | Pings 5 modules (cpp_core, redis, db, exchange_api, frontend_socket) |
| HEANSettings | `src/hean/config.py` | Pydantic BaseSettings from `.env`. All features gated by flags |
| SymbiontXBridge | `src/hean/symbiont_x/bridge.py` | GA evolution bridge. `_evaluate_fitness()` (line 358) evaluates ALL genomes on SAME static history — BUG |
| BacktestEngine | `src/hean/symbiont_x/backtesting/backtest_engine.py` | Backtesting for Symbiont X genomes |
| PortfolioAccounting | `src/hean/portfolio/accounting.py` | Position tracking, PnL calculation |
| DuckDBStore | `src/hean/storage/duckdb_store.py` | Persistence for ticks, physics, brain analyses |
| Logging | `src/hean/logging/__init__.py` | `get_logger(__name__)` with request_id/trace_id context |

### Critical Code Patterns You MUST Follow

```python
# Logging — ALWAYS use this, never bare print() or stdlib logging
from hean.logging import get_logger
logger = get_logger(__name__)

# Event publishing
await self._bus.publish(Event(event_type=EventType.SIGNAL, data={"signal": signal_obj, "key": "value"}))

# Event subscription
self._bus.subscribe(EventType.SIGNAL, self._on_signal)

# Event handler signature — ALWAYS async, takes Event
async def _on_signal(self, event: Event) -> None:
    data = event.data
    signal = data.get("signal")

# Config — add to HEANSettings in src/hean/config.py
archon_enabled: bool = Field(default=True, description="Enable ARCHON orchestrator")

# Ruff: line-length=100, py311 target
# mypy: strict mode (disallow_untyped_defs, strict_equality)
# asyncio_mode = "auto" in pytest — no @pytest.mark.asyncio needed
```

### EventBus Behavior You Must Understand

1. **Fast-path events** (`SIGNAL`, `ORDER_REQUEST`, `ORDER_FILLED`) bypass queues — dispatched immediately via `_dispatch_fast()`. This means they are processed SYNCHRONOUSLY in the caller's context.
2. **Handler exceptions** (bus.py line 476-482): If a handler raises, it's logged but **the signal is lost**. No retry, no dead-letter.
3. **No acknowledgment**: After `_dispatch()`, there's no confirmation that handlers succeeded.
4. **Circuit breaker**: At 95% queue utilization, LOW priority events are dropped.

### TradingSystem Integration Points

In `src/hean/main.py`, `TradingSystem.__init__()` (line ~91):
```python
class TradingSystem:
    def __init__(self, mode="run", bus=None):
        self._bus = bus or EventBus()
        self._accounting = PortfolioAccounting(...)
        self._execution_router = ...
        self._order_manager = ...
        self._killswitch = ...
        self._strategies: list[BaseStrategy] = []
        # ... ~50 more fields
```

In `start()` (line ~392): Components initialized in order, subscriptions set up.
In `stop()` (line ~1103): Reverse-order shutdown.

---

## PROBLEMS ARCHON SOLVES

| # | Problem | Where in code | Impact |
|---|---------|---------------|--------|
| 1 | **Signals lost on handler exception** | `bus.py:476-482` — exception logged, signal gone | Missed trades, silent failures |
| 2 | **No signal lifecycle tracking** | No correlation IDs, no stage timestamps | Cannot debug "why didn't my signal execute?" |
| 3 | **No component health overview** | `HealthMonitor` pings 5 modules, ignores strategies/risk/execution | Blind to degraded components |
| 4 | **Symbiont X GA runs idle** | `bridge.py:358-417` — same fitness for all genomes | Evolution doesn't actually optimize |
| 5 | **No state reconciliation** | `PositionReconciler` exists but limited | Local state can drift from exchange |
| 6 | **No audit trail** | Decisions/signals not persistently logged | Cannot replay or post-mortem |
| 7 | **No central decision-making** | Components autonomous, no coordinator | No adaptive strategy activation/deactivation |

---

## ARCHON ARCHITECTURE

```
src/hean/archon/
    __init__.py                    # Exports: Archon, SignalPipelineManager, etc.
    archon.py                      # Main Archon class — entry point, wires sub-systems
    protocols.py                   # ComponentState enum, ArchonComponent protocol
    directives.py                  # DirectiveType enum, Directive/DirectiveAck dataclasses
    signal_pipeline.py             # SignalStage enum, SignalTrace dataclass
    signal_pipeline_manager.py     # SignalPipelineManager — guaranteed delivery tracking
    dead_letter_queue.py           # DeadLetterQueue — failed/blocked/timed-out signals
    reconciler.py                  # ArchonReconciler — position/balance/order sync
    heartbeat.py                   # HeartbeatRegistry — component liveness tracking
    health_matrix.py               # HealthMatrix — composite health score (0-100)
    cortex.py                      # Cortex — decision engine (strategy activation, risk mode)
    genome_director.py             # GenomeDirector — directed Symbiont X evolution via backtesting
    chronicle.py                   # Chronicle — audit trail with DuckDB persistence

src/hean/api/routers/archon.py    # FastAPI router: /api/v1/archon/*
tests/test_archon/                 # Test directory
    __init__.py
    test_signal_pipeline.py
    test_heartbeat.py
    test_health_matrix.py
    test_reconciler.py
    test_cortex.py
    test_genome_director.py
    test_chronicle.py
    test_archon_integration.py
```

---

## IMPLEMENTATION SPEC — PHASE BY PHASE

### PHASE 0: Foundation

**Files to create:** `src/hean/archon/__init__.py`, `protocols.py`, `directives.py`, `signal_pipeline.py`
**Files to modify:** `src/hean/core/types.py` (add EventTypes), `src/hean/config.py` (add settings)

#### 0.1. Add EventTypes to `src/hean/core/types.py`

Add these AFTER the existing `META_STRATEGY_UPDATE` entry (line 81):

```python
    # Archon orchestration events
    ARCHON_DIRECTIVE = "archon_directive"
    ARCHON_HEARTBEAT = "archon_heartbeat"
    SIGNAL_PIPELINE_UPDATE = "signal_pipeline_update"
    RECONCILIATION_ALERT = "reconciliation_alert"
```

#### 0.2. Add settings to `src/hean/config.py`

Add to `HEANSettings` class:

```python
    # ARCHON Brain-Orchestrator
    archon_enabled: bool = Field(default=True, description="Enable ARCHON orchestrator")
    archon_signal_pipeline_enabled: bool = Field(default=True, description="Enable signal lifecycle tracking")
    archon_reconciliation_enabled: bool = Field(default=True, description="Enable periodic state reconciliation")
    archon_cortex_enabled: bool = Field(default=True, description="Enable Cortex decision engine")
    archon_cortex_interval_sec: int = Field(default=30, description="Cortex decision loop interval")
    archon_heartbeat_interval_sec: float = Field(default=5.0, description="Component heartbeat interval")
    archon_signal_timeout_sec: float = Field(default=10.0, description="Signal stage timeout before dead-letter")
    archon_max_active_signals: int = Field(default=1000, description="Max concurrent tracked signals")
    archon_reconciliation_interval_sec: int = Field(default=30, description="Position reconciliation interval")
    archon_chronicle_enabled: bool = Field(default=True, description="Enable audit trail")
    archon_chronicle_max_memory: int = Field(default=10000, description="Max in-memory chronicle entries")
```

#### 0.3. Create `src/hean/archon/protocols.py`

```python
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
```

#### 0.4. Create `src/hean/archon/directives.py`

```python
"""Directive types for ARCHON → component communication."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


class DirectiveType(str, Enum):
    """Types of directives ARCHON can issue."""
    ACTIVATE_STRATEGY = "activate_strategy"
    DEACTIVATE_STRATEGY = "deactivate_strategy"
    UPDATE_STRATEGY_PARAMS = "update_strategy_params"
    SET_RISK_MODE = "set_risk_mode"
    QUARANTINE_SYMBOL = "quarantine_symbol"
    INITIATE_RECONCILIATION = "initiate_reconciliation"
    TRIGGER_EVOLUTION_CYCLE = "trigger_evolution_cycle"
    PAUSE_TRADING = "pause_trading"
    RESUME_TRADING = "resume_trading"


@dataclass
class Directive:
    """Command from ARCHON to a component."""
    directive_type: DirectiveType
    target_component: str
    params: dict[str, Any] = field(default_factory=dict)
    directive_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    issued_at: datetime = field(default_factory=datetime.utcnow)
    requires_ack: bool = True


@dataclass
class DirectiveAck:
    """Acknowledgment of a directive."""
    directive_id: str
    component_id: str
    success: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    executed_at: datetime = field(default_factory=datetime.utcnow)
```

#### 0.5. Create `src/hean/archon/signal_pipeline.py`

```python
"""Signal lifecycle tracking types."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


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
        self.stages.append(StageRecord(
            stage=stage,
            timestamp=now,
            details=details or {},
        ))

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
```

---

### PHASE 1: Signal Pipeline Manager (HIGHEST PRIORITY)

**File:** `src/hean/archon/signal_pipeline_manager.py`

**What it does:** Subscribes to EventBus events and tracks every signal from GENERATED to terminal state. Detects timeouts. Maintains dead-letter queue. Provides metrics.

**CRITICAL DESIGN CONSTRAINT:** Signal Pipeline is **PASSIVE** — it only LISTENS to existing events. It does NOT modify the event flow. It does NOT add latency to the fast-path. It subscribes as an additional handler alongside existing handlers.

**How correlation_id propagation works:**
- When Pipeline sees a SIGNAL event, it generates a `correlation_id` and stores it keyed by `(strategy_id, symbol, side, timestamp)` — a "fingerprint"
- When Pipeline sees ORDER_REQUEST, it matches by fingerprint to find the correlation_id
- When Pipeline sees ORDER_FILLED/REJECTED, it matches by `order_id` from ORDER_PLACED stage
- This approach requires NO modification to existing event publishers

**Implementation:**

```python
"""Signal Pipeline Manager — guaranteed delivery tracking."""

import asyncio
import time
from collections import OrderedDict
from typing import Any

from hean.archon.signal_pipeline import SignalStage, SignalTrace
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class DeadLetterQueue:
    """Stores signals that failed to complete the pipeline."""

    def __init__(self, max_size: int = 500) -> None:
        self._entries: list[SignalTrace] = []
        self._max_size = max_size

    def add(self, trace: SignalTrace) -> None:
        self._entries.append(trace)
        if len(self._entries) > self._max_size:
            self._entries.pop(0)

    @property
    def size(self) -> int:
        return len(self._entries)

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self._entries[-n:]]

    def clear(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count


class SignalPipelineManager:
    """Tracks signal lifecycle from GENERATED to terminal state.

    PASSIVE observer — subscribes to EventBus events but does NOT
    modify the event flow or add latency to fast-path dispatch.

    Features:
    - Correlation ID tracking for end-to-end signal tracing
    - Stage transition timestamps for latency measurement
    - Timeout detection for stale signals
    - Dead letter queue for failed/blocked signals
    - Aggregate metrics: fill rate, avg latency, block rate
    """

    def __init__(
        self,
        bus: EventBus,
        max_active: int = 1000,
        stage_timeout_sec: float = 10.0,
    ) -> None:
        self._bus = bus
        self._max_active = max_active
        self._stage_timeout = stage_timeout_sec

        # Active signals being tracked: correlation_id -> SignalTrace
        self._active: OrderedDict[str, SignalTrace] = OrderedDict()

        # Fingerprint index for matching events without correlation_id
        # Key: (strategy_id, symbol, side) -> correlation_id
        # Used to link ORDER_REQUEST back to SIGNAL
        self._fingerprint_index: dict[tuple[str, str, str], str] = {}

        # Order ID index: order_id -> correlation_id
        self._order_index: dict[str, str] = {}

        # Dead letter queue
        self.dead_letters = DeadLetterQueue()

        # Completed signals (ring buffer for recent history)
        self._completed: list[SignalTrace] = []
        self._max_completed = 200

        # Metrics
        self._metrics = {
            "signals_tracked": 0,
            "signals_completed": 0,  # Reached ORDER_FILLED or POSITION_OPENED
            "signals_blocked": 0,    # RISK_BLOCKED
            "signals_rejected": 0,   # ORDER_REJECTED
            "signals_timed_out": 0,
            "signals_evicted": 0,    # Evicted from active due to max_active
            "total_latency_ms": 0.0,
            "completed_count_for_avg": 0,
        }

        self._running = False
        self._timeout_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start pipeline tracking. Subscribe to events."""
        self._running = True

        # Subscribe to signal lifecycle events
        self._bus.subscribe(EventType.SIGNAL, self._on_signal)
        self._bus.subscribe(EventType.ORDER_REQUEST, self._on_order_request)
        self._bus.subscribe(EventType.RISK_BLOCKED, self._on_risk_blocked)
        self._bus.subscribe(EventType.ORDER_PLACED, self._on_order_placed)
        self._bus.subscribe(EventType.ORDER_FILLED, self._on_order_filled)
        self._bus.subscribe(EventType.ORDER_REJECTED, self._on_order_rejected)
        self._bus.subscribe(EventType.ORDER_CANCELLED, self._on_order_cancelled)
        self._bus.subscribe(EventType.POSITION_OPENED, self._on_position_opened)

        # Background task: check for timed-out signals
        self._timeout_task = asyncio.create_task(self._timeout_loop())

        logger.info("[SignalPipeline] Started — tracking signal lifecycle")

    async def stop(self) -> None:
        """Stop pipeline tracking."""
        self._running = False
        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass

        # Unsubscribe
        self._bus.unsubscribe(EventType.SIGNAL, self._on_signal)
        self._bus.unsubscribe(EventType.ORDER_REQUEST, self._on_order_request)
        self._bus.unsubscribe(EventType.RISK_BLOCKED, self._on_risk_blocked)
        self._bus.unsubscribe(EventType.ORDER_PLACED, self._on_order_placed)
        self._bus.unsubscribe(EventType.ORDER_FILLED, self._on_order_filled)
        self._bus.unsubscribe(EventType.ORDER_REJECTED, self._on_order_rejected)
        self._bus.unsubscribe(EventType.ORDER_CANCELLED, self._on_order_cancelled)
        self._bus.unsubscribe(EventType.POSITION_OPENED, self._on_position_opened)

        stats = self.get_status()
        logger.info(
            f"[SignalPipeline] Stopped — "
            f"tracked={stats['signals_tracked']}, "
            f"completed={stats['signals_completed']}, "
            f"blocked={stats['signals_blocked']}, "
            f"dead_letters={stats['dead_letter_count']}"
        )

    # ── Event handlers ──────────────────────────────────────────────

    async def _on_signal(self, event: Event) -> None:
        """Track a newly generated signal."""
        data = event.data
        signal = data.get("signal")
        if not signal:
            return

        # Extract fields (handle both Signal objects and dicts)
        strategy_id = getattr(signal, "strategy_id", "") or data.get("strategy_id", "")
        symbol = getattr(signal, "symbol", "") or data.get("symbol", "")
        side = getattr(signal, "side", "") or data.get("side", "")
        confidence = getattr(signal, "confidence", 0.0)

        trace = SignalTrace(
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            confidence=confidence,
        )
        trace.advance(SignalStage.GENERATED)

        # Store in active
        self._active[trace.correlation_id] = trace

        # Create fingerprint for matching subsequent events
        fp = (strategy_id, symbol, side)
        self._fingerprint_index[fp] = trace.correlation_id

        self._metrics["signals_tracked"] += 1

        # Evict oldest if over limit
        while len(self._active) > self._max_active:
            evicted_id, evicted = self._active.popitem(last=False)
            evicted.advance(SignalStage.DEAD_LETTER, {"reason": "evicted_max_active"})
            self.dead_letters.add(evicted)
            self._metrics["signals_evicted"] += 1
            # Clean up indices
            self._cleanup_indices(evicted_id, evicted)

    async def _on_order_request(self, event: Event) -> None:
        """Track signal advancing to risk-approved ORDER_REQUEST."""
        corr_id = self._match_event(event)
        if corr_id and corr_id in self._active:
            self._active[corr_id].advance(SignalStage.RISK_APPROVED, {
                "signal_id": event.data.get("signal_id", ""),
            })

    async def _on_risk_blocked(self, event: Event) -> None:
        """Track signal blocked by risk layer."""
        corr_id = self._match_event(event)
        if corr_id and corr_id in self._active:
            trace = self._active.pop(corr_id)
            trace.advance(SignalStage.RISK_BLOCKED, {
                "reason": event.data.get("reason", "unknown"),
                "risk_state": event.data.get("risk_state", ""),
            })
            self.dead_letters.add(trace)
            self._metrics["signals_blocked"] += 1
            self._cleanup_indices(corr_id, trace)

    async def _on_order_placed(self, event: Event) -> None:
        """Track order placed on exchange."""
        corr_id = self._match_event(event)
        order_id = event.data.get("order_id", "")
        if corr_id and corr_id in self._active:
            self._active[corr_id].advance(SignalStage.ORDER_PLACED, {
                "order_id": order_id,
            })
            self._active[corr_id].order_id = order_id
            # Index by order_id for fill/reject matching
            if order_id:
                self._order_index[order_id] = corr_id

    async def _on_order_filled(self, event: Event) -> None:
        """Track order filled — signal pipeline success."""
        corr_id = self._match_by_order_id(event) or self._match_event(event)
        if corr_id and corr_id in self._active:
            trace = self._active.pop(corr_id)
            trace.advance(SignalStage.ORDER_FILLED, {
                "fill_price": event.data.get("fill_price", event.data.get("price", 0)),
                "fill_qty": event.data.get("fill_qty", event.data.get("qty", 0)),
            })
            self._complete_signal(corr_id, trace)

    async def _on_order_rejected(self, event: Event) -> None:
        """Track order rejected by exchange."""
        corr_id = self._match_by_order_id(event) or self._match_event(event)
        if corr_id and corr_id in self._active:
            trace = self._active.pop(corr_id)
            trace.advance(SignalStage.ORDER_REJECTED, {
                "reason": event.data.get("reason", "unknown"),
            })
            self.dead_letters.add(trace)
            self._metrics["signals_rejected"] += 1
            self._cleanup_indices(corr_id, trace)

    async def _on_order_cancelled(self, event: Event) -> None:
        """Track order cancelled."""
        corr_id = self._match_by_order_id(event) or self._match_event(event)
        if corr_id and corr_id in self._active:
            trace = self._active.pop(corr_id)
            trace.advance(SignalStage.ORDER_CANCELLED, {
                "reason": event.data.get("reason", ""),
            })
            self.dead_letters.add(trace)
            self._cleanup_indices(corr_id, trace)

    async def _on_position_opened(self, event: Event) -> None:
        """Track position opened — final success state."""
        # Try to match by recent completed signals
        pass  # Optional: link position_id to trace

    # ── Matching logic ──────────────────────────────────────────────

    def _match_event(self, event: Event) -> str | None:
        """Match event to active signal trace via fingerprint."""
        data = event.data
        # Try direct correlation_id first (if we add it later)
        corr_id = data.get("_correlation_id")
        if corr_id:
            return corr_id

        # Match by fingerprint
        strategy_id = data.get("strategy_id", "")
        symbol = data.get("symbol", "")
        side = data.get("side", "")

        # Try extracting from nested objects
        signal = data.get("signal")
        if signal:
            strategy_id = strategy_id or getattr(signal, "strategy_id", "")
            symbol = symbol or getattr(signal, "symbol", "")
            side = side or getattr(signal, "side", "")

        order_request = data.get("order_request")
        if order_request:
            strategy_id = strategy_id or getattr(order_request, "strategy_id", "")
            symbol = symbol or getattr(order_request, "symbol", "")
            side = side or getattr(order_request, "side", "")

        fp = (strategy_id, symbol, side)
        return self._fingerprint_index.get(fp)

    def _match_by_order_id(self, event: Event) -> str | None:
        """Match event by order_id."""
        order_id = event.data.get("order_id", "")
        return self._order_index.get(order_id) if order_id else None

    def _cleanup_indices(self, corr_id: str, trace: SignalTrace) -> None:
        """Remove index entries for a completed/dead signal."""
        fp = (trace.strategy_id, trace.symbol, trace.side)
        if self._fingerprint_index.get(fp) == corr_id:
            del self._fingerprint_index[fp]
        if trace.order_id and trace.order_id in self._order_index:
            del self._order_index[trace.order_id]

    def _complete_signal(self, corr_id: str, trace: SignalTrace) -> None:
        """Move signal to completed list and update metrics."""
        self._completed.append(trace)
        if len(self._completed) > self._max_completed:
            self._completed.pop(0)

        self._metrics["signals_completed"] += 1
        self._metrics["total_latency_ms"] += trace.latency_ms
        self._metrics["completed_count_for_avg"] += 1
        self._cleanup_indices(corr_id, trace)

    # ── Timeout detection ───────────────────────────────────────────

    async def _timeout_loop(self) -> None:
        """Periodically check for signals stuck in non-terminal stage."""
        while self._running:
            try:
                await asyncio.sleep(2.0)  # Check every 2 seconds
                now = time.time()
                timed_out_ids: list[str] = []

                for corr_id, trace in self._active.items():
                    if trace.stages:
                        last_ts = trace.stages[-1].timestamp.timestamp()
                        if now - last_ts > self._stage_timeout:
                            timed_out_ids.append(corr_id)

                for corr_id in timed_out_ids:
                    trace = self._active.pop(corr_id)
                    prev_stage = trace.current_stage.value
                    trace.advance(SignalStage.ORDER_TIMEOUT, {
                        "stuck_at": prev_stage,
                        "timeout_sec": self._stage_timeout,
                    })
                    self.dead_letters.add(trace)
                    self._metrics["signals_timed_out"] += 1
                    self._cleanup_indices(corr_id, trace)
                    logger.warning(
                        f"[SignalPipeline] Signal {corr_id[:8]} timed out "
                        f"at stage '{prev_stage}' after {self._stage_timeout}s"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SignalPipeline] Timeout loop error: {e}", exc_info=True)

    # ── Status / Metrics ────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Get pipeline status for API."""
        tracked = self._metrics["signals_tracked"]
        completed = self._metrics["signals_completed"]
        avg_latency = 0.0
        if self._metrics["completed_count_for_avg"] > 0:
            avg_latency = (
                self._metrics["total_latency_ms"]
                / self._metrics["completed_count_for_avg"]
            )

        return {
            "active_count": len(self._active),
            "dead_letter_count": self.dead_letters.size,
            "signals_tracked": tracked,
            "signals_completed": completed,
            "signals_blocked": self._metrics["signals_blocked"],
            "signals_rejected": self._metrics["signals_rejected"],
            "signals_timed_out": self._metrics["signals_timed_out"],
            "signals_evicted": self._metrics["signals_evicted"],
            "fill_rate_pct": round((completed / tracked * 100) if tracked > 0 else 0.0, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "recent_dead_letters": self.dead_letters.recent(10),
            "active_signals": [
                t.to_dict() for t in list(self._active.values())[-10:]
            ],
        }

    def get_trace(self, correlation_id: str) -> dict[str, Any] | None:
        """Get trace for a specific signal by correlation_id."""
        if correlation_id in self._active:
            return self._active[correlation_id].to_dict()
        for t in reversed(self._completed):
            if t.correlation_id == correlation_id:
                return t.to_dict()
        for t in self.dead_letters._entries:
            if t.correlation_id == correlation_id:
                return t.to_dict()
        return None
```

**Tests for Phase 1** (`tests/test_archon/test_signal_pipeline.py`):

Test the following scenarios:
1. Signal GENERATED → ORDER_REQUEST → ORDER_PLACED → ORDER_FILLED (happy path)
2. Signal GENERATED → RISK_BLOCKED → dead letter
3. Signal GENERATED → ORDER_PLACED → ORDER_REJECTED → dead letter
4. Signal GENERATED → (no further events) → TIMEOUT → dead letter
5. Max active signals eviction
6. Metrics accuracy: fill_rate_pct, avg_latency_ms
7. `get_trace()` finds active, completed, and dead-letter signals
8. `get_status()` returns correct counts

Each test creates an `EventBus`, starts it, creates `SignalPipelineManager`, publishes events manually, and asserts trace stages. Use `await asyncio.sleep(0.01)` between publishes to let handlers process.

---

### PHASE 2: Heartbeat + Health Matrix

**File:** `src/hean/archon/heartbeat.py`

HeartbeatRegistry: tracks `component_id → last_heartbeat_timestamp`. Methods:
- `register(component_id, interval_sec)` — register component
- `beat(component_id, metadata=None)` — record heartbeat
- `get_status() -> dict[str, dict]` — per-component: `last_beat_ago_sec`, `missed_beats`, `healthy`
- `get_unhealthy() -> list[str]` — list of dead/unresponsive components

**File:** `src/hean/archon/health_matrix.py`

HealthMatrix: aggregates health from EventBus, HeartbeatRegistry, and component health_check() calls. Computes composite score 0-100.

Score formula:
- 40% EventBus health (not degraded, no circuit breaker, low drop rate)
- 30% Component heartbeats (% of components that are healthy)
- 20% Signal Pipeline fill rate (> 50% is good)
- 10% Recent handler errors (0 errors = full score)

Methods:
- `async def start()` — begins periodic health assessment
- `async def stop()`
- `def get_composite_score() -> float` — 0-100
- `async def get_full_status() -> dict` — detailed breakdown

---

### PHASE 3: Chronicle (Audit Trail)

**File:** `src/hean/archon/chronicle.py`

Chronicle: subscribes to KEY events and stores structured records. In-memory ring buffer + optional DuckDB persistence.

Events to chronicle:
- `SIGNAL` — signal generated (who, what, confidence)
- `ORDER_REQUEST` — risk approved
- `RISK_BLOCKED` — risk rejected (reason)
- `ORDER_FILLED` — execution success (price, qty)
- `ORDER_REJECTED` — exchange rejected
- `KILLSWITCH_TRIGGERED` — emergency halt
- `ARCHON_DIRECTIVE` — ARCHON decision

Each record: `{timestamp, event_type, correlation_id, strategy_id, symbol, details}`.

Methods:
- `async def start() / stop()`
- `def query(event_type=None, symbol=None, strategy_id=None, limit=50) -> list[dict]`
- `def get_signal_journey(correlation_id) -> list[dict]` — full trace

---

### PHASE 4: Reconciler

**File:** `src/hean/archon/reconciler.py`

ArchonReconciler: periodic loops that compare local state with exchange state.

Three reconciliation loops:
1. **Position Reconciliation** (every `archon_reconciliation_interval_sec`):
   - Call `bybit_http.get_positions()`
   - Compare with `accounting.get_positions()`
   - Log discrepancies, publish `RECONCILIATION_ALERT` event
2. **Balance Reconciliation** (every 60s):
   - Call `bybit_http.get_wallet_balance()`
   - Compare with local equity tracking
3. **Order Reconciliation** (every 15s):
   - Call `bybit_http.get_open_orders()`
   - Compare with `order_manager` state

On discrepancy:
- Log WARNING with details
- Publish `Event(event_type=EventType.RECONCILIATION_ALERT, data={...})`
- Cortex can decide whether to auto-fix or alert

Dependencies: `PortfolioAccounting`, `OrderManager`, `BybitHTTPClient` — all available from TradingSystem.

---

### PHASE 5: Cortex (Decision Engine)

**File:** `src/hean/archon/cortex.py`

Cortex: runs a periodic decision loop (every `archon_cortex_interval_sec` seconds). Reads system state, makes strategic decisions, issues directives.

Decision loop pseudocode:
```
every 30 seconds:
    1. Read HealthMatrix composite score
    2. Read SignalPipeline fill rate and dead letter count
    3. Read RiskGovernor state
    4. Read per-strategy performance (from accounting)
    5. Read Physics market phase

    DECISIONS:
    - If health_score < 40: issue PAUSE_TRADING directive
    - If health_score recovered > 70 and was paused: issue RESUME_TRADING
    - If strategy X has Sharpe < 0 over last 50 trades: issue DEACTIVATE_STRATEGY
    - If fill_rate < 30%: log warning, increase signal_timeout
    - If market phase == "distribution" and risk_state == NORMAL: recommend SOFT_BRAKE
    - If dead_letter_count growing fast: investigate stuck handlers
```

Methods:
- `async def run_decision_loop()` — main loop (run as asyncio.Task)
- `async def stop()`
- `def get_status() -> dict` — current directives, last decision, mode
- `async def _evaluate() -> list[Directive]` — one evaluation cycle
- `async def _execute_directive(directive)` — apply directive

**IMPORTANT:** Cortex decisions are ADVISORY by default. It publishes `ARCHON_DIRECTIVE` events. Components that opt-in can listen and react. No forced overrides of existing risk/execution logic.

---

### PHASE 6: Genome Director

**File:** `src/hean/archon/genome_director.py`

GenomeDirector: replaces the broken `_evaluate_fitness` in SymbiontXBridge with backtesting-based evaluation.

Key change: Instead of evaluating all genomes on the same historical PnL data, each genome's parameters are tested via `BacktestEngine.run_fast()` with those specific params.

Fitness = weighted multi-objective:
- 35% Sharpe ratio (normalized: 3.0 = perfect)
- 25% Profit factor (normalized: 3.0 = perfect)
- 20% Win rate (0-1)
- 20% Max drawdown penalty (20% DD = 0 score)

Promotion criteria:
- New genome fitness must be > current * 1.15 (15% improvement minimum)
- Backtest must have >= 20 trades
- Max drawdown must be < 10%

---

### PHASE 7: Main Archon Class + Integration

**File:** `src/hean/archon/archon.py`

```python
"""ARCHON — Central Brain-Orchestrator."""

import asyncio
from typing import Any

from hean.archon.chronicle import Chronicle
from hean.archon.cortex import Cortex
from hean.archon.health_matrix import HealthMatrix
from hean.archon.heartbeat import HeartbeatRegistry
from hean.archon.reconciler import ArchonReconciler
from hean.archon.signal_pipeline_manager import SignalPipelineManager
from hean.config import HEANSettings
from hean.core.bus import EventBus
from hean.logging import get_logger

logger = get_logger(__name__)


class Archon:
    """Central orchestration brain for HEAN Trading System.

    Wraps around existing components without modifying them.
    Adds: signal tracking, health monitoring, reconciliation,
    strategic decisions, and audit trail.
    """

    def __init__(self, bus: EventBus, settings: HEANSettings) -> None:
        self._bus = bus
        self._settings = settings
        self._running = False

        self.signal_pipeline: SignalPipelineManager | None = None
        self.heartbeat: HeartbeatRegistry | None = None
        self.health_matrix: HealthMatrix | None = None
        self.cortex: Cortex | None = None
        self.reconciler: ArchonReconciler | None = None
        self.chronicle: Chronicle | None = None

    async def start(self, **components: Any) -> None:
        """Start ARCHON sub-systems based on settings."""
        self._running = True
        s = self._settings

        if s.archon_signal_pipeline_enabled:
            self.signal_pipeline = SignalPipelineManager(
                bus=self._bus,
                max_active=s.archon_max_active_signals,
                stage_timeout_sec=s.archon_signal_timeout_sec,
            )
            await self.signal_pipeline.start()

        self.heartbeat = HeartbeatRegistry(
            default_interval=s.archon_heartbeat_interval_sec,
        )

        self.health_matrix = HealthMatrix(
            bus=self._bus,
            heartbeat=self.heartbeat,
            signal_pipeline=self.signal_pipeline,
        )
        await self.health_matrix.start()

        if s.archon_chronicle_enabled:
            self.chronicle = Chronicle(
                bus=self._bus,
                max_memory=s.archon_chronicle_max_memory,
            )
            await self.chronicle.start()

        if s.archon_reconciliation_enabled:
            accounting = components.get("accounting")
            order_manager = components.get("order_manager")
            bybit_http = components.get("bybit_http")
            if accounting and order_manager and bybit_http:
                self.reconciler = ArchonReconciler(
                    bus=self._bus,
                    accounting=accounting,
                    order_manager=order_manager,
                    bybit_http=bybit_http,
                    interval_sec=s.archon_reconciliation_interval_sec,
                )
                await self.reconciler.start()

        if s.archon_cortex_enabled:
            self.cortex = Cortex(
                bus=self._bus,
                health_matrix=self.health_matrix,
                signal_pipeline=self.signal_pipeline,
                interval_sec=s.archon_cortex_interval_sec,
            )
            await self.cortex.start()

        logger.info("[ARCHON] Brain-Orchestrator activated")

    async def stop(self) -> None:
        """Stop all sub-systems in reverse order."""
        self._running = False

        if self.cortex:
            await self.cortex.stop()
        if self.reconciler:
            await self.reconciler.stop()
        if self.chronicle:
            await self.chronicle.stop()
        if self.health_matrix:
            await self.health_matrix.stop()
        if self.signal_pipeline:
            await self.signal_pipeline.stop()

        logger.info("[ARCHON] Brain-Orchestrator deactivated")

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive ARCHON status."""
        return {
            "running": self._running,
            "signal_pipeline": (
                self.signal_pipeline.get_status()
                if self.signal_pipeline else None
            ),
            "health": (
                self.health_matrix.get_composite_score()
                if self.health_matrix else None
            ),
            "heartbeats": (
                self.heartbeat.get_status()
                if self.heartbeat else None
            ),
            "cortex": (
                self.cortex.get_status()
                if self.cortex else None
            ),
            "reconciler_active": self.reconciler is not None,
            "chronicle_active": self.chronicle is not None,
        }
```

#### Integration into TradingSystem (`src/hean/main.py`)

Add to `__init__`:
```python
from hean.archon.archon import Archon
self._archon: Archon | None = None
```

Add at the END of `start()` method (after all other components are initialized):
```python
# ARCHON — Central Brain Orchestrator
if self._settings.archon_enabled:
    from hean.archon.archon import Archon
    self._archon = Archon(bus=self._bus, settings=self._settings)
    await self._archon.start(
        accounting=self._accounting,
        order_manager=self._order_manager,
        bybit_http=getattr(self._execution_router, '_bybit_http', None),
    )
```

Add at the BEGINNING of `stop()` method (before other components):
```python
if self._archon:
    await self._archon.stop()
```

#### API Router (`src/hean/api/routers/archon.py`)

```python
"""ARCHON API endpoints."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/archon", tags=["archon"])


@router.get("/status")
async def archon_status(request: Request) -> dict:
    """Get ARCHON overall status."""
    facade = request.state.engine_facade
    ts = getattr(facade, '_trading_system', None)
    archon = getattr(ts, '_archon', None) if ts else None
    if not archon:
        return {"active": False, "error": "ARCHON not initialized"}
    return archon.get_status()


@router.get("/pipeline")
async def pipeline_status(request: Request) -> dict:
    """Get Signal Pipeline status with dead letters."""
    facade = request.state.engine_facade
    ts = getattr(facade, '_trading_system', None)
    archon = getattr(ts, '_archon', None) if ts else None
    if not archon or not archon.signal_pipeline:
        return {"active": False}
    return archon.signal_pipeline.get_status()


@router.get("/pipeline/trace/{correlation_id}")
async def signal_trace(request: Request, correlation_id: str) -> dict:
    """Get full trace for a specific signal."""
    facade = request.state.engine_facade
    ts = getattr(facade, '_trading_system', None)
    archon = getattr(ts, '_archon', None) if ts else None
    if not archon or not archon.signal_pipeline:
        return {"error": "Pipeline not active"}
    trace = archon.signal_pipeline.get_trace(correlation_id)
    return trace or {"error": "Trace not found"}


@router.get("/health")
async def health_matrix(request: Request) -> dict:
    """Get comprehensive health matrix."""
    facade = request.state.engine_facade
    ts = getattr(facade, '_trading_system', None)
    archon = getattr(ts, '_archon', None) if ts else None
    if not archon or not archon.health_matrix:
        return {"active": False}
    return await archon.health_matrix.get_full_status()


@router.get("/chronicle")
async def chronicle_query(
    request: Request,
    event_type: str | None = None,
    symbol: str | None = None,
    strategy_id: str | None = None,
    limit: int = 50,
) -> dict:
    """Query audit trail."""
    facade = request.state.engine_facade
    ts = getattr(facade, '_trading_system', None)
    archon = getattr(ts, '_archon', None) if ts else None
    if not archon or not archon.chronicle:
        return {"active": False}
    return {
        "entries": archon.chronicle.query(
            event_type=event_type,
            symbol=symbol,
            strategy_id=strategy_id,
            limit=limit,
        )
    }
```

Register in `src/hean/api/main.py`:
```python
from hean.api.routers import archon
app.include_router(archon.router, prefix=API_PREFIX)
```

---

## CONSTRAINTS — DO NOT VIOLATE

1. **DO NOT modify `bus.py`** — ARCHON is passive, subscribes alongside existing handlers
2. **DO NOT modify existing strategies** — ARCHON observes, doesn't intercept
3. **DO NOT add latency** to fast-path events (SIGNAL, ORDER_REQUEST, ORDER_FILLED)
4. **ALL new settings** go into `HEANSettings` in `config.py` with sensible defaults
5. **ALL logging** via `from hean.logging import get_logger; logger = get_logger(__name__)`
6. **Ruff format** — line-length 100, no trailing whitespace
7. **mypy strict** — all functions fully typed
8. **asyncio_mode = "auto"** — no `@pytest.mark.asyncio` in tests
9. **Each phase must be independently testable** — don't depend on later phases
10. **Graceful degradation** — if ARCHON fails to start, TradingSystem continues normally. Wrap in try/except.

---

## TESTING COMMANDS

```bash
# Run all ARCHON tests
pytest tests/test_archon/ -v

# Run single test file
pytest tests/test_archon/test_signal_pipeline.py -v

# Run with coverage
pytest tests/test_archon/ --cov=src/hean/archon -v

# Lint
ruff check src/hean/archon/
ruff format src/hean/archon/

# Type check
mypy src/hean/archon/
```

---

## EXECUTION ORDER

Implement in this exact order. Each phase must pass its tests before moving to the next:

1. **Phase 0** → Create directory, types, config. Run: `ruff check src/hean/archon/ && mypy src/hean/archon/`
2. **Phase 1** → SignalPipelineManager. Run: `pytest tests/test_archon/test_signal_pipeline.py -v`
3. **Phase 2** → Heartbeat + HealthMatrix. Run: `pytest tests/test_archon/test_heartbeat.py tests/test_archon/test_health_matrix.py -v`
4. **Phase 3** → Chronicle. Run: `pytest tests/test_archon/test_chronicle.py -v`
5. **Phase 4** → Reconciler. Run: `pytest tests/test_archon/test_reconciler.py -v`
6. **Phase 5** → Cortex. Run: `pytest tests/test_archon/test_cortex.py -v`
7. **Phase 6** → GenomeDirector. Run: `pytest tests/test_archon/test_genome_director.py -v`
8. **Phase 7** → Archon main class + integration. Run: `pytest tests/test_archon/ -v && make test-quick`
9. **Final** → Verify ALL existing tests still pass: `make test-quick`
