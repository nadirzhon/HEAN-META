# archon -- ARCHON Brain-Orchestrator

Central coordination layer that monitors system health, tracks signal lifecycles, reconciles state with the exchange, and issues strategic directives to components.

## Architecture

ARCHON wraps around existing trading components without modifying them. It subscribes to EventBus events as a passive observer and publishes advisory `ARCHON_DIRECTIVE` events that components opt into. The `Archon` class is the top-level entry point and manages six sub-systems: SignalPipelineManager, HeartbeatRegistry, HealthMatrix, Cortex, ArchonReconciler, and Chronicle. Each sub-system can be independently enabled or disabled via configuration flags.

## Key Classes

- `Archon` -- Main orchestrator; initializes and manages sub-system lifecycle (start/stop). Exposes `get_status()` for API introspection.
- `Cortex` -- Strategic decision engine. Runs a periodic evaluation loop that checks health scores, signal fill rates, and dead letter counts. Issues Directives to transition between system modes (AGGRESSIVE / NORMAL / DEFENSIVE / EMERGENCY).
- `HealthMatrix` -- Aggregates health from EventBus status, HeartbeatRegistry, and SignalPipelineManager into a composite score (0-100). Score formula: 40% bus health + 30% heartbeat health + 20% pipeline fill rate + 10% error rate.
- `HeartbeatRegistry` -- Tracks component liveness. Components register with an expected heartbeat interval; missed beats mark them as unhealthy. Used by HealthMatrix for the composite score.
- `SignalPipelineManager` -- Passive observer that tracks the full signal lifecycle from GENERATED through RISK_APPROVED/RISK_BLOCKED to ORDER_FILLED or DEAD_LETTER. Measures end-to-end latency and fill rate. Maintains a DeadLetterQueue for failed signals.
- `ArchonReconciler` -- Runs three background loops to reconcile local state with the exchange: position reconciliation (configurable interval, default 30s), balance reconciliation (60s), and order reconciliation (15s). Publishes `RECONCILIATION_ALERT` on discrepancy.
- `Chronicle` -- Audit trail that records key trading decisions as `ChronicleEntry` records in an in-memory ring buffer (configurable max, default 10,000). Supports querying by event type, symbol, strategy, or correlation ID to trace a signal's full journey.
- `Directive` / `DirectiveType` -- Data model for commands issued by Cortex (PAUSE_TRADING, RESUME_TRADING, ACTIVATE_STRATEGY, SET_RISK_MODE, etc.).
- `GenomeDirector` -- Directs Symbiont X evolution by running fast backtests with genome-specific parameters for fitness evaluation.
- `SignalStage` / `SignalTrace` -- Signal lifecycle types used by SignalPipelineManager.

## Events

| Event | Direction | Description |
|-------|-----------|-------------|
| SIGNAL | Subscribes | Begins tracking a new signal through the pipeline |
| ORDER_REQUEST | Subscribes | Signal advanced to risk-approved stage |
| RISK_BLOCKED | Subscribes | Signal blocked by risk layer (moved to dead letter) |
| ORDER_PLACED | Subscribes | Order placed on exchange |
| ORDER_FILLED | Subscribes | Pipeline success -- signal completed its lifecycle |
| ORDER_REJECTED | Subscribes | Order rejected by exchange |
| ORDER_CANCELLED | Subscribes | Order cancelled |
| POSITION_OPENED | Subscribes | Final success state |
| KILLSWITCH_TRIGGERED | Subscribes (Chronicle) | Recorded in audit trail |
| ARCHON_DIRECTIVE | Publishes | Advisory directive from Cortex to components |
| RECONCILIATION_ALERT | Publishes | Discrepancy detected between local and exchange state |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| ARCHON_SIGNAL_PIPELINE_ENABLED | true | Enable signal lifecycle tracking |
| ARCHON_MAX_ACTIVE_SIGNALS | 1000 | Max concurrent signals tracked |
| ARCHON_SIGNAL_TIMEOUT_SEC | 10 | Timeout before moving signal to dead letter |
| ARCHON_HEARTBEAT_INTERVAL_SEC | 5 | Default heartbeat interval for components |
| ARCHON_CHRONICLE_ENABLED | true | Enable audit trail recording |
| ARCHON_CHRONICLE_MAX_MEMORY | 10000 | Max chronicle entries in ring buffer |
| ARCHON_RECONCILIATION_ENABLED | true | Enable state reconciliation loops |
| ARCHON_RECONCILIATION_INTERVAL_SEC | 30 | Position reconciliation interval |
| ARCHON_CORTEX_ENABLED | true | Enable strategic decision engine |
| ARCHON_CORTEX_INTERVAL_SEC | 30 | Decision loop evaluation interval |
