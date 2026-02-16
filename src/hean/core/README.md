# core -- EventBus, Types, and System Infrastructure

Foundation layer providing the event-driven communication backbone, shared data types, market regime detection, context aggregation, and component lifecycle management.

## Architecture

Every component in HEAN communicates through the `EventBus` defined in this package. The bus supports multi-priority queues (CRITICAL > NORMAL > LOW), a circuit breaker that drops LOW events at 95% utilization, and fast-path dispatch that bypasses queues entirely for time-critical events (SIGNAL, ORDER_REQUEST, ORDER_FILLED). The `types` module defines all shared DTOs (Event, Signal, OrderRequest, Order, Position, etc.) and the `EventType` enum. The `ContextAggregator` fuses outputs from Physics, Brain, Oracle, OFI, and Causal engines into a `UnifiedMarketContext` per symbol, publishing `CONTEXT_READY` events for strategies.

## Key Classes

- `EventBus` (`bus.py`) -- Async pub/sub bus with multi-priority queues. Supports `subscribe()`, `unsubscribe()`, `publish()`, `flush()`, `start()`, `stop()`. Tracks metrics (events published/dropped/delayed/processed, handler errors). Sync handlers run in a thread pool to avoid blocking the event loop.
- `EventType` (`types.py`) -- String enum of all event types (TICK, SIGNAL, ORDER_REQUEST, ORDER_FILLED, PHYSICS_UPDATE, BRAIN_ANALYSIS, ARCHON_DIRECTIVE, etc.).
- `Event` (`types.py`) -- Dataclass: `event_type`, `timestamp`, `data` dict.
- `Signal` (`types.py`) -- Pydantic model for trading signals: strategy_id, symbol, side, entry_price, stop_loss, take_profit, confidence (0-1), urgency (0-1), prefer_maker.
- `OrderRequest` (`types.py`) -- Pydantic model for risk-approved order requests with validators for side, size, order_type, and price-for-limit validation.
- `Order`, `Position`, `EquitySnapshot` (`types.py`) -- Pydantic models for order lifecycle, position state, and equity snapshots.
- `RegimeDetector` (`regime.py`) -- Detects market regime (RANGE / NORMAL / IMPULSE) from price volatility and return acceleration. Subscribes to TICK events and publishes REGIME_UPDATE.
- `ContextAggregator` (`context_aggregator.py`) -- Central integration hub that subscribes to PHYSICS_UPDATE, BRAIN_ANALYSIS, ORACLE_PREDICTION, OFI_UPDATE, and CAUSAL_SIGNAL events. Builds `UnifiedMarketContext` per symbol and publishes CONTEXT_READY.
- `ComponentRegistry` (`system/component_registry.py`) -- Manages advanced component lifecycle (RLRiskManager, DynamicOracleWeighting, StrategyAllocator). Handles conditional imports for optional dependencies.
- `OrderFlowImbalance` (`ofi.py`) -- Calculates order flow imbalance from tick data.

## Sub-packages

- `system/` -- ComponentRegistry, HealthMonitor, ErrorAnalyzer, RedisStateManager
- `intelligence/` -- Oracle/ML signal fusion (see [intelligence/README.md](intelligence/README.md))
- `arb/` -- Triangular arbitrage scanner
- `execution/` -- Iceberg order support
- `evolution/` -- Compiler bridge for evolved strategies
- `network/` -- Global sync, proxy sharding, scouter
- `telemetry/` -- Self-insight telemetry

## Events

| Event | Direction | Description |
|-------|-----------|-------------|
| TICK | Published by exchange WS | Market tick data (symbol, price, volume, bid, ask) |
| REGIME_UPDATE | Published by RegimeDetector | Market regime change (RANGE/NORMAL/IMPULSE) |
| CONTEXT_READY | Published by ContextAggregator | Unified market context ready for strategies |
| CONTEXT_UPDATE | Published by various | Sub-typed context updates (oracle, sentiment, regime, physics) |
| ERROR | Published by EventBus | Handler exception or system error |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| RL_RISK_ENABLED | false | Enable RL risk manager component |
| ORACLE_DYNAMIC_WEIGHTING | false | Enable dynamic oracle weight adaptation |
