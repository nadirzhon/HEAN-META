# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HEAN is an event-driven crypto trading system for Bybit Testnet. All trades execute on testnet with virtual funds. The system consists of a Python backend (FastAPI + async event bus), a native iOS SwiftUI app, and optional Docker deployment with Redis.

## Commands

```bash
make install              # pip install -e ".[dev]"
make test                 # pytest (671 tests, ~10 min full suite)
make test-quick           # pytest excluding Bybit connection tests
make lint                 # ruff check src/ && mypy src/
make run                  # python -m hean.main run
make dev                  # docker-compose with dev profile
make smoke                # ./scripts/smoke_test.sh (run before Docker rebuild)
```

Single test execution:
```bash
pytest tests/test_api.py -v                        # Single file
pytest tests/test_api.py::test_health -v           # Single function
pytest -k "impulse" -v                             # Pattern match
pytest tests/test_truth_layer_invariants.py -v     # Truth layer invariants
```

Use `python3` (not `python`) on macOS. Tests use `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.

Docker:
```bash
docker-compose up -d --build    # Build and start (API + Redis)
docker-compose logs -f          # View logs
make docker-clean               # Remove containers and volumes
make prod-with-monitoring       # Production + Prometheus/Grafana
```

## Architecture

```
Bybit WebSocket (ticks, orderbook, funding)
         │
         ▼
   ┌─────────────┐
   │  Event Bus   │ ◄── Priority queues + fast-path dispatch
   └──────┬──────┘      (SIGNAL, ORDER_REQUEST, ORDER_FILLED bypass normal queue)
          │
    ┌─────┼──────────────────────────────────┐
    │     │     │         │         │         │
    ▼     ▼     ▼         ▼         ▼         ▼
Strategies Risk Execution Portfolio Physics  Brain
    │     │     │         │
    │     │     ▼         │
    │     │  Bybit HTTP   │
    └─────┴──────────────┘
```

**Signal chain:** TICK → Strategy → filter cascade → SIGNAL → RiskGovernor → ORDER_REQUEST → ExecutionRouter → Bybit HTTP → ORDER_FILLED → Position update

### Key Entry Points

- `src/hean/main.py` — CLI entrypoint + `TradingSystem` class that instantiates and wires all components
- `src/hean/api/main.py` — FastAPI server with WebSocket; registers 21 routers under `/api/v1/`
- `src/hean/api/engine_facade.py` — Unified interface to TradingSystem; `get_facade()` used by all routers
- `src/hean/core/bus.py` — Async `EventBus` with multi-priority queues and circuit breaker

### Event-Driven Design

All components communicate via `EventBus` (`src/hean/core/bus.py`). Types in `src/hean/core/types.py`:

| Category | Events |
|----------|--------|
| Market | `TICK`, `FUNDING`, `FUNDING_UPDATE`, `ORDER_BOOK_UPDATE`, `REGIME_UPDATE`, `CONTEXT_UPDATE`, `CANDLE` |
| Strategy | `SIGNAL`, `STRATEGY_PARAMS_UPDATED` |
| Risk | `ORDER_REQUEST`, `RISK_BLOCKED`, `RISK_ALERT` |
| Execution | `ORDER_PLACED`, `ORDER_FILLED`, `ORDER_CANCELLED`, `ORDER_REJECTED` |
| Portfolio | `POSITION_OPENED`, `POSITION_CLOSED`, `POSITION_UPDATE`, `POSITION_CLOSE_REQUEST`, `PNL_UPDATE`, `EQUITY_UPDATE`, `ORDER_DECISION`, `ORDER_EXIT_DECISION` |
| System | `STOP_TRADING`, `KILLSWITCH_TRIGGERED`, `KILLSWITCH_RESET`, `ERROR`, `STATUS`, `HEARTBEAT` |
| Intelligence | `CONTEXT_READY`, `PHYSICS_UPDATE`, `BRAIN_ANALYSIS`, `META_LEARNING_PATCH`, `ORACLE_PREDICTION`, `OFI_UPDATE`, `CAUSAL_SIGNAL` |
| Council | `COUNCIL_REVIEW`, `COUNCIL_RECOMMENDATION` |

### Core Modules

- **`src/hean/strategies/`** — 11 strategies, each gated by a settings flag:
  - `ImpulseEngine` — Momentum with 12-layer deterministic filter cascade (`impulse_filters.py`), 70-95% signal rejection
  - `FundingHarvester`, `BasisArbitrage` — Funding rate and basis spread arbitrage
  - `MomentumTrader`, `CorrelationArbitrage`, `EnhancedGrid`, `HFScalping`
  - `InventoryNeutralMM`, `RebateFarmer`, `LiquiditySweep`, `SentimentStrategy`
- **`src/hean/risk/`** — `RiskGovernor` (state machine: NORMAL → SOFT_BRAKE → QUARANTINE → HARD_STOP), `KillSwitch` (>20% drawdown), `PositionSizer`, `KellyCriterion`, `DepositProtector`, `SmartLeverage`
- **`src/hean/execution/`** — `router_bybit_only.py` is the production router (with idempotency). `router.py` is the generic router interface
- **`src/hean/exchange/bybit/`** — `http.py` (REST with instrument/leverage caching), `ws_public.py` (market data), `ws_private.py` (order/position updates)
- **`src/hean/portfolio/`** — Accounting, capital allocation, profit capture
- **`src/hean/physics/`** — Market thermodynamics: temperature, entropy, phase detection (accumulation/markup/distribution/markdown), Szilard engine, participant classifier, anomaly detector, temporal stack
- **`src/hean/brain/`** — Claude AI periodic market analysis (requires `ANTHROPIC_API_KEY`); rule-based fallback when disabled
- **`src/hean/council/`** — Multi-agent AI council for trade review and consensus decisions
- **`src/hean/storage/`** — DuckDB persistence for ticks, physics snapshots, brain analyses
- **`src/hean/symbiont_x/`** — Genetic algorithm strategy evolution (genome lab, immune system, decision ledger). Has its own test suite but is not wired into main trading flow
- **`src/hean/core/intelligence/`** — MetaLearningEngine, CausalInferenceEngine, MultimodalSwarm exist but are **disabled** in `main.py` (output not consumed by any strategy)

## Configuration

Settings in `src/hean/config.py` via Pydantic `BaseSettings` (loaded from `.env`):

```bash
# Required
BYBIT_API_KEY, BYBIT_API_SECRET    # Exchange credentials
BYBIT_TESTNET=true                  # Always testnet
LIVE_CONFIRM=YES                    # Required for trading

# Capital
INITIAL_CAPITAL=300                 # Starting USDT

# AI (optional)
ANTHROPIC_API_KEY=...               # Enables Claude Brain
BRAIN_ENABLED=true
BRAIN_ANALYSIS_INTERVAL=30          # Seconds between analyses

# Strategies (each has an enable flag)
IMPULSE_ENGINE_ENABLED=true
FUNDING_HARVESTER_ENABLED=true
BASIS_ARBITRAGE_ENABLED=true
# ... HF_SCALPING_ENABLED, MOMENTUM_TRADER_ENABLED, etc.
```

## API Endpoints

All under `/api/v1/` prefix. Key endpoints:

| Endpoint | Response shape |
|----------|---------------|
| `GET /engine/status` | `{status, running, equity, daily_pnl, initial_capital}` |
| `GET /orders/positions` | `[{position_id, symbol, side, size, entry_price, ...}]` |
| `GET /orders` | `[{order_id, symbol, side, qty, price, status, ...}]` |
| `GET /strategies` | `[{strategy_id, type, enabled}]` |
| `GET /risk/governor/status` | `{risk_state, level, reason_codes, quarantined_symbols, can_clear}` |
| `GET /risk/killswitch/status` | `{triggered, reasons, thresholds, current_metrics}` |
| `GET /trading/why` | Complex diagnostic: engine state, killswitch, last activity, reason codes |
| `GET /trading/metrics` | `{counters: {last_1m, last_5m, session}}` with signals/orders/fills |
| `GET /physics/state?symbol=X` | Temperature, entropy, phase, Szilard profit |
| `GET /physics/participants?symbol=X` | Whale/MM/retail breakdown |
| `GET /physics/anomalies?limit=N` | `{anomalies: [...], active_count}` |
| `GET /temporal/stack` | `{levels: {"5": {...}, "4": {...}}}` |
| `GET /brain/analysis` | Latest AI market analysis |
| `GET /council/status` | Council decision status |

WebSocket: `ws://localhost:8000/ws` — topics: `system_status`, `order_decisions`, `ai_catalyst`

## iOS App

SwiftUI app in `ios/`, targeting iOS 17+. Open `ios/HEAN.xcodeproj` in Xcode.

### Architecture
- **DIContainer** (`Core/DI/DIContainer.swift`) — `@EnvironmentObject` dependency injection
- **APIClient** (`Core/Networking/APIClient.swift`) — Actor-based HTTP client
- **APIEndpoints** (`Core/Networking/APIEndpoints.swift`) — Centralized endpoint enum
- **Command Center** — 5 tabs: Live, Mind, Action, X-Ray, Settings

### Critical iOS Patterns
- Backend `snake_case` → iOS `camelCase`: always add `CodingKeys`
- Backend field names diverge: `position_id`→`id`, `current_price`→`markPrice`, `size`→`quantity`
- Use `decodeIfPresent` with defaults for optional fields
- Custom `init(from:)` with fallback keys (try `"id"` then `"position_id"`)
- Case-insensitive enum decoding (`"buy"/"BUY"/"Buy"`)
- `Services.swift` is the compiled consolidated service file; `Services/Live/*.swift` files are NOT in the build
- `DesignSystem/` contains compiled color/formatting extensions
- ~158 Swift files in build phase — not all `.swift` files in directory are compiled; check `project.pbxproj`

## Docker Services

`docker-compose.yml` defines: `api` (port 8000), `redis` (port 6379), plus optional `symbiont-testnet`, `collector`, `physics`, `brain`, `risk-svc`.

Production monitoring via `docker-compose.production.yml` adds Prometheus + Grafana.

## Code Conventions

- Ruff for linting (line-length 100, py311 target). Rules: E, W, F, I, B, C4, UP
- mypy strict mode (disallow_untyped_defs, strict_equality)
- `asyncio_mode = "auto"` in pytest — all async tests auto-detected
- Strategies inherit from `BaseStrategy` (`src/hean/strategies/base.py`)
- All exchange interactions go through `BybitHTTPClient` (never direct HTTP calls)
- `DRY_RUN=true` is default — blocks real order placement with a hard RuntimeError
