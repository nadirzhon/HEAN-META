# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HEAN is a production-grade, event-driven crypto trading system for Bybit Testnet. All trades execute on testnet with virtual funds — no paper trading mode exists. The system includes a Python backend (FastAPI + async event bus), a React/Vite web dashboard, and a native iOS SwiftUI app.

## Commands

### Development
```bash
make install          # Install dependencies: pip install -e ".[dev]"
make test             # Run tests: pytest
make lint             # Lint + type check: ruff check src/ && mypy src/
make run              # Run CLI: python -m hean.main run
make dev              # Start development environment with hot reload
```

### Single Test Execution
```bash
pytest tests/test_api.py -v                    # Run single test file
pytest tests/test_api.py::test_health -v       # Run single test function
pytest tests/test_truth_layer_invariants.py -v # Truth layer tests
```

### Docker
```bash
docker-compose up -d --build    # Build and start (API + UI + Redis)
docker-compose logs -f          # View logs
make docker-clean               # Remove containers and volumes
make prod-up                    # Start production environment
make prod-with-monitoring       # Production + Prometheus/Grafana
```

### Smoke Test (run before Docker rebuild)
```bash
./scripts/smoke_test.sh
```

## Architecture

```
FastAPI Backend → Engine Facade → Event Bus (priority queues)
                                      ↓
          ┌────────────┬──────────────┼──────────────┬────────────┐
          ↓            ↓              ↓              ↓            ↓
     Strategies   Risk Mgmt     Execution      Portfolio     Physics
          ↓            ↓              ↓              ↓            ↓
          └────────────┴──────────────┴──────────────┴────────────┘
                                   ↓
                           Bybit Exchange (Testnet)
```

### Key Entry Points
- `src/hean/main.py` — CLI and `TradingSystem` orchestrator (instantiates all components)
- `src/hean/api/main.py` — FastAPI server with WebSocket support
- `src/hean/api/engine_facade.py` — Unified interface to TradingSystem; `get_facade()` provides access from routers
- `src/hean/core/bus.py` — Async `EventBus` with multi-priority queues and circuit breaker

### Core Modules
- `src/hean/strategies/` — Trading strategies (ImpulseEngine, FundingHarvester, BasisArbitrage, plus 5 more)
- `src/hean/risk/` — Risk management (RiskGovernor, KillSwitch, PositionSizer, KellyCriterion)
- `src/hean/execution/` — Order routing (`router_bybit_only.py` is the primary production router)
- `src/hean/portfolio/` — Capital allocation, accounting, profit capture
- `src/hean/exchange/bybit/` — Bybit HTTP (`http.py`) and WebSocket (`ws_public.py`, `ws_private.py`) clients
- `src/hean/physics/` — Market thermodynamics (PhysicsEngine, ParticipantClassifier, AnomalyDetector, TemporalStack)
- `src/hean/brain/` — Claude AI analysis client (periodic market analysis with rule-based fallback)
- `src/hean/storage/` — DuckDB persistence layer for ticks, physics snapshots, brain analyses

### Event-Driven Design
All components communicate via `EventBus` (`src/hean/core/bus.py`). Event types defined in `src/hean/core/types.py`:
- **Market:** `TICK`, `FUNDING`, `ORDER_BOOK_UPDATE`, `REGIME_UPDATE`, `CONTEXT_UPDATE`
- **Strategy:** `SIGNAL`, `STRATEGY_PARAMS_UPDATED`
- **Risk:** `ORDER_REQUEST`, `RISK_BLOCKED`, `RISK_ALERT`
- **Execution:** `ORDER_PLACED`, `ORDER_FILLED`, `ORDER_CANCELLED`, `ORDER_REJECTED`
- **Portfolio:** `POSITION_OPENED`, `POSITION_CLOSED`, `POSITION_UPDATE`, `PNL_UPDATE`, `EQUITY_UPDATE`

Flow: Strategies publish SIGNAL → RiskGovernor filters → ExecutionRouter places ORDER_REQUEST → Exchange fills

## Configuration

Settings in `src/hean/config.py` via Pydantic `BaseSettings`. Key env vars:
- `BYBIT_API_KEY`, `BYBIT_API_SECRET` — Exchange credentials
- `BYBIT_TESTNET=true` — Always use testnet
- `LIVE_CONFIRM=YES` — Required for live trading
- `INITIAL_CAPITAL` — Starting capital (default: 300 USDT)
- `ANTHROPIC_API_KEY` — Optional, enables Claude Brain analysis
- `BRAIN_ENABLED` — Enable/disable brain module (default: true)
- `BRAIN_ANALYSIS_INTERVAL` — Seconds between analyses (default: 30)

## Risk Management States

RiskGovernor uses graduated states: `NORMAL` → `SOFT_BRAKE` → `QUARANTINE` → `HARD_STOP`

KillSwitch triggers on >20% drawdown from initial capital.

## API Routers

All routers registered in `src/hean/api/main.py` under `/api/v1/` prefix:

| Group | Routers |
|-------|---------|
| Core | engine, trading, strategies, risk, risk_governor |
| Analysis | analytics, graph_engine, telemetry |
| Physics | physics, temporal |
| AI/ML | brain, causal_inference, meta_learning, multimodal_swarm, singularity |
| System | system, market, changelog, storage |

Key endpoints:
- `GET /api/v1/engine/status` — Engine state, equity, PnL
- `GET /api/v1/orders/positions` — `{"positions": [dict]}`
- `GET /api/v1/orders` — `{"orders": [dict]}`
- `GET /api/v1/strategies` — `[dict]` with `strategy_id`, `type`, `enabled`
- `GET /api/v1/risk/governor/status` — `risk_state`, `level`, `reason_codes`
- `GET /api/v1/physics/state?symbol=X` — Temperature, entropy, phase, Szilard profit
- `GET /api/v1/physics/participants?symbol=X` — Participant breakdown (whales, MM, retail)
- `GET /api/v1/physics/anomalies?limit=N` — Market anomalies with `{anomalies: [...], active_count}`
- `GET /api/v1/temporal/stack` — Temporal levels as `{levels: {"5": {...}, "4": {...}}}`
- `GET /api/v1/brain/thoughts` — AI analysis history
- `GET /api/v1/brain/analysis` — Latest brain analysis
- `GET /api/v1/storage/ticks?symbol=X` — Historical tick data from DuckDB

## Testing

```bash
pytest                                        # All tests
pytest --cov=src/hean --cov-report=html      # With coverage
pytest -k "test_api" -v                       # Tests matching pattern
```

Tests use `asyncio_mode = "auto"` — async tests work without `@pytest.mark.asyncio`.

Note: Use `python3` (not `python`) on macOS.

## iOS App

The iOS app lives in `ios/` and is a SwiftUI app targeting iOS 17+.

### Build & Run
Open `ios/HEAN.xcodeproj` in Xcode. The app connects to the FastAPI backend via `APIClient`.

### Architecture
- **DIContainer** (`Core/DI/DIContainer.swift`) — Dependency injection with `@EnvironmentObject`
- **APIClient** (`Core/Networking/APIClient.swift`) — Actor-based HTTP client with generic `get<T: Decodable>()` method
- **APIEndpoints** (`Core/Networking/APIEndpoints.swift`) — Centralized endpoint enum
- **Models** (`Models/`) — Codable structs matching backend responses
- **Views** — Tab-based: Dashboard, Brain, Map (Gravity), Players, Settings

### Critical Compilation Notes
- Xcode project (`project.pbxproj`) has ~48 compiled files — not all `.swift` files in the directory are compiled
- `Services.swift` is the COMPILED consolidated service file; individual `Services/Live/*.swift` are NOT in the build
- Extensions in `Extensions/` folder must be explicitly added to pbxproj to compile
- `DesignSystem/` contains compiled color/formatting extensions (duplicates of `Extensions/` are safe)

### Backend-to-iOS Field Mapping
- Backend uses `snake_case`; iOS uses `camelCase` — always add `CodingKeys`
- Backend field names often differ: `position_id`→`id`, `current_price`→`markPrice`, `size`→`quantity`
- Use `decodeIfPresent` with defaults for optional backend fields
- Use custom `init(from:)` decoders with fallback keys when field names vary
- Case-insensitive enum decoding for values like `"buy"`/`"BUY"`/`"Buy"`

## Docker Services

`docker-compose.yml` defines:
- **hean-api** — Python FastAPI backend (port 8000)
- **hean-ui** — React/Vite dashboard (port 3000)
- **hean-redis** — Redis for state/caching (port 6379)

Production adds Prometheus + Grafana monitoring (`docker-compose.production.yml`).

## Symbiont X

`src/hean/symbiont_x/` is an advanced subsystem with:
- Genome Lab (genetic algorithm for strategy evolution)
- Immune System (circuit breakers, reflexes, constitution)
- Nervous System (event envelope, health sensors, WS connectors)
- Decision Ledger (auditable trade decision recording)
- Capital Allocator (portfolio-level rebalancing)
- Adversarial Twin (stress testing, survival scoring)
- Regime Brain (market regime classification)
