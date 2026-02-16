# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See also: `GEMINI.md` for additional Gemini-specific project context.

## Project Overview

HEAN is an event-driven crypto trading system for Bybit Testnet. All trades execute on testnet with virtual funds. Three frontends:
- **Python backend** — FastAPI + async EventBus + Redis
- **iOS app** — SwiftUI native (iOS 17+)
- **Web dashboard** — Next.js 15 + React 19 + Zustand + Tailwind 4 (at `dashboard/`)

## Commands

```bash
make install              # pip install -e ".[dev]"
make test                 # pytest (~680 tests)
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

Formatting (separate from lint):
```bash
ruff format .             # Auto-format all Python files
ruff check --fix src/     # Auto-fix lint issues where possible
```

ML training (requires optional deps):
```bash
python3 scripts/train_oracle.py --symbol BTCUSDT --days 30 --model-type tcn
python3 scripts/train_rl_risk.py --timesteps 50000
python3 scripts/promote_model.py --experiment oracle-tcn --metric val_accuracy --output models/tcn_production.pt
```

Use `python3` (not `python`) on macOS. Tests use `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.

Docker:
```bash
docker-compose up -d --build                          # Build and start (API + Redis)
docker-compose --profile training up -d mlflow        # Start MLflow tracking server
docker-compose logs -f                                # View logs
make docker-clean                                     # Remove containers and volumes
make prod-with-monitoring                             # Production + Prometheus/Grafana
```

iOS (build from CLI):
```bash
xcodebuild -project ios/HEAN.xcodeproj -scheme HEAN \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -quiet build
```

Dashboard:
```bash
cd dashboard && npm install && npm run dev    # Next.js dev server on port 3001
cd dashboard && npm run build                 # Production build
```

## Architecture

```
Bybit WebSocket (ticks, orderbook, funding)
         │
         ▼
   ┌─────────────┐
   │  Event Bus   │ ◄── Priority queues (Critical/Normal/Low) + fast-path dispatch
   └──────┬──────┘      (SIGNAL, ORDER_REQUEST, ORDER_FILLED bypass normal queue)
          │
          ├──── ServiceEventBridge (bidirectional: EventBus ↔ Redis Streams)
          │
    ┌─────┼──────────────────────────────────────────┐
    │     │     │         │         │         │       │
    ▼     ▼     ▼         ▼         ▼         ▼       ▼
Strategies Risk Execution Portfolio Physics  Brain  Oracle
    │     │     │         │                          (Hybrid)
    │     │     ▼         │
    │     │  Bybit HTTP   │
    └─────┴──────────────┘
```

**Signal chain:** TICK → Strategy → filter cascade → SIGNAL → RiskGovernor → ORDER_REQUEST → ExecutionRouter → Bybit HTTP → ORDER_FILLED → Position update

### Microservice Decomposition (Docker)

When running via Docker, the system decomposes into independent services communicating via Redis Streams:
- `api` — FastAPI gateway (REST + WebSocket), port 8000
- `symbiont-testnet` — Core trading logic container
- `redis` — Central message broker and state store, port 6379
- `collector` — Bybit WebSocket market data ingestion (`services/collector/`)
- `physics` — Market thermodynamics calculations (`services/physics/`)
- `brain` — AI-based decision making via Claude (`services/brain/`)
- `risk-svc` — Dedicated risk management service (`services/risk/`)
- `oracle` — Hybrid price+narrative AI signal service (`services/oracle/`)
- `mlflow` (training profile) — MLflow tracking server, port 5000

Each microservice lives in `services/<name>/` with its own `main.py`. They communicate exclusively via Redis Streams — no direct service-to-service calls.

`ServiceEventBridge` (`src/hean/core/system/service_event_bridge.py`) bridges between the in-process EventBus and Redis Streams bidirectionally:
- **Inbound** (Redis → EventBus): `physics:*`, `brain:analysis`, `oracle:signals`, `risk:policy_updates`
- **Outbound** (EventBus → Redis): `TICK`, `SIGNAL`, `ORDER_FILLED`, `POSITION_OPENED/CLOSED`, `KILLSWITCH_TRIGGERED`, `PNL_UPDATE`

In local dev mode (`make run`), all components run in a single process via `TradingSystem`.

### Key Entry Points

- `src/hean/main.py` — CLI entrypoint + `TradingSystem` class that instantiates and wires all components (~3800 lines)
- `src/hean/api/main.py` — FastAPI server with WebSocket; registers routers under `/api/v1/`
- `src/hean/api/engine_facade.py` — Unified interface to TradingSystem; `get_facade()` used by all routers
- `src/hean/core/bus.py` — Async `EventBus` with multi-priority queues and circuit breaker
- `src/hean/core/system/component_registry.py` — `ComponentRegistry` manages component lifecycle (init → start → stop) with dependency ordering

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
| Intelligence | `CONTEXT_READY`, `PHYSICS_UPDATE`, `BRAIN_ANALYSIS`, `ORACLE_PREDICTION`, `ORACLE_SIGNAL_V1`, `RISK_POLICY_UPDATE_V1` |
| Council | `COUNCIL_REVIEW`, `COUNCIL_RECOMMENDATION` |

`CONTEXT_UPDATE` carries a `type` field for sub-routing: `oracle_predictions`, `ollama_sentiment`, `finbert_sentiment`, `rl_risk_adjustment`, `regime_update`, `physics_state`.

**When adding features, follow event-driven principles:**
- Publish events to `EventBus` to signal state changes or actions
- Create handlers that subscribe to specific events
- Avoid direct synchronous calls between major components (e.g., a Strategy should not directly call an Execution module)

### Core Modules

- **`src/hean/strategies/`** — 11 strategies, each gated by a settings flag:
  - `ImpulseEngine` — Momentum with 12-layer deterministic filter cascade (`impulse_filters.py`), 70-95% signal rejection
  - `FundingHarvester`, `BasisArbitrage` — Funding rate and basis spread arbitrage
  - `MomentumTrader`, `CorrelationArbitrage`, `EnhancedGrid`, `HFScalping`
  - `InventoryNeutralMM`, `RebateFarmer`, `LiquiditySweep`, `SentimentStrategy`
  - Supporting modules: `manager.py` (strategy lifecycle), `edge_confirmation.py`, `multi_factor_confirmation.py`, `physics_aware_positioner.py`, `physics_signal_filter.py`
- **`src/hean/risk/`** — `RiskGovernor` (state machine: NORMAL → SOFT_BRAKE → QUARANTINE → HARD_STOP), `KillSwitch` (>20% drawdown), `PositionSizer`, `KellyCriterion`, `DepositProtector`, `SmartLeverage`, `RLRiskManager` (optional PPO-based risk param adjustment), `TradingRiskEnv` (Gymnasium env for RL training)
- **`src/hean/execution/`** — `router_bybit_only.py` is the production router (with idempotency). `router.py` is the generic router interface. `position_reconciliation.py` syncs local state with exchange
- **`src/hean/exchange/bybit/`** — `http.py` (REST with instrument/leverage caching), `ws_public.py` (market data), `ws_private.py` (order/position updates)
- **`src/hean/portfolio/`** — Accounting, capital allocation, profit capture
- **`src/hean/physics/`** — Market thermodynamics: temperature, entropy, phase detection (accumulation/markup/distribution/markdown), Szilard engine, participant classifier, anomaly detector, temporal stack
- **`src/hean/brain/`** — Claude AI periodic market analysis (requires `ANTHROPIC_API_KEY`); rule-based fallback when disabled
- **`src/hean/council/`** — Multi-agent AI council for trade review and consensus decisions
- **`src/hean/sentiment/`** — `SentimentAnalyzer` (FinBERT), `OllamaSentimentClient` (local LLM via Ollama), news/reddit/twitter clients, `SentimentAggregator`
- **`src/hean/core/intelligence/`** — `OracleIntegration` (hybrid 4-source signal fusion), `OracleEngine` (TCN + fingerprinting), `TCPriceReversalPredictor` (PyTorch TCN), `CorrelationEngine`, `VolatilityPredictor`
- **`src/hean/storage/`** — DuckDB persistence for ticks, physics snapshots, brain analyses
- **`src/hean/symbiont_x/`** — Genetic algorithm strategy evolution (genome lab, immune system, decision ledger, backtesting engine). Has its own test suite but is not wired into main trading flow

### Hybrid Oracle (Signal Fusion)

`OracleIntegration` (`src/hean/core/intelligence/oracle_integration.py`) fuses 4 signal sources with weighted ensemble:

| Source | Weight | Input |
|--------|--------|-------|
| TCN (price reversal) | 40% | `TICK` events → `TCPriceReversalPredictor` |
| FinBERT (text sentiment) | 20% | `CONTEXT_UPDATE` with `type=finbert_sentiment` |
| Ollama (local LLM) | 20% | `CONTEXT_UPDATE` with `type=ollama_sentiment` |
| Claude Brain | 20% | `BRAIN_ANALYSIS` events (keyword extraction) |

Signals are only published when combined confidence exceeds 0.6. Stale sources (>10 min) are excluded from the ensemble. TCN model weights can be loaded from a trained checkpoint via `TCN_MODEL_PATH` config.

## Configuration

Settings in `src/hean/config.py` via Pydantic `BaseSettings` (`HEANSettings` class, loaded from `.env`):

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
BRAIN_ANALYSIS_INTERVAL=60          # Seconds between analyses

# Oracle (trained models)
TCN_MODEL_PATH=                     # Path to trained TCN weights (.pt)
LSTM_MODEL_PATH=                    # Path to trained LSTM weights (.h5)

# Ollama (local LLM sentiment, alongside Brain)
OLLAMA_ENABLED=false                # Requires running Ollama server
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_SENTIMENT_INTERVAL=300       # Seconds between analyses

# RL Risk Manager (trained PPO agent)
RL_RISK_ENABLED=false               # Requires trained model
RL_RISK_MODEL_PATH=                 # Path to PPO model (.zip)
RL_RISK_ADJUST_INTERVAL=60          # Seconds between adjustments

# Strategies (each has an enable flag)
IMPULSE_ENGINE_ENABLED=true
FUNDING_HARVESTER_ENABLED=true
BASIS_ARBITRAGE_ENABLED=true
# ... HF_SCALPING_ENABLED, MOMENTUM_TRADER_ENABLED, etc.
```

All new configurable parameters must be added to `HEANSettings` in `src/hean/config.py` with sensible defaults. Do not hardcode values.

### Environment Files

Two env files are used:
- **`.env`** — Main config loaded by `HEANSettings` (Bybit keys, strategy flags, AI keys, capital). Copy from `.env.example`
- **`backend.env`** — Docker-specific overrides (auth, CORS, runtime flags). Copy from `backend.env.example`

In Docker, the `api` and `symbiont-testnet` services both load `.env`. `backend.env` provides additional Docker-specific overrides. Both files are needed for Docker deployments.

## API Endpoints

All under `/api/v1/` prefix. New endpoints go in dedicated files within `src/hean/api/routers/` and must be registered in `src/hean/api/main.py`.

Key endpoints:

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
| `GET /brain/analysis` | Latest AI market analysis |
| `GET /council/status` | Council decision status |
| `POST /engine/start` | Start trading engine |
| `POST /engine/stop` | Stop trading engine |

WebSocket: `ws://localhost:8000/ws` — topics: `system_status`, `order_decisions`, `ai_catalyst`

## iOS App

SwiftUI app in `ios/`, targeting iOS 17+. Open `ios/HEAN.xcodeproj` in Xcode. ~162 Swift files in build phase.

### Architecture
- **DIContainer** (`Core/DI/DIContainer.swift`) — `@EnvironmentObject` dependency injection
- **APIClient** (`Core/Networking/APIClient.swift`) — Actor-based HTTP client
- **APIEndpoints** (`Core/Networking/APIEndpoints.swift`) — Centralized endpoint enum
- **Command Center** — 6 tabs: Live, Mind, Action, X-Ray, Genesis, Settings

### Critical iOS Patterns
- Backend `snake_case` → iOS `camelCase`: always add `CodingKeys`
- Backend field names diverge: `position_id`→`id`, `current_price`→`markPrice`, `size`→`quantity`
- Use `decodeIfPresent` with defaults for optional fields
- Custom `init(from:)` with fallback keys (try `"id"` then `"position_id"`)
- Case-insensitive enum decoding (`"buy"/"BUY"/"Buy"`)
- `Services.swift` is the compiled consolidated service file; `Services/Live/*.swift` files are NOT in the build
- `DesignSystem/` contains compiled color/formatting extensions (`Color(hex:)`, `Double.asCurrency`, etc.)
- Not all `.swift` files in the directory are compiled; check `project.pbxproj` for the source of truth
- When adding new Swift files, manually add PBXBuildFile + PBXFileReference + PBXGroup entries to `project.pbxproj`
- Build IDs follow sequential pattern: `B1000xxx` (file ref), `A1000xxx` (build file), `E1000xxx` (groups)
- Complex Canvas closures cause Swift type-checker timeouts — extract into separate private structs with explicit type annotations
- Bilingual strings via `L.isRussian` in `DesignSystem/Strings.swift`

### Genesis Tab
Cinematic 5-scene animated intro (`Features/Genesis/`):
1. Chaos — chaotic market waves (Canvas + TimelineView)
2. Spark of Reason — order emerges from chaos
3. Architecture of Consciousness — 6 agent nodes with neural connections
4. Signal Path — pulse through Analysis→Strategy→Risk→Execution pipeline
5. Conscious Growth — nodes converge, equity curve draws

Uses pure SwiftUI animations (TimelineView, Canvas, Shape with animatableData). Auto-play mode with 8s per scene + manual navigation.

## Web Dashboard

Next.js 15 app in `dashboard/` with React 19, Zustand state management, Tailwind 4, Framer Motion, and Recharts.

Key structure:
- `src/store/heanStore.ts` — Zustand store for global state
- `src/services/api.ts` + `websocket.ts` — Backend communication
- `src/components/tabs/` — CockpitTab, NeuroMapTab, TacticalTab, BlackBoxTab
- `src/components/cockpit/` — EquityChart, PositionsSummary, LiveFeed, MetricRow
- `src/components/neuromap/` — NeuroMapView, PulsingNode, AnimatedConnection (visual agent topology)
- `src/components/ui/` — GlassCard, MetricCard, AnimatedNumber, StatusBadge

Runs on port 3001 (`npm run dev`). Connects to backend API on port 8000.

## CI/CD

GitHub Actions workflow in `.github/workflows/docker-build-deploy.yml`:
- **On push/PR to main:** runs ruff check, mypy, pytest with coverage
- **On push to main:** builds multi-platform Docker images (amd64/arm64), pushes to GHCR, Trivy security scan, Kubernetes deploy

## Code Conventions

- Ruff for linting (line-length 100, py311 target). Rules: E, W, F, I, B, C4, UP. Use `ruff format .` for formatting separately from `ruff check`
- mypy strict mode (disallow_untyped_defs, strict_equality)
- `asyncio_mode = "auto"` in pytest — all async tests auto-detected
- Logging: use `from hean.logging import get_logger; logger = get_logger(__name__)` — never bare `print()` or stdlib `logging` directly
- Strategies inherit from `BaseStrategy` (`src/hean/strategies/base.py`)
- All exchange interactions go through `BybitHTTPClient` (never direct HTTP calls)
- New AI/ML modules should be gated by config flags and wrapped in try/except for optional deps
- `DRY_RUN=true` is default — blocks real order placement with a hard RuntimeError
