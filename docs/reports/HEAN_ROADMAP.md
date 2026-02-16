# HEAN Project Roadmap

> Full-system audit completed 2026-02-08. This document captures the current state of every subsystem and defines the integration path to make everything work together.

---

## Current State Overview

| Layer | Files | Integration | Health |
|-------|-------|-------------|--------|
| **Python Backend (Core)** | ~120 .py | 85% | Working, some orphans |
| **EventBus** | 70 event types | 52% | Core flow OK, advanced modules disconnected |
| **Strategies** | 8 strategies | 90% | Active, connected to bus |
| **Risk Management** | 6 modules | 80% | Working, missing event publishing |
| **Execution** | router_bybit_only.py (primary) | 85% | Active, fills processing OK |
| **Physics Engine** | 12 files | 30% | Runs but outputs unused |
| **Brain (Claude AI)** | 4 files | 40% | Partial - only ImpulseEngine consumes |
| **Symbiont X** | 37 files | 0% | COMPLETELY ISOLATED |
| **Web UI (React)** | ~20 components | 70% | Core dashboard OK, physics components orphaned |
| **iOS App** | 76/96 compiled | 40% | Models OK, 74% endpoints missing |
| **Docker** | 9 Dockerfiles | 85% | 2 blockers, 1 security issue |
| **Tests** | 671 tests, 73 files | 16% coverage | All pass, API mismatch found |

---

## Phase 1: Critical Fixes (Blockers)

These prevent the system from working correctly. Fix first.

### 1.1 EventBus: POSITION_CLOSE_REQUEST orphan
**Severity:** CRITICAL
**Files:** `src/hean/core/intelligence/oracle_integration.py:183,214`
**Problem:** Oracle publishes POSITION_CLOSE_REQUEST but NO component subscribes to it. Close requests go into the void.
**Fix:** Add handler in `src/hean/main.py` TradingSystem to subscribe to POSITION_CLOSE_REQUEST and route to ExecutionRouter.

### 1.2 EventBus: Physics event type mismatch
**Severity:** CRITICAL
**Files:** `src/hean/physics/engine.py:173`, `src/hean/core/context_aggregator.py:82`
**Problem:** PhysicsEngine publishes `CONTEXT_UPDATE` but ContextAggregator subscribes to `PHYSICS_UPDATE`. Events never reach the aggregator.
**Fix:** Change PhysicsEngine to publish `PHYSICS_UPDATE` event type, or change ContextAggregator to listen for `CONTEXT_UPDATE` with `context_type="physics"`.

### 1.3 EventBus: CONTEXT_READY has no consumers
**Severity:** CRITICAL
**Files:** `src/hean/core/context_aggregator.py:300`, `src/hean/strategies/base.py`
**Problem:** ContextAggregator publishes unified context from Physics+Brain+Oracle+OFI+Causal, but NO STRATEGY subscribes to it. All that intelligence is wasted.
**Fix:** Add CONTEXT_READY subscription in BaseStrategy. Pass context data to strategy `generate_signal()` method.

### 1.4 API Response Structure Mismatch
**Severity:** CRITICAL
**Files:** `src/hean/api/routers/trading.py`, `tests/test_api.py`
**Problem:** API returns raw `list[dict]` for positions and orders. Tests expect `{"positions": [...]}`. iOS app expects raw arrays. Inconsistency causes confusion.
**Fix:** Standardize: either wrap all responses or document raw arrays. Update tests and iOS models to match.

### 1.5 Security: API keys committed to repo
**Severity:** CRITICAL
**Files:** `.env.symbiont:5-6`
**Problem:** Real Bybit testnet API keys committed in plain text.
**Fix:** Rotate keys immediately. Add `.env.symbiont` to `.gitignore`. Use `.env.symbiont.example` template only.

### 1.6 Docker: Missing ui.env reference
**Severity:** CRITICAL
**Files:** `docker-deploy.sh:62`
**Problem:** Deployment script requires `ui.env` which doesn't exist. Deploy will fail.
**Fix:** Remove `ui.env` from required_files list in docker-deploy.sh (UI uses build-time env vars).

---

## Phase 2: Integration of Disconnected Modules

These modules exist and work individually but aren't connected to the main system.

### 2.1 Symbiont X Integration (37 files, 0% integrated)
**Severity:** HIGH
**Directory:** `src/hean/symbiont_x/`
**Problem:** Entire module operates in a parallel universe. Has its own WS connectors, execution kernel, portfolio allocator - none connected to EventBus.
**Plan:**
1. Add `EventBus` parameter to `HEANSymbiontX.__init__()`
2. Bridge `NervousSystem.ws_connectors` to subscribe to main EventBus TICK events
3. Bridge `ExecutionKernel` to publish ORDER_REQUEST via main EventBus
4. Bridge `DecisionLedger` to log decisions from main system events
5. Bridge `RegimeClassifier` to consume REGIME_UPDATE events
6. Add Symbiont X startup to `TradingSystem.__init__()` in main.py (behind config flag)

### 2.2 RiskGovernor Event Publishing
**Severity:** HIGH
**Files:** `src/hean/risk/risk_governor.py`
**Problem:** RiskGovernor logs state changes but never publishes `RISK_ALERT` or `RISK_BLOCKED` events. UI and other components can't react.
**Fix:** Publish `RISK_ALERT` when state transitions (NORMAL->SOFT_BRAKE->QUARANTINE->HARD_STOP). Publish `RISK_BLOCKED` when blocking a signal.

### 2.3 Brain Module: Expand to all strategies
**Severity:** HIGH
**Files:** `src/hean/brain/`, `src/hean/strategies/base.py`
**Problem:** Brain analysis only consumed by ImpulseEngine. FundingHarvester, BasisArbitrage, and 5 other strategies ignore it.
**Fix:** Add `BRAIN_ANALYSIS` subscription to BaseStrategy. Apply brain confidence as a signal weight multiplier.

### 2.4 Physics -> Strategy Pipeline
**Severity:** HIGH
**Problem:** Physics engine calculates temperature, entropy, phase, Szilard profit, participant breakdown, anomalies - none influence trading decisions.
**Fix:** After fixing Phase 1.2 and 1.3, strategies will receive unified context via CONTEXT_READY. Add physics-based signal filters (e.g., don't trade when entropy is extreme).

### 2.5 META_LEARNING_PATCH orphan
**Severity:** MEDIUM
**Files:** `src/hean/core/intelligence/meta_learning_engine.py:429`
**Problem:** Meta-learning generates parameter patches but nothing applies them.
**Fix:** Add subscriber in TradingSystem that applies patches to strategy parameters via STRATEGY_PARAMS_UPDATED event.

### 2.6 Unintegrated Python Modules
**Severity:** MEDIUM
**Problem:** Several modules exist but are never imported by any other module:
- `src/hean/portfolio/rl_allocator.py` - RL-based portfolio allocation (never imported)
- `src/hean/core/intelligence/transformer_predictor.py` - Transformer price predictor (never imported)
- `src/hean/funding_arbitrage/` - Cross-exchange funding arb (never imported from main)
- `src/hean/google_trends/` - Google Trends strategy (never imported from main)
- `src/hean/ml_predictor/` - LSTM predictor (never imported from main)
- `src/hean/sentiment/` - Sentiment analysis (never imported from main)

**Fix:** For each module, either:
1. Add to TradingSystem initialization (behind config flags)
2. Register as additional strategies in `src/hean/strategies/__init__.py`
3. Document as experimental/future in ROADMAP

---

## Phase 3: Web UI Completion

### 3.1 Physics Dashboard Integration
**Severity:** HIGH
**Files:** `apps/ui/src/app/components/trading/DashboardV2.tsx` + 6 physics components
**Problem:** 7 fully-built physics visualization components (TemperatureGauge, EntropyGauge, PhaseIndicator, GravityMap, PlayersRadar, BrainTimeline, DashboardV2) are orphaned - never imported or rendered.
**Fix:**
1. After Phase 2.4, backend will publish physics_update via WebSocket
2. Add `physics_update` and `brain_update` to WsTopic type in `apps/ui/src/app/api/client.ts`
3. Add route for DashboardV2 in App.tsx (e.g., toggle view or second tab)
4. Verify `usePhysicsWebSocket` hook connects properly

### 3.2 WebSocket Topic Alignment
**Severity:** MEDIUM
**Files:** `apps/ui/src/app/api/client.ts:38-48`, backend WS handler
**Problem:** Frontend subscribes to `physics_update` and `brain_update` topics that backend doesn't emit.
**Fix:** Add these topics to backend WebSocket broadcast in `src/hean/api/services/websocket_service.py`. Subscribe to corresponding EventBus events.

---

## Phase 4: iOS App Completion

### 4.1 Missing API Endpoints (28+ endpoints)
**Severity:** HIGH
**Files:** `ios/HEAN/Core/Networking/APIEndpoints.swift`
**Problem:** iOS defines only 22 endpoints but backend has 85+. Critical missing:
- Engine control: `/engine/start`, `/stop`, `/pause`, `/resume`
- Bulk operations: `/orders/cancel-all`, `/orders/close-all-positions`
- Telemetry: `/telemetry/*`
- Trading diagnostics: `/trading/why`, `/trading/metrics`
**Fix:** Add all missing endpoints to APIEndpoints.swift enum. Priority: engine control, then telemetry, then analytics.

### 4.2 Engine Control UI
**Severity:** HIGH
**Problem:** iOS has no way to start/stop/pause the trading engine.
**Fix:** Add engine control buttons to SettingsView with confirmation dialogs.

### 4.3 Hidden Functional Views
**Severity:** MEDIUM
**Problem:** StrategiesView, MarketsView, TradeView, RiskDashboardView, SignalFeedView all exist and work but are not accessible in navigation.
**Fix:** Add navigation links from appropriate tabs (e.g., Strategies from Action tab, Markets from Live tab).

### 4.4 WebSocket Topic Expansion
**Severity:** MEDIUM
**Problem:** iOS subscribes to only 4 WS topics (orders, positions, signals, risk_events). Backend publishes 10+ topics.
**Fix:** Add subscriptions for: `system_heartbeat`, `account_state`, `physics_events`, `brain_events`, `temporal_events`.

### 4.5 Orphaned iOS Files (20 files)
**Severity:** LOW
**Problem:** 20 Swift files exist in ios/ directory but are not compiled in pbxproj.
**Fix:** Either add to build (if needed) or delete to reduce confusion. Priority removals: duplicate Extensions/, individual Live/ service files (consolidated in Services.swift).

---

## Phase 5: Testing & Quality

### 5.1 Fix API Test Expectations
**Severity:** HIGH
**Files:** `tests/test_api.py`, `tests/test_api_routers.py`
**Problem:** Tests accept HTTP 500 as "passing". When engine is initialized, response structure assertions will fail.
**Fix:** Create proper integration tests that initialize engine facade. Validate actual response structures.

### 5.2 Increase Test Coverage (16% -> 50%+)
**Severity:** MEDIUM
**Problem:** 35,920 statements, only 5,615 covered. Critical modules at 0%:
- `api/main.py` (847 LOC, 0%)
- `api/engine_facade.py` (275 LOC, 0%)
- `main.py` (2000+ LOC, 0%)
- All physics modules (0%)
- All brain modules (0%)
**Fix:** Prioritize tests for: engine_facade, main.py signal handling, risk_governor state transitions, physics engine.

### 5.3 Symbiont X Test Coverage
**Severity:** LOW
**Problem:** Only genome_lab and decision_ledger have tests. 6 modules untested.
**Fix:** Add tests after Phase 2.1 integration.

---

## Phase 6: Docker & Deployment Hardening

### 6.1 Fix Cargo.lock for Rust Service
**Severity:** HIGH
**Files:** `rust_services/api_gateway/`, `.dockerignore`
**Problem:** Rust Dockerfile copies Cargo.lock but it doesn't exist and .dockerignore blocks it.
**Fix:** Generate Cargo.lock, commit it, update .dockerignore.

### 6.2 Non-root User in All Dockerfiles
**Severity:** MEDIUM
**Problem:** `api/Dockerfile` and `Dockerfile.testnet` run as root.
**Fix:** Add user creation and switch (like api/Dockerfile.optimized does).

### 6.3 Health Check Consistency
**Severity:** MEDIUM
**Problem:** Different scripts use `/health` vs `/api/v1/health`. Shell variable interpolation in CMD healthchecks.
**Fix:** Standardize to `/health` everywhere. Fix shell form in service healthchecks.

### 6.4 Missing docker-compose.monitoring.yml
**Severity:** LOW
**Files:** `Makefile:13`
**Problem:** Makefile references `docker-compose.monitoring.yml` which doesn't exist.
**Fix:** Create it or update Makefile targets to use production compose with `--profile monitoring`.

### 6.5 Add tini to All Production Images
**Severity:** LOW
**Problem:** Only `api/Dockerfile.optimized` uses tini for proper signal handling.
**Fix:** Add tini to all production Dockerfiles.

---

## Phase 7: Cleanup & Polish

### 7.1 Remove ~100 Orphaned Documentation Files
**Problem:** Root directory has 100+ abandoned `.md` files (Russian and English) from various development phases. These are deleted in git but untracked copies exist.
**Fix:** Clean up untracked files. Move any valuable content to `docs/archive/`.

### 7.2 Consolidate Duplicate Code
**Problem:**
- iOS: Services duplicated in both `Services.swift` and `Services/Live/*.swift`
- Backend: Two execution routers (`router.py` and `router_bybit_only.py`)
- Multiple `.env` example files with overlapping content
**Fix:** Delete unused duplicates. Document which files are authoritative.

### 7.3 Document Environment Variables
**Severity:** LOW
**Problem:** Several env vars used in code but not documented:
- `ABSOLUTE_PLUS_ENABLED`, `META_LEARNING_AUTO_PATCH`, `MULTI_SYMBOL_ENABLED`
- `BRAIN_ENABLED`, `BRAIN_ANALYSIS_INTERVAL`
**Fix:** Add all config vars to .env.example files.

---

## Integration Architecture (Target State)

```
                              HEAN Target Architecture

    ┌─────────────────────────────────────────────────────────────────┐
    │                         EVENT BUS                               │
    │  TICK ─── SIGNAL ─── ORDER_REQUEST ─── ORDER_FILLED            │
    │  PHYSICS_UPDATE ─── BRAIN_ANALYSIS ─── CONTEXT_READY           │
    │  RISK_ALERT ─── RISK_BLOCKED ─── EQUITY_UPDATE                 │
    │  META_LEARNING_PATCH ─── POSITION_CLOSE_REQUEST                │
    └─────────┬───────────────────────────────────────┬───────────────┘
              │                                       │
    ┌─────────▼─────────┐               ┌─────────────▼──────────────┐
    │   Market Data      │               │    Context Aggregator       │
    │   (Bybit WS)       │               │  Physics + Brain + Oracle   │
    │     ↓ TICK          │               │  + OFI + Causal             │
    └─────────┬──────────┘               │     ↓ CONTEXT_READY         │
              │                           └──────────────┬─────────────┘
              │                                          │
    ┌─────────▼──────────────────────────────────────────▼──────────────┐
    │                       STRATEGIES (8)                               │
    │  ImpulseEngine | FundingHarvester | BasisArbitrage | CorrelationArb│
    │  InventoryNeutralMM | LiquiditySweep | RebateFarmer | MultiFactor │
    │  + GoogleTrends | Sentiment | MLPredictor (Phase 2)               │
    │                ↓ SIGNAL                                            │
    └────────────────┬───────────────────────────────────────────────────┘
                     │
    ┌────────────────▼───────────────────────────────────────────────────┐
    │                    RISK MANAGEMENT                                 │
    │  RiskGovernor (state machine) → RISK_ALERT / RISK_BLOCKED          │
    │  KillSwitch (drawdown guard) → KILLSWITCH_TRIGGERED                │
    │  PositionSizer | KellyCriterion | DynamicRisk                      │
    │                ↓ ORDER_REQUEST                                      │
    └────────────────┬───────────────────────────────────────────────────┘
                     │
    ┌────────────────▼───────────────────────────────────────────────────┐
    │                    EXECUTION                                       │
    │  ExecutionRouter (router_bybit_only.py) → Bybit HTTP               │
    │  PositionMonitor | SignalDecay | AdaptiveTTL | SmartExecution       │
    │                ↓ ORDER_PLACED / ORDER_FILLED                        │
    └────────────────┬───────────────────────────────────────────────────┘
                     │
    ┌────────────────▼───────────────────────────────────────────────────┐
    │                    PORTFOLIO                                        │
    │  PortfolioAccounting | ProfitCapture | Allocator | RLAllocator      │
    │                ↓ POSITION_OPENED / PNL_UPDATE / EQUITY_UPDATE       │
    └────────────────┬───────────────────────────────────────────────────┘
                     │
    ┌────────────────▼───────────────────────────────────────────────────┐
    │                    SYMBIONT X (Phase 2.1)                           │
    │  GenomeLab | ImmuneSys | DecisionLedger | CapitalAllocator         │
    │  RegimeBrain | AdversarialTwin | NervousSystem                     │
    │  ↕ EventBus bridge                                                  │
    └────────────────────────────────────────────────────────────────────┘
                     │
    ┌────────────────▼───────────────────────────────────────────────────┐
    │                    FRONTENDS                                        │
    │                                                                     │
    │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
    │  │ FastAPI + WS  │   │  React/Vite  │   │  iOS SwiftUI │           │
    │  │ Port 8000     │   │  Port 3000   │   │  Xcode       │           │
    │  │ 85+ endpoints │   │  Dashboard   │   │  5 tabs      │           │
    │  │ WebSocket     │   │  + Physics   │   │  + engine    │           │
    │  │ broadcast     │   │  dashboard   │   │  controls    │           │
    │  └──────────────┘   └──────────────┘   └──────────────┘           │
    └────────────────────────────────────────────────────────────────────┘
```

---

## Priority Execution Order

| # | Phase | Items | Effort | Impact |
|---|-------|-------|--------|--------|
| 1 | **1.1-1.6** | Critical fixes (6 items) | 1-2 days | Unblocks everything |
| 2 | **2.2** | RiskGovernor events | 2-3 hours | Risk visibility |
| 3 | **2.3-2.4** | Brain + Physics -> Strategies | 1 day | Smarter trading |
| 4 | **3.1-3.2** | Web UI physics dashboard | 1 day | Full visualization |
| 5 | **4.1-4.2** | iOS engine control + endpoints | 2 days | Mobile control |
| 6 | **2.1** | Symbiont X integration | 3-5 days | Evolution + immune system |
| 7 | **2.6** | Unintegrated modules | 2-3 days | More strategies |
| 8 | **5.1-5.2** | Test coverage | 3-5 days | Production safety |
| 9 | **6.1-6.5** | Docker hardening | 1-2 days | Deployment reliability |
| 10 | **4.3-4.5** | iOS completion | 2-3 days | Full mobile app |
| 11 | **7.1-7.3** | Cleanup & polish | 1-2 days | Code quality |

---

## Metrics to Track

- **EventBus Integration:** 52% -> 95% (all modules connected)
- **Test Coverage:** 16% -> 50%+ (critical paths covered)
- **iOS API Coverage:** 26% -> 80%+ (all critical endpoints)
- **Web UI Components:** 70% -> 100% (physics dashboard live)
- **Symbiont X Integration:** 0% -> 80% (EventBus bridged)
- **Docker Deploy Success:** Blocked -> Clean deploy

---

*Generated by full-system audit on 2026-02-08. All findings verified by 6 parallel audit agents covering: Python backend, EventBus, iOS app, React UI, Docker infra, and test suite.*
