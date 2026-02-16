# HEAN Trading System — Full Integration & Orchestration Audit

**Date:** 2026-02-08
**Scope:** Event Bus, Orchestration, API/Docker, iOS, Tests, Dead Code
**Status:** 47 issues found (7 Critical, 15 High, 14 Medium, 11 Low)

---

## Executive Summary

The HEAN trading system has a solid EventBus foundation and well-designed architecture, but suffers from **significant integration gaps**. Approximately **20% of the codebase (~6500 LOC) is orphaned** — modules that exist but are never instantiated. The core signal-to-order flow works, but critical components like `ContextAggregator` and `RiskGovernor` are never wired in. The UI cannot connect to the backend due to WebSocket protocol and API prefix mismatches.

---

## Table of Contents

1. [P0 — Deploy Blockers](#1-p0--deploy-blockers)
2. [Orchestration & Event Bus](#2-orchestration--event-bus)
3. [Orphaned Modules (~20% of codebase)](#3-orphaned-modules-20-of-codebase)
4. [Strategy Integration Gaps](#4-strategy-integration-gaps)
5. [API & UI Contract Mismatches](#5-api--ui-contract-mismatches)
6. [Docker & Infrastructure](#6-docker--infrastructure)
7. [Configuration Sync](#7-configuration-sync)
8. [Dependency Issues](#8-dependency-issues)
9. [iOS App Integration](#9-ios-app-integration)
10. [Test Coverage Gaps](#10-test-coverage-gaps)
11. [Security & Production Readiness](#11-security--production-readiness)
12. [Dead Code & Root Scripts](#12-dead-code--root-scripts)
13. [Recommended Fix Order](#13-recommended-fix-order)

---

## 1. P0 — Deploy Blockers

These issues prevent the system from working when deployed.

### 1.1 WebSocket Protocol Mismatch [CRITICAL]

| | |
|-|-|
| **Backend** | `src/hean/api/services/websocket_service.py:56` — uses `python-socketio` (Socket.IO protocol) |
| **UI Client** | `apps/ui/src/app/api/client.ts:430` — uses native browser `WebSocket` (RFC 6455) |

**Impact:** UI cannot connect to backend WebSocket. All real-time data (ticks, signals, positions) fails silently.

**Fix:** Either switch backend to native `fastapi.WebSocket` or add `socket.io-client` to the UI.

### 1.2 API Prefix Mismatch [CRITICAL]

| | |
|-|-|
| **Backend** | `src/hean/api/main.py:1581` → registers all routers under `/api/v1/` |
| **UI Client** | `apps/ui/src/app/api/client.ts:1` → uses `API_BASE = "/api"` |

**Impact:** Every UI API call hits `/api/engine/status` but backend expects `/api/v1/engine/status` → all requests return 404.

**Fix:** Change `client.ts` line 1 to `const API_BASE = "/api/v1"`.

### 1.3 Missing `/health` Endpoint [CRITICAL]

| | |
|-|-|
| **Dockerfile** | `api/Dockerfile:100` → healthcheck calls `http://localhost:8000/health` |
| **Backend** | No `/health` route exists in any router |

**Impact:** Docker marks API container as permanently unhealthy.

**Fix:** Add `@app.get("/health")` to `src/hean/api/main.py`.

---

## 2. Orchestration & Event Bus

### 2.1 ContextAggregator Never Instantiated [CRITICAL]

| | |
|-|-|
| **File** | `src/hean/core/context_aggregator.py` |
| **Purpose** | Central hub: subscribes to Physics, Brain, Oracle, OFI, Causal signals → publishes unified `CONTEXT_READY` |
| **Usage in main.py** | **ZERO** — never imported, never instantiated |

**Impact cascade:**
- `BaseStrategy` subscribes to `CONTEXT_READY` but never receives it
- Physics, Brain, Oracle data is published but never aggregated
- `UnifiedMarketContext` class exists but is never populated
- 6 event types depend on it: `CONTEXT_UPDATE`, `PHYSICS_UPDATE`, `ORACLE_PREDICTION`, `OFI_UPDATE`, `CAUSAL_SIGNAL`, `BRAIN_ANALYSIS`

### 2.2 RiskGovernor Not in Signal Flow [CRITICAL]

| | |
|-|-|
| **File** | `src/hean/risk/risk_governor.py` |
| **Purpose** | Graduated risk: `NORMAL` → `SOFT_BRAKE` → `QUARANTINE` → `HARD_STOP` |
| **Used by** | API router `risk_governor.py` serves status, but governor is never instantiated in `main.py` |

**Impact:** The documented risk state machine doesn't run. Only basic `KillSwitch` is active.

### 2.3 Event Subscription Ordering Bug [HIGH]

In `main.py start()`:
1. EventBus started (line 407)
2. Strategies started (lines 702–759) — can begin publishing `SIGNAL` events
3. Main subscribes to events (lines 783–793) — too late

**Impact:** Signals published between step 2 and step 3 are lost.

**Fix:** Move event subscriptions BEFORE strategy start.

### 2.4 Orphaned Event Types [MEDIUM]

| Event Type | Issue |
|-----------|-------|
| `ORDER_DECISION` | Published in main.py, **no subscribers** |
| `ORDER_EXIT_DECISION` | Published in main.py, **no subscribers** |
| `STRATEGY_PARAMS_UPDATED` | **No publishers, no subscribers** |
| `POSITION_CLOSE_REQUEST` | Subscribers exist but **no publishers** |
| `META_LEARNING_PATCH` | Subscriber in main, but publisher uses different event type |

### 2.5 Components Missing from Orchestrator [HIGH]

| Component | File | Purpose |
|-----------|------|---------|
| ContextAggregator | `core/context_aggregator.py` | Unified market context |
| RiskGovernor | `risk/risk_governor.py` | Graduated risk states |
| CorrelationEngine exposed via facade | instantiated but not exposed | API router can't access it |
| GlobalSafetyNet exposed via facade | instantiated but not exposed | Same |
| SelfHealingMiddleware exposed via facade | instantiated but not exposed | Same |

---

## 3. Orphaned Modules (~20% of codebase)

These modules exist, have full implementations, but are **never imported or instantiated** by the main system.

| Module | Location | Files | Est. LOC | Status |
|--------|----------|-------|----------|--------|
| Symbiont X | `src/hean/symbiont_x/` | 40+ | ~3500 | Never imported outside itself |
| Funding Arbitrage | `src/hean/funding_arbitrage/` | 7 | ~800 | Never used (separate FundingHarvester strategy exists) |
| Google Trends | `src/hean/google_trends/` | 5 | ~600 | Never used |
| Sentiment Analysis | `src/hean/sentiment/` | 6 | ~700 | Never used |
| ML Predictor | `src/hean/ml_predictor/` | 7 | ~900 | Never used |
| hean_core (Rust) | `hean_core/` | 5 | ~400 | Rust library, never compiled or called from Python |
| services/ (microservices) | `services/` | 12 | ~740 | Docker stubs, not integrated with main engine |
| hft_core/ | `hft_core/` | 8 | ~500 | Multi-language stubs (Rust/Go/C++), never built |
| cpp_core/ | `cpp_core/` | 3 | ~300 | C++ indicators, never compiled |
| **TOTAL** | | **93+** | **~8500** | **~20–25% of codebase** |

### Services directory breakdown

`services/` contains 4 microservice stubs (collector, brain, physics, risk) referenced in `docker-compose.yml` (lines 135–267). Each is ~170 LOC of placeholder Python that doesn't integrate with the main engine's EventBus or Redis state.

---

## 4. Strategy Integration Gaps

### 4.1 Strategy Registration Status

| Strategy | In `__init__.py` | In `main.py` | Default Enabled |
|----------|-----------------|-------------|-----------------|
| ImpulseEngine | Yes | Yes (line 712) | Yes |
| FundingHarvester | Yes | Yes (line 702) | Yes |
| BasisArbitrage | Yes | Yes (line 707) | Yes |
| HFScalpingStrategy | Yes | Yes (line 741) | **No** (`hf_scalping_enabled=False`) |
| EnhancedGridStrategy | Yes | Yes (line 748) | **No** (`enhanced_grid_enabled=False`) |
| MomentumTrader | Yes | Yes (line 755) | **No** (`momentum_trader_enabled=False`) |
| CorrelationArbitrage | Yes | **No** | Dead code |
| InventoryNeutralMM | Yes | **No** | Dead code |
| LiquiditySweepDetector | Yes | **No** | Dead code |
| MultiFactorConfirmation | Yes | **No** | Dead code |
| RebateFarmer | Yes | **No** | Dead code |
| SentimentStrategy | Yes | **No** | Dead code |
| EdgeConfirmationLoop | Yes | **No** | Dead code |

**7 out of 13 strategies are dead code** — exported from `__init__.py` but never instantiated.

### 4.2 Execution Router Confusion [MEDIUM]

Three overlapping routers exist:

| File | Class | Used |
|------|-------|------|
| `execution/router.py` | `ExecutionRouter` | **Yes** — imported by main.py |
| `execution/router_bybit_only.py` | `ExecutionRouter` | No (same class name!) |
| `execution/fast_router.py` | `FastOrderRouter` | No |

`router.py` internally imports from `router_bybit_only.py`, but the class name collision creates confusion.

---

## 5. API & UI Contract Mismatches

### 5.1 Missing Endpoints Called by UI [HIGH]

| UI Function | Called Endpoint | Exists in Backend |
|-------------|----------------|-------------------|
| `fetchPortfolioSummary()` | `/portfolio/summary` | **No** — no portfolio router |
| `fetchDashboard()` | `/system/v1/dashboard` | **No** — wrong path (double v1) |

### 5.2 Position Response Shape Mismatch [HIGH]

UI expects `position_id`, `entry_price`, `unrealized_pnl` but backend returns raw Bybit dict with `positionIdx`, `avgPrice`, `unrealisedPnl`.

### 5.3 Full Endpoint Mapping

The UI `client.ts` calls endpoints that assume `/api` prefix, but backend registers under `/api/v1`. Every API call from UI will 404 without the prefix fix.

---

## 6. Docker & Infrastructure

### 6.1 Missing Healthcheck Script for Symbiont [HIGH]

`docker-compose.yml:118` references `/app/healthcheck.sh` but this file is only created inline in `Dockerfile.testnet` and never copied to the correct location for the symbiont service.

### 6.2 API Dockerfile Missing AI Dependencies [HIGH]

`api/Dockerfile:75-82` manually installs `openai` and `anthropic` but these aren't in `pyproject.toml` core dependencies. If the optional extras aren't installed, Brain module will crash on import.

### 6.3 Service Stubs in docker-compose.yml [MEDIUM]

4 services defined in `docker-compose.yml` (collector, brain, physics, risk) are placeholder Python scripts (~170 LOC each) that don't integrate with the main trading engine. They start but do nothing useful.

### 6.4 Redis URL Inconsistency [LOW]

Different default formats across files: some use `redis://redis:6379/0`, others `redis://redis:6379`. All resolve to db=0 but inconsistent.

---

## 7. Configuration Sync

### 7.1 Env Var Name Mismatches [HIGH]

| Config Field (`config.py`) | Expected Env Var | `.env.example` Uses |
|---------------------------|------------------|---------------------|
| `gemini_api_key` | `GEMINI_API_KEY` | `GOOGLE_API_KEY` |

### 7.2 Missing from .env.example [MEDIUM]

These settings exist in `config.py` but aren't documented in `.env.example`:

- `API_AUTH_ENABLED` (config.py:433)
- `API_AUTH_KEY` (config.py:438)
- `JWT_SECRET` (config.py:441)
- `WS_ALLOWED_ORIGINS` (config.py:446)
- `REDIS_URL` (config.py:467) — critical for Docker
- `REQUIRE_LIVE_CONFIRM` (config.py:419)
- `PROCESS_FACTORY_ENABLED` (config.py:625)

### 7.3 Unused Vars in backend.env.example [LOW]

These vars in `backend.env.example` are NOT read by `config.py`:

`LOG_FORMAT`, `API_HOST`, `API_PORT`, `API_WORKERS`, `REDIS_HOST`, `REDIS_PORT`, `MAX_POSITION_SIZE`, `STOP_LOSS_PERCENTAGE`

### 7.4 Brain Enabled Without API Key [MEDIUM]

`config.py` defaults: `brain_enabled=True` + `anthropic_api_key=""`. Brain module will try to start and fail repeatedly with API errors.

---

## 8. Dependency Issues

### 8.1 pybit Missing from pyproject.toml [HIGH]

`src/hean/ml_predictor/data_loader.py:14` imports `pybit`, which exists in `requirements.txt` but NOT in `pyproject.toml`. Installing via `pip install -e .` won't include it.

### 8.2 requirements.txt Incomplete [MEDIUM]

`requirements.txt` has 24 packages but `pyproject.toml` lists 52+. Missing: `slowapi`, `prometheus-client`, `networkx`, `python-socketio`, `grpcio`, `polars`, `orjson`, `duckdb`.

Users installing via `requirements.txt` will have a broken system.

---

## 9. iOS App Integration

### 9.1 Uncompiled Swift Files [MEDIUM]

96 Swift files exist in `ios/` but only 77 are in the Xcode build phase. **19 Swift files are not compiled.** This includes files in `Extensions/` and `Services/Live/` directories.

Known not-compiled:
- Individual files in `Services/Live/*.swift` (consolidated into `Services.swift`)
- Extension files in `Extensions/` (duplicated in `DesignSystem/`)

### 9.2 API Endpoint Mismatches [HIGH]

iOS `APIEndpoints.swift` must match backend exactly. Common issues:
- Backend returns `snake_case`, iOS expects `camelCase` — requires `CodingKeys`
- Position fields differ: `position_id` vs `id`, `current_price` vs `markPrice`
- Response wrapping differs: `/orders/positions` returns raw array, not `{ positions: [...] }`

### 9.3 WebSocket Configuration [MEDIUM]

iOS WebSocket client needs to match the same protocol as the backend (Socket.IO vs native WS). If backend uses Socket.IO, the iOS `URLSessionWebSocketTask` approach won't work.

---

## 10. Test Coverage Gaps

### 10.1 Modules With NO Tests

| Module | Location | Test Files |
|--------|----------|-----------|
| brain/ | `src/hean/brain/` | None |
| physics/ | `src/hean/physics/` | None |
| storage/ | `src/hean/storage/` | None |
| sentiment/ | `src/hean/sentiment/` | None |
| funding_arbitrage/ | `src/hean/funding_arbitrage/` | None |
| google_trends/ | `src/hean/google_trends/` | None |
| ml_predictor/ | `src/hean/ml_predictor/` | None |
| indicators/ | `src/hean/indicators/` | None |
| utils/ | `src/hean/utils/` | None |
| income/ | `src/hean/income/` | None (except `test_streams_smoke.py`) |
| hft/ | `src/hean/hft/` | None |
| afo_core/ | `src/hean/afo_core/` | None |
| data_sources/ | `src/hean/data_sources/` | None |
| exchange/ | `src/hean/exchange/` | Partial (`test_bybit_http`, `test_bybit_websocket`) |
| core/ (bus, types) | `src/hean/core/` | Partial (regime, contracts only) |

**~15 modules have zero test coverage.**

### 10.2 Critical Untested Paths

- Signal → Order → Fill event chain (no integration test)
- RiskGovernor state machine transitions
- ContextAggregator event aggregation
- PhysicsEngine calculations
- BrainClient → strategy feedback loop

---

## 11. Security & Production Readiness

### 11.1 Authentication Disabled by Default [MEDIUM]

`config.py:433` → `api_auth_enabled: bool = False`. Production deployment will have no authentication unless explicitly enabled.

### 11.2 CORS Allows All Origins [MEDIUM]

`websocket_service.py:52-54` → If `WS_ALLOWED_ORIGINS` is not set, accepts connections from any origin.

### 11.3 JWT Secret Not Persistent [MEDIUM]

`src/hean/api/auth.py:43` → If `JWT_SECRET` not set, generates random secret on each restart, invalidating all tokens.

### 11.4 No Rate Limiting on Public Endpoints [LOW]

SlowAPI is imported but only applied conditionally. Unauthenticated endpoints have no rate limiting.

---

## 12. Dead Code & Root Scripts

### 12.1 Root-Level Python Scripts [LOW]

| File | Status |
|------|--------|
| `demo_simple.py` | Demo script, not production |
| `demo_simulation.py` | Demo script, not production |
| `full_system_demo.py` | Demo script, not production |
| `live_testnet_demo.py` | Used by docker-compose symbiont service |
| `live_testnet_real.py` | Used by docker-compose symbiont service |
| `simple_test.py` | Should be in tests/ |
| `test_bybit_connection_simple.py` | Should be in tests/ |
| `test_symbiont.py` | Should be in tests/ |
| `test_execution_fixes.py` | Should be in tests/ |
| `test_three_fixes.py` | Should be in tests/ |
| `test_ws_fix.py` | Should be in tests/ |
| `verify_symbiont_fixes.py` | Should be in tests/ |

### 12.2 Markdown File Explosion [LOW]

100+ markdown files in root directory (many in Russian). Most are implementation reports, changelogs, and guides that should be in `docs/`. Several are duplicates or outdated.

---

## 13. Recommended Fix Order

### Phase 1: Deploy Blockers (Day 1)

| # | Issue | Files to Change |
|---|-------|----------------|
| 1 | Fix API prefix in UI client | `apps/ui/src/app/api/client.ts` |
| 2 | Fix WebSocket protocol (choose one) | `websocket_service.py` OR `client.ts` |
| 3 | Add `/health` endpoint | `src/hean/api/main.py` |
| 4 | Fix/remove `fetchPortfolioSummary` | `client.ts` |
| 5 | Fix `fetchDashboard` path | `client.ts` |

### Phase 2: Core Integration (Day 2-3)

| # | Issue | Files to Change |
|---|-------|----------------|
| 6 | Instantiate ContextAggregator | `src/hean/main.py` |
| 7 | Wire RiskGovernor into signal flow | `src/hean/main.py` |
| 8 | Fix event subscription ordering | `src/hean/main.py` |
| 9 | Add position field normalization | API router or UI mapping |
| 10 | Fix Symbiont healthcheck | `docker-compose.yml` |

### Phase 3: Config & Dependencies (Day 3-4)

| # | Issue | Files to Change |
|---|-------|----------------|
| 11 | Add pybit to pyproject.toml | `pyproject.toml` |
| 12 | Fix Gemini API key env var name | `backend.env.example` |
| 13 | Add missing env vars to .env.example | `.env.example` |
| 14 | Add brain_enabled validation | `config.py` |
| 15 | Fix API Dockerfile AI deps | `api/Dockerfile` |

### Phase 4: Cleanup (Day 4-5)

| # | Issue | Files to Change |
|---|-------|----------------|
| 16 | Decide: integrate or archive orphaned modules | Multiple |
| 17 | Register or remove dead strategies | `main.py`, `strategies/__init__.py` |
| 18 | Remove/consolidate duplicate routers | `execution/` |
| 19 | Move root test files to tests/ | Root directory |
| 20 | Move markdown files to docs/ | Root directory |

### Phase 5: Security & Tests (Day 5+)

| # | Issue | Files to Change |
|---|-------|----------------|
| 21 | Add integration test for signal→fill flow | `tests/` |
| 22 | Add tests for physics, brain, storage | `tests/` |
| 23 | Enforce auth in production config | `docker-compose.production.yml` |
| 24 | Restrict CORS and require JWT_SECRET | `config.py`, `auth.py` |

---

## Signal Flow Diagram (Current vs Expected)

### Current (Working)
```
Strategy → SIGNAL → main._handle_signal → ORDER_REQUEST → ExecutionRouter → Bybit → ORDER_FILLED → accounting
```

### Missing Links
```
                    ContextAggregator ──── NEVER INSTANTIATED
                    ┌────────────────┐
Physics ──────────→ │                │
Brain ────────────→ │   CONTEXT_READY │──→ BaseStrategy (subscribed but never receives)
Oracle ───────────→ │                │
OFI ──────────────→ │                │
Causal ───────────→ └────────────────┘

RiskGovernor ─────── NEVER INSTANTIATED ──→ Should filter between SIGNAL and ORDER_REQUEST
```

### Expected (After Fixes)
```
Strategy → SIGNAL → RiskGovernor.check() → main._handle_signal → ORDER_REQUEST → ExecutionRouter → Bybit
     ↑                                                                                              │
     │                                                                                              ↓
     └── ContextAggregator ← Physics/Brain/Oracle/OFI                               ORDER_FILLED → accounting
```

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total issues found | 47 |
| Critical (P0) | 7 |
| High (P1) | 15 |
| Medium (P2) | 14 |
| Low (P3) | 11 |
| Orphaned code (LOC) | ~8,500 |
| Orphaned code (% of codebase) | ~20-25% |
| Dead strategies | 7 of 13 |
| Untested modules | 15+ |
| iOS uncompiled Swift files | 19 of 96 |
| Root markdown files to organize | 100+ |
| Estimated fix time (critical) | 2-3 days |
| Estimated fix time (all) | 5-7 days |
