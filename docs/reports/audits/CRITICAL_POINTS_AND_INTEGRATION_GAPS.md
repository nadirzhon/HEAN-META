# HEAN Trading System - Critical Points & Integration Gaps

## Executive Summary

This document provides a comprehensive analysis of the HEAN trading system, identifying critical architectural points and areas where code integration is missing, incomplete, or broken.

**Analysis Date:** 2026-01-30
**Last Updated:** 2026-01-31
**Total Issues Found:** 80+
**Critical Issues:** 25 (6 FIXED)
**High Priority Issues:** 30

## Fixed Issues (2026-01-31)

| Issue | Status | File |
|-------|--------|------|
| 4 unregistered API routers | **FIXED** | `src/hean/api/main.py` |
| fast_router.py hard C++ dependency | **FIXED** | `src/hean/execution/fast_router.py` |
| singularity.py creating new EngineFacade | **FIXED** | `src/hean/api/routers/singularity.py` |
| Unsafe attribute access in routers | **FIXED** | `causal_inference.py`, `meta_learning.py`, `multimodal_swarm.py` |

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Critical Integration Gaps](#2-critical-integration-gaps)
3. [Unregistered API Routers](#3-unregistered-api-routers)
4. [Orphaned Modules (Never Used)](#4-orphaned-modules-never-used)
5. [Stub/Mock Implementations](#5-stubmock-implementations)
6. [C++ Module Dependencies](#6-c-module-dependencies)
7. [Event Bus Integration Gaps](#7-event-bus-integration-gaps)
8. [Frontend-Backend Mismatches](#8-frontend-backend-mismatches)
9. [Configuration Issues](#9-configuration-issues)
10. [Recommendations](#10-recommendations)

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Frontend (React/Vite)                     │
│              apps/ui/src/app/ - TypeScript                  │
└──────────────────────┬──────────────────────────────────────┘
                       │ WebSocket + REST
┌──────────────────────▼──────────────────────────────────────┐
│                  API Gateway (FastAPI)                      │
│              src/hean/api/main.py                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│            Engine Facade (Orchestration)                    │
│              src/hean/api/engine_facade.py                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Event Bus (Central Hub)                        │
│              src/hean/core/bus.py                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Strategies    Risk System    Exchange
   (3 active)    (RiskGovernor)  (Bybit)
```

### Key Entry Points

| Component | File | Status |
|-----------|------|--------|
| Main Trading System | `src/hean/main.py` | Active |
| API Server | `src/hean/api/main.py` | Active |
| Engine Facade | `src/hean/api/engine_facade.py` | Active |
| Event Bus | `src/hean/core/bus.py` | Active |

---

## 2. Critical Integration Gaps

### 2.1 Severity Matrix

| Issue Type | Count | Severity | Impact |
|------------|-------|----------|--------|
| Unregistered API Routers | 4 | CRITICAL | 12 endpoints unreachable |
| Orphaned Modules | 20+ | HIGH | Dead code, maintenance burden |
| Stub Implementations | 15+ | HIGH | Features don't work |
| Missing Event Bus Connections | 8+ | HIGH | Systems isolated |
| C++ Dependency Failures | 5+ | CRITICAL | Execution may fail |
| TODO/FIXME Markers | 50+ | MEDIUM | Incomplete features |

### 2.2 Critical Data Flow Breaks

```
BROKEN: ML Predictor → (nowhere)
        Strategy defined but never instantiated

BROKEN: Sentiment Analysis → (nowhere)
        6 modules built, never connected to trading

BROKEN: Google Trends → (nowhere)
        Strategy exists, never registered

BROKEN: Oracle Integration → (nowhere)
        Never subscribed to EventBus

BROKEN: Causal Inference Router → (not registered)
        Imported but not included in app
```

---

## 3. Unregistered API Routers

**Status: FIXED (2026-01-31)**

~~**Location:** `src/hean/api/main.py` (lines 1506-1531)~~

~~These routers are imported in `src/hean/api/routers/__init__.py` but **NOT registered** with the FastAPI app:~~

All 4 previously unregistered routers are now registered:

### 3.1 Causal Inference Router
- **File:** `src/hean/api/routers/causal_inference.py`
- **Endpoints:**
  - `GET /causal-inference/stats`
  - `GET /causal-inference/relationships`
  - `GET /causal-inference/pre-echoes`
- **Impact:** Causal inference data inaccessible via API

### 3.2 Meta Learning Router
- **File:** `src/hean/api/routers/meta_learning.py`
- **Endpoints:**
  - `GET /meta-learning/state`
  - `GET /meta-learning/weights`
  - `GET /meta-learning/patches`
- **Impact:** Meta-learning system status hidden

### 3.3 Multimodal Swarm Router
- **File:** `src/hean/api/routers/multimodal_swarm.py`
- **Endpoints:**
  - `GET /multimodal-swarm/stats`
  - `GET /multimodal-swarm/tensors/{symbol}`
  - `GET /multimodal-swarm/modality-weights`
- **Impact:** Swarm consensus data inaccessible

### 3.4 Singularity Router
- **File:** `src/hean/api/routers/singularity.py`
- **Endpoints:**
  - `GET /api/metamorphic/sel`
  - `GET /api/causal/graph`
  - `GET /api/atomic/clusters`
- **Impact:** Advanced AI metrics hidden

**Fix Required:** Add to `src/hean/api/main.py`:
```python
app.include_router(causal_inference.router, prefix="/api", tags=["causal"])
app.include_router(meta_learning.router, prefix="/api", tags=["meta"])
app.include_router(multimodal_swarm.router, prefix="/api", tags=["swarm"])
app.include_router(singularity.router, prefix="/api", tags=["singularity"])
```

---

## 4. Orphaned Modules (Never Used)

### 4.1 ML Predictor System
**Directory:** `src/hean/ml_predictor/`

| File | Issue |
|------|-------|
| `predictor.py:201` | `TODO: Fetch from exchange API or database` |
| `features.py:267` | `TODO: Properly map normalized values back` |
| `trainer.py:155` | Uses fake sample data instead of real |
| `strategy.py` | Class defined, never imported by main system |

**Integration Status:** COMPLETELY ISOLATED - No references in:
- `src/hean/main.py`
- `src/hean/api/routers/`
- `src/hean/process_factory/`

### 4.2 Sentiment Analysis System
**Directory:** `src/hean/sentiment/`

| File | Purpose | Status |
|------|---------|--------|
| `analyzer.py` | NLP sentiment analysis | Never called |
| `aggregator.py` | Combine sentiment sources | Never called |
| `news_client.py` | Fetch news articles | Never called |
| `reddit_client.py` | Reddit scraper | Never called |
| `twitter_client.py` | Twitter API client | Never called |
| `__init__.py` | Package init | Imports exist |

**Associated Strategy:** `src/hean/strategies/sentiment_strategy.py` (8.8k LOC)
- Fully implemented
- Never instantiated in TradingSystem

### 4.3 Google Trends System
**Directory:** `src/hean/google_trends/`

| File | Issue |
|------|-------|
| `strategy.py:305` | `# TODO: Integrate with real-time price feed from bus` |

**Integration Status:** Never imported by main system

### 4.4 Funding Arbitrage System
**Directory:** `src/hean/funding_arbitrage/`

| File | Purpose | Status |
|------|---------|--------|
| `strategy.py` | Funding rate arbitrage | Never used |
| `binance_funding.py` | Binance integration | Never used |
| `okx_funding.py` | OKX integration | Never used |
| `aggregator.py` | Multi-exchange aggregation | Never used |

**Note:** Main system only uses Bybit, making these modules obsolete.

### 4.5 Symbiont X Alternative System
**Directory:** `src/hean/symbiont_x/`

This appears to be an alternative autonomous trading framework with unclear integration to the main TradingSystem.

| Component | File | Issue |
|-----------|------|-------|
| Main Symbiont | `symbiont.py:182` | `# TODO: Actually run tests` |
| Shutdown | `symbiont.py:293` | `# await self.ws_connector.disconnect()  # TODO: implement` |
| Portfolio | `capital_allocator/portfolio.py:234` | `TODO: Требует historical PnL data` |
| Rebalancer | `capital_allocator/rebalancer.py:236` | `# TODO: Implement proper performance tracking` |

---

## 5. Stub/Mock Implementations

### 5.1 Test Worlds (Completely Fake)
**File:** `src/hean/symbiont_x/adversarial_twin/test_worlds.py`

All three test worlds return **hardcoded metrics** instead of actual results:

```python
# ReplayWorld (lines 186-198)
total_trades = 100
winning_trades = 65
total_pnl = 1250.0  # HARDCODED!

# PaperWorld (lines 301-313)
total_trades = 85
winning_trades = 50
total_pnl = 850.0  # HARDCODED!

# MicroRealWorld (lines 419-430)
total_trades = 45
winning_trades = 24
total_pnl = 12.50  # HARDCODED!
```

**TODO markers in file:**
- Line 183: `# TODO: Implement actual backtest logic`
- Line 298: `# TODO: Implement actual paper trading logic`
- Line 416: `# TODO: Implement actual micro-real trading logic`
- Lines 252, 369, 492, 503: Multiple `TODO: Implement`

### 5.2 AI Factory Mock Implementation
**File:** `src/hean/ai/factory.py`

- Line 108: `# STUB: In real implementation, replay events with candidate strategy`
- Generates mock metrics instead of real evaluations
- Never actually replays events with strategies

**File:** `src/hean/ai/canary.py`

- Line 111: `# STUB: In real implementation, calculate actual Sharpe from returns`

### 5.3 Code Generation Engine Stubs
**File:** `src/hean/core/intelligence/codegen_engine.py`

| Line | Issue |
|------|-------|
| 423 | `holding_time_seconds=0.0,  # TODO: Calculate from timestamps` |
| 424 | `market_conditions={},  # TODO: Extract from regime/volatility` |
| 494 | `# TODO: Evaluate performance metrics from shadow testing` |
| 513 | `# TODO: Actual integration logic would go here` |

### 5.4 Stress Tests Framework
**File:** `src/hean/symbiont_x/adversarial_twin/stress_tests.py`

- Line 118: `# TODO: Implement actual stress test logic`
- Framework structure present, no actual logic

---

## 6. C++ Module Dependencies

### 6.1 Critical C++ Dependencies

The system has optional C++ modules for high-performance operations. Missing C++ modules cause issues:

**Package Init:** `src/hean/cpp_modules/__init__.py` (lines 15-43)
- Graceful fallback for missing C++ modules
- System continues with **50-100x performance degradation**

### 6.2 Hard Failures Without C++ - **FIXED**

| File | Issue | Severity | Status |
|------|-------|----------|--------|
| `src/hean/execution/fast_router.py` | ~~`RuntimeError` if C++ unavailable~~ Python fallback added | ~~CRITICAL~~ **FIXED** |
| `src/hean/indicators/fast_indicators.py:10-12` | Imports `indicators_cpp` | HIGH |
| `src/hean/core/intelligence/metamorphic_integration.py:19` | Imports `graph_engine_py` | HIGH |
| `src/hean/api/routers/system.py:135` | Imports `get_cpp_status` | MEDIUM |

**Fast Router Critical Path:**
```python
# src/hean/execution/fast_router.py
from hean.cpp_modules import order_router_cpp  # Line 10-17

if not order_router_cpp:
    raise RuntimeError("C++ order router required")  # Line 33
```

**Impact:** Order execution completely fails without C++ modules.

---

## 7. Event Bus Integration Gaps

### 7.1 Systems Not Connected to EventBus

These systems are defined but never subscribed to the EventBus in `src/hean/main.py`:

| System | File | Issue |
|--------|------|-------|
| Oracle Integration | `core/intelligence/oracle_integration.py` | Never instantiated |
| Metamorphic Integration | `core/intelligence/metamorphic_integration.py` | Never instantiated |
| Correlation Engine | Variable `_correlation_engine` | Never initialized (line 139) |
| Safety Net | Variable `_safety_net` | Never initialized (line 140) |

### 7.2 Main.py Uninitialized Systems

**File:** `src/hean/main.py`

```python
# Lines 144-145 - Defined but never started:
_meta_learning_engine = None
_causal_inference_engine = None
_multimodal_swarm = None
```

These engines have routers (unregistered) that try to access them, but the engines themselves are never instantiated.

### 7.3 Volatility Predictor Integration
**File:** `src/hean/core/intelligence/volatility_predictor.py`

- Line 145: `# TODO: Integrate with GraphEngineWrapper to get feature vector`
- Cannot get actual feature vectors from graph engine

---

## 8. Frontend-Backend Mismatches

### 8.1 WebSocket Duplication

The system has **two WebSocket implementations**:

1. **Native FastAPI WebSocket** (`src/hean/api/main.py` lines 57-229)
   - Custom `ConnectionManager` class
   - Actually used for real-time updates

2. **Socket.io Service** (`src/hean/api/services/websocket_service.py`)
   - Initialized at line 504
   - **Barely used** - duplicates functionality
   - Should be removed

### 8.2 HTTP-WebSocket Synchronization

**Issue:** Race condition between HTTP responses and WebSocket broadcasts

```
Frontend calls: POST /engine/start
  ↓
Router returns: 200 OK immediately
  ↓
EventBus publishes: ENGINE_STARTED (async)
  ↓
WebSocket broadcasts: system_status topic (delayed)

Result: Frontend may display stale state
```

### 8.3 Services Underutilization

| Service | Status | Issue |
|---------|--------|-------|
| `market_data_store.py` | Initialized | Working |
| `trading_metrics.py` | Initialized | Underutilized - not all API calls tracked |
| `websocket_service.py` | Initialized | Redundant - unused |

---

## 9. Configuration Issues

### 9.1 Configuration Sprawl
**File:** `src/hean/config.py`

- **200+ configuration fields**
- Many combinations untested
- Risk of invalid states

### 9.2 Disabled-by-Default Critical Features

| Feature | Config Key | Default | Issue |
|---------|-----------|---------|-------|
| Meta Learning | `absolute_plus_enabled` | `False` | System exists but disabled |
| AI Factory | `ai_factory_enabled` | `False` | Cannot generate strategies |
| Auto Patch | `meta_learning_auto_patch` | `False` | Requires writable volume |

---

## 10. Recommendations

### 10.1 Immediate Actions (CRITICAL)

1. **Register Missing Routers**
   - Add 4 routers to `src/hean/api/main.py`
   - Or delete if not needed

2. **Fix C++ Dependencies**
   - Build C++ modules OR
   - Add proper Python fallbacks for `fast_router.py`

3. **Remove Test World Hardcoding**
   - Implement actual backtest/paper/micro-real logic
   - Replace hardcoded metrics

### 10.2 Short-Term Actions (HIGH)

1. **Connect Orphaned Systems**
   - Either integrate ML Predictor, Sentiment, Google Trends
   - Or delete them to reduce maintenance burden

2. **Initialize Event Bus Subscribers**
   - Connect `_meta_learning_engine`, `_causal_inference_engine`, `_multimodal_swarm`
   - Or remove their routers

3. **Remove Duplicate WebSocket**
   - Keep native FastAPI WebSocket
   - Delete Socket.io wrapper

4. **Implement AI Factory Properly**
   - Replace stub with real event replay
   - Calculate actual performance metrics

### 10.3 Medium-Term Actions (MEDIUM)

1. **Complete Symbiont X Integration**
   - Clarify relationship with main TradingSystem
   - Implement all TODO items or deprecate

2. **Add Integration Tests**
   - Test data flow from strategy → execution → WebSocket
   - Verify all API endpoints reach business logic

3. **Configuration Validation**
   - Add Pydantic validators for dangerous combinations
   - Document all configuration options

4. **Clean Up Dead Code**
   - Remove unused exchange implementations (Binance, OKX)
   - Remove incomplete features or mark experimental

---

## Appendix: File Reference

### Critical Files Requiring Attention

| File | Priority | Issue |
|------|----------|-------|
| `src/hean/api/main.py` | CRITICAL | Missing router registrations |
| `src/hean/execution/fast_router.py` | CRITICAL | Hard C++ dependency |
| `src/hean/symbiont_x/adversarial_twin/test_worlds.py` | HIGH | Hardcoded metrics |
| `src/hean/ai/factory.py` | HIGH | Stub implementation |
| `src/hean/ml_predictor/strategy.py` | HIGH | Orphaned module |
| `src/hean/sentiment/` | HIGH | 6 orphaned files |
| `src/hean/core/intelligence/codegen_engine.py` | MEDIUM | Multiple TODOs |
| `src/hean/api/services/websocket_service.py` | MEDIUM | Redundant |

### Files Safe to Delete (Unused)

- `src/hean/funding_arbitrage/binance_funding.py`
- `src/hean/funding_arbitrage/okx_funding.py`
- `src/hean/api/services/websocket_service.py` (after removing references)

---

## Changelog

| Date | Author | Changes |
|------|--------|---------|
| 2026-01-30 | Claude Code Analysis | Initial comprehensive audit |
