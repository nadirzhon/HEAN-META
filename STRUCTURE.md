# HEAN — Logical Structure & Module Connections

## 1. Three Frontends, One Backend

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTENDS                             │
│                                                          │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────┐ │
│  │ iOS App  │   │ Web Dashboard│   │ Other Clients    │ │
│  │ SwiftUI  │   │ Next.js 15   │   │ (Telegram, etc.) │ │
│  │ iOS 17+  │   │ :3001        │   │                  │ │
│  └────┬─────┘   └──────┬───────┘   └────────┬─────────┘ │
│       │                │                     │           │
│       └────────────────┼─────────────────────┘           │
│                        ▼                                 │
│              ┌──────────────────┐                        │
│              │  FastAPI Gateway │                        │
│              │  REST + WebSocket│                        │
│              │  :8000           │                        │
│              └────────┬─────────┘                        │
└───────────────────────┼──────────────────────────────────┘
                        ▼
                 BACKEND (Python)
```

---

## 2. Core — EventBus

All components communicate via a central event bus, never calling each other directly.

```
                    ┌─────────────────────┐
                    │      EventBus       │
                    │  (core/bus.py)      │
                    │                     │
                    │  Priority Queues:   │
                    │  ├─ CRITICAL ◀──── SIGNAL, ORDER_REQUEST, ORDER_FILLED
                    │  ├─ NORMAL  ◀──── TICK, POSITION_*, PNL_*
                    │  └─ LOW     ◀──── HEARTBEAT, STATUS
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────┴──┐    ┌──────┴────┐    ┌─────┴──────┐
     │ Subscribers│    │ Emitters  │    │ ServiceEvent│
     │ (handlers) │    │           │    │ Bridge      │
     └────────────┘    └───────────┘    │ (↔ Redis)   │
                                        └─────────────┘
```

Fast-path dispatch bypasses queues entirely for `SIGNAL`, `ORDER_REQUEST`, and `ORDER_FILLED` events, ensuring minimal latency for the trading hot path. Circuit breaker drops LOW events when queue utilization exceeds 95%.

---

## 3. Signal Chain (TICK → ORDER_FILLED)

The main pipeline from market data to executed order:

```
 Bybit WebSocket                          Bybit HTTP REST
      │                                        ▲
      ▼                                        │
 ┌─────────┐    ┌────────────┐    ┌────────────┴───┐    ┌───────────┐
 │  TICK    │───▶│ Strategies │───▶│ RiskGovernor   │───▶│ Execution │
 │  Events  │    │ (11 total) │    │ (State Machine)│    │ Router    │
 └─────────┘    └──────┬─────┘    └───────┬────────┘    └─────┬─────┘
                       │                  │                    │
                    SIGNAL           ORDER_REQUEST        ORDER_FILLED
                  (filtered)         (approved)          (confirmed)
                       │                  │                    │
                       │                  │                    ▼
                       │                  │              ┌───────────┐
                       │                  │              │ Portfolio  │
                       │                  │              │ Accounting │
                       │                  │              └───────────┘
                       │                  │
                  70-95% rejection    NORMAL → SOFT_BRAKE
                  (filter cascade)   → QUARANTINE → HARD_STOP
```

---

## 4. Full Event Flow

```
                         Bybit Exchange
                     ┌───────┴────────┐
                     │  ws_public.py  │  ws_private.py
                     │  (ticks, book) │  (orders, positions)
                     └───────┬────────┘
                             │
                        ┌────▼─────┐
                        │ EventBus │◄─────────────────────────────┐
                        └────┬─────┘                              │
           ┌─────────────────┼────────────────────────┐           │
           │                 │                        │           │
     ┌─────▼─────┐    ┌─────▼──────┐           ┌─────▼─────┐    │
     │ Strategies │    │  Physics   │           │   Brain   │    │
     │ (11 total) │    │  Engine    │           │  (Claude) │    │
     │            │    │            │           │           │    │
     │ TICK ──▶   │    │ TICK ──▶   │           │ Periodic  │    │
     │  analyze   │    │  temp,     │           │  analysis │    │
     │  filter    │    │  entropy,  │           │           │    │
     │  SIGNAL ──▶│    │  phase     │           │BRAIN_     │    │
     └─────┬──────┘    │PHYSICS_ ──▶│           │ANALYSIS──▶│    │
           │           │UPDATE      │           └───────────┘    │
           │           └────────────┘                             │
     ┌─────▼──────────┐                                          │
     │ Oracle          │  (4-source fusion)                      │
     │ Integration     │  TCN 40% + FinBERT 20%                  │
     │                 │  + Ollama 20% + Claude 20%              │
     │ ORACLE_         │                                          │
     │ PREDICTION ─────▶──────────────────────────────────────────┤
     └────────────────┘                                           │
           │                                                      │
     ┌─────▼──────────┐     ┌─────────────┐     ┌────────────┐  │
     │ RiskGovernor    │────▶│ Execution   │────▶│ Bybit HTTP │  │
     │                 │     │ Router      │     │ (order)    │  │
     │ NORMAL/BRAKE/   │     │             │     └──────┬─────┘  │
     │ QUARANTINE/STOP │     │ Idempotent  │            │        │
     └────────────────┘      │ + Retry     │      ORDER_FILLED   │
                             └─────────────┘            │        │
                                                        ▼        │
                             ┌──────────────┐    ┌──────────┐    │
                             │  Portfolio   │◄───│ Position │    │
                             │  Accounting  │    │ Reconcile│────┘
                             │              │    └──────────┘
                             │ PnL, Equity, │
                             │ Allocation   │
                             └──────────────┘
```

---

## 5. Module Map: `src/hean/`

```
src/hean/
│
├── main.py ─────────────── Entry point: TradingSystem (wires everything, ~3800 lines)
├── config.py ───────────── HEANSettings (1150+ lines, Pydantic BaseSettings, loads .env)
├── logging.py ──────────── Custom logger (get_logger)
│
├── core/ ───────────────── CORE INFRASTRUCTURE
│   ├── bus.py ──────────── EventBus (priority queues + circuit breaker)
│   ├── types.py ────────── EventType enum (98+ event types) + DTOs
│   ├── regime.py ───────── RegimeDetector (NORMAL/IMPULSE/RANGE/CRASH)
│   ├── timeframes.py ──── CandleAggregator (1m, 5m, 15m, 1h)
│   ├── context_aggregator.py  Fuses Brain + Physics + TCN + OFI + Causal signals
│   ├── market_context.py ── UnifiedMarketContext dataclass
│   ├── feedback_agent.py ── Learns from trading outcomes
│   ├── agent_registry.py ── Tracks active AI agents
│   │
│   ├── system/ ─────────── System Lifecycle
│   │   ├── component_registry.py  ComponentRegistry (init → start → stop with deps)
│   │   ├── health_monitor.py ──── System health monitoring
│   │   ├── error_analyzer.py ──── Error diagnosis
│   │   ├── redis_state.py ──────── Redis state management
│   │   └── service_event_bridge.py  EventBus ↔ Redis Streams bridge
│   │
│   ├── intelligence/ ──── AI/ML Signal Fusion
│   │   ├── oracle_integration.py  Hybrid 4-source fusion (TCN+FinBERT+Ollama+Claude)
│   │   ├── oracle_engine.py ───── Core oracle (TCN + fingerprinting)
│   │   ├── tcn_predictor.py ───── Temporal Convolutional Network (PyTorch)
│   │   ├── transformer_predictor.py  Transformer sequence model
│   │   ├── dynamic_oracle.py ──── Adaptive oracle with performance tracking
│   │   ├── correlation_engine.py ── Real-time Pearson for pair trading
│   │   ├── volatility_predictor.py  Vol forecasting (ONNX)
│   │   ├── causal_inference_engine.py  Granger causality + transfer entropy
│   │   ├── graph_engine.py ─────── Knowledge graph (lead-lag detection)
│   │   ├── meta_learning_engine.py  Recursive meta-learning
│   │   ├── multimodal_swarm.py ─── Multi-agent consensus
│   │   └── market_genome.py ────── Market state synthesis
│   │
│   ├── evolution/ ──────── Genetic Evolution
│   │   ├── genome_lab.py ── Genetic algorithm strategy tuning
│   │   ├── immune_system.py  Self-healing recovery mechanisms
│   │   └── nervous_system.py  Signal propagation coordination
│   │
│   └── telemetry/ ──────── Self-Insight
│       └── self_insight.py  Telemetry for Brain/Council analysis
│
├── strategies/ ─────────── 11 TRADING STRATEGIES
│   ├── base.py ─────────── BaseStrategy ABC (all inherit from this)
│   ├── manager.py ──────── StrategyManager (lifecycle: init → run → stop)
│   ├── impulse_engine.py ─ ImpulseEngine — Momentum + 12-layer filter cascade
│   ├── impulse_filters.py  Deterministic 12-layer filter cascade
│   ├── funding_harvester.py  FundingHarvester — Funding rate arbitrage
│   ├── basis_arbitrage.py ── BasisArbitrage — Futures/spot basis trading
│   ├── hf_scalping.py ───── HFScalping — High-frequency (40-60 trades/day)
│   ├── enhanced_grid.py ─── EnhancedGrid — Range-bound grid trading
│   ├── momentum_trader.py ── MomentumTrader — Trend following
│   ├── inventory_neutral_mm.py  InventoryNeutralMM — Market making
│   ├── correlation_arb.py ── CorrelationArbitrage — Pair trading
│   ├── rebate_farmer.py ──── RebateFarmer — Maker fee rebate capture
│   ├── liquidity_sweep.py ── LiquiditySweep — Institutional sweep detection
│   ├── sentiment_strategy.py  SentimentStrategy — Sentiment-based
│   ├── edge_confirmation.py ─ Edge validation before order
│   ├── multi_factor_confirmation.py  Multi-factor signal confirmation
│   ├── physics_aware_positioner.py ── Physics-based position sizing
│   └── physics_signal_filter.py ───── Physics-based signal filtering
│
├── risk/ ───────────────── RISK MANAGEMENT (20 files)
│   ├── risk_governor.py ── RiskGovernor state machine:
│   │                       NORMAL → SOFT_BRAKE → QUARANTINE → HARD_STOP
│   ├── killswitch.py ──── Emergency shutdown (>30% drawdown testnet)
│   ├── limits.py ───────── RiskLimits (max positions, orders, hold time)
│   ├── position_sizer.py ─ Position sizing (Kelly + regime + leverage)
│   ├── kelly_criterion.py  Kelly fractional sizing (recursive, confidence-scaled)
│   ├── immune_system.py ── Self-healing: isolate bad strategies → recover → restore
│   ├── recovery_engine.py  Recovery mechanism after immune isolation
│   ├── deposit_protector.py  Never go below initial capital
│   ├── capital_preservation.py  Reduce risk on drawdown/PF degradation
│   ├── smart_leverage.py ── Dynamic leverage (3-5x for high-quality only)
│   ├── multi_level_protection.py  Layered: per-strategy, hourly, daily
│   ├── rl_risk_manager.py ── PPO-based risk adjustment (optional)
│   ├── gym_env.py ────────── Gymnasium env for RL training
│   ├── dynamic_risk.py ──── Adaptive risk parameters
│   ├── tail_risk.py ──────── Catastrophe simulation + GlobalSafetyNet
│   ├── doomsday_sandbox.py ─ Doomsday scenario simulation
│   ├── anomaly_detector.py ─ General anomaly detection
│   ├── price_anomaly_detector.py  Price gaps, spikes, crashes
│   └── physics_position_sizer.py  Adjust size by temperature/entropy/phase
│
├── execution/ ──────────── ORDER EXECUTION (19 files)
│   ├── router_bybit_only.py  Production router (idempotent, retry-safe)
│   ├── router.py ────────── Generic router interface
│   ├── fast_router.py ──── Low-latency fast routing
│   ├── smart_execution.py ── TWAP vs Market vs Limit selection
│   ├── smart_order_selector.py  Market vs limit based on edge
│   ├── twap_executor.py ──── Time-weighted average price (large orders)
│   ├── order_manager.py ──── Order lifecycle management
│   ├── position_reconciliation.py  Sync local state with exchange
│   ├── position_monitor.py ── Force-close stale positions (TTL)
│   ├── slippage_estimator.py  Slippage prediction
│   ├── edge_estimator.py ──── Edge estimation
│   ├── signal_decay.py ────── Signal strength degradation over time
│   ├── adaptive_ttl.py ────── Adaptive order TTL
│   ├── order_timing.py ────── Timing optimization
│   ├── maker_retry_queue.py ── Retry failed maker orders
│   ├── atomic_executor.py ──── Atomic transaction execution
│   └── execution_diagnostics.py  Execution quality analysis
│
├── portfolio/ ──────────── PORTFOLIO MANAGEMENT (14 files)
│   ├── accounting.py ───── PortfolioAccounting (PnL, equity, drawdown)
│   ├── allocator.py ────── CapitalAllocator (1/N or performance-weighted)
│   ├── strategy_capital_allocator.py  Dynamic per-strategy allocation
│   ├── profit_capture.py ── Profit locking on targets
│   ├── profit_target_tracker.py  Track profit targets
│   ├── smart_reinvestor.py ─ Reinvest profits (85% default)
│   ├── rebalancer.py ────── Portfolio rebalancing
│   ├── capital_pressure.py  Pressure metrics
│   ├── decision_memory.py ── Block bad contexts after loss streaks
│   ├── strategy_memory.py ── Strategy performance memory
│   ├── meta_strategy_brain.py  Meta-brain for strategy selection
│   ├── evolution_bridge.py ── Bridge to Symbiont X
│   └── rl_allocator.py ──── RL-based allocation (optional)
│
├── physics/ ────────────── MARKET PHYSICS (12 files)
│   ├── engine.py ───────── PhysicsEngine (orchestrates T/S/phase, publishes PHYSICS_UPDATE)
│   ├── temperature.py ──── Market temperature (kinetic energy of price*volume)
│   ├── entropy.py ──────── Market entropy (volume distribution disorder)
│   ├── phase_detector.py ── Phase: ACCUMULATION → MARKUP → DISTRIBUTION → MARKDOWN
│   ├── szilard.py ──────── Szilard engine (max profit = T * I, information theory)
│   ├── participant_classifier.py  Retail / Whale / Algo detection
│   ├── anomaly_detector.py ─ Market anomaly detection
│   ├── temporal_stack.py ── Multi-timeframe temporal analysis
│   ├── cross_market.py ─── Cross-exchange dynamics
│   ├── emotion_arbitrage.py  Emotional pricing signals
│   └── rust_bridge.py ──── Optional Rust/C++ integration
│
├── brain/ ──────────────── AI BRAIN (Claude API)
│   ├── claude_client.py ── Anthropic API (periodic market analysis)
│   ├── models.py ───────── BrainAnalysis, BrainThought, TradingSignal
│   ├── snapshot.py ─────── Market snapshot formatter for LLM prompts
│   └── self_awareness_context.py  System introspection for AI
│
├── council/ ────────────── AI COUNCIL (multi-agent review)
│   ├── council.py ──────── AICouncil orchestrator
│   ├── members.py ──────── AI agents (Claude, Gemini, GPT, DeepSeek)
│   ├── review.py ───────── Consensus review logic
│   ├── prompts.py ──────── Role-specific system prompts
│   ├── executor.py ─────── Applies safe recommendations
│   └── introspector.py ─── Self-reflection (collects system state for review)
│
├── archon/ ─────────────── ARCHON BRAIN-ORCHESTRATOR
│   ├── archon.py ───────── Central coordination layer
│   ├── signal_pipeline.py  Signal lifecycle: GENERATED → ... → FILLED/DEAD_LETTER
│   ├── signal_pipeline_manager.py  End-to-end signal tracking
│   ├── cortex.py ───────── Strategic decision engine (system mode control)
│   ├── health_matrix.py ── Composite health score (0-100)
│   ├── heartbeat.py ────── Component liveness tracking
│   ├── reconciler.py ──── Local vs exchange state reconciliation
│   ├── chronicle.py ────── Audit trail for key trading decisions
│   ├── directives.py ──── PAUSE/RESUME/ACTIVATE/etc.
│   ├── genome_director.py  Directed evolution for Symbiont X genomes
│   └── protocols.py ────── Protocol definitions
│
├── exchange/ ───────────── EXCHANGE CONNECTORS
│   ├── bybit/
│   │   ├── http.py ─────── BybitHTTPClient (REST, instrument cache, circuit breaker)
│   │   ├── ws_public.py ── Public WebSocket (ticks, orderbook, funding)
│   │   ├── ws_private.py ─ Private WebSocket (orders, positions, reconnection)
│   │   ├── models.py ──── Bybit-specific Pydantic models
│   │   ├── integration.py  Integration helpers
│   │   └── price_feed.py ─ Price feed management
│   ├── executor.py ─────── SmartLimitExecutor (geometric slippage prediction)
│   └── models.py ───────── Exchange-level models
│
├── sentiment/ ──────────── SENTIMENT ANALYSIS
│   ├── analyzer.py ─────── FinBERT sentiment analyzer
│   ├── aggregator.py ──── Multi-source aggregation
│   ├── models.py ───────── Sentiment data models
│   ├── news_client.py ─── News scraping
│   ├── reddit_client.py ── Reddit sentiment
│   └── twitter_client.py ─ Twitter sentiment
│
├── storage/ ────────────── PERSISTENT STORAGE
│   └── duckdb_store.py ── DuckDB (ticks, physics, brain analyses)
│
├── observability/ ──────── OBSERVABILITY & MONITORING (12+ files)
│   ├── health.py ───────── HealthCheck (system readiness)
│   ├── health_score.py ─── Aggregated health score (0-100)
│   ├── metrics.py ──────── SystemMetrics (signals, orders, fills counters)
│   ├── signal_rejection_telemetry.py  Track rejections with reason codes
│   ├── money_critical_log.py ────── Critical financial events
│   ├── latency_histogram.py ─────── Latency tracking
│   ├── no_trade_report.py ────────── Diagnose why no trades occurred
│   ├── metrics_exporter.py ───────── MetricsExporter (file/Prometheus)
│   ├── prometheus_server.py ──────── Prometheus metrics server
│   └── monitoring/
│       └── self_healing.py ───────── Self-healing monitors
│
├── symbiont_x/ ─────────── GENETIC EVOLUTION ENGINE
│   ├── symbiont.py ─────── HEANSymbiontX orchestrator
│   ├── bridge.py ───────── Bridge to main trading system
│   ├── kpi_system.py ──── KPI tracking
│   ├── genome_lab/ ─────── Genetic algorithm (genomes, mutation, crossover)
│   ├── immune_system/ ─── Circuit breakers, risk constitution, reflexes
│   ├── decision_ledger/ ── Decision tracking and replay
│   ├── backtesting/ ────── Backtest engine for genome evaluation
│   ├── capital_allocator/  Genome-specific capital allocation
│   ├── adversarial_twin/ ─ Stress tests and survival scoring
│   ├── nervous_system/ ─── Health sensors and WS connectors
│   ├── regime_brain/ ──── Regime classification
│   └── execution_kernel/ ─ Execution kernel for evolved strategies
│
├── income/ ─────────────── INCOME STREAMS
│   └── streams.py ──────── Funding, Rebate, Basis, Volatility
│
├── api/ ────────────────── API GATEWAY
│   ├── main.py ─────────── FastAPI app + 23 routers + WebSocket
│   ├── engine_facade.py ── EngineFacade (unified TradingSystem interface)
│   ├── auth.py ─────────── JWT/API key authentication
│   ├── schemas.py ──────── Pydantic response schemas
│   ├── routers/ ────────── 23 route files
│   │   ├── engine.py ──── /engine/start, /engine/stop, /engine/status
│   │   ├── trading.py ─── /trading/why, /trading/metrics
│   │   ├── strategies.py  /strategies
│   │   ├── risk.py ────── /risk/killswitch/status
│   │   ├── risk_governor.py  /risk/governor/status
│   │   ├── physics.py ─── /physics/state
│   │   ├── brain.py ───── /brain/analysis
│   │   ├── council.py ─── /council/status
│   │   ├── archon.py ──── /archon/status
│   │   ├── analytics.py ─ /analytics/*
│   │   ├── market.py ──── /market/*
│   │   ├── storage.py ─── /storage/*
│   │   ├── telemetry.py ─ /telemetry/*
│   │   ├── graph_engine.py  /graph/*
│   │   ├── causal_inference.py  /causal/*
│   │   ├── meta_learning.py  /meta-learning/*
│   │   ├── multimodal_swarm.py  /swarm/*
│   │   ├── singularity.py  /singularity/*
│   │   ├── temporal.py ─── /temporal/*
│   │   ├── meta_brain.py ─ /meta-brain/*
│   │   ├── changelog.py ── /changelog/*
│   │   └── system.py ──── /system/*
│   ├── services/ ────────── Internal API services
│   │   ├── websocket_service.py  WebSocket event forwarding
│   │   ├── ws_manager.py ──────── ConnectionManager (topic pub/sub)
│   │   ├── trading_metrics.py ─── Trading metrics aggregation
│   │   ├── market_data_store.py ── In-memory market data cache
│   │   └── prometheus_metrics.py ─ Prometheus metrics updater
│   └── telemetry/ ──────── Telemetry service
│       └── service.py ──── Event recording + WebSocket publishing
│
└── [supporting modules]
    ├── indicators/ ─────── Technical indicators (TA-Lib wrapper)
    ├── hft/ ────────────── HFT utilities (shared memory, circuit breaker)
    ├── ml/ ─────────────── ML training utilities
    ├── ml_predictor/ ──── ML prediction utilities
    ├── funding_arbitrage/  Funding rate arbitrage
    ├── agent_generation/ ── Auto-improvement (ImprovementCatalyst)
    ├── ai/ ────────────── Agent generation utilities
    └── process_factory/ ── Experimental process generation
```

---

## 6. Docker Mode (Microservices)

```
┌───────────────────────────────────────────────────────┐
│                   Docker Network                       │
│                                                        │
│  ┌─────────┐   ┌───────────────┐   ┌──────────────┐  │
│  │  api     │   │ symbiont-     │   │  collector   │  │
│  │ :8000    │   │ testnet       │   │ (Bybit WS)   │  │
│  │ FastAPI  │   │ (trading)     │   └──────┬───────┘  │
│  └────┬─────┘   └───────┬──────┘          │          │
│       │                 │                  │          │
│       └─────────────────┼──────────────────┘          │
│                         ▼                              │
│              ┌──────────────────┐                      │
│              │     Redis        │  ◄── Single broker   │
│              │     :6379        │      for all events   │
│              │  Redis Streams   │                      │
│              └────────┬─────────┘                      │
│                       │                                │
│       ┌───────────────┼───────────────┐               │
│       │               │               │               │
│  ┌────▼────┐    ┌─────▼─────┐   ┌────▼─────┐        │
│  │ physics │    │   brain   │   │ risk-svc │        │
│  │(thermo- │    │ (Claude   │   │(Governor)│        │
│  │dynamics)│    │  analysis)│   │          │        │
│  └─────────┘    └───────────┘   └──────────┘        │
└───────────────────────────────────────────────────────┘
```

**ServiceEventBridge** (`src/hean/core/system/service_event_bridge.py`) translates:
- **Inbound** (Redis → EventBus): `physics:*`, `brain:analysis`, `oracle:signals`, `risk:policy_updates`
- **Outbound** (EventBus → Redis): `TICK`, `SIGNAL`, `ORDER_FILLED`, `POSITION_OPENED/CLOSED`, `KILLSWITCH_TRIGGERED`, `PNL_UPDATE`

In local dev mode (`make run`), all components run in a single process — no Redis needed.

---

## 7. Module Dependency Map

```
                    ┌──────────┐
                    │  config  │ ◄── All modules read settings
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │ EventBus │ ◄── Central nervous system
                    └────┬─────┘
                         │
    ┌────────────────────┼────────────────────────────────┐
    │          │         │         │         │             │
    ▼          ▼         ▼         ▼         ▼             ▼
Strategies → Risk → Execution → Portfolio  Physics     Brain
    │                    │                    │            │
    │                    ▼                    │            │
    │              Exchange/Bybit             │            │
    │              (http + ws)                │            │
    │                                        │            │
    └──────────────┬─────────────────────────┘            │
                   ▼                                      │
            Oracle Integration ◄──────────────────────────┘
            (4-source AI fusion)
                   │
                   ▼
              ARCHON (orchestrator)
              Council (AI council)
```

---

## 8. iOS App — Screen Structure

```
HEANApp (ios/HEAN/)
  └── ContentView (TabView)
        ├── Tab 1: Live ─── Equity, positions, trades (realtime)
        ├── Tab 2: Mind ─── AI insights, Brain analysis
        ├── Tab 3: Action ── Order management, strategy control
        ├── Tab 4: X-Ray ── Diagnostics, KillSwitch, risk levels
        └── Tab 5: Settings ── Configuration

  Additional Screens:
    ├── Genesis ────── 5-scene cinematic intro (Canvas + TimelineView)
    ├── Strategies ─── Strategy list & management
    ├── Signals ────── Signal feed
    ├── Brain ──────── Brain analysis details
    ├── Risk ───────── Risk dashboard
    ├── Markets ────── Market scanner
    ├── Anomalies ──── Anomaly alerts
    ├── Players ────── Participant classifier
    ├── GravityMap ─── Market gravity visualization
    ├── TemporalStack ─ Temporal market analysis
    └── AIAssistant ── AI chat interface

  Architecture:
    DIContainer (@EnvironmentObject)
      → APIClient (Actor-based HTTP)
        → APIEndpoints (centralized enum)
    WebSocketManager → real-time events
    Services.swift → consolidated service layer (COMPILED source of truth)
    DesignSystem/ → Theme, Components, Motion, Strings (bilingual RU/EN)
```

**Key iOS patterns:**
- Backend `snake_case` → iOS `camelCase`: always add `CodingKeys`
- `decodeIfPresent` with defaults for optional fields
- Custom `init(from:)` with fallback keys
- Case-insensitive enum decoding
- Xcode project: 78 build entries (77 Swift + 1 Assets)

---

## 9. Web Dashboard Structure

```
dashboard/ (Next.js 15 + React 19 + TypeScript)
  ├── src/
  │   ├── app/ ────────────── Next.js app directory
  │   ├── store/
  │   │   └── heanStore.ts ── Zustand global state
  │   ├── services/
  │   │   ├── api.ts ──────── REST client → :8000
  │   │   └── websocket.ts ── WS client → ws://:8000/ws
  │   ├── components/
  │   │   ├── tabs/
  │   │   │   ├── CockpitTab ── Equity, metrics, positions
  │   │   │   ├── NeuroMapTab ── Agent topology (graph visualization)
  │   │   │   ├── TacticalTab ── Order book, execution
  │   │   │   └── BlackBoxTab ── Hidden signals, debug
  │   │   ├── cockpit/
  │   │   │   ├── EquityChart ── Equity curve (Recharts)
  │   │   │   ├── PositionsSummary
  │   │   │   ├── LiveFeed ──── Real-time events
  │   │   │   └── MetricRow
  │   │   ├── neuromap/
  │   │   │   ├── NeuroMapView ── Agent network
  │   │   │   ├── PulsingNode ── Agent nodes (Framer Motion)
  │   │   │   └── AnimatedConnection
  │   │   └── ui/ ── GlassCard, MetricCard, AnimatedNumber, StatusBadge
  │   ├── types/ ─── TypeScript interfaces
  │   └── utils/ ─── Utility functions
  └── Tech: Tailwind 4, Framer Motion, Recharts
      Runs on port 3001
```

---

## 10. Event Types Reference

| Category | Events |
|----------|--------|
| **Market** | `TICK`, `FUNDING`, `FUNDING_UPDATE`, `ORDER_BOOK_UPDATE`, `REGIME_UPDATE`, `CANDLE`, `CONTEXT_UPDATE` |
| **Strategy** | `SIGNAL`, `STRATEGY_PARAMS_UPDATED` |
| **Risk** | `ORDER_REQUEST`, `RISK_BLOCKED`, `RISK_ALERT` |
| **Execution** | `ORDER_PLACED`, `ORDER_FILLED`, `ORDER_CANCELLED`, `ORDER_REJECTED` |
| **Portfolio** | `POSITION_OPENED`, `POSITION_CLOSED`, `POSITION_UPDATE`, `EQUITY_UPDATE`, `PNL_UPDATE`, `ORDER_DECISION`, `ORDER_EXIT_DECISION` |
| **System** | `STOP_TRADING`, `KILLSWITCH_TRIGGERED`, `KILLSWITCH_RESET`, `ERROR`, `STATUS`, `HEARTBEAT` |
| **Intelligence** | `BRAIN_ANALYSIS`, `CONTEXT_READY`, `PHYSICS_UPDATE`, `ORACLE_PREDICTION`, `CAUSAL_SIGNAL` |
| **Orchestration** | `ARCHON_DIRECTIVE`, `ARCHON_HEARTBEAT`, `SIGNAL_PIPELINE_UPDATE`, `RECONCILIATION_ALERT` |
| **Immune** | `IMMUNE_STATE_UPDATE`, `RECOVERY_PLAN_STARTED`, `RECOVERY_COMPLETED`, `PREDICTIVE_DANGER`, `CAPITAL_REALLOCATED` |
| **Council** | `COUNCIL_REVIEW`, `COUNCIL_RECOMMENDATION` |

`CONTEXT_UPDATE` carries a `type` field for sub-routing: `oracle_predictions`, `ollama_sentiment`, `finbert_sentiment`, `rl_risk_adjustment`, `regime_update`, `physics_state`.

---

## 11. API Endpoints (all under `/api/v1/`)

| Method | Endpoint | Response | Purpose |
|--------|----------|----------|---------|
| GET | `/engine/status` | `{status, running, equity, daily_pnl, initial_capital}` | System status |
| GET | `/orders/positions` | `[{position_id, symbol, side, qty, ...}]` | Open positions |
| GET | `/orders` | `[{order_id, symbol, side, qty, price, status, ...}]` | Open orders |
| GET | `/strategies` | `[{strategy_id, type, enabled}]` | Strategy list |
| GET | `/risk/governor/status` | `{risk_state, level, reason_codes, ...}` | Governor state |
| GET | `/risk/killswitch/status` | `{triggered, reasons, thresholds, ...}` | KillSwitch |
| GET | `/trading/why` | Complex diagnostic | Why no trading? |
| GET | `/trading/metrics` | `{counters: {last_1m, last_5m, session}}` | Metrics |
| GET | `/physics/state?symbol=X` | `{temperature, entropy, phase, ...}` | Market physics |
| GET | `/brain/analysis` | Latest AI analysis | Brain thoughts |
| GET | `/council/status` | Council decision status | Council review |
| POST | `/engine/start` | `{status: "started"}` | Start trading |
| POST | `/engine/stop` | `{status: "stopped"}` | Stop trading |
| WS | `/ws` | Stream events | Topics: `system_status`, `order_decisions`, `ai_catalyst` |

---

## 12. Configuration System

`HEANSettings` in `src/hean/config.py` — Pydantic BaseSettings, loads from `.env`:

| Category | Key Settings |
|----------|-------------|
| **Exchange** | `BYBIT_API_KEY`, `BYBIT_API_SECRET`, `BYBIT_TESTNET=true` |
| **Capital** | `INITIAL_CAPITAL=300`, `reinvest_rate=0.85` |
| **Risk (Iron Rules)** | `max_daily_drawdown_pct=15%`, `max_trade_risk_pct=1%`, `max_leverage=3x`, `killswitch_drawdown_pct=30%` |
| **Strategies** | `IMPULSE_ENGINE_ENABLED`, `FUNDING_HARVESTER_ENABLED`, ... (11 flags) |
| **Execution** | `maker_first=true`, `maker_ttl_ms=150`, `allow_taker_fallback=true` |
| **AI/ML** | `ANTHROPIC_API_KEY`, `BRAIN_ENABLED`, `OLLAMA_ENABLED`, `RL_RISK_ENABLED` |
| **Microservices** | `physics_source`, `brain_source`, `risk_source` (`local` or `redis`) |
| **API** | `api_auth_enabled`, `api_auth_key`, `jwt_secret` |
| **Safety** | `DRY_RUN=true` (default, blocks real orders), `LIVE_CONFIRM=YES` |

---

## 13. Key Architectural Principles

1. **Event-Driven** — All components communicate via EventBus, never direct calls
2. **Dual-Mode** — Same code runs as monolith (`make run`) or microservices (Docker + Redis Streams)
3. **Multi-Layer Protection** — Signal → 12-layer filter → RiskGovernor → KillSwitch → DepositProtector → ImmuneSystem
4. **Configuration-First** — All parameters in `HEANSettings`, no hardcoded values
5. **AI-Optional** — Brain, Council, Oracle all gated by config flags with graceful fallbacks
6. **Type-Safe** — mypy strict, Pydantic validation on all inputs
7. **DRY_RUN Default** — Blocks real order placement with RuntimeError unless explicitly disabled
8. **Observability Built-In** — Metrics, telemetry, health checks, self-healing, latency tracking
9. **Testnet Only** — All trading executes on Bybit Testnet with virtual funds

---

## 14. Package READMEs

| Package | README |
|---------|--------|
| ARCHON Orchestrator | [src/hean/archon/README.md](src/hean/archon/README.md) |
| Core Infrastructure | [src/hean/core/README.md](src/hean/core/README.md) |
| Trading Strategies | [src/hean/strategies/README.md](src/hean/strategies/README.md) |
| Risk Management | [src/hean/risk/README.md](src/hean/risk/README.md) |
| Order Execution | [src/hean/execution/README.md](src/hean/execution/README.md) |
| Exchange Connectors | [src/hean/exchange/README.md](src/hean/exchange/README.md) |
| Portfolio Accounting | [src/hean/portfolio/README.md](src/hean/portfolio/README.md) |
| Market Physics | [src/hean/physics/README.md](src/hean/physics/README.md) |
| Brain (LLM Analysis) | [src/hean/brain/README.md](src/hean/brain/README.md) |
| AI Council | [src/hean/council/README.md](src/hean/council/README.md) |
| Sentiment Analysis | [src/hean/sentiment/README.md](src/hean/sentiment/README.md) |
| Oracle Intelligence | [src/hean/core/intelligence/README.md](src/hean/core/intelligence/README.md) |
| Symbiont X Evolution | [src/hean/symbiont_x/README.md](src/hean/symbiont_x/README.md) |
| Observability | [src/hean/observability/README.md](src/hean/observability/README.md) |
| Storage | [src/hean/storage/README.md](src/hean/storage/README.md) |
| API Gateway | [src/hean/api/README.md](src/hean/api/README.md) |

---

## 15. Development Commands

```bash
make install              # pip install -e ".[dev]"
make test                 # pytest (~680 tests)
make test-quick           # pytest excluding Bybit connection tests
make lint                 # ruff check src/ && mypy src/
make run                  # python -m hean.main run (local, single process)
make dev                  # docker-compose with dev profile
make smoke                # ./scripts/smoke_test.sh
make docker-clean         # Clean containers and volumes
make prod-with-monitoring # Production + Prometheus/Grafana
```

```bash
# iOS build
xcodebuild -project ios/HEAN.xcodeproj -scheme HEAN \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -quiet build

# Dashboard
cd dashboard && npm install && npm run dev    # port 3001
cd dashboard && npm run build                 # production
```















  ┌────────────┬─────────────────────────────┬───────────────────────────────────────────────────────────────────┐
  │   Модуль   │            Путь             │                            Назначение                             │
  ├────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ brain      │ src/hean/brain/             │ AI-анализ рынка через Claude                                      │
  ├────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ physics    │ src/hean/physics/           │ Термодинамика рынка (температура, энтропия, фазы)                 │
  ├────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ oracle     │ src/hean/core/intelligence/ │ Гибридный 4-source signal fusion (TCN + FinBERT + Ollama + Brain) │
  ├────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ risk       │ src/hean/risk/              │ RiskGovernor, KillSwitch, PositionSizer, Kelly                    │
  ├────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ strategies │ src/hean/strategies/        │ 11 стратегий (ImpulseEngine, FundingHarvester и т.д.)             │
  ├────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ execution  │ src/hean/execution/         │ Роутер ордеров, reconciliation                                    │
  ├────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ exchange   │ src/hean/exchange/bybit/    │ HTTP + WebSocket клиенты Bybit                                    │
  ├────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ portfolio  │ src/hean/portfolio/         │ Учёт позиций, капитал, PnL                                        │
  ├────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ council    │ src/hean/council/           │ Мульти-агентный AI совет для ревью сделок                         │
  ├────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ sentiment  │ src/hean/sentiment/         │ FinBERT, Ollama, новости, Reddit, Twitter                         │
  ├────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ bus        │ src/hean/core/bus.py        │ EventBus — центральная нервная система                            │
  ├────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ storage    │ src/hean/storage/           │ DuckDB персистентность                                            │
  ├────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ symbiont_x │ src/hean/symbiont_x/        │ Генетическая эволюция стратегий                                   │
  ├────────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ api        │ src/hean/api/               │ FastAPI + WebSocket                                               │
  └────────────┴─────────────────────────────┴───────────────────────────────────────────────────────────────────┘

  А в Docker-микросервисах (services/):

  ┌───────────┬──────────────────────────────────────────────┐
  │  Сервис   │                     Путь                     │
  ├───────────┼──────────────────────────────────────────────┤
  │ collector │ services/collector/ — сбор данных с Bybit WS │
  ├───────────┼──────────────────────────────────────────────┤
  │ physics   │ services/physics/ — отдельный сервис физики  │
  ├───────────┼──────────────────────────────────────────────┤
  │ brain     │ services/brain/ — AI-сервис                  │
  ├───────────┼──────────────────────────────────────────────┤
  │ risk-svc  │ services/risk/ — сервис управления рисками   │
  ├───────────┼──────────────────────────────────────────────┤
  │ oracle    │ services/oracle/ — hybrid AI сигналы         │
  └───────────┴────────────────────────────────────────