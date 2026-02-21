# HEAN Alpha Scout — Persistent Memory

## Architecture Quick Reference
- EventBus: `backend/packages/hean-core/src/hean/core/bus.py` — 3 priority queues (CRITICAL/NORMAL/LOW), circuit breaker at 95% utilization, BusHealthStatus struct with queue_utilization_pct + drop_rate_pct
- BusCanary: `backend/packages/hean-core/src/hean/core/bus_canary.py` — heartbeat RTT probe, P99 window of 200 samples, exposes stalls_detected + p99_rtt_ms
- DLQ: `backend/packages/hean-core/src/hean/core/bus_dlq.py` — in-memory, CRITICAL_TYPES frozenset, retry_all() method
- RiskSentinel: `backend/packages/hean-risk/src/hean/risk/risk_sentinel.py` — pre-computes RiskEnvelope, 8 risk components
- OrderManager: `backend/packages/hean-execution/src/hean/execution/order_manager.py` — pre_register_placement() / consume_pending_placement(), 30s stale timeout
- PositionReconciler: `backend/packages/hean-execution/src/hean/execution/position_reconciliation.py` — periodic reconciliation, ReconciliationResult with missing_locally/missing_on_exchange
- EventType.RISK_ENVELOPE is now CRITICAL priority (fix applied)
- Signal has `metadata: dict` field — can embed composite scoring without schema changes
- OFI_UPDATE and CAUSAL_SIGNAL event types already exist in types.py

## Alpha Ideas Explored (Session 1 — 2026-02-21)
See: [alpha-briefs-session1.md](alpha-briefs-session1.md)

1. BusLoadAlpha — Bus utilization / canary RTT as volatility pre-signal. Novel; maps to VPIN microstructure research.
2. ConvergenceScoreAlpha — Count of simultaneously firing strategies as confirmation signal. Supported by ensemble ML literature.
3. RetryDeltaAlpha — Price drift between DLQ retry and original signal submission. Mean-reversion angle supported by arrival price TCA research.
4. PhantomFillArbitrage — Analyze reconciliation-discovered positions as natural experiments. Empirical, no strong academic precedent.
5. EnvelopeAccuracyAlpha — Tighter risk envelope (from dedup fix) enables better Kelly sizing = compounding capital efficiency edge.

## Key Academic References Found
- VPIN (Easley, Lopez de Prado, O'Hara): volume-synchronized order flow toxicity, predicts price jumps — directly applicable to BusLoadAlpha
- Queue Imbalance paper (Gould & Bonart, arXiv:1512.03492): queue depth predicts one-tick-ahead price direction
- Ensemble methods in FinRL (arXiv:2501.10709): model agreement boosts Sharpe — supports ConvergenceScoreAlpha
- TCA arrival price benchmarks (Talos 2024): ~13 bps average slippage from arrival price — sets floor for RetryDeltaAlpha edge
- Perpetual futures pricing (arXiv:2310.11771v2): random-maturity arbitrage, funding rate dynamics
- Deep LOB forecasting microstructural guide (Tandf, 2025): LOBFrame framework, microstructure features for prediction

## Alpha Ideas Explored (Session 2 — 2026-02-22)
See: [alpha-briefs-session2.md](alpha-briefs-session2.md)

A. HawkesLiquidationPredictor — self-exciting Hawkes process on liquidation events, predicts cascade bursts 15-90s ahead.
B. CrossAssetTransferEntropyRouter — real-time causal TE from BTC→ALT, extends existing cross_market.py. CAUSAL_SIGNAL EventType already exists.
C. LOBShapeContrastiveFingerprintStrategy — contrastive encoder on 10-20 level order book shapes. New service lob-fingerprint/, publishes as 5th Oracle source.
D. OnChainSOPRMeanReversionStrategy — STH-SOPR extremes (>1.02 / <0.98) as mean-reversion signals. Glassnode/Coinmetrics free API, 1h resolution.
E. GammaExposureGravityStrategy — Deribit options dealer GEX maps strike-level price attractors/repellers. New gamma service, 15-min polling.

## Key Academic References (Session 2)
- arXiv:2502.04027 — Markov-modulated Hawkes process for burst detection in crypto
- arXiv:2312.16190 — Hawkes-based LOB forecasting (self-excitation in order placement)
- arXiv:2512.01112 — Autodeleveraging formal model (Oct 2025 $2.1B in 12 min episode)
- SSRN:5611392 — Oct 2025 cascade anatomy: $19B OI erased in 36h
- arXiv:2602.00776 — Stable cross-asset LOB microstructure patterns (Sharpe validated on Bybit)
- arXiv:2506.05764 — 100ms Bybit LOB snapshots: depth imbalance predicts short-horizon moves
- arXiv:2407.18645 — Contrastive learning of asset embeddings from financial time series
- arXiv:2403.09267 — Deep LOB forecasting microstructural guide (LOBFrame, 2025)
- arXiv:2602.07046 — Transfer entropy: BTC dominant information driver in crypto-equity
- SSRN:5576424 — Out-of-sample funding rate predictability (DAR models)
- arXiv:2506.14614 — Pricing options on crypto futures (Black-Scholes failures)
- Coinmetrics / Glassnode: SOPR, NUPL, Realized Price — free API tier at 1h resolution

## Architecture Notes (Session 2 discoveries)
- cross_market.py already computes Pearson correlation + propagation delay — TE is a direct extension
- anomaly_detector.py already detects LIQUIDATION_CASCADE — Hawkes adds predictive pre-signal
- OFI_UPDATE EventType already exists (types.py line 68) — can carry TE-weighted OFI
- CAUSAL_SIGNAL EventType already exists (types.py line 69) — ready for TE output
- EmotionArbitrage covers news wave timing — SIR model would be redundant
- Deribit REST API is free and real-time; Glassnode free tier is 1h resolution
- FundingHarvester already partially captures DAR-predicted funding rate mean reversion

## Known Dead Ends
- "System CPU load as volatility proxy" — too indirect; bus RTT is a better HEAN-native signal
- Pure "phantom fill profit" study — too few samples on testnet to reach statistical significance quickly
- Miner hash rate proxy for BTC — 2-week difficulty adjustment lag, too slow for HEAN
- Pure SIR narrative model — EmotionArbitrage.py already covers the wave-based analog
- Funding rate DAR predictability as standalone strategy — FundingHarvester overlaps substantially
