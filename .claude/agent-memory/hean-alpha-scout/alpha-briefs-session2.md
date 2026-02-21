# Alpha Briefs — Session 2 (2026-02-22)

Topic: Revolutionary alpha sources for crypto — cross-domain, microstructure, on-chain.
Target: HIGH Sharpe (>2.0), LOW correlation with existing 11 strategies.

---

## Brief A: HawkesLiquidationPredictor

**Core Hypothesis:** Liquidations are self-exciting — each forced close
increases margin pressure on nearby leveraged positions, creating measurable
Hawkes-process clustering that predicts the NEXT liquidation burst 15-90
seconds in advance, enabling HEAN to either fade the exhaustion or ride the cascade.

**Sources:**
- arXiv:2502.04027 — Markov-modulated Hawkes process for burst detection
- arXiv:2312.16190 — Hawkes-based LOB forecasting
- SSRN:5611392 — Oct 2025 liquidation cascade anatomy ($19B OI erased in 36h)
- arXiv:2512.01112 — Autodeleveraging formal model

**Integration path:**
- New EventType: LIQUIDATION_BURST_SIGNAL
- New service: `services/hawkes/` consuming CoinGlass liquidation stream API
- Subscribes to PHYSICS_UPDATE (cascade anomaly already detected by anomaly_detector.py)
- Publishes to EventBus → strategies can subscribe

---

## Brief B: CrossAssetTransferEntropyRouter

**Core Hypothesis:** Bitcoin's information dominance over altcoins is
time-varying and measurable in real-time via transfer entropy; when BTC→ALT
transfer entropy spikes, alts predictably follow BTC's direction within
30-120 seconds, producing a systematic lead-lag strategy uncorrelated with
momentum or funding.

**Sources:**
- arXiv:2602.07046 — Transfer entropy shows BTC as dominant information driver
- CrossMarketImpulse already in physics engine (Pearson correlation + delay detection)
- Extends existing cross_market.py from Pearson to causal transfer entropy

**Integration path:**
- Extend `packages/hean-physics/src/hean/physics/cross_market.py`
- Add TransferEntropyCalculator class using sliding-window discretized TE
- New EventType: CAUSAL_SIGNAL already exists — populate with TE direction + lag_ms
- LiquiditySweep or new strategy subscribes to CAUSAL_SIGNAL with TE filter

---

## Brief C: LOBShapeContrastiveFingerprintStrategy

**Core Hypothesis:** Order book "shapes" (the 2D depth profile across
10-20 price levels) cluster into a small set of recurring fingerprint types
(accumulation wall, vacuum, iceberg, squeeze), each with statistically
distinct 5-minute forward return distributions; a contrastive-trained
encoder can identify these in real-time and generate directional signals
with Sharpe >2.5 in cross-asset backtest.

**Sources:**
- arXiv:2602.00776 — Stable cross-asset microstructure patterns (BTC/LTC/ETC)
- arXiv:2506.05764 — 100ms LOB snapshots from Bybit, depth imbalance predicts moves
- arXiv:2407.18645 — Contrastive learning of asset embeddings from time series
- arXiv:2403.09267 — Deep LOB microstructural guide (LOBFrame)

**Integration path:**
- New service: `services/lob-fingerprint/`
- Train contrastive model offline on Bybit historical LOB data (available via API)
- Publishes ORACLE_SIGNAL_V1 with fingerprint_type + direction + confidence
- OracleIntegration adds as 5th fusion source (new weight slot)

---

## Brief D: OnChainSOPRMeanReversionStrategy

**Core Hypothesis:** Bitcoin's Short-Term Holder SOPR (ratio of realized
price to paid price for coins moved within 155 days) reliably marks local
exhaustion points: SOPR spikes above 1.02 indicate panic profit-taking
momentum that mean-reverts within 4-12 hours; SOPR dips below 0.98
indicate capitulation bottoms. These on-chain signals are structurally
uncorrelated with orderbook microstructure signals.

**Sources:**
- Glassnode SOPR data (API available, free tier: 1h resolution)
- Coinmetrics SOPR (free alternative, 1h granularity)
- Academic basis: realized price / cost basis distribution theory (Lopez de Prado)

**Integration path:**
- New client: `packages/hean-intelligence/src/hean/intelligence/onchain_client.py`
- Poll Glassnode or Coinmetrics API every hour (free tier sufficient)
- Publish CONTEXT_UPDATE with type="onchain_sopr"
- SentimentStrategy or new OnChainStrategy subscribes + generates signals

---

## Brief E: GammaExposureGravityStrategy

**Core Hypothesis:** Aggregated options dealer gamma exposure at specific
strike prices creates predictable "gravitational" forces on perpetual
futures prices — strikes with large net negative dealer gamma act as
attractors (price is pulled toward them before expiry), while large
positive gamma strikes create repellers. This can predict 4-24h price
range with abnormal accuracy.

**Sources:**
- MenthorQ GEX (Gamma Exposure) model for BTC options
- Deribit options data (REST API, free, real-time OI per strike)
- CME 2025 article on BTC basis trading + gamma effects
- arXiv:2506.14614 — Pricing options on cryptocurrency futures

**Integration path:**
- New service: `services/gamma/`
- Fetches Deribit options chain every 15 minutes
- Computes net dealer GEX per strike (assumes dealers short calls, long puts)
- Identifies GEX gravity zones; publishes CONTEXT_UPDATE type="gamma_gravity"
- New GammaGravityStrategy: fade moves away from gravity zones, ride approaches

---

## Session 2 Honorable Mentions

- **DEX Liquidity Migration Alpha**: Track Uniswap/Curve liquidity pool TVL
  shifts as leading indicator of spot demand shifts. High implementation cost.
- **Miner Hash Rate Proxy**: BTC difficulty adjustment timing predicts
  miner capitulation / accumulation. 2-week lag — too slow for HEAN.
- **Funding Rate DAR Predictability** (SSRN:5576424): Out-of-sample
  funding rate prediction via double-autoregressive models. FundingHarvester
  already captures this partially; marginal improvement only.
- **SIR Narrative Model**: Epidemiological model for meme/narrative spread.
  EmotionArbitrage already covers the wave-based analog. Interesting but
  would be redundant with existing emotion_arbitrage.py.
