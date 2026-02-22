# Alpha Briefs — Session 3 (2026-02-22)
## Focus: Microstructure-Specific Alpha for Bybit Testnet with $500 Capital

---

## Alpha Brief 1: FundingRatePressurePredictor

**Core Hypothesis**: The *rate of change* of open interest combined with the *direction* of recent price action predicts the sign and magnitude of the NEXT funding rate settlement 30-90 minutes before it occurs, with enough precision to enter and exit with better timing than FundingHarvester's current EWMA-only model.

**What FundingHarvester already does** (per audit):
- EWMA of historical funding rates
- Momentum of recent 3-period vs older 3-period rates
- Time-of-day cyclical features (sin/cos hour)
- Trend persistence count
- Cooldown: 4 hours per symbol, max 6 signals/day

**What it does NOT do**:
- Use OI derivative (rate of OI change, not just current OI)
- Use long/short ratio divergence from OI direction
- Use funding rate cross-market basis (Bybit vs Binance) as mean-reversion anchor
- Use price premium/discount to spot as predictor (perp premium = perp price minus spot price)
- Model the "funding clamp" behavior: Bybit caps funding at ±0.75% per 8h, so extreme rates have higher reversion probability

**Inspiration/Source**:
- SSRN:5576424 — DAR (Distributional Autoregressive) models for funding rate prediction outperform ARMA on crypto out-of-sample
- Bybit API: `/v5/market/funding/history`, `/v5/market/open-interest`, CoinGlass L/S ratio (already wired in)
- Perpetual basis = (perp_price - spot_price) / spot_price — if basis > funding_rate, longs are overpaying and rate will fall

**Why It Might Work**:
The funding rate in perpetual futures is a market-clearing mechanism. When OI is rising AND price is trending up AND the long/short ratio is >0.65, the market is accumulating levered longs who will collectively pay the next funding. The pressure is visible 1-2 hours before settlement. The perp-spot premium is the most direct leading indicator because the funding formula uses it. The CoinGlass collector is already wired but only uses 10min-old OI snapshots; adding 1-minute OI polling from Bybit REST and computing dOI/dt gives a much sharper signal.

**Data Requirements**:
- Bybit `/v5/market/open-interest` — 1-minute polling, free, no key required
- Bybit `/v5/market/fundingRate` — already fetched
- Binance `/fapi/v1/fundingRate` — for cross-exchange basis anchor (free)
- CoinGlass long/short ratio — already wired (requires API key for real data)
- Bybit spot price (BTCUSDT on spot) — available from public WS

**Proposed Methodology**:
1. Every 60 seconds: fetch Bybit OI, compute dOI/dt over 5-min and 15-min windows
2. Compute perp-spot premium: (perp_price - spot_price) / spot_price
3. Fetch cross-exchange funding basis: Bybit_rate - Binance_rate
4. Feature vector: [dOI_5m, dOI_15m, perp_premium, LS_ratio, cross_basis, hour_sin, hour_cos, persistence]
5. Train a lightweight Ridge regression or GBM on 30 days of historical data (Bybit REST history goes back 200 periods)
6. Signal: if predicted_rate > current_rate + 0.0001 (longs will pay more), go SHORT 90 min before settlement; exit 5 min after settlement
7. Size: 2-3% of capital per trade (Kelly fraction given 60-65% win rate expected)

**HEAN Integration Path**:
- New module: `packages/hean-strategies/src/hean/strategies/funding_rate_predictor.py`
- Inherits `BaseStrategy`, subscribes to `FUNDING`, `TICK`, `PHYSICS_UPDATE`
- New background task in strategy `__init__` to poll OI every 60s via http_client
- Publishes to existing `FUNDING_UPDATE` event or adds `FUNDING_PREDICTION` to types.py
- Can augment FundingHarvester's `predict_next_funding_ml()` by passing features via Signal.metadata
- Cross-exchange basis calculation lives in a new `services/funding_predictor/` microservice or inline

**Potential Risks & Challenges**:
- Bybit testnet may have different funding dynamics than mainnet (fewer participants means noisier rates)
- Perp-spot premium requires Bybit SPOT data — need to verify testnet has spot market for BTC
- Binance cross-basis uses mainnet data (testnet Binance is not always reliable)
- Ridge regression trained on 30 days is brittle to regime shifts — needs rolling retrain
- CoinGlass free tier has rate limits; mock data (already in code) will reduce signal quality

**Estimated Daily PnL with $500 Capital**:
- 1-2 trades/day (one per 8h funding cycle, select only high-confidence)
- Position size: $25-50 per trade (5-10% capital, 2x leverage)
- Expected edge per trade: 0.08-0.15% from better timing (enter 90min before vs 2h before)
- Daily estimate: $1-3 incremental over what FundingHarvester already captures
- Compounding: the real edge is tighter timing allows larger leverage with same risk budget

**Estimated Alpha Decay**: 12-24 months. This is structural: the funding mechanism is not going away, and the perp-spot premium is a mechanical relationship. The edge decays only if competitors with faster OI polling crowd out the 90-minute window.

**First Step**: Pull 200 historical funding periods from Bybit REST, correlate perp-spot premium at t-90min with actual funding rate at t. If R² > 0.25, the signal is real. Should take 2-3 hours of data work.

---

## Alpha Brief 2: OFI-VPIN Adverse Selection Gate

**Core Hypothesis**: The existing ImpulseEngine kills 97% of signals through 12 filter layers. Most of those kills are correct, but the ordering and structure of filters creates false positives (letting through bad trades) AND false negatives (blocking good trades). By replacing/augmenting the filter cascade with a real-time Order Flow Imbalance (OFI) + VPIN composite score, we can reduce false negatives by 15-20% without increasing false positives, directly translating to more executed trades at the same (or better) win rate.

**What exists**:
- consensus_swarm.py: VPIN computed as `ofi_result.imbalance_strength` — this is a simplified proxy, not true VPIN
- OFI_UPDATE EventType exists in types.py (line 68) but appears unused by strategies directly
- ImpulseEngine: 12 filter layers including spread, volatility expansion, volume, OFI, etc.
- participant_classifier.py: classifies MM vs institutional vs arb vs retail

**The gap**: True VPIN (Easley, Lopez de Prado, O'Hara 2012) uses VOLUME-SYNCHRONIZED buckets, not time-synchronized. The distinction matters: time-based OFI measures activity, volume-based VPIN measures toxicity. High VPIN = informed traders dominating = adverse selection risk for market makers, but DIRECTIONAL opportunity for position takers.

**Inspiration/Source**:
- Easley, Lopez de Prado, O'Hara (2012) "Flow Toxicity and Liquidity in a High Frequency World" — VPIN predicts Flash Crash 77 minutes ahead
- arXiv:1512.03492 (Gould & Bonart) — queue imbalance predicts one-tick-ahead direction with 52-54% accuracy on HFT timescales
- arXiv:2506.05764 — 100ms Bybit LOB snapshots: depth imbalance at levels 1-5 predicts 10-second returns

**Why It Might Work**:
When informed traders (institutions, whales) are active, they leave a fingerprint in the order book: the buy volume in each fixed-dollar-amount "bucket" diverges from sell volume. This imbalance, computed per bucket, gives VPIN. A high VPIN reading means: (a) an informed move is in progress, (b) market makers are withdrawing liquidity, (c) the next few minutes will be directional. For ImpulseEngine, substituting the filter "did volume spike?" with "is VPIN > 0.7?" would improve precision because VPIN filters out noise-driven volume spikes.

**Data Requirements**:
- Bybit public trade stream: tick-by-tick trades with side (buyer-initiated vs seller-initiated) — already received via ws_public.py
- ORDER_BOOK_UPDATE events — already subscribed by PhysicsEngine
- No external API required — pure computation on existing data streams

**Proposed Methodology**:
1. Create `VPINCalculator` class: maintain a rolling window of volume buckets (bucket_size = $50K notional for BTC)
2. For each bucket: `V_buy` = buyer-initiated volume, `V_sell` = seller-initiated volume
3. VPIN = rolling mean of |V_buy - V_sell| / (V_buy + V_sell) over last 50 buckets
4. Companion: Compute Kyle's Lambda = price_impact per unit of signed volume: Δprice / |net_order_flow|
5. Gate logic: if VPIN > 0.65 AND lambda_z_score > 1.5 → INFORMED_FLOW detected
6. In ImpulseEngine: add as filter layer 13 (INFORMED_FLOW_GATE); allow trades only when VPIN > 0.65 (momentum direction confirmed) OR VPIN < 0.35 (low toxicity, good for mean reversion trades)
7. For strategies expecting directional moves: use VPIN as a SIZE MULTIPLIER (high VPIN = larger size on momentum trades)

**HEAN Integration Path**:
- New module: `packages/hean-physics/src/hean/physics/vpin_calculator.py`
- PhysicsEngine subscribes and computes VPIN every 50 ticks, publishes via OFI_UPDATE event (already exists)
- ImpulseEngine subscribes to OFI_UPDATE, caches VPIN score
- New filter class `VPINFilter(BaseFilter)` in impulse_filters.py — single `allow()` method
- HFScalpingStrategy: use VPIN < 0.4 as confirmation (low toxicity = good for scalping)

**Potential Risks & Challenges**:
- Bybit testnet volume is much lower than mainnet → bucket sizes need tuning (try $5K not $50K)
- Trade side classification: Bybit marks taker side, not necessarily informed side — need to cross-reference with order book movement
- VPIN has a known lag on very short timescales; the 50-bucket window is ~5-20 minutes at testnet volumes
- Kyle's Lambda is noisy when computed on individual assets — need at least 200 ticks to stabilize

**Estimated Daily PnL with $500 Capital**:
- Indirect: reduces false negatives in ImpulseEngine by ~15%
- If ImpulseEngine currently generates 2-3 profitable trades/day, VPIN gate could add 0.5-1 more per day
- At current win rate and $1-3 profit per trade: +$0.50-$2.00/day incremental
- The bigger edge: VPIN as size multiplier on high-confidence moves could increase avg win by 20-30%

**Estimated Alpha Decay**: 24-36 months. VPIN is public knowledge (2012 paper) but computationally non-trivial and rarely implemented correctly. On testnet the edge is stable longer because there is very little competing HF flow.

**First Step**: Add trade-side tracking to the existing TICK handler: count buyer_volume vs seller_volume in rolling $5K buckets for BTCUSDT. Log VPIN every 100 trades for 24 hours. Then compute correlation between VPIN and 5-minute subsequent price change. If correlation > 0.15, proceed with integration.

---

## Alpha Brief 3: Liquidation Heatmap Gravity Strategy

**Core Hypothesis**: Open interest concentrated at specific price levels (visible via Bybit's liquidation price distribution endpoint) creates predictable "gravity wells" — price will be magnetically drawn to these clusters because market makers front-run the forced liquidations, and once triggered, the cascade creates momentum that overshoots before reversing. A strategy that enters WITH the move 30-60 seconds before the cluster is reached, and exits 60-90 seconds after the cluster is swept, captures both the momentum and the reversal.

**What exists**:
- anomaly_detector.py: detects LIQUIDATION_CASCADE post-hoc from 5 consecutive directional high-volume trades
- CoinGlass collector: fetches 24h historical liquidation totals — no per-level resolution
- The existing detector is REACTIVE; this idea is PREDICTIVE

**The gap**: Bybit provides a `/v5/market/risk-limit` endpoint and `/v5/market/delivery-price` but NOT direct liquidation heatmaps. However, CoinGlass Pro API has a "Liquidation Heatmap" endpoint (`/api/pro/v1/futures/liquidation_heatmap`) that shows price levels where leveraged positions will be liquidated. The key insight: OI clustering at price X means: when price reaches X, a cascade happens, price spikes through X+ε, then reverses to X-δ.

**Inspiration/Source**:
- arXiv:2512.01112 — Formal model of autodeleveraging cascade dynamics, shows price overshoot of 0.3-0.8% beyond initial cluster
- SSRN:5611392 — Oct 2025 cascade anatomy: cascade initiates at cluster, completes 12 minutes later
- CoinGlass Liquidation Heatmap API — provides $price_level → $OI_at_risk mapping, updated every 15 minutes
- Empirical observation: BTC liquidation clusters at round numbers ($1000, $500 intervals) due to retail leverage habits

**Why It Might Work**:
Market makers with access to OI-at-risk data will begin accumulating positions in advance of a large cluster because the liquidation cascade is a guaranteed source of one-sided flow. When $500M of long liquidations are clustered at $85,000 BTC, and price is at $85,500, any selling pressure that breaks $85,000 will trigger a cascade of forced selling → price drops to $84,500 (0.6% overshoot). The pattern is mechanical and repeatable. The edge is not in knowing WHERE the cluster is (CoinGlass shows this publicly) but in entering BEFORE the cascade with high confidence that the cluster will be hit.

**Data Requirements**:
- CoinGlass Pro API: Liquidation Heatmap — requires Pro API key (~$100/month) or reverse-engineering the public heatmap visual
- Alternative (FREE): Bybit open interest by price level — NOT directly available, but can be APPROXIMATED by tracking the price levels where OI spikes when price approaches them
- Bybit `/v5/market/open-interest?intervalTime=5min` — free, no key
- Existing LIQUIDATION_CASCADE anomaly events from anomaly_detector.py — to calibrate cascade dynamics

**Proposed Methodology**:
1. Build a synthetic liquidation heatmap: for each symbol, track the distribution of entry prices of recent trades with high volume (proxy for leveraged entries). Cluster by price level. This approximates where leveraged positions were opened.
2. When price approaches within 0.5% of a high-OI cluster:
   - If price has been falling (cluster is BELOW current price = long liquidation cluster): prepare SHORT entry
   - If price has been rising (cluster is ABOVE current price = short liquidation cluster): prepare LONG entry
3. Entry trigger: price breaks through the cluster by 0.05% (first confirmation of cascade start)
4. Target: cluster ± 0.3% (the expected overshoot, calibrated from anomaly_detector cascade data)
5. Stop: cluster ∓ 0.15% (if cascade reverses immediately, exit)
6. Exit: when LIQUIDATION_CASCADE anomaly deactivates (TTL=300s) or target reached

**HEAN Integration Path**:
- New strategy: `packages/hean-strategies/src/hean/strategies/liquidation_gravity.py`
- Inherits BaseStrategy, subscribes to `TICK`, `PHYSICS_UPDATE` (for LIQUIDATION_CASCADE anomaly), `ORDER_BOOK_UPDATE`
- New background task to build synthetic heatmap from trade history (already available via TICK stream)
- Listen to `PHYSICS_UPDATE` for existing `LIQUIDATION_CASCADE` anomaly flag → use as entry trigger
- Existing OFI_UPDATE from ConsensusSwarm can confirm cascade direction
- No new EventTypes needed

**Potential Risks & Challenges**:
- On testnet with low volume, liquidation clusters are thin — may get false triggers
- CoinGlass Pro API cost ($100/month) may not be justified for testnet; synthetic heatmap is a free but noisier approximation
- Cascade timing is highly variable: 12 minutes (mainnet) vs unknown (testnet)
- The entry-after-first-break approach has significant slippage risk in fast-moving cascades
- Over-optimization risk: clusters shift as positions are opened/closed

**Estimated Daily PnL with $500 Capital**:
- Frequency: 1-3 cascade events per day on BTC/ETH
- Win rate: 55-65% (cascade direction is predictable, timing is not)
- Position: 5-10% capital (2x leverage) = $50-100 at risk
- Avg gain: 0.3% per trade = $0.15-0.30 per $50 position
- Daily estimate: $0.90-$2.70 from 3 trades
- Higher frequency on volatile days: up to $5-8

**Estimated Alpha Decay**: 6-12 months. Liquidation cascade trading is becoming increasingly crowded on mainnet as retail "liq hunter" bots proliferate. On testnet, no competition for months.

**First Step**: Mine the last 7 days of TICK events from DuckDB storage. Identify all LIQUIDATION_CASCADE anomaly events. For each event: record (1) price at cascade detection, (2) price peak/trough within 5 minutes, (3) price 10 minutes after cascade ends. Calculate average overshoot and reversal. If overshoot > 0.2% and reversal > 0.1%, the pattern is tradeable.

---

## Alpha Brief 4: Regime-Adaptive Strategy Portfolio Rotator

**Core Hypothesis**: The current system runs ALL enabled strategies simultaneously, creating internal conflicts (opposing signals on the same symbol). A meta-layer that EXCLUSIVELY activates the highest-expected-value strategy set for the detected regime — and SILENCES all others — eliminates self-cancellation and concentrates capital on the strategies with the best regime fit.

**What exists** (per audit):
- Each strategy has `_allowed_regimes` set (e.g., HFScalping allows RANGE+NORMAL; ImpulseEngine allows IMPULSE)
- RiskGovernor reduces position size in RANGE by 30%
- PhysicsEngine emits REGIME_UPDATE events
- CorrelationArb and other strategies individually gate on regime
- BUT: multiple strategies remain active in the same regime and can generate conflicting signals

**The gap**: There is no EXCLUSIVE activation logic. In RANGE regime, HFScalping, FundingHarvester, EnhancedGrid, CorrelationArb, RebateFarmer, and InventoryNeutralMM can all be active simultaneously. They don't coordinate. The 97% signal kill rate from ImpulseEngine's filters is PARTLY caused by this: the risk system kills one side of conflicting signals.

**Why It Might Work**:
This is the PORTFOLIO CONSTRUCTION problem applied to strategies rather than assets. Academic finance (Brinson 1991, Grinold 1999) shows that combining uncorrelated strategies improves Sharpe only when they are genuinely uncorrelated — if they are trading the same instruments in the same regime, they add noise, not signal. By activating only the REGIME-OPTIMAL strategy set, capital is concentrated, signal-to-noise rises, and the 97% kill rate falls to perhaps 80%.

**Inspiration/Source**:
- Grinold & Kahn "Active Portfolio Management": IC (Information Coefficient) is regime-specific; strategy selection should maximize regime-conditional IC
- arXiv:2501.10709 (FinRL ensemble): regime-conditional strategy switching outperforms static ensembles
- Lopez de Prado "Advances in Financial Machine Learning" Chapter 15: strategy rotation based on HMM regime detection

**Regime-Strategy Mapping** (proposed):
```
IMPULSE/markup:     [ImpulseEngine, MomentumTrader] only — momentum is king
RANGE/accumulation: [FundingHarvester, RebateFarmer, EnhancedGrid] only — harvest static alpha
DISTRIBUTION:       [LiquiditySweep, CorrelationArb] only — fade exhausted moves
MARKDOWN:           [ImpulseEngine(short only), LiquidationGravity] — ride the down
HIGH_ENTROPY:       [InventoryNeutralMM] only — maximum spread capture, no directional bets
```

**Data Requirements**:
- Existing REGIME_UPDATE events from PhysicsEngine — already emitted
- Existing PHYSICS_UPDATE events with phase detection (accumulation/markup/distribution/markdown)
- Strategy performance metrics per regime — need to build a rolling tracker (regime → [win_rate, avg_pnl])

**Proposed Methodology**:
1. New `StrategyRotator` class in `packages/hean-strategies/src/hean/strategies/manager.py` (extend existing StrategyManager)
2. Subscribe to REGIME_UPDATE and PHYSICS_UPDATE events
3. Maintain per-strategy, per-regime performance statistics in a rolling window (last 50 trades per regime)
4. On regime change: compute expected_value[strategy][regime] = win_rate × avg_win - loss_rate × avg_loss
5. Activate strategies where expected_value > threshold (e.g., > $0.50 per trade in that regime)
6. Silence strategies where expected_value < threshold via strategy.enabled = False or by ignoring their signals in a new MetaGate
7. Capital reallocation: when fewer strategies are active, allow larger position sizes per active strategy

**HEAN Integration Path**:
- Extend `packages/hean-strategies/src/hean/strategies/manager.py` with `StrategyRotator` mixin
- New `RegimePerformanceTracker` dataclass tracking per-strategy, per-regime rolling stats
- Hook into ORDER_FILLED events to track actual realized PnL per strategy per regime
- Use existing REGIME_UPDATE → call `rotator.on_regime_change(new_regime, new_phase)`
- RiskSentinel's `_active_strategy_ids` list is already the mechanism for enabling/disabling strategies (lines 99-107 of risk_sentinel.py)

**Potential Risks & Challenges**:
- Cold start: needs 50+ trades per regime to have reliable statistics — could take weeks on testnet
- Regime mis-classification by PhysicsEngine → wrong strategy set activated → period of poor performance
- Chicken-and-egg: can't evaluate strategies that are silenced → use epsilon-exploration (10% time, activate a random secondary strategy)
- Rapid regime oscillation (e.g., bouncing between RANGE/IMPULSE every few minutes) causes constant switching overhead
- Coordination with existing 12-filter ImpulseEngine: those filters are strategy-internal and not affected by the rotator

**Estimated Daily PnL with $500 Capital**:
- Does not directly generate new trades — amplifies existing strategies
- By concentrating capital: if each active strategy currently uses 3-5% of capital, exclusive activation allows 8-15% per trade
- If win rate holds at ~55%: doubling position size doubles PnL from winning trades
- Realistic estimate: +$3-7/day incremental on current $5-15/day baseline
- The bigger long-term gain: compounding at higher capital utilization

**Estimated Alpha Decay**: This is a structural improvement, not a market inefficiency — it does not decay. The regime-strategy mapping may need retuning as crypto market structure evolves, but the concept is durable.

**First Step**: Query DuckDB for all filled orders in the last 30 days. Group by (strategy_id, detected_regime at time of entry). Compute win_rate and avg_pnl per group. Build a simple matrix. If any strategy has win_rate > 60% in one regime and < 40% in another, the rotator has evidence to work with.

---

## Alpha Brief 5: Micro-Mean-Reversion at 1-5 Second Timescales (Bid-Ask Bounce Harvesting)

**Core Hypothesis**: In liquid crypto perpetual markets, individual large market orders (>5x median size) temporarily displace the mid-price by 1-3 basis points before market makers replenish liquidity. This 1-3 bps move, which occurs within 1-5 seconds of the large order, is predictable, repeatable, and large enough to be profitable at 5-10x leverage for positions held < 5 seconds.

**What exists**:
- HFScalpingStrategy: 30-second cooldown, targets 25 bps moves over 5-tick window
- ParticipantClassifier: already identifies WHALE trades (>20x median)
- AnomalyDetector: WHALE_INFLOW anomaly triggers at 10x median volume

**The gap**: HFScalping operates at the 5-tick (multiple seconds) timescale targeting 25bps moves. This idea operates at the 1-5 SECOND timescale targeting 1-3bps moves with leverage. The mechanics are completely different: this exploits the TEMPORARY PRICE IMPACT of a single large trade, not a momentum move. The closest analogy is the bid-ask bounce captured by market makers.

**Inspiration/Source**:
- Glosten & Harris (1988) "Estimating the Components of the Bid-Ask Spread" — proves that large trades have temporary price impact that reverts within seconds
- Kyle (1985) "Continuous Auctions and Insider Trading" — lambda (price impact per unit of signed volume) determines reversion speed
- arXiv:2506.05764 — Bybit LOB snapshots: at 100ms resolution, depth imbalance has 52% directional accuracy on 10-second horizon
- Hasbrouck (1991) "Measuring the Information Content of Stock Trades" — temporary vs permanent price impact model

**Why It Might Work**:
When a whale market-buys $500K of BTC in a single order, it consumes all ask liquidity from best ask to +3bps above mid. The mid-price jumps 3bps. Market makers immediately place new ask orders at the new level, but simultaneously, other market participants post limit sell orders at the impact price (they see the price as temporarily elevated). Within 1-5 seconds, the price reverts 1-2bps toward the pre-impact mid. The trade: go SHORT within 500ms of detecting a WHALE_INFLOW_BUY, target +1.5bps reversion, stop at -1bps.

**Crypto-specific advantage**: Unlike equities, crypto perpetuals settle continuously with no fixed tick sizes. The minimum meaningful price move on BTC is ~$0.10 on a $90,000 price → 0.0001bps. This means the 3bps temporary impact is enormous relative to the minimum move. Also, Bybit Testnet has thinner liquidity than mainnet, so large orders have MORE temporary impact, making this MORE reliable on testnet.

**Data Requirements**:
- Bybit trade stream with side (already available via ws_public.py): need buyer/seller initiation flag
- Bybit LOB stream at 200ms update frequency (Bybit provides level-1 and level-25 book): need best-bid/ask + sizes
- ORDER_BOOK_UPDATE already subscribed by PhysicsEngine
- No external APIs needed

**Proposed Methodology**:
1. Extend AnomalyDetector's WHALE_INFLOW detection to also record the DIRECTION of the whale trade (buy vs sell initiated)
2. Build `ImpactReversionTracker`: on WHALE_INFLOW detection, record pre-impact mid, immediately measure post-impact mid, record impact_bps
3. After 500ms, enter AGAINST the whale's direction (short after large buy, long after large sell)
4. Target: entry_price + impact_bps * 0.5 (capture 50% of reversion)
5. Stop: entry_price - impact_bps * 0.33 (tight stop — reversion should happen within 2 seconds or not at all)
6. Force exit after 5 seconds regardless (no overnight holds)
7. Only trade if impact_bps > 1.5 (minimum viable reversion) AND bid-ask spread < 2bps (execution feasible)

**HEAN Integration Path**:
- Extend `packages/hean-physics/src/hean/physics/anomaly_detector.py`: add `side` to WHALE_INFLOW anomaly details (already has `details` dict)
- New strategy: `packages/hean-strategies/src/hean/strategies/impact_reversion.py`
- Inherits BaseStrategy with `_tick_interval_ms = 50` (override for ultra-fast processing)
- Subscribe to PHYSICS_UPDATE (to catch WHALE_INFLOW anomaly), ORDER_BOOK_UPDATE
- Signal metadata: `type: "impact_reversion"`, `hold_seconds_max: 5`, `impact_bps: float`
- New TP/SL logic: time-based exit at 5 seconds (need to track entry timestamp in position)
- Risk: maximum 1 concurrent impact reversion trade (per symbol, they don't stack)

**Potential Risks & Challenges**:
- Bybit testnet execution latency: on testnet, REST order placement takes 100-500ms, which may consume most of the 1-5 second reversion window. LIMIT orders preferred but may not fill if price moves fast.
- Whale classification on testnet may trigger frequently on small absolute values (e.g., 10x median might be only $5K on testnet)
- The temporary impact assumption holds only when the book is being REPLENISHED. If the whale has more to sell (iceberg order), the impact is permanent, not temporary.
- Very short hold time creates high turnover → need careful fee accounting (Bybit maker: -0.01%, taker: 0.055%)
- The 97% signal kill rate from ImpulseEngine filters would destroy this strategy — impact reversion signals must bypass the 12-filter cascade

**Estimated Daily PnL with $500 Capital**:
- Frequency: 5-20 whale trades per day on BTC (testnet dependent)
- Win rate: 58-65% (temporary impact is a well-documented phenomenon)
- Position: 3-5% capital with 5x leverage = $75-125 notional per trade
- Target: +1.5bps = $0.11-$0.19 per trade
- Daily with 10 trades: $1.10-$1.90
- The real scaling: increase position to 10% capital → $2.20-$3.80/day
- Limitation on testnet: fewer whale trades than mainnet

**Estimated Alpha Decay**: 6-18 months on mainnet (many HFT shops already run this). On Bybit Testnet specifically: likely 12+ months because institutional HFT does not operate on testnet.

**First Step**: Instrument the existing WHALE_INFLOW anomaly to log the direction (buy/sell) and the 5-second price change following detection. Run for 48 hours on live testnet. If the average price change in the 5 seconds after a buy-initiated whale is negative (reversion), and the magnitude is > 1bps, the edge is confirmed. This requires zero new code beyond adding 5 lines to anomaly_detector.py.

---

## Honorable Mentions (Promising but Need More Research)

### A. Cross-Exchange Basis Trade (Bybit vs Binance Testnet)
- Bybit testnet funding often diverges from Binance testnet by 2-5x (thin markets)
- Challenge: Binance testnet requires separate API setup; not currently integrated
- Edge: on mainnet, basis convergence trades yield 0.05-0.2% per 8h settlement; on testnet even larger
- Why not in top 5: requires dual-exchange execution infrastructure not yet in HEAN

### B. Options-Like Synthetic Structure via Perpetuals
- Construct a delta-hedged position: buy spot (or long perp) + short perp at 2x size = synthetic put-like payoff
- This is complex and requires accurate Greeks calculation
- Why not in top 5: $500 capital is too small for the hedging cost to make sense

### C. Micro-Arbitrage Between Bybit Perpetual and Bybit Spot
- On Bybit, perpetual and spot markets exist side-by-side
- When perp trades at 0.05% premium to spot, go short perp + long spot (convergence trade)
- Edge: pure arbitrage with no directional risk
- Why not in top 5: requires simultaneous cross-market order execution (perp + spot), which HEAN's execution router does not currently support

### D. Open Interest Momentum Signal
- OI rising + price rising = leveraged momentum, follow it
- OI rising + price falling = short squeeze building, bet on reversal
- Simple to implement as an augmentation to FundingHarvester
- Why not in top 5: partially covered by existing OI_SPIKE anomaly detector; needs differentiation

### E. Time-of-Day Seasonal Alpha
- Bybit testnet has strong time-of-day patterns: Asia hours (00:00-08:00 UTC) = low volume, high spread, grid strategies best; US hours (14:00-22:00 UTC) = trending moves, momentum strategies best
- Why not in top 5: easy to implement as a schedule filter on existing strategies, but not novel alpha — more of an enhancement
