# Alpha Briefs — Session 1 (2026-02-21)

Topic: Alpha angles derived from HEAN reliability fixes.

## Brief 1: BusLoadAlpha
Derived from: Event priority upgrade fix for RISK_ENVELOPE.
Core idea: Bus utilization and canary RTT are real-time proxies for system-wide
order flow intensity, which precedes volatility expansions (per VPIN literature).
Integration: New BUS_PRESSURE event type; BusCanary already measures RTT P99.
Data: bus._metrics["events_dropped"], bus._health.queue_utilization_pct, canary.p99_rtt_ms
Expected edge: 15-30 bps by widening spreads or pausing new entries before spikes.
Alpha decay: Medium — proprietary to HEAN's own infrastructure, not crowdable.

## Brief 2: ConvergenceScoreAlpha
Derived from: In-flight order pre-registration (OrderManager._pending_placements).
Core idea: Count of strategies simultaneously attempting orders = agreement score.
Integration: Middleware intercepts SIGNAL events, builds convergence counter,
emits CONTEXT_UPDATE with type="convergence_score".
Expected edge: 10-25 bps confidence boost on convergent signals.
Alpha decay: Medium-low — standard ensemble technique, but HEAN-specific combination.

## Brief 3: RetryDeltaAlpha
Derived from: DLQ auto-retry recovering failed ORDER_REQUEST events.
Core idea: Track Δprice = (retry_price - original_price) to classify regime.
If Δprice > 0 (price moved against original direction), mean reversion likely → hold.
If Δprice < 0 (favorable drift), momentum signal → re-enter aggressively.
Integration: DLQ.retry_all() hook adds price_at_retry to retried event metadata.
Expected edge: 5-20 bps by routing retries smarter; avoid re-entry into momentum crashes.
Alpha decay: Low — very few systems have DLQ with price tagging.

## Brief 4: PhantomFillArbitrage
Derived from: Phantom order reconciliation discovering untracked exchange positions.
Core idea: Log every reconciliation-discovered position with timestamp, price,
market regime at discovery. Build P&L distribution of these "accidental" entries.
If mean P&L > 0, it implies entry price correlates with favorable market timing.
Integration: PositionReconciler publishes RECONCILIATION_ALERT with phantom details.
Expected edge: Diagnostic first; if profitable, could inform intentional entry randomization.
Alpha decay: Very low — purely observational; no crowding risk.

## Brief 5: EnvelopeAccuracyAlpha (Capital Efficiency Compounding)
Derived from: ORDER_FILLED dedup ensuring single envelope recompute.
Core idea: With accurate (non-double-counted) equity, Kelly sizing targets the
true optimal fraction. Study the delta between old double-counted size and
true Kelly size across a sample of trades. Measure Sharpe improvement.
Integration: Log pre/post dedup envelope_multiplier in trade records.
Expected edge: 3-8% CAGR improvement from compound sizing accuracy over 6 months.
Alpha decay: None — this is a pure operational improvement.
