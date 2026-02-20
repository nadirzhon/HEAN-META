# Health Check Findings — 2026-02-21

## Session: Live Docker System at http://localhost:8000

### Raw Endpoint Responses Summary

| Endpoint | HTTP | Key Values |
|---|---|---|
| /risk/governor/status | 200 | state=NORMAL, recommended_action="Risk governor not initialized" |
| /risk/killswitch/status | 200 | triggered=false, drawdown_pct threshold=8.0 (config says 30%!) |
| /autopilot/status | 200 | mode=learning, 900s in mode, 0 decisions |
| /physics/state?symbol=BTCUSDT | 200 | temperature=0, phase=unknown, should_trade=false, "Engine not running" |
| /brain/analysis | 200 | last analysis 23h ago, no live analysis |
| /strategies | 200 | 10 strategies enabled, all zeros |

### Vulnerability: KillSwitch API Threshold Mismatch
- API reports `thresholds.drawdown_pct: 8.0`
- Config has `killswitch_drawdown_pct = 30.0`
- Likely the API response serializes a different attribute (the `max_daily_drawdown_pct`-derived
  value or a legacy hardcoded field), not the actual `settings.killswitch_drawdown_pct`
- Risk: Operators monitoring this endpoint could have false confidence in protection levels

### Vulnerability: RiskGovernor Not Fully Initialized
- `recommended_action` returns "Risk governor not initialized"
- This means `check_and_update()` has never been called (peak_equity=0, initial_capital=0)
- The governor's `_initial_capital == 0.0` guard on first call has not fired
- Despite engine running for 15+ minutes with signals flowing

### Vulnerability: Physics Microservice Split
- In-process API returns `temperature=0, phase=unknown, should_trade=false, "Engine not running"`
- Autopilot context shows `physics_temperature=227,155,974` (live value from microservice)
- The API physics endpoint reads from the in-process PhysicsEngine, not the physics microservice
- This creates a split-brain: Autopilot sees real physics data; API monitoring shows stale zeros

### Vulnerability: AutoPilot Stuck in LEARNING
- System has been running ~15 min (900s) in LEARNING mode
- LEARNING mode can only transition to CONSERVATIVE
- AutoPilot coordinator's learning_period_sec defaults to 3600s (1 hour)
- Zero decisions made in learning period — bandit arms all at alpha=1.0, beta=1.0 (uninformed priors)
- This is by design but means the system is effectively unguided for the first hour

### Vulnerability: Brain Analysis Staleness
- Last brain analysis: 2026-02-20T22:13 (23+ hours before health check on 2026-02-21)
- Brain appears to have run once at startup and not since
- This could indicate BRAIN_ENABLED=false, missing ANTHROPIC_API_KEY, or an interval/crash issue

### Warning: Physics Temperature Scale
- Autopilot context shows `physics_temperature: 227,155,974`
- Brain shows `T=111583764685` for ADAUSDT
- These are raw unscaled values from MarketTemperature — not abnormal for the physics model
  but extremely confusing for human operators monitoring dashboards
- The physics API endpoint normalizes via RollingNormalizer; the raw values in context don't

### Observation: Signal-to-Order Funnel
- 323 signals generated in ~15 min session
- 12 signals blocked by risk
- 0 orders created
- This suggests filter cascade (12-layer impulse filters) is rejecting all signals,
  OR position sizer is returning 0 for all signals
- Top signal source: inventory_neutral_mm (234 signals) — this strategy generates many small signals
