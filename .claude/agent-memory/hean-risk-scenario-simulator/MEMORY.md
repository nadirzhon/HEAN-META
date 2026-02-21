# HEAN Risk Scenario Simulator — Agent Memory

## Key File Paths

- Risk Governor: `/Users/macbookpro/Desktop/HEAN/backend/packages/hean-risk/src/hean/risk/risk_governor.py`
- KillSwitch: `/Users/macbookpro/Desktop/HEAN/backend/packages/hean-risk/src/hean/risk/killswitch.py`
- Physics Engine: `/Users/macbookpro/Desktop/HEAN/backend/packages/hean-physics/src/hean/physics/engine.py`
- AutoPilot Coordinator: `/Users/macbookpro/Desktop/HEAN/backend/packages/hean-core/src/hean/core/autopilot/coordinator.py`
- AutoPilot State Machine: `/Users/macbookpro/Desktop/HEAN/backend/packages/hean-core/src/hean/core/autopilot/state.py`
- Config: `/Users/macbookpro/Desktop/HEAN/backend/packages/hean-core/src/hean/config.py`
- Packages live in `/Users/macbookpro/Desktop/HEAN/backend/packages/hean-{risk,physics,core,api}/`

## Confirmed Threshold Values (from config.py)

- `killswitch_drawdown_pct` = **30%** (from initial capital, HARD_STOP)
- `max_daily_drawdown_pct` = **15%** (daily pause, auto-resets after 5 min)
- `RiskGovernor` SOFT_BRAKE: 10% drawdown from HWM
- `RiskGovernor` QUARANTINE: 15% drawdown from HWM
- `RiskGovernor` HARD_STOP: 20% drawdown from HWM
- `KillSwitch` API response `equity_drop` threshold = 15.0 (from `thresholds.equity_drop`)

## Critical Discrepancy Found (Health Check 2026-02-21)

The `/api/v1/risk/killswitch/status` endpoint reports `thresholds.drawdown_pct: 8.0` and `thresholds.equity_drop: 15.0`.
However the actual config values are `killswitch_drawdown_pct=30%` and `max_daily_drawdown_pct=15%`.
The API response `drawdown_pct: 8.0` does NOT match config — this is likely a hardcoded default in the API serializer,
not the live config value. Needs investigation.

## System State at First Health Check (2026-02-21)

- Engine running LIVE mode (not dry-run), equity ~500 USDT
- RiskGovernor: NORMAL, not initialized warning ("Risk governor not initialized" in recommended_action)
- KillSwitch: not triggered, current_drawdown=0.0%
- AutoPilot: stuck in LEARNING mode for ~15 min (900s), never transitioned, zero decisions
- Physics: reports "Engine not running" for all symbols despite engine running — microservice split issue
- Brain: last analysis was 23+ hours ago (2026-02-20T22:13, current date 2026-02-21)
- Physics temperature in autopilot context: 227,155,974 (wildly abnormal — likely raw unscaled value)
- Strategies: 10-11 enabled, zero trades (all win_rate=0, profit_factor=0)
- 323 signals generated session-wide, 12 blocks, 0 orders created
- Council: disabled
- Signals dominated by inventory_neutral_mm (234/323), risk_svc (35/323)

## Architectural Patterns Confirmed

- RiskGovernor uses HIGH WATER MARK drawdown (from peak equity), not initial capital
- KillSwitch uses HARD_STOP (from initial capital) + DAILY_PAUSE (from daily high)
- AutoPilot is a separate meta-layer; LEARNING -> CONSERVATIVE is the only first transition
- Physics service runs as a separate microservice in Docker; the in-process API endpoint
  returns zeroed/stale data when the physics microservice is separate
- Brain analysis is periodic (Brain interval config), not continuous

## EventBus Queue Sizing (confirmed from bus.py)

- `bus_max_queue_size` = 50,000 (config default)
- critical_queue = maxsize // 5 = **10,000 slots**
- normal_queue = maxsize // 2 = **25,000 slots**
- low_queue = maxsize = **50,000 slots**
- FAST_PATH_EVENTS (bypass queue entirely): SIGNAL, ORDER_REQUEST, ORDER_FILLED, ENRICHED_SIGNAL, ORDER_DECISION
- CRITICAL priority (never dropped, wait up to 5s): ORDER_*, POSITION_*, RISK_ALERT, RISK_ENVELOPE, POSITION_CLOSE_REQUEST, EQUITY_UPDATE, SIGNAL, ENRICHED_SIGNAL
- LOW priority dropped when circuit open; NORMAL dropped after 1s timeout

## Key Fixes Confirmed in Codebase (2026-02-21)

- `RISK_ENVELOPE` and `POSITION_CLOSE_REQUEST` now CRITICAL priority in EVENT_PRIORITY_MAP (bus.py:107-108)
- `_safe_call_handler` now re-raises instead of swallowing exceptions → DLQ now populates (bus.py:827-831)
- Two-phase RiskSentinel startup: `start()` caches envelope, `publish_initial_envelope()` publishes after strategies subscribe (risk_sentinel.py:113-182)
- `InFlightTracker` prevents burst-beyond-max_open_positions in window between envelope recompute and fill (inflight_tracker.py)
- `EventBus.stop()` now explicitly cancels the processing task (bus.py line 905) — previously, setting `_running=False` was insufficient
- `DeadLetterQueue` auto-retry uses exponential backoff: `2^retry_count * 10s` (bus_dlq.py:601)
- DLQ `permanent_failures` stored separately (not re-queued) once `max_retries` exhausted (bus_dlq.py:584-596)

## InFlightTracker Key Facts

- Located: `packages/hean-core/src/hean/core/inflight_tracker.py`
- `reserve()` is synchronous (not async) — safe under GIL for asyncio-concurrent callers
- GC interval = 5s, reservation timeout = 30s
- Auto-releases on ORDER_FILLED / ORDER_REJECTED / ORDER_CANCELLED events
- `can_open_position(current, max)` checks `current + in_flight < max`

## _close_position_at_price Gap (main.py:3673)

On Bybit HTTP failure, function logs CRITICAL and returns without marking position closed.
Position stays tracked in PortfolioAccounting. PositionReconciler must detect the drift.
No retry logic in the close function itself — single attempt only.

## Critical Dead Code Found (2026-02-22, $1000/day risk analysis)

### GlobalSafetyNet (tail_risk.py) — DEAD CODE
- `_activate_safety_net()` (line 163): logs + publishes REGIME_UPDATE but does NOT
  reduce any actual position sizes (PositionSizer never reads GlobalSafetyNet.get_size_multiplier())
- `_initiate_hedge_positions()` (line 217): creates Signal object but NEVER publishes it
  to EventBus — signal is created and immediately garbage collected
- GlobalSafetyNet is NOT integrated into PositionSizer.calculate_size() or calculate_size_v2()

### HARD_STOP Does NOT Close Positions
- RiskGovernor._escalate_to(HARD_STOP) publishes RISK_ALERT but does NOT publish
  POSITION_CLOSE_REQUEST for open positions (risk_governor.py:688-736)
- KillSwitch._trigger() publishes STOP_TRADING + KILLSWITCH_TRIGGERED but also does
  NOT close existing positions (killswitch.py:304-345)
- Only protection for existing positions at HARD_STOP: max_hold_seconds=900 TTL

### FundingHarvester State Divergence (funding_harvester.py:393)
- self._positions[symbol] = side is always written even if signal was blocked by RiskLimits
- Creates internal/real state mismatch: strategy thinks it holds a position it doesn't
- No auto-close signal on funding rate sign inversion (funding_harvester.py:307-311)

### KillSwitch Spread Check is Absolute, Not Relative (killswitch.py:201)
- Threshold: spread_pct > 2.0% (absolute)
- BTC/ETH never reach 2% spread in normal crisis (typical spread 0.01-0.1%)
- 10x spread widening from 0.1% → 1% does NOT trigger the check
- Need EMA baseline + relative threshold (e.g., >5x baseline AND >0.5%)

### MultiLevelProtection Does NOT Close Existing Positions
- check_all_protections() only blocks NEW ORDER_REQUEST events
- Existing open positions are not affected by hourly/daily limits
- Only position closure mechanism when HARD_STOP: max_hold_seconds TTL (15 min)

## Risk Parameter Summary (confirmed from config.py 2026-02-22)

- max_trade_risk_pct = 1.0%, max_leverage = 3.0x, max_open_positions = 10
- killswitch HARD_STOP = 30% from initial capital (DepositProtector)
- RiskGovernor HARD_STOP = 20% from HWM (risk_governor.py:340)
- max_concurrent_risk = 10 * 1% = 10% equity at risk simultaneously
- Max leverage for P(ruin) < 1% = 2-3x (current config is at the boundary)
- SmartLeverage max_leverage_2x edge: 25bps, 3x: 35bps, 4x: 50bps

## Known Gaps / Warnings

- See `health-check-findings.md` for detailed vulnerability list
- See `backend/docs/stress-test-scenarios.md` for full stress test design (6 scenarios)
