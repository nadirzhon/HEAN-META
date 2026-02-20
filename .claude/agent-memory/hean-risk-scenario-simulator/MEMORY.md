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

## Known Gaps / Warnings

- See `health-check-findings.md` for detailed vulnerability list
