# Pull Request: Small Capital Profit Mode + Auto-Deploy to Mac

## Summary

This PR includes two major features:

1. **Small Capital Profit Mode** - Cost-aware execution system for viable small deposit trading
2. **Auto-Deploy to Mac** - GitHub Self-Hosted Runner for automated deployment on push to main

---

## Part 1: Small Capital Profit Mode

### Overview

Implements comprehensive cost-aware execution that prevents trades where fees + spread + slippage dominate edge. Makes the system viable for small deposits by enforcing maker-first execution and providing full observability for every trading decision.

### What Changed

**New Backend Modules (Python):**
- `src/hean/execution/cost_engine.py` (220 lines) - Estimates fees, spread, slippage in bps
- `src/hean/execution/market_filters.py` (147 lines) - Detects stale data, low liquidity, wide spreads
- `src/hean/execution/trade_gating.py` (275 lines) - Enforces edge ≥ cost × multiplier logic

**Modified Backend Files:**
- `src/hean/config.py` (+69 lines) - 14 new feature flags with safe defaults
- `src/hean/execution/router.py` (+7 lines) - Initialize trade gating modules
- `src/hean/api/routers/trading.py` (+35 lines) - Enhanced /trading/why endpoint
- `backend.env` (+14 lines) - Environment variables

**New Frontend Component:**
- `apps/ui/src/app/components/trading/SmallCapitalPanel.tsx` (171 lines) - UI panel showing cost vs edge metrics

**Testing & Documentation:**
- `scripts/smoke_test_small_capital.py` (327 lines) - Comprehensive verification
- `SMALL_CAPITAL_MODE_IMPLEMENTATION.md` - Complete implementation guide

### Key Features

✅ **Cost Engine** - Estimates total trading costs (fees + spread + slippage)
✅ **Market Filters** - Blocks trades on stale data, low liquidity, wide spreads
✅ **Trade Gating** - Requires edge ≥ cost × multiplier (default 4x)
✅ **Observability** - Enhanced `/trading/why` endpoint with cost/edge metrics
✅ **UI Component** - Real-time cost vs edge visualization
✅ **Feature Flags** - Backwards compatible, safe defaults

### Environment Variables (New)

```bash
SMALL_CAPITAL_MODE=true                    # Enable cost-aware gating
MIN_NOTIONAL_USD=10.0                      # Bybit minimum
MAKER_ONLY_DEFAULT=true                    # Force maker orders
COST_EDGE_MULTIPLIER=4.0                   # Require edge ≥ cost × 4
MAX_SPREAD_BPS=8.0                         # Block if spread > 8 bps
MAX_SLIPPAGE_ESTIMATE_BPS=20.0             # Block if slippage > 20 bps
STALE_TICK_MAX_AGE_SEC=2                   # Block if tick > 2 sec old
ALLOW_TAKER_IF_EDGE_STRONG=false           # Default: no taker orders
TAKER_EDGE_MULTIPLIER=8.0                  # Require edge ≥ cost × 8 for taker
MAKER_LIMIT_CHASE_RETRIES=2                # Retry maker 2x
MAKER_LIMIT_CHASE_TIMEOUT_SEC=5            # 5 sec timeout per retry
```

### API Changes

**Enhanced: GET /trading/why**

Now includes `small_capital_mode` section:

```json
{
  "small_capital_mode": {
    "enabled": true,
    "avg_cost_bps": 12.5,
    "avg_edge_bps": 52.3,
    "edge_cost_ratio": 4.18,
    "top_block_reasons": [
      {"reason": "EDGE_TOO_LOW_FOR_COST", "count": 15},
      {"reason": "SPREAD_TOO_WIDE", "count": 8}
    ],
    "maker_fill_rate": 0.73,
    "decision_counts": {
      "create": 42,
      "skip": 15,
      "block": 23,
      "total": 80
    }
  }
}
```

### Trade Decision Logic

```
total_cost_bps = (2 × fee_bps) + spread_bps + slippage_bps
required_edge_bps = total_cost_bps × COST_EDGE_MULTIPLIER

if expected_edge_bps >= required_edge_bps:
    CREATE order (allow trade)
else:
    BLOCK trade with reason "EDGE_TOO_LOW_FOR_COST"
```

### Non-Negotiable Constraints Met

✅ No breaking changes to existing REST/WS endpoints
✅ No refactoring of EngineFacade/Risk/Strategies
✅ UI components preserved, only additions
✅ No "guaranteed profit" claims
✅ Backwards compatible with feature flags

---

## Part 2: Auto-Deploy to Mac via Self-Hosted Runner

### Overview

Automated deployment pipeline that triggers on push to main, executing `docker compose up -d --build --remove-orphans` on macOS runner.

### What Changed

**New Files:**

1. **`.github/workflows/deploy_on_macos_runner.yml`** (180 lines)
   - GitHub Actions workflow for auto-deploy
   - Triggers: push to main + workflow_dispatch
   - Runs on: [self-hosted, macOS, nadir-mac]
   - Steps: checkout → diagnostics → deploy → verify → health checks

2. **`scripts/runner_setup_macos.md`** (350 lines)
   - Step-by-step guide for installing GitHub runner on Mac
   - Configuration with critical label "nadir-mac"
   - Service installation (auto-start on boot)
   - Troubleshooting and security best practices

3. **`scripts/deploy_local.sh`** (135 lines, executable)
   - Local deployment script for testing
   - Mimics workflow: diagnostics → deploy → verify
   - Supports --rebuild flag for no-cache builds
   - Color-coded output, health checks

**Modified:**
- `README.md` - Added "Auto-Deploy to Mac via Self-Hosted Runner" section

### How It Works (End-to-End)

1. **One-time setup:** Install runner on Mac with label `nadir-mac`
2. **Runner service:** Runs as background service (auto-starts on boot)
3. **Push to main:** Triggers GitHub Actions workflow
4. **Deploy:** Executes `docker compose up -d --build --remove-orphans`
5. **Verify:** Health checks, logs, container status
6. **Cleanup:** Removes dangling images
7. **Summary:** Shows deployment result

### Security Features

✅ **Only push to main** - Never on pull_request (prevents fork attacks)
✅ **Label targeting** - Runs only on `nadir-mac` runner (isolation)
✅ **Concurrency control** - Prevents overlapping deployments
✅ **Timeout (30 min)** - Prevents stuck workflows
✅ **Health checks** - Validates deployment success

### Workflow Features

- System diagnostics (Docker version, disk usage)
- Container health checks (API: 8000, UI: 3000)
- Deployment logs (last 200 lines)
- Automatic cleanup (dangling images)
- Deployment summary (commit, branch, status)
- Manual trigger support (workflow_dispatch with rebuild option)

### Security Rationale

**Why no `pull_request` trigger?**
- Prevents malicious code execution from untrusted forks
- Self-hosted runners execute code on your machine
- Only trusted code from main branch is deployed

**Why label targeting (`nadir-mac`)?**
- Ensures only your authorized runner executes deploys
- Prevents accidental runs on wrong machines
- Allows multiple runners with different purposes

**Why concurrency control?**
- Prevents race conditions from simultaneous deploys
- Ensures clean state before each deployment
- Cancels in-progress runs when new push arrives

**Why 30-minute timeout?**
- Prevents resource exhaustion from hung workflows
- Forces failure detection on stuck deployments
- Reasonable time for Docker build + deploy

### Testing

**Automated (GitHub Actions):**
```bash
git commit --allow-empty -m "Test auto-deploy"
git push origin main
```

**Manual (Local):**
```bash
./scripts/deploy_local.sh           # Normal deploy
./scripts/deploy_local.sh --rebuild # Force rebuild
```

**Verification:**
1. Check Actions tab in GitHub for workflow run
2. Verify containers: `docker ps | grep hean`
3. Check health: `curl http://localhost:8000/health`
4. Access UI: http://localhost:3000

---

## Files Changed

**Total: 14 files changed, +2,598 lines**

### Small Capital Mode (10 files)
- ✅ `SMALL_CAPITAL_MODE_IMPLEMENTATION.md` (new, 550 lines)
- ✅ `src/hean/config.py` (modified, +69 lines)
- ✅ `src/hean/execution/cost_engine.py` (new, 220 lines)
- ✅ `src/hean/execution/market_filters.py` (new, 147 lines)
- ✅ `src/hean/execution/trade_gating.py` (new, 275 lines)
- ✅ `src/hean/execution/router.py` (modified, +7 lines)
- ✅ `src/hean/api/routers/trading.py` (modified, +35 lines)
- ✅ `backend.env` (modified, +14 lines)
- ✅ `apps/ui/src/app/components/trading/SmallCapitalPanel.tsx` (new, 171 lines)
- ✅ `scripts/smoke_test_small_capital.py` (new, 327 lines)

### Auto-Deploy (4 files)
- ✅ `.github/workflows/deploy_on_macos_runner.yml` (new, 180 lines)
- ✅ `scripts/runner_setup_macos.md` (new, 350 lines)
- ✅ `scripts/deploy_local.sh` (new, 135 lines, executable)
- ✅ `README.md` (modified, +118 lines)

---

## Risk Assessment

### Small Capital Mode Risks

**Low Risk:**
- ✅ Backwards compatible (feature flag controlled)
- ✅ No breaking changes to existing endpoints
- ✅ Safe defaults (can be disabled with `SMALL_CAPITAL_MODE=false`)
- ✅ Isolated modules (no refactoring of core systems)

**Mitigation:**
- Comprehensive smoke tests before deployment
- Gradual rollout (enable for subset of symbols first)
- Monitoring via `/trading/why` endpoint

### Auto-Deploy Risks

**Security Risks (Mitigated):**
- ❌ Code execution from forks → ✅ No `pull_request` trigger
- ❌ Unauthorized runner access → ✅ Label targeting (`nadir-mac`)
- ❌ Concurrent deploys → ✅ Concurrency control
- ❌ Hung deployments → ✅ 30-minute timeout

**Operational Risks (Low):**
- Runner offline → Workflow fails gracefully (no auto-deploy)
- Docker daemon down → Health checks catch failure
- Port conflicts → Pre-flight checks in workflow

**Mitigation:**
- Manual testing with `./scripts/deploy_local.sh`
- Monitor GitHub Actions logs
- Health checks before marking success

---

## Testing Performed

### Small Capital Mode
✅ Module imports (cost_engine, market_filters, trade_gating)
✅ Config loading (all 14 feature flags)
✅ REST endpoint `/trading/why` includes new fields
✅ WebSocket connection and subscriptions
✅ Cost calculation logic (fees + spread + slippage)
✅ Edge vs cost gating (4x multiplier)
✅ Market filters (stale data, wide spreads)

### Auto-Deploy
✅ Workflow syntax validation
✅ Local deploy script testing
✅ Health check endpoints (API, UI)
✅ Container startup and verification
✅ Log tailing and diagnostics
✅ Cleanup (dangling images)

---

## Deployment Steps

### 1. Small Capital Mode

**Before merge:**
```bash
# Run smoke tests
python3 scripts/smoke_test_small_capital.py
```

**After merge:**
```bash
# Rebuild Docker images
docker compose down
docker compose build --no-cache api hean-ui
docker compose up -d

# Verify deployment
curl http://localhost:8000/trading/why | jq '.small_capital_mode'
```

**Expected output:**
```json
{
  "enabled": true,
  "avg_cost_bps": null,
  "avg_edge_bps": null,
  "edge_cost_ratio": null,
  "top_block_reasons": [],
  "decision_counts": {"create": 0, "skip": 0, "block": 0, "total": 0}
}
```

### 2. Auto-Deploy

**One-time setup:**
```bash
# Install runner on Mac
# See scripts/runner_setup_macos.md for detailed steps

mkdir -p ~/actions-runner && cd ~/actions-runner
# Download from GitHub (Settings → Actions → Runners)
./config.sh --url https://github.com/nadirzhon/HEAN-META --token TOKEN --labels "nadir-mac"
./svc.sh install
./svc.sh start
```

**Verify auto-deploy:**
```bash
# Test with empty commit
git commit --allow-empty -m "Test auto-deploy"
git push origin main

# Check GitHub Actions tab
# Verify containers on Mac
docker ps | grep hean
```

---

## Alternatives Considered

### Small Capital Mode
1. **Hardcoded cost assumptions** → Rejected: Not flexible, no observability
2. **External cost API** → Rejected: Latency, dependency
3. **Feature flag per component** → Selected: Backwards compatible, gradual rollout

### Auto-Deploy
1. **GitHub-hosted runner** → Rejected: No access to local Mac
2. **Manual deployment** → Rejected: Slow, error-prone
3. **Self-hosted runner** → Selected: Automated, secure, reliable

---

## Next Steps

1. **Review PR** - Ensure all changes are acceptable
2. **Merge to main** - Triggers auto-deploy (if runner installed)
3. **Run smoke tests** - Verify Small Capital Mode works
4. **Monitor metrics** - Check `/trading/why` for cost/edge data
5. **Install runner** - If not already done (one-time setup)
6. **Test auto-deploy** - Push test commit to main

---

## Questions for Review

1. ✅ Are the security measures for auto-deploy sufficient?
2. ✅ Should we add more health checks to the workflow?
3. ✅ Is the `COST_EDGE_MULTIPLIER=4.0` default appropriate?
4. ✅ Should we add more block reasons to market filters?
5. ✅ Any concerns about backwards compatibility?

---

## References

- Small Capital Mode Guide: `SMALL_CAPITAL_MODE_IMPLEMENTATION.md`
- Runner Setup Guide: `scripts/runner_setup_macos.md`
- Workflow File: `.github/workflows/deploy_on_macos_runner.yml`
- Local Deploy Script: `scripts/deploy_local.sh`

---

**Created by:** Claude 4.5 (Principal Quant/Systems Engineer + DevOps/Platform Engineer)
**Date:** 2026-01-22
**Branch:** `claude/small-capital-profit-mode-xJMw0`
**Commits:** 2 (678db4f + a012806)
