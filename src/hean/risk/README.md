# risk -- Risk Management

Graduated risk management system with a state machine (RiskGovernor), adaptive killswitch, Kelly-optimal position sizing, leverage control, and optional RL-based parameter adjustment.

## Architecture

Risk management is layered. The `RiskGovernor` operates as a state machine with four levels: NORMAL, SOFT_BRAKE, QUARANTINE, and HARD_STOP. It evaluates equity drawdown, per-symbol performance, and error rates to transition between states. The `KillSwitch` acts as a safety net, triggering only on catastrophic drawdown from initial capital (configurable, default 30%) or excessive daily loss. `PositionSizer` integrates Kelly Criterion, regime-based adjustments, smart leverage, capital preservation mode, and a hard cap on combined multipliers (MAX_TOTAL_MULTIPLIER = 3.0) to prevent multiplicative explosion. The optional `RLRiskManager` uses a trained PPO agent to dynamically adjust leverage, position size multipliers, and stop-loss placement based on market conditions observed through the Physics engine.

## Key Classes

- `RiskGovernor` (`risk_governor.py`) -- State machine: NORMAL -> SOFT_BRAKE (50% sizing) -> QUARANTINE (per-symbol blocking) -> HARD_STOP (emergency halt). Tracks high water mark for accurate drawdown calculation.
- `KillSwitch` (`killswitch.py`) -- Adaptive safety net. Hard-kills only on catastrophic loss; daily drawdown triggers a pause with 5-minute cooldown auto-reset. Auto-reset is enabled by default for testnet.
- `PositionSizer` (`position_sizer.py`) -- Calculates position size integrating Kelly Criterion, DynamicRiskManager, SmartLeverageManager, and CapitalPreservationMode. Caps total multiplier at 3.0x.
- `KellyCriterion` (`kelly_criterion.py`) -- Recursive Kelly with fractional sizing (0.25x-0.5x), confidence-based scaling, streak tracking, and Bayesian win rate estimation. Per-strategy performance tracking.
- `DepositProtector` (`deposit_protector.py`) -- Ensures equity never falls below initial capital. Highest priority protection mechanism.
- `SmartLeverageManager` (`smart_leverage.py`) -- Allows 3-5x leverage only for very high-quality signals. Multiple safety checks: edge requirements, regime, drawdown, profit factor, volatility.
- `RLRiskManager` (`rl_risk_manager.py`) -- PPO-based risk parameter adjustment. Adapts leverage (1-10x), size multiplier (0.5-2x), and stop-loss (0.5-10%). Falls back to rule-based logic if model unavailable.
- `TradingRiskEnv` (`gym_env.py`) -- Gymnasium environment for training the RL risk agent. 10-dimensional observation space, 3-dimensional continuous action space.
- `DynamicRiskManager` (`dynamic_risk.py`) -- Adaptive risk parameters based on recent performance.
- `CapitalPreservationMode` (`capital_preservation.py`) -- Activates conservative sizing when capital is at risk.
- `PriceAnomalyDetector` (`price_anomaly_detector.py`) -- Detects price gaps, spikes, and flash crashes. Can block trading and reduce position sizes.

## Events

| Event | Direction | Description |
|-------|-----------|-------------|
| SIGNAL | Subscribes (RiskGovernor) | Evaluates signal against risk limits before approving |
| ORDER_REQUEST | Publishes | Risk-approved order request sent to execution |
| RISK_BLOCKED | Publishes | Signal blocked by risk layer (with reason code) |
| RISK_ALERT | Publishes | Risk state change or warning |
| KILLSWITCH_TRIGGERED | Publishes | Emergency halt triggered |
| KILLSWITCH_RESET | Publishes | Killswitch deactivated |
| PNL_UPDATE | Subscribes | Tracks PnL for drawdown calculation |
| EQUITY_UPDATE | Subscribes | Tracks equity for killswitch evaluation |
| CONTEXT_UPDATE | Subscribes (RLRiskManager) | Market conditions for RL observation |
| PHYSICS_UPDATE | Subscribes (RLRiskManager) | Temperature/entropy/phase for RL observation |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| RISK_GOVERNOR_ENABLED | true | Enable graduated risk governor |
| KILLSWITCH_DRAWDOWN_PCT | 30 | Catastrophic drawdown threshold (% from initial) |
| MAX_LEVERAGE | 5.0 | Maximum allowed leverage |
| RL_RISK_ENABLED | false | Enable RL-based risk parameter adjustment |
| RL_RISK_MODEL_PATH | "" | Path to trained PPO model (.zip) |
| RL_RISK_ADJUST_INTERVAL | 60 | Seconds between RL adjustments |
