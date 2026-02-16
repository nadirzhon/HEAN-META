# strategies -- Trading Strategies

Eleven trading strategies, each inheriting from `BaseStrategy` and gated by an individual settings flag. Strategies subscribe to market events via EventBus and publish `SIGNAL` events when entry conditions are met.

## Architecture

All strategies extend `BaseStrategy` (defined in `base.py`), which provides automatic subscription to TICK, FUNDING, REGIME_UPDATE, CONTEXT_READY, and PHYSICS_UPDATE events. The base class implements the SSD (Singular Spectral Determinism) lens that filters noise at the physics level -- ticks in "silent" mode are dropped before reaching strategy logic, while "laplace" mode boosts signal confidence for deterministic predictions. Price anomaly detection is also handled at the base level. Strategies call `_publish_signal(signal)` to emit signals, which applies SSD metadata and anomaly-based size reduction before publishing to EventBus. The `StrategyAllocator` (in `manager.py`) dynamically reallocates capital across strategies based on performance (Sharpe, profit factor, win rate) and market phase alignment.

## Key Classes

- `BaseStrategy` (`base.py`) -- Abstract base class. Provides SSD lens filtering, price anomaly detection, CONTEXT_READY handling, and `_publish_signal()` for uniform signal emission. Subclasses implement `on_tick()` and `on_funding()`.
- `StrategyAllocator` (`manager.py`) -- Dynamic capital allocator. Tracks per-strategy performance via `StrategyPerformance`, rebalances every 5 minutes (configurable), respects min 5% / max 40% allocation limits, and gives bonus allocation to strategies suited for the current market phase.
- `StrategyPerformance` (`manager.py`) -- Per-strategy tracker: win rate, profit factor, Sharpe ratio, drawdown, composite score.

### Strategy Implementations

| Strategy | File | Description |
|----------|------|-------------|
| ImpulseEngine | `impulse_engine.py` | Momentum strategy with 12-layer deterministic filter cascade (70-95% signal rejection) |
| FundingHarvester | `funding_harvester.py` | Captures funding rate arbitrage opportunities |
| BasisArbitrage | `basis_arbitrage.py` | Exploits basis spread between spot and futures |
| MomentumTrader | `momentum_trader.py` | Trend-following momentum strategy |
| CorrelationArbitrage | `correlation_arb.py` | Statistical arbitrage on correlated pairs |
| EnhancedGrid | `enhanced_grid.py` | Enhanced grid trading strategy |
| HFScalping | `hf_scalping.py` | High-frequency scalping |
| InventoryNeutralMM | `inventory_neutral_mm.py` | Inventory-neutral market making |
| RebateFarmer | `rebate_farmer.py` | Maker rebate farming via limit orders |
| LiquiditySweep | `liquidity_sweep.py` | Detects and trades liquidity sweep events |
| SentimentStrategy | `sentiment_strategy.py` | Trades on FinBERT/Ollama/Brain sentiment signals |

### Supporting Modules

- `impulse_filters.py` -- 12-layer deterministic filter cascade for ImpulseEngine
- `edge_confirmation.py` -- Confirms strategy edge before signal emission
- `multi_factor_confirmation.py` -- Multi-factor signal confirmation (physics, sentiment, oracle)
- `physics_aware_positioner.py` -- Physics-based position sizing adjustments
- `physics_signal_filter.py` -- Physics-based signal filtering (blocks signals in unfavorable phases)

## Events

| Event | Direction | Description |
|-------|-----------|-------------|
| TICK | Subscribes | Price/volume updates from exchange |
| FUNDING | Subscribes | Funding rate updates |
| REGIME_UPDATE | Subscribes | Market regime changes |
| CONTEXT_READY | Subscribes | Unified market context from ContextAggregator |
| PHYSICS_UPDATE | Subscribes | Physics state for SSD lens |
| SIGNAL | Publishes | Trading signal emitted when entry conditions met |
| POSITION_CLOSED | Subscribes (Allocator) | Tracks completed trades for performance scoring |
| EQUITY_UPDATE | Subscribes (Allocator) | Tracks total capital for allocation |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| IMPULSE_ENGINE_ENABLED | true | Enable ImpulseEngine strategy |
| FUNDING_HARVESTER_ENABLED | true | Enable FundingHarvester strategy |
| BASIS_ARBITRAGE_ENABLED | true | Enable BasisArbitrage strategy |
| MOMENTUM_TRADER_ENABLED | false | Enable MomentumTrader strategy |
| CORRELATION_ARB_ENABLED | false | Enable CorrelationArbitrage strategy |
| ENHANCED_GRID_ENABLED | false | Enable EnhancedGrid strategy |
| HF_SCALPING_ENABLED | false | Enable HFScalping strategy |
| INVENTORY_NEUTRAL_MM_ENABLED | false | Enable InventoryNeutralMM strategy |
| REBATE_FARMER_ENABLED | false | Enable RebateFarmer strategy |
| LIQUIDITY_SWEEP_ENABLED | false | Enable LiquiditySweep strategy |
| SENTIMENT_STRATEGY_ENABLED | false | Enable SentimentStrategy strategy |
