# portfolio -- Portfolio Accounting and Capital Allocation

Tracks portfolio equity, realized/unrealized PnL, drawdown, per-strategy performance, capital allocation, and profit capture with intra-session compounding.

## Architecture

`PortfolioAccounting` is the central ledger, tracking all positions, equity snapshots, and PnL with thread-safe mutations via `asyncio.Lock`. It maintains per-strategy metrics (PnL, trades, wins, losses) and per-strategy-per-regime tracking for detailed performance analysis. The `CapitalAllocator` distributes capital across enabled strategies with adaptive weight adjustment based on rolling profit factor and market regime. `ProfitCapture` locks in profits when targets are reached and supports `IntraSessionCompounding`, which reinvests 50% of realized profits every 5% gain within a session for compound growth.

## Key Classes

- `PortfolioAccounting` (`accounting.py`) -- Thread-safe portfolio ledger. Tracks positions, cash balance, realized PnL, total fees, peak equity, daily start equity. Provides `get_equity()`, `get_positions()`, `get_strategy_metrics()`. Includes metrics cache with 5-second TTL for performance.
- `CapitalAllocator` (`allocator.py`) -- Allocates capital to strategies with adaptive weights. Uses rolling profit factor over 30-day window, integrates with `StrategyMemory` for historical performance, and applies `CapitalPressure` for short-term adjustments.
- `ProfitCapture` (`profit_capture.py`) -- Locks profits when configurable targets are reached. Prevents giving back gains.
- `IntraSessionCompounding` (`profit_capture.py`) -- Reinvests 50% of realized profits every 5% equity gain within a session for accelerated compound growth.
- `CapitalPressure` (`capital_pressure.py`) -- Transient, short-term capital adjustment layer.
- `StrategyMemory` (`strategy_memory.py`) -- Historical performance memory for strategies.
- `DecisionMemory` (`decision_memory.py`) -- Remembers past decisions to avoid repeat mistakes.
- `MetaStrategyBrain` (`meta_strategy_brain.py`) -- Meta-level strategy decision making.
- `SmartReinvestor` (`smart_reinvestor.py`) -- Intelligent profit reinvestment logic.
- `ProfitTargetTracker` (`profit_target_tracker.py`) -- Tracks progress toward profit targets.
- `PortfolioRebalancer` (`rebalancer.py`) -- Portfolio-level rebalancing.

## Events

| Event | Direction | Description |
|-------|-----------|-------------|
| ORDER_FILLED | Subscribes | Updates positions and PnL on fill |
| POSITION_OPENED | Publishes | New position added to portfolio |
| POSITION_CLOSED | Publishes | Position removed, realized PnL recorded |
| PNL_UPDATE | Publishes | Periodic PnL snapshot |
| EQUITY_UPDATE | Publishes | Equity change notification |
| ORDER_DECISION | Publishes | Entry decision record |
| ORDER_EXIT_DECISION | Publishes | Exit decision record |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| INITIAL_CAPITAL | 300 | Starting USDT capital |
