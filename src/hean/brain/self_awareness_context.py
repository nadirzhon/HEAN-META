"""
Self-Awareness Context Aggregator

This module is responsible for gathering all necessary data points from across
the HEAN system to provide a comprehensive "self-awareness" context for the
Brain (AI) to analyze its own performance and code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hean.portfolio.strategy_capital_allocator import StrategyPerformance
    from hean.symbiont_x.backtesting.backtest_engine import BacktestResult

@dataclass
class SystemPerformanceMetrics:
    """High-level performance metrics of the entire system."""
    total_pnl: float
    roi_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    active_strategies: int

@dataclass
class SelfAwarenessContext:
    """
    A complete snapshot of the system's state, performance, and code
    for meta-analysis by the Brain.
    """
    timestamp: float
    market_regime: dict[str, Any]
    system_performance: SystemPerformanceMetrics
    strategy_performance: list[StrategyPerformance]
    worst_performing_strategy_code: str | None = None
    symbiont_backtest_results: list[BacktestResult] = field(default_factory=list)
    recent_errors: list[str] = field(default_factory=list)

class ContextAggregator:
    """
    Gathers context from various system components.

    This class will need access to the main TradingSystem object or its
    constituent components (Portfolio, Strategy Allocator, etc.) to
    gather the required data.
    """

    def __init__(self, trading_system: Any):
        # NOTE: Using 'Any' to avoid circular dependency with main.py's TradingSystem
        # This should be properly typed with a Protocol in a real implementation.
        self.trading_system = trading_system
        self.max_code_lines = 150 # Limit code snippet size

    def _get_worst_performing_strategy(self) -> StrategyPerformance | None:
        """Identifies the worst performing strategy based on recent PnL or Sharpe."""
        if not self.trading_system.strategy_allocator:
            return None

        performance_data = self.trading_system.strategy_allocator.get_performance_metrics()
        if not performance_data:
            return None

        # Simple logic: worst by sharpe ratio if available, else by PnL
        return min(performance_data, key=lambda p: p.sharpe_ratio if p.sharpe_ratio is not None else p.realized_pnl)

    def _read_strategy_source_code(self, strategy_id: str) -> str | None:
        """
        Reads the source code of a given strategy file.

        NOTE: This is a simplified placeholder. A real implementation would need a
        robust way to map a strategy ID/name to its source file path.
        """
        # This mapping would need to be maintained or discovered dynamically.
        strategy_file_map = {
            "ImpulseEngine": "src/hean/strategies/impulse_engine.py",
            "MomentumTrader": "src/hean/strategies/momentum_trader.py",
            "FundingHarvester": "src/hean/strategies/funding_harvester.py",
            # ... etc. for all strategies
        }

        file_path = strategy_file_map.get(strategy_id)
        if not file_path:
            return f"# Source code for '{strategy_id}' not found."

        try:
            with open(file_path, encoding='utf-8') as f:
                lines = f.readlines()
                # Return a snippet, not the whole file
                return "".join(lines[:self.max_code_lines])
        except FileNotFoundError:
            return f"# File not found: {file_path}"
        except Exception as e:
            return f"# Error reading file {file_path}: {e}"

    def build_context(self) -> SelfAwarenessContext:
        """Constructs the full self-awareness context."""
        import time

        # 1. Get market regime from Physics engine
        physics_state = self.trading_system.physics_engine.get_latest_state()

        # 2. Get overall system performance from Portfolio
        portfolio = self.trading_system.portfolio
        stats = portfolio.get_statistics() # Assuming this method exists

        system_perf = SystemPerformanceMetrics(
            total_pnl=stats.get('total_pnl', 0.0),
            roi_pct=stats.get('roi_pct', 0.0),
            sharpe_ratio=stats.get('sharpe_ratio', 0.0),
            max_drawdown_pct=stats.get('max_drawdown_pct', 0.0),
            win_rate=stats.get('win_rate', 0.0),
            active_strategies=len(portfolio.get_active_strategies()),
        )

        # 3. Get individual strategy performance
        strategy_perf = self.trading_system.strategy_allocator.get_performance_metrics()

        # 4. Get code of the worst performing strategy
        worst_strategy = self._get_worst_performing_strategy()
        worst_strategy_code = None
        if worst_strategy:
            worst_strategy_code = self._read_strategy_source_code(worst_strategy.strategy_id)

        # 5. Get recent backtest results from Symbiont X (placeholder)
        # In a real system, this would be fetched from a shared store where
        # the evolution runner saves its results.
        symbiont_results = []

        # 6. Get recent critical errors (placeholder)
        recent_errors = self.trading_system.health_monitor.get_recent_critical_errors(limit=5)

        return SelfAwarenessContext(
            timestamp=time.time(),
            market_regime=physics_state,
            system_performance=system_perf,
            strategy_performance=strategy_perf,
            worst_performing_strategy_code=worst_strategy_code,
            symbiont_backtest_results=symbiont_results,
            recent_errors=[str(e) for e in recent_errors],
        )
