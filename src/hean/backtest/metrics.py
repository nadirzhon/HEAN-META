"""Backtest metrics calculation."""

import json
from pathlib import Path
from typing import Any

from hean.core.types import EquitySnapshot, Order, Position
from hean.logging import get_logger
from hean.portfolio.accounting import PortfolioAccounting
from hean.portfolio.allocator import CapitalAllocator
from hean.strategies.base import BaseStrategy

logger = get_logger(__name__)


class BacktestMetrics:
    """Calculates backtest performance metrics."""

    def __init__(
        self,
        accounting: PortfolioAccounting | None = None,
        paper_broker=None,
        strategies: dict[str, BaseStrategy] | None = None,
        allocator: CapitalAllocator | None = None,
        execution_router=None,
    ) -> None:
        """Initialize metrics calculator.

        Args:
            accounting: Portfolio accounting instance for PnL tracking
            paper_broker: Paper broker instance for execution metrics
            strategies: Dictionary mapping strategy_id to Strategy instance
            allocator: Capital allocator for weight history
            execution_router: Execution router instance for execution diagnostics
        """
        self._equity_history: list[EquitySnapshot] = []
        self._orders: list[Order] = []
        self._positions: list[Position] = []
        self._accounting = accounting
        self._paper_broker = paper_broker
        self._strategies: dict[str, BaseStrategy] = strategies or {}
        self._allocator = allocator
        self._execution_router = execution_router

    def record_equity(self, snapshot: EquitySnapshot) -> None:
        """Record an equity snapshot."""
        self._equity_history.append(snapshot)

    def record_order(self, order: Order) -> None:
        """Record an order."""
        self._orders.append(order)

    def record_position(self, position: Position) -> None:
        """Record a position."""
        self._positions.append(position)

    def calculate(self) -> dict[str, Any]:
        """Calculate all metrics."""
        logger.info(
            f"[DEBUG_METRICS] calculate() called, paper_broker={self._paper_broker is not None}, execution_router={self._execution_router is not None}, equity_history_len={len(self._equity_history)}"
        )
        metrics: dict[str, Any] = {}

        # CRITICAL: Calculate total_trades even if no equity history
        # Use paper broker as primary source
        total_trades = 0
        if self._paper_broker:
            try:
                fill_stats = self._paper_broker.get_fill_stats()
                total_trades = fill_stats.get("total_fills", 0)
                logger.info(f"[DEBUG_METRICS] Total trades from paper broker: {total_trades}")
            except Exception as e:
                logger.error(f"[DEBUG_METRICS] Error getting fill stats: {e}", exc_info=True)

        if self._equity_history:
            initial_equity = self._equity_history[0].equity
            final_equity = self._equity_history[-1].equity
            total_return = (final_equity - initial_equity) / initial_equity * 100

            # Calculate max drawdown
            peak = initial_equity
            max_dd = 0.0
            max_dd_pct = 0.0
            for snapshot in self._equity_history:
                if snapshot.equity > peak:
                    peak = snapshot.equity
                dd = peak - snapshot.equity
                dd_pct = (dd / peak * 100) if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
                    max_dd_pct = dd_pct

            # Calculate win rate and total trades
            # CRITICAL FIX: Use multiple sources to count trades, prioritize paper_broker
            filled_orders = []

            # Method 1: Try paper broker first (most reliable source)
            if self._paper_broker:
                try:
                    fill_stats = self._paper_broker.get_fill_stats()
                    total_fills = fill_stats.get("total_fills", 0)
                    logger.info(
                        f"[DEBUG_METRICS] Paper broker fill stats: {fill_stats}, total_fills={total_fills}"
                    )
                    if total_fills > 0:
                        logger.info(
                            f"[DEBUG_METRICS] Using {total_fills} fills from paper broker for total_trades"
                        )
                        filled_orders = [None] * total_fills
                except Exception as e:
                    logger.error(
                        f"[DEBUG_METRICS] Error getting fill stats from paper broker: {e}",
                        exc_info=True,
                    )

            # Method 2: Try OrderManager if paper broker didn't work
            if (
                len(filled_orders) == 0
                and self._execution_router
                and hasattr(self._execution_router, "_order_manager")
            ):
                filled_orders = self._execution_router._order_manager.get_filled_orders()
                logger.info(f"[DEBUG] Found {len(filled_orders)} filled orders from OrderManager")

            # Method 3: Try accounting (count closed positions)
            if len(filled_orders) == 0 and self._accounting:
                # Count trades from accounting - use strategy metrics
                strategy_metrics = self._accounting.get_strategy_metrics()
                if strategy_metrics:
                    total_trades_from_accounting = sum(
                        m.get("trades", 0) for m in strategy_metrics.values()
                    )
                    logger.info(
                        f"[DEBUG] Found {total_trades_from_accounting} trades from accounting strategy_metrics"
                    )
                    if total_trades_from_accounting > 0:
                        filled_orders = [None] * int(total_trades_from_accounting)

            # Method 4: Fallback to self._orders
            if len(filled_orders) == 0:
                from hean.core.types import OrderStatus

                filled_orders = [
                    o
                    for o in self._orders
                    if hasattr(o, "status")
                    and (
                        o.status == OrderStatus.FILLED
                        or str(o.status).lower() == "filled"
                        or (hasattr(o.status, "value") and o.status.value == "filled")
                    )
                ]
                logger.info(f"[DEBUG] Found {len(filled_orders)} filled orders from self._orders")

            if filled_orders:
                # Simplified: count profitable vs unprofitable
                # In real system, would track PnL per order
                wins = len([o for o in filled_orders if o.avg_fill_price])  # Placeholder
                win_rate = (wins / len(filled_orders) * 100) if filled_orders else 0.0
            else:
                win_rate = 0.0

            # Calculate profit factor from accounting if available
            profit_factor = 1.0  # Default
            if self._accounting:
                strategy_metrics = self._accounting.get_strategy_metrics()
                if strategy_metrics:
                    # Aggregate profit factor across strategies (weighted by trades)
                    total_wins = 0.0
                    total_losses = 0.0
                    for strat_metrics in strategy_metrics.values():
                        wins = strat_metrics.get("wins", 0)
                        losses = strat_metrics.get("losses", 0)
                        total_wins += wins
                        total_losses += losses

                    if total_losses > 0:
                        profit_factor = total_wins / total_losses
                    elif total_wins > 0:
                        profit_factor = total_wins
                    else:
                        profit_factor = 1.0
                else:
                    # Fallback: calculate from PnL if available
                    if hasattr(self._accounting, "_strategy_pnl"):
                        total_pnl = sum(self._accounting._strategy_pnl.values())
                        if total_pnl > 0:
                            profit_factor = 1.5  # Simplified estimate
                        else:
                            profit_factor = 0.8  # Simplified estimate

            # Calculate expectancy (simplified)
            expectancy = 0.0  # Placeholder

            # Calculate Sharpe ratio (simplified, would need returns series)
            sharpe_ratio = 0.0  # Placeholder

            metrics.update(
                {
                    "initial_equity": initial_equity,
                    "final_equity": final_equity,
                    "total_return_pct": total_return,
                    "max_drawdown": max_dd,
                    "max_drawdown_pct": max_dd_pct,
                    "win_rate_pct": win_rate,
                    "profit_factor": profit_factor,
                    "expectancy": expectancy,
                    "sharpe_ratio": sharpe_ratio,
                    "total_trades": max(
                        len(filled_orders), total_trades
                    ),  # Use max of both sources
                    "total_positions": len(self._positions),
                }
            )

        # If no equity history, still set total_trades
        if not self._equity_history:
            metrics["total_trades"] = total_trades
            logger.info(f"[DEBUG_METRICS] Set total_trades={total_trades} (no equity history)")

        # Add execution metrics if paper broker is available
        execution_metrics: dict[str, Any] = {}
        if self._paper_broker:
            fill_stats = self._paper_broker.get_fill_stats()
            execution_metrics.update(fill_stats)

        # Add execution diagnostics if router is available
        if self._execution_router:
            diagnostics = self._execution_router.get_diagnostics()
            diagnostics_snapshot = diagnostics.snapshot()
            execution_metrics.update(
                {
                    "maker_fill_rate": diagnostics_snapshot.get("maker_fill_rate", 0.0),
                    "avg_time_to_fill_ms": diagnostics_snapshot.get("avg_time_to_fill_ms", 0.0),
                    "volatility_rejection_rate": diagnostics_snapshot.get(
                        "volatility_rejection_rate", 0.0
                    ),
                    "volatility_rejections_soft": diagnostics_snapshot.get(
                        "volatility_rejections_soft", 0.0
                    ),
                    "volatility_rejections_hard": diagnostics_snapshot.get(
                        "volatility_rejections_hard", 0.0
                    ),
                    "maker_attempted": diagnostics_snapshot.get("maker_attempted", 0.0),
                    "maker_filled": diagnostics_snapshot.get("maker_filled", 0.0),
                    "maker_expired": diagnostics_snapshot.get("maker_expired", 0.0),
                }
            )

            # Add retry queue metrics
            retry_queue = self._execution_router.get_retry_queue()
            execution_metrics.update(
                {
                    "retry_success_rate": retry_queue.get_retry_success_rate(),
                    "retry_queue_size": retry_queue.get_queue_size(),
                }
            )

        if execution_metrics:
            metrics["execution"] = execution_metrics

        # Add per-strategy metrics if accounting is available
        if self._accounting:
            strategy_metrics = self._accounting.get_strategy_metrics()
            if strategy_metrics:
                # Add strategy-specific extended metrics if available
                for strategy_id, strategy in self._strategies.items():
                    if strategy_id in strategy_metrics:
                        # Check if strategy has get_metrics method
                        if hasattr(strategy, "get_metrics"):
                            try:
                                extended_metrics = strategy.get_metrics()
                                if extended_metrics:
                                    # Store with strategy_id as key for extended metrics
                                    strategy_metrics[strategy_id][f"{strategy_id}_metrics"] = (
                                        extended_metrics
                                    )
                            except Exception as e:
                                logger.warning(f"Failed to get metrics from {strategy_id}: {e}")

                metrics["strategies"] = strategy_metrics

        # Add weight history if available
        if self._allocator:
            weight_history = self._allocator.get_weight_history()
            if weight_history:
                metrics["weight_history"] = weight_history

        return metrics

    def print_report(self, metrics: dict[str, Any] | None = None) -> None:
        """Print a formatted backtest report.

        Args:
            metrics: Optional pre-calculated metrics dict. If None, will calculate internally.
        """
        if metrics is None:
            metrics = self.calculate()

        print("\n" + "=" * 80)
        print("BACKTEST REPORT")
        print("=" * 80)
        print(f"Initial Equity:     ${metrics.get('initial_equity', 0):,.2f}")
        print(f"Final Equity:       ${metrics.get('final_equity', 0):,.2f}")
        print(f"Total Return:       {metrics.get('total_return_pct', 0):.2f}%")
        print(
            f"Max Drawdown:       ${metrics.get('max_drawdown', 0):,.2f} ({metrics.get('max_drawdown_pct', 0):.2f}%)"
        )
        print(f"Win Rate:           {metrics.get('win_rate_pct', 0):.2f}%")
        print(f"Total Trades:       {metrics.get('total_trades', 0)}")
        print(f"Total Positions:    {metrics.get('total_positions', 0)}")

        # Print NoTradeReport summary if no trades
        if metrics.get("total_trades", 0) == 0:
            from hean.observability.no_trade_report import no_trade_report

            summary = no_trade_report.get_summary()
            print("\n" + "-" * 80)
            print("NO-TRADE DIAGNOSIS (0 trades detected)")
            print("-" * 80)
            if summary.pipeline_counters:
                print("Pipeline Counters:")
                for counter, count in sorted(summary.pipeline_counters.items()):
                    print(f"  {counter}: {count}")
            if summary.pipeline_per_strategy:
                print("\nPipeline Counters per Strategy:")
                for strategy_id, counters in summary.pipeline_per_strategy.items():
                    print(f"  {strategy_id}:")
                    for counter, count in sorted(counters.items()):
                        print(f"    {counter}: {count}")
            if summary.totals:
                print("\nBlock Reasons (totals):")
                for reason, count in sorted(summary.totals.items()):
                    print(f"  {reason}: {count}")
            print("-" * 80)
        else:
            # Even with trades, show pipeline counters for debugging
            from hean.observability.no_trade_report import no_trade_report

            summary = no_trade_report.get_summary()
            if summary.pipeline_counters:
                print("\n" + "-" * 80)
                print("PIPELINE TRACE")
                print("-" * 80)
                print("Pipeline Counters:")
                for counter, count in sorted(summary.pipeline_counters.items()):
                    print(f"  {counter}: {count}")
                if summary.pipeline_per_strategy:
                    print("\nPipeline Counters per Strategy:")
                    for strategy_id, counters in summary.pipeline_per_strategy.items():
                        print(f"  {strategy_id}:")
                        for counter, count in sorted(counters.items()):
                            print(f"    {counter}: {count}")
                print("-" * 80)

        # Print execution metrics
        if "execution" in metrics:
            exec_metrics = metrics["execution"]
            print("\nExecution:")
            print(f"  Maker Fills:              {exec_metrics.get('maker_fills', 0)}")
            print(f"  Taker Fills:              {exec_metrics.get('taker_fills', 0)}")
            print(
                f"  Maker Fill Rate:          {exec_metrics.get('maker_fill_rate', exec_metrics.get('maker_fill_rate_pct', 0)):.2f}%"
            )
            print(
                f"  Avg Time To Fill:          {exec_metrics.get('avg_time_to_fill_ms', 0):.2f} ms"
            )
            print(
                f"  Soft Volatility Blocks:   {int(exec_metrics.get('volatility_rejections_soft', 0))}"
            )
            print(
                f"  Hard Volatility Blocks:   {int(exec_metrics.get('volatility_rejections_hard', 0))}"
            )
            print(f"  Retry Success Rate:       {exec_metrics.get('retry_success_rate', 0):.2f}%")

        # Print per-strategy table
        if "strategies" in metrics and metrics["strategies"]:
            print("\n" + "-" * 80)
            print("PER-STRATEGY PERFORMANCE")
            print("-" * 80)
            print(
                f"{'Strategy':<20} {'Return%':<12} {'Trades':<10} {'WinRate%':<12} {'PF':<10} {'MaxDD%':<10}"
            )
            print("-" * 80)

            for strategy_id, strat_metrics in metrics["strategies"].items():
                print(
                    f"{strategy_id:<20} "
                    f"{strat_metrics['return_pct']:>10.2f}% "
                    f"{int(strat_metrics['trades']):>10} "
                    f"{strat_metrics['win_rate_pct']:>10.2f}% "
                    f"{strat_metrics['profit_factor']:>10.2f} "
                    f"{strat_metrics['max_drawdown_pct']:>10.2f}%"
                )

                # Print per-regime PnL if available
                if "regime_pnl" in strat_metrics and strat_metrics["regime_pnl"]:
                    for regime, pnl in strat_metrics["regime_pnl"].items():
                        print(f"  └─ {regime}: ${pnl:,.2f}")

                # Print strategy-specific extended metrics if available
                # Look for metrics with pattern "{strategy_id}_metrics"
                extended_metrics_key = f"{strategy_id}_metrics"
                if extended_metrics_key in strat_metrics:
                    extended_metrics = strat_metrics[extended_metrics_key]
                    # Impulse engine specific metrics
                    if strategy_id == "impulse_engine":
                        if "avg_time_in_trade_sec" in extended_metrics:
                            print(
                                f"  └─ Avg Time in Trade: {extended_metrics.get('avg_time_in_trade_sec', 0):.1f}s"
                            )
                        if "be_stop_hit_rate_pct" in extended_metrics:
                            print(
                                f"  └─ BE Stop Hit Rate: {extended_metrics.get('be_stop_hit_rate_pct', 0):.2f}%"
                            )
                    # Other strategies can add their own extended metrics display here

            print("-" * 80)

        # Print weight history if available
        if self._allocator:
            weight_history = self._allocator.get_weight_history()
            if weight_history:
                print("\n" + "-" * 80)
                print("CAPITAL ALLOCATION WEIGHT HISTORY")
                print("-" * 80)
                # Print first, last, and key points
                print(f"Initial weights: {weight_history[0]}")
                if len(weight_history) > 1:
                    print(f"Final weights: {weight_history[-1]}")
                # Print summary of weight changes
                if len(weight_history) > 2:
                    strategies = [k for k in weight_history[0].keys() if k != "_date"]
                    print("\nWeight evolution:")
                    for strategy_id in strategies:
                        initial = weight_history[0].get(strategy_id, 0.0)
                        final = weight_history[-1].get(strategy_id, 0.0)
                        change = final - initial
                        print(f"  {strategy_id}: {initial:.1%} → {final:.1%} ({change:+.1%})")
                print("-" * 80)

        print("=" * 80 + "\n")

    def save_json(self, filepath: str) -> None:
        """Save metrics to JSON file."""
        metrics = self.calculate()

        # Ensure directory exists
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"Metrics saved to {filepath}")
