"""No-trade / signal block reporting utilities.

Tracks reasons why signals are blocked so that backtests and evaluations
can report where opportunity is being filtered out.

Extended with comprehensive tracing counters for "Attempt → Order → Fill" pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from hean.logging import get_logger

logger = get_logger(__name__)


# Canonical reason keys expected by requirements. We don't strictly enforce
# the set, but documenting them here keeps usage consistent.
NO_TRADE_REASONS = {
    "risk_limits_reject",
    "daily_attempts_reject",
    "cooldown_reject",
    "edge_reject",
    "filter_reject",
    "regime_gating_reject",
    "maker_edge_reject",
    "spread_reject",
    "volatility_soft_penalty",
    "volatility_hard_reject",
    "protection_block",
    "decision_memory_block",
}

# Extended counters for detailed pipeline tracing
PIPELINE_COUNTERS = {
    "signals_emitted",
    "signals_rejected_risk",
    "signals_rejected_daily_attempts",
    "signals_rejected_cooldown",
    "signals_blocked_decision_memory",
    "signals_blocked_protection",
    "execution_soft_vol_blocks",
    "execution_hard_vol_blocks",
    "orders_created",
    "maker_orders_placed",
    "maker_orders_cancelled_ttl",
    "maker_orders_filled",
    "taker_orders_placed",
    "taker_orders_filled",
    "positions_opened",
    "positions_closed",
}


@dataclass
class NoTradeSummary:
    """Summary DTO returned by NoTradeReport.get_summary()."""

    totals: dict[str, int]
    per_strategy: dict[str, dict[str, int]]
    per_symbol: dict[str, dict[str, int]]
    pipeline_counters: dict[str, int]
    pipeline_per_strategy: dict[str, dict[str, int]]


class NoTradeReport:
    """Aggregates reasons why signals are blocked / not traded.

    This is intentionally lightweight and in-memory only. For each increment we
    track:
        - global totals per reason
        - per-strategy per-reason counts
        - per-symbol per-reason counts
        - pipeline counters (signals_emitted, orders_created, etc.)
        - pipeline counters per strategy

    The same instance is reused across a process; run-level helpers (backtest /
    evaluation) should call reset() before starting a new run to avoid
    cross-run leakage in tests.
    """

    def __init__(self) -> None:
        self._totals: defaultdict[str, int] = defaultdict(int)
        self._per_strategy: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._per_symbol: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # Pipeline counters for detailed tracing
        self._pipeline_counters: defaultdict[str, int] = defaultdict(int)
        self._pipeline_per_strategy: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def reset(self) -> None:
        """Reset all counters.

        Useful for tests or when running multiple evaluations in a single
        process. CLI entrypoints typically run a single backtest/evaluation per
        process so this is primarily for tests.
        """
        self._totals.clear()
        self._per_strategy.clear()
        self._per_symbol.clear()
        self._pipeline_counters.clear()
        self._pipeline_per_strategy.clear()

    def increment(self, reason: str, symbol: str | None, strategy_id: str | None) -> None:
        """Increment counter for a blocked signal.

        Args:
            reason: Canonical reason string (see NO_TRADE_REASONS).
            symbol: Trading symbol, if known.
            strategy_id: Strategy identifier, if known.
        """
        if not reason:
            return

        self._totals[reason] += 1

        if strategy_id:
            self._per_strategy[strategy_id][reason] += 1

        if symbol:
            self._per_symbol[symbol][reason] += 1

    def increment_pipeline(self, counter: str, strategy_id: str | None = None) -> None:
        """Increment a pipeline counter.

        Args:
            counter: Counter name (see PIPELINE_COUNTERS).
            strategy_id: Strategy identifier, if known.
        """
        if not counter:
            return

        self._pipeline_counters[counter] += 1

        if strategy_id:
            self._pipeline_per_strategy[strategy_id][counter] += 1

    def get_summary(self) -> NoTradeSummary:
        """Return aggregated counters for reporting.

        Returns:
            NoTradeSummary with:
                - totals: global counts per reason
                - per_strategy: mapping strategy_id -> {reason -> count}
                - per_symbol: mapping symbol -> {reason -> count}
                - pipeline_counters: global pipeline counters
                - pipeline_per_strategy: mapping strategy_id -> {counter -> count}
        """
        # Convert defaultdicts to regular dicts for a clean API surface
        totals = dict(self._totals)
        per_strategy: dict[str, dict[str, int]] = {
            strat: dict(reasons) for strat, reasons in self._per_strategy.items()
        }
        per_symbol: dict[str, dict[str, int]] = {
            sym: dict(reasons) for sym, reasons in self._per_symbol.items()
        }
        pipeline_counters = dict(self._pipeline_counters)
        pipeline_per_strategy: dict[str, dict[str, int]] = {
            strat: dict(counters) for strat, counters in self._pipeline_per_strategy.items()
        }

        return NoTradeSummary(
            totals=totals,
            per_strategy=per_strategy,
            per_symbol=per_symbol,
            pipeline_counters=pipeline_counters,
            pipeline_per_strategy=pipeline_per_strategy,
        )


# Global singleton used across the system.
no_trade_report = NoTradeReport()
