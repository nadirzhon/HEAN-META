"""Active capital pressure layer for short‑term capital tilting.

This module sits *on top* of the slower, daily `CapitalAllocator` logic and
provides a short‑term, transient multiplier per strategy:

    - Inputs (per strategy):
        - Last N trade PnLs (short‑term profit factor / trend)
        - Recent drawdown observations
        - Optional context key from `DecisionMemory` (regime / spread / vol / hour)

    - Output:
        - A transient multiplier in [min_multiplier, max_multiplier] which is
          applied to the base adaptive weight *before* final normalization.

Design goals:
    - Deterministic, explainable rules (no black boxes).
    - React quickly to clusters of losses in the *same* context.
    - Provide a small, temporary boost when short‑term PF is strong and
      drawdown is low / stable.
    - Multipliers decay smoothly back toward 1.0 over time.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from statistics import fmean

from hean.config import settings
from hean.logging import get_logger
from hean.observability.metrics import metrics as system_metrics
from hean.portfolio.decision_memory import ContextKey

logger = get_logger(__name__)


@dataclass
class TradeSample:
    """Single trade sample used for short‑term PF and context tracking."""

    pnl: float
    context: ContextKey | None
    timestamp: datetime


@dataclass
class StrategyPressureState:
    """Per‑strategy transient pressure state."""

    # Rolling trade history (shared window with decision memory)
    trades: deque[TradeSample] = field(
        default_factory=lambda: deque(maxlen=settings.memory_window_trades)
    )
    # Recent drawdown observations (percentage values)
    drawdowns: deque[float] = field(default_factory=lambda: deque(maxlen=16))
    # Current capital multiplier (1.0 = neutral)
    multiplier: float = 1.0


class CapitalPressure:
    """Active capital pressure controller.

    This layer produces a *multiplier* per strategy that tilts capital
    allocation based on very recent performance signals while decaying
    back toward neutral over time.
    """

    def __init__(
        self,
        pf_window_trades: int = 15,
        boost_pct: float = 0.2,
        boost_pf_threshold: float = 1.2,
        max_multiplier: float = 1.3,
        min_multiplier: float = 0.3,
        cut_multiplier: float = 0.5,
        decay_rate: float = 0.3,
        dd_stable_tolerance: float = 0.5,
    ) -> None:
        """Initialize capital pressure parameters.

        Args:
            pf_window_trades: Number of most recent trades to use for short‑term PF.
            boost_pct: Size of boost when conditions are good (e.g. 0.2 = +20%).
            boost_pf_threshold: PF threshold above which we consider PF "strong".
            max_multiplier: Upper bound on pressure multiplier.
            min_multiplier: Lower bound on pressure multiplier (never fully disabled).
            cut_multiplier: Multiplier applied on immediate cut events
                (e.g. 0.5 halves capital after 2 losses in same context).
            decay_rate: Fraction of distance toward 1.0 that decays per update.
            dd_stable_tolerance: Maximum allowed *increase* in drawdown between
                observations to still consider drawdown "stable".
        """
        self._states: dict[str, StrategyPressureState] = defaultdict(StrategyPressureState)
        self._pf_window_trades = max(1, pf_window_trades)
        self._boost_pct = boost_pct
        self._boost_pf_threshold = boost_pf_threshold
        self._max_multiplier = max_multiplier
        self._min_multiplier = min_multiplier
        self._cut_multiplier = cut_multiplier
        self._decay_rate = decay_rate
        self._dd_stable_tolerance = dd_stable_tolerance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _state(self, strategy_id: str) -> StrategyPressureState:
        return self._states[strategy_id]

    def _short_term_pf(self, state: StrategyPressureState) -> float:
        """Compute short‑term PF from the last N trades."""
        if not state.trades:
            return 1.0

        recent = list(state.trades)[-self._pf_window_trades :]
        wins = sum(t.pnl for t in recent if t.pnl > 0)
        losses = abs(sum(t.pnl for t in recent if t.pnl < 0))

        if losses == 0:
            return wins if wins > 0 else 1.0
        return wins / losses

    def _apply_boost(self, strategy_id: str) -> None:
        state = self._state(strategy_id)
        before = state.multiplier
        new_multiplier = before * (1.0 + self._boost_pct)
        new_multiplier = max(self._min_multiplier, min(self._max_multiplier, new_multiplier))
        if new_multiplier > before:
            state.multiplier = new_multiplier
            system_metrics.increment("pressure_boost_events")
            logger.debug(
                "CapitalPressure boost for %s: %.3f -> %.3f",
                strategy_id,
                before,
                new_multiplier,
            )
            self._update_avg_metric()

    def _apply_cut(self, strategy_id: str) -> None:
        state = self._state(strategy_id)
        before = state.multiplier
        new_multiplier = before * self._cut_multiplier
        # Ensure we never go below the configured floor
        new_multiplier = max(self._min_multiplier, new_multiplier)
        if new_multiplier < before:
            state.multiplier = new_multiplier
            system_metrics.increment("pressure_cut_events")
            logger.warning(
                "CapitalPressure cut for %s after context losses: %.3f -> %.3f",
                strategy_id,
                before,
                new_multiplier,
            )
            self._update_avg_metric()

    def _update_avg_metric(self) -> None:
        """Update avg_pressure_multiplier gauge."""
        if not self._states:
            system_metrics.set_gauge("avg_pressure_multiplier", 1.0)
            return

        avg = fmean(state.multiplier for state in self._states.values())
        system_metrics.set_gauge("avg_pressure_multiplier", float(avg))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record_trade(
        self,
        strategy_id: str,
        pnl: float,
        context: ContextKey | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a completed trade for short‑term PF and context tracking.

        This method should be called from the execution / accounting layer
        whenever a trade is finalized.

        Implements the "2 losses in same context" rule:
            - If two *consecutive* losses occur in the same context,
              we immediately cut capital for that strategy (but do not
              fully disable it).
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        state = self._state(strategy_id)
        sample = TradeSample(pnl=float(pnl), context=context, timestamp=timestamp)
        state.trades.append(sample)

        # Check for consecutive losses in same context
        if sample.pnl < 0 and context is not None:
            losses_same_context = 0
            for t in reversed(state.trades):
                if t.context != context:
                    break
                if t.pnl < 0:
                    losses_same_context += 1
                else:
                    break

            if losses_same_context >= 2:
                self._apply_cut(strategy_id)

    def update_drawdown(self, strategy_id: str, drawdown_pct: float) -> None:
        """Update recent drawdown and apply boost logic when appropriate.

        If short‑term PF is strong AND drawdown is not accelerating upward,
        we apply a temporary capital boost.
        """
        state = self._state(strategy_id)
        drawdown_pct = float(drawdown_pct)

        prev_dd = state.drawdowns[-1] if state.drawdowns else None
        state.drawdowns.append(drawdown_pct)

        # Compute short‑term PF
        pf = self._short_term_pf(state)

        # Determine if drawdown is "stable"
        dd_change = drawdown_pct - prev_dd if prev_dd is not None else 0.0
        dd_stable = dd_change <= self._dd_stable_tolerance

        if pf >= self._boost_pf_threshold and dd_stable:
            self._apply_boost(strategy_id)

    def decay_all(self) -> None:
        """Decay all multipliers back toward 1.0.

        Called once per allocator rebalance (typically daily).
        """
        if not self._states:
            return

        for strategy_id, state in self._states.items():
            m = state.multiplier
            if abs(m - 1.0) < 1e-6:
                continue
            # Move a fraction of the distance toward 1.0
            new_m = m + (1.0 - m) * self._decay_rate
            # Keep within bounds
            new_m = max(self._min_multiplier, min(self._max_multiplier, new_m))
            state.multiplier = new_m
            logger.debug(
                "CapitalPressure decay for %s: %.3f -> %.3f",
                strategy_id,
                m,
                new_m,
            )

        self._update_avg_metric()

    def update_from_metrics(self, strategy_metrics: dict[str, dict[str, float]]) -> None:
        """Update internal state from per‑strategy metrics.

        This is intended to be called by `CapitalAllocator.update_weights`
        once per rebalance cycle.
        """
        if not strategy_metrics:
            return

        # First decay all multipliers toward neutral
        self.decay_all()

        # Then update drawdown observations per strategy
        for strategy_id, metrics in strategy_metrics.items():
            dd = float(metrics.get("max_drawdown_pct", 0.0))
            self.update_drawdown(strategy_id, dd)

        # Ensure avg metric is updated even if no boost/cut fired
        self._update_avg_metric()

    def get_multiplier(self, strategy_id: str) -> float:
        """Get current pressure multiplier for a strategy."""
        return self._state(strategy_id).multiplier

    def reset(self) -> None:
        """Reset all pressure state (primarily for tests)."""
        self._states.clear()
        self._update_avg_metric()
