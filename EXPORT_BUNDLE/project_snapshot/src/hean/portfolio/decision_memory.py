"""Decision memory for context-aware trade blocking and penalties.

This layer tracks per-strategy, per-context performance and applies
deterministic, explainable penalties to avoid repeating bad decisions.

Context key:
    (regime, spread_bucket, volatility_bucket, hour_bucket)

Rolling statistics per (strategy_id, context):
    - trades_count
    - wins
    - losses
    - pnl_sum
    - max_drawdown_pct (over last N trades in this context)
    - loss_streak

API:
    - record_trade(position, regime, spread_bps, volatility, timestamp)
    - context_score(strategy_id, context) -> [-1.0, +1.0]
    - is_context_blocked(strategy_id, context) -> bool
    - penalty_multiplier(strategy_id, context) -> [0.0, 1.0]

Rules:
    - PF < 1.0 over last N trades → negative score
    - loss_streak >= K → temporary block
    - drawdown > threshold → cooldown (block)
    - blocks expire after configurable time
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Hashable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from hean.config import settings
from hean.core.regime import Regime
from hean.core.types import Position
from hean.logging import get_logger
from hean.observability.metrics import metrics

logger = get_logger(__name__)


ContextKey = tuple[str, str, str, str]


@dataclass
class ContextStats:
    """Rolling statistics for a single (strategy, context) pair."""

    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    pnl_sum: float = 0.0
    max_drawdown_pct: float = 0.0
    loss_streak: int = 0
    last_trade_at: datetime | None = None
    block_until: datetime | None = None
    # Rolling last-N trade PnLs for PF / drawdown computation
    pnl_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=settings.memory_window_trades)
    )


class DecisionMemory:
    """Context-aware decision memory.

    This module is intentionally deterministic and simple:
    it does not change risk limits, only *reduces* position size
    or temporarily blocks trading in clearly bad contexts.
    """

    def __init__(self) -> None:
        # (strategy_id, context_key) -> ContextStats
        self._stats: dict[tuple[str, ContextKey], ContextStats] = defaultdict(ContextStats)
        # Track penalty values for metrics
        self._penalty_sum: float = 0.0
        self._penalty_count: int = 0

    # ------------------------------------------------------------------
    # Context handling
    # ------------------------------------------------------------------
    @staticmethod
    def _bucket_spread(spread_bps: float | None) -> str:
        """Bucket spread into coarse bands (in bps).

        Buckets are 0–5, 5–10, 10–20, 20–50, 50+.
        """
        if spread_bps is None:
            return "spread:unknown"

        spread = max(0.0, float(spread_bps))
        if spread < 5:
            return "spread:0-5"
        if spread < 10:
            return "spread:5-10"
        if spread < 20:
            return "spread:10-20"
        if spread < 50:
            return "spread:20-50"
        return "spread:50+"

    @staticmethod
    def _bucket_volatility(volatility: float | None) -> str:
        """Bucket volatility into coarse bands.

        Uses simple absolute volatility bands:
            < 0.5%  → low
            0.5–1%  → medium
            1–2%    → high
            > 2%    → extreme
        """
        if volatility is None:
            return "vol:unknown"

        v = abs(float(volatility))
        if v < 0.005:
            return "vol:low"
        if v < 0.01:
            return "vol:med"
        if v < 0.02:
            return "vol:high"
        return "vol:extreme"

    @staticmethod
    def _bucket_hour(timestamp: datetime | None) -> str:
        """Bucket hour-of-day (UTC) into 4-hour windows."""
        if timestamp is None:
            return "hour:unknown"

        hour = timestamp.hour
        bucket = (hour // 4) * 4
        return f"hour:{bucket:02d}-{(bucket + 4) % 24:02d}"

    @staticmethod
    def _normalize_regime(regime: Regime | str | None) -> str:
        if regime is None:
            return "regime:unknown"
        if isinstance(regime, Regime):
            return f"regime:{regime.value}"
        return f"regime:{str(regime)}"

    def build_context(
        self,
        regime: Regime | str | None,
        spread_bps: float | None,
        volatility: float | None,
        timestamp: datetime | None,
    ) -> ContextKey:
        """Build a normalized context key from raw features."""
        return (
            self._normalize_regime(regime),
            self._bucket_spread(spread_bps),
            self._bucket_volatility(volatility),
            self._bucket_hour(timestamp),
        )

    # ------------------------------------------------------------------
    # Core recording
    # ------------------------------------------------------------------
    def record_trade(
        self,
        position: Position,
        regime: Regime,
        spread_bps: float | None,
        volatility: float | None,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a completed trade into decision memory.

        Args:
            position: Closed position with realized_pnl and strategy_id.
            regime: Regime at entry/exit (we use most recent known).
            spread_bps: Spread at entry (approximate).
            volatility: Volatility at entry (approximate).
            timestamp: Close timestamp (defaults to now).
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        strategy_id = position.strategy_id
        context = self.build_context(regime, spread_bps, volatility, timestamp)
        key = (strategy_id, context)
        stats = self._stats[key]

        pnl = float(position.realized_pnl)
        stats.trades_count += 1
        stats.pnl_sum += pnl
        stats.pnl_history.append(pnl)
        stats.last_trade_at = timestamp

        if pnl > 0:
            stats.wins += 1
            stats.loss_streak = 0
        elif pnl < 0:
            stats.losses += 1
            stats.loss_streak += 1

        # Recompute max_drawdown_pct over rolling history to stay local
        if stats.pnl_history:
            cum = 0.0
            peak = 0.0
            max_dd_pct = 0.0
            for p in stats.pnl_history:
                cum += p
                if cum > peak:
                    peak = cum
                drawdown = peak - cum
                if peak > 0:
                    dd_pct = (drawdown / peak) * 100.0
                    if dd_pct > max_dd_pct:
                        max_dd_pct = dd_pct
            stats.max_drawdown_pct = max_dd_pct

        # Apply blocking rules
        now = datetime.utcnow()
        block = False
        if stats.loss_streak >= settings.memory_loss_streak:
            block = True
            logger.warning(
                "DecisionMemory: loss streak %s >= %s for (%s, %s) – blocking context",
                stats.loss_streak,
                settings.memory_loss_streak,
                strategy_id,
                context,
            )

        if stats.max_drawdown_pct > settings.memory_drawdown_threshold_pct:
            block = True
            logger.warning(
                "DecisionMemory: drawdown %.2f%% > %.2f%% for (%s, %s) – cooldown",
                stats.max_drawdown_pct,
                settings.memory_drawdown_threshold_pct,
                strategy_id,
                context,
            )

        if block:
            stats.block_until = now + timedelta(hours=settings.memory_block_hours)
            metrics.increment("memory_blocks_total")

    # ------------------------------------------------------------------
    # Scoring / control
    # ------------------------------------------------------------------
    def _get_stats(self, strategy_id: str, context: Hashable) -> ContextStats | None:
        if not isinstance(context, tuple) or len(context) != 4:
            # Defensive: if caller passes raw tuple/dict, we can't safely index.
            return None
        key = (strategy_id, context)  # type: ignore[arg-type]
        return self._stats.get(key)

    def context_score(self, strategy_id: str, context: ContextKey) -> float:
        """Compute a context score in [-1.0, +1.0].

        Heuristics (all deterministic):
            - PF < 1.0 → negative contribution
            - loss_streak → negative contribution
            - drawdown_pct → negative contribution

        Positive PF can provide a mild positive bias, but we never
        *increase* risk based on this score – it is only used to
        scale DOWN size (see penalty_multiplier).
        """
        stats = self._get_stats(strategy_id, context)
        if not stats or stats.trades_count == 0:
            return 0.0

        # Profit factor over local history
        wins_sum = sum(p for p in stats.pnl_history if p > 0)
        losses_sum = abs(sum(p for p in stats.pnl_history if p < 0))
        if losses_sum == 0:
            pf = wins_sum if wins_sum > 0 else 1.0
        else:
            pf = wins_sum / losses_sum

        score = 0.0

        # PF contribution: map PF<1 into [-0.5, 0), PF>1 into (0, +0.5]
        if pf >= 1.0:
            # pf in [1, 3] → score in [0, 0.5]
            score += min(0.5, 0.25 * (pf - 1.0))
        else:
            # pf in [0, 1) → score in [-0.5, 0)
            score -= min(0.5, 0.5 * (1.0 - pf))

        # Drawdown contribution (penalize up to -0.3)
        if stats.max_drawdown_pct > 0:
            dd_norm = min(1.0, stats.max_drawdown_pct / settings.memory_drawdown_threshold_pct)
            score -= 0.3 * dd_norm

        # Loss streak contribution (penalize up to -0.3)
        if stats.loss_streak > 0:
            ls_norm = min(1.0, stats.loss_streak / settings.memory_loss_streak)
            score -= 0.3 * ls_norm

        # Clamp to [-1, 1]
        if score > 1.0:
            score = 1.0
        elif score < -1.0:
            score = -1.0

        return score

    def is_context_blocked(self, strategy_id: str, context: ContextKey) -> bool:
        """Return True if this context is currently blocked for the strategy."""
        stats = self._get_stats(strategy_id, context)
        if not stats or stats.block_until is None:
            return False

        now = datetime.utcnow()
        if now >= stats.block_until:
            # Expired block – clean up
            stats.block_until = None
            return False

        return True

    def penalty_multiplier(self, strategy_id: str, context: ContextKey) -> float:
        """Return a size penalty multiplier in [0.0, 1.0].

        Important invariants:
            - We NEVER increase size above baseline (no >1.0 multipliers).
            - Strongly negative scores can push size close to zero,
              but we keep a small non-zero floor to allow recovery
              unless the context is explicitly blocked.
        """
        if self.is_context_blocked(strategy_id, context):
            return 0.0

        score = self.context_score(strategy_id, context)

        if score >= 0.0:
            # Never boost size based on positive context – only reduce.
            return 1.0

        # Map score in [-1, 0) to multiplier in [min_floor, 1.0)
        min_floor = 0.1
        multiplier = 1.0 + score  # score=-1 → 0, score=0 → 1
        if multiplier < min_floor:
            multiplier = min_floor
        if multiplier > 1.0:
            multiplier = 1.0
        return multiplier

    # ------------------------------------------------------------------
    # New Phase 2 API methods (requested API)
    # ------------------------------------------------------------------
    def record_close(
        self,
        strategy_id: str,
        context_key: ContextKey,
        pnl: float,
        timestamp: datetime,
    ) -> None:
        """Record a position close with PnL for a given strategy and context.

        Args:
            strategy_id: Strategy identifier
            context_key: Context key tuple (regime, spread_bucket, vol_bucket, hour_bucket)
            pnl: Realized PnL from the closed position
            timestamp: Close timestamp
        """
        key = (strategy_id, context_key)
        stats = self._stats[key]

        stats.trades_count += 1
        stats.pnl_sum += pnl
        stats.pnl_history.append(pnl)
        stats.last_trade_at = timestamp

        if pnl > 0:
            stats.wins += 1
            stats.loss_streak = 0
        elif pnl < 0:
            stats.losses += 1
            stats.loss_streak += 1

        # Recompute max_drawdown_pct over rolling history
        if stats.pnl_history:
            cum = 0.0
            peak = 0.0
            max_dd_pct = 0.0
            for p in stats.pnl_history:
                cum += p
                if cum > peak:
                    peak = cum
                drawdown = peak - cum
                if peak > 0:
                    dd_pct = (drawdown / peak) * 100.0
                    if dd_pct > max_dd_pct:
                        max_dd_pct = dd_pct
            stats.max_drawdown_pct = max_dd_pct

        # Apply blocking rules
        now = datetime.utcnow()
        block = False
        if stats.loss_streak >= settings.memory_loss_streak:
            block = True
            logger.warning(
                "DecisionMemory: loss streak %s >= %s for (%s, %s) – blocking context",
                stats.loss_streak,
                settings.memory_loss_streak,
                strategy_id,
                context_key,
            )

        if stats.max_drawdown_pct > settings.memory_drawdown_threshold_pct:
            block = True
            logger.warning(
                "DecisionMemory: drawdown %.2f%% > %.2f%% for (%s, %s) – cooldown",
                stats.max_drawdown_pct,
                settings.memory_drawdown_threshold_pct,
                strategy_id,
                context_key,
            )

        if block:
            stats.block_until = now + timedelta(hours=settings.memory_block_hours)
            metrics.increment("memory_blocks_total")

    def penalty(self, strategy_id: str, context_key: ContextKey) -> float:
        """Get penalty multiplier for a strategy and context.

        Returns a multiplier in [0.0, 1.0] that should be applied to position size.
        Returns 0.0 if context is blocked, otherwise returns a value based on
        historical performance in this context.

        Args:
            strategy_id: Strategy identifier
            context_key: Context key tuple (regime, spread_bucket, vol_bucket, hour_bucket)

        Returns:
            Penalty multiplier in [0.0, 1.0]
        """
        multiplier = self.penalty_multiplier(strategy_id, context_key)

        # Track for metrics
        self._penalty_sum += multiplier
        self._penalty_count += 1
        if self._penalty_count > 0:
            avg_penalty = self._penalty_sum / self._penalty_count
            metrics.set_gauge("memory_penalty_avg", avg_penalty)

        return multiplier

    def blocked(self, strategy_id: str, context_key: ContextKey) -> bool:
        """Check if a context is currently blocked for a strategy.

        Blocks expire after a configurable time (memory_block_hours).

        Args:
            strategy_id: Strategy identifier
            context_key: Context key tuple (regime, spread_bucket, vol_bucket, hour_bucket)

        Returns:
            True if context is blocked, False otherwise
        """
        return self.is_context_blocked(strategy_id, context_key)
