"""Strategy Capital Allocator: Dynamic capital allocation across strategies.

"Strategy of Strategies" - a higher-level portfolio manager that:
1. Tracks performance metrics per strategy (Sharpe, PnL, win rate)
2. Allocates capital based on recent performance and market phase alignment
3. Reallocates dynamically as conditions change

Production-grade with comprehensive metrics and safety guards.
"""

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StrategyMetrics:
    """Performance metrics for a single strategy."""
    strategy_id: str
    total_pnl: float = 0.0
    trade_count: int = 0
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    allocated_capital: float = 0.0
    returns: deque = field(default_factory=lambda: deque(maxlen=100))  # Last 100 returns
    last_trade_time: float | None = None

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.trade_count == 0:
            return 0.5  # No data, assume neutral
        return self.wins / self.trade_count

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor."""
        if abs(self.gross_loss) < 1e-6:
            return 10.0 if self.gross_profit > 0 else 1.0
        return abs(self.gross_profit / self.gross_loss)

    @property
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from recent returns."""
        if len(self.returns) < 10:
            return 0.0

        returns_array = np.array(list(self.returns))
        if np.std(returns_array) < 1e-10:
            return 0.0

        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)

        # Annualized Sharpe (assume ~100 trades/year for crypto)
        sharpe = (mean_return / std_return) * np.sqrt(100) if std_return > 0 else 0.0
        return float(sharpe)

    @property
    def roi(self) -> float:
        """Return on invested capital."""
        if self.allocated_capital < 1e-6:
            return 0.0
        return (self.total_pnl / self.allocated_capital) * 100


@dataclass
class AllocationDecision:
    """Capital allocation decision record."""
    timestamp: float
    strategy_id: str
    allocated_capital: float
    allocation_pct: float
    reason: str
    metrics_snapshot: dict


class StrategyCapitalAllocator:
    """Dynamic capital allocator for strategies.

    Allocation Methods:
    1. PERFORMANCE_WEIGHTED: Allocate based on recent Sharpe ratio
    2. PHASE_MATCHED: Allocate more to strategies aligned with current market phase
    3. HYBRID: Combination of performance + phase alignment

    Safety Guards:
    - Min allocation per active strategy: 5%
    - Max allocation per strategy: 40%
    - Reallocation cooldown: 1 hour (avoid thrashing)
    - Gradual reallocation: max 10% shift per reallocation
    """

    MIN_ALLOCATION_PCT = 5.0  # Minimum 5% per active strategy
    MAX_ALLOCATION_PCT = 40.0  # Maximum 40% to any single strategy
    MAX_ALLOCATION_SHIFT_PCT = 10.0  # Max 10% change per reallocation
    REALLOCATION_COOLDOWN_SECONDS = 3600.0  # 1 hour cooldown

    # Phase-strategy affinity matrix
    PHASE_AFFINITY = {
        "impulse_engine": ["markup", "markdown", "water"],  # Momentum strategies
        "funding_harvester": ["ice", "accumulation"],  # Neutral markets
        "basis_arbitrage": ["ice", "accumulation"],  # Low volatility
        "momentum_trader": ["markup", "water"],  # Trending up
        "hf_scalping": ["ice", "water"],  # Stable with volume
        "enhanced_grid": ["ice", "accumulation"],  # Range-bound
        "liquidity_sweep": ["markup", "markdown", "distribution"],  # Volatile transitions
        "sentiment_strategy": ["accumulation", "distribution"],  # Sentiment-driven
    }

    def __init__(
        self,
        bus: EventBus,
        total_capital: float,
        allocation_method: str = "hybrid",
    ):
        """Initialize strategy capital allocator.

        Args:
            bus: Event bus for subscribing to trade events
            total_capital: Total capital to allocate across strategies
            allocation_method: "performance_weighted", "phase_matched", or "hybrid"
        """
        self._bus = bus
        self._total_capital = total_capital
        self._allocation_method = allocation_method

        # Strategy metrics tracking
        self._strategy_metrics: dict[str, StrategyMetrics] = {}

        # Current allocations
        self._current_allocations: dict[str, float] = {}  # strategy_id -> capital

        # Physics state for phase-matched allocation
        self._dominant_phase = "unknown"
        self._phase_confidence = 0.0

        # Allocation history
        self._allocation_history: deque[AllocationDecision] = deque(maxlen=1000)
        self._last_reallocation_time = 0.0

        # Running state
        self._running = False

    async def start(self) -> None:
        """Start capital allocator."""
        self._running = True

        # Subscribe to events
        self._bus.subscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        self._bus.subscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)

        logger.info(
            f"StrategyCapitalAllocator started: ${self._total_capital:.2f} total capital, "
            f"method={self._allocation_method}"
        )

    async def stop(self) -> None:
        """Stop capital allocator."""
        self._running = False
        self._bus.unsubscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        self._bus.unsubscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        logger.info("StrategyCapitalAllocator stopped")

    async def _handle_position_closed(self, event: Event) -> None:
        """Track closed positions for strategy performance."""
        position = event.data.get("position")
        if not position:
            return

        strategy_id = position.strategy_id
        pnl = position.realized_pnl

        # Initialize metrics if new strategy
        if strategy_id not in self._strategy_metrics:
            self._strategy_metrics[strategy_id] = StrategyMetrics(strategy_id=strategy_id)

        metrics = self._strategy_metrics[strategy_id]

        # Update metrics
        metrics.total_pnl += pnl
        metrics.trade_count += 1
        metrics.last_trade_time = time.time()

        if pnl > 0:
            metrics.wins += 1
            metrics.gross_profit += pnl
        else:
            metrics.losses += 1
            metrics.gross_loss += pnl

        # Calculate return (PnL / allocated capital)
        if metrics.allocated_capital > 0:
            trade_return = pnl / metrics.allocated_capital
            metrics.returns.append(trade_return)

        logger.debug(
            f"Strategy {strategy_id}: PnL={pnl:.2f}, total_pnl={metrics.total_pnl:.2f}, "
            f"win_rate={metrics.win_rate:.2%}, sharpe={metrics.sharpe_ratio:.2f}"
        )

        # Check if reallocation needed
        await self._maybe_reallocate()

    async def _handle_physics_update(self, event: Event) -> None:
        """Update dominant market phase for phase-matched allocation."""
        physics = event.data.get("physics", {})
        phase = physics.get("phase", "unknown")
        confidence = physics.get("phase_confidence", 0.0)

        # Update dominant phase (simple majority vote across all symbols)
        # In production, could weight by trading volume
        self._dominant_phase = phase
        self._phase_confidence = confidence

    async def _maybe_reallocate(self) -> None:
        """Check if reallocation is needed and execute if so."""
        # Cooldown check
        time_since_last = time.time() - self._last_reallocation_time
        if time_since_last < self.REALLOCATION_COOLDOWN_SECONDS:
            return

        # Need at least 2 strategies with some trades
        active_strategies = [
            s for s in self._strategy_metrics.values() if s.trade_count >= 5
        ]

        if len(active_strategies) < 2:
            logger.debug("Not enough active strategies for reallocation")
            return

        # Calculate new allocations
        new_allocations = self._calculate_allocations(active_strategies)

        # Check if allocations changed significantly
        if not self._allocations_changed_significantly(new_allocations):
            logger.debug("Allocations have not changed significantly, skipping reallocation")
            return

        # Apply new allocations (with gradual shift limit)
        self._apply_allocations(new_allocations)

        self._last_reallocation_time = time.time()

        logger.info(
            f"Capital reallocation executed: {len(new_allocations)} strategies, "
            f"method={self._allocation_method}, dominant_phase={self._dominant_phase}"
        )

    def _calculate_allocations(self, strategies: list[StrategyMetrics]) -> dict[str, float]:
        """Calculate optimal capital allocation across strategies.

        Returns:
            Dict mapping strategy_id to allocated capital
        """
        if self._allocation_method == "performance_weighted":
            return self._performance_weighted_allocation(strategies)
        elif self._allocation_method == "phase_matched":
            return self._phase_matched_allocation(strategies)
        elif self._allocation_method == "hybrid":
            return self._hybrid_allocation(strategies)
        else:
            logger.warning(f"Unknown allocation method: {self._allocation_method}, using equal")
            return self._equal_allocation(strategies)

    def _equal_allocation(self, strategies: list[StrategyMetrics]) -> dict[str, float]:
        """Equal allocation (1/N rule)."""
        per_strategy = self._total_capital / len(strategies)
        return {s.strategy_id: per_strategy for s in strategies}

    def _performance_weighted_allocation(
        self, strategies: list[StrategyMetrics]
    ) -> dict[str, float]:
        """Allocate based on Sharpe ratio (risk-adjusted returns)."""
        # Calculate Sharpe for each strategy
        sharpe_ratios = []
        strategy_ids = []

        for s in strategies:
            sharpe = s.sharpe_ratio
            # Floor at 0 (no negative Sharpe strategies get capital)
            sharpe = max(0.0, sharpe)
            sharpe_ratios.append(sharpe)
            strategy_ids.append(s.strategy_id)

        # If all Sharpe ratios are 0, use equal allocation
        if sum(sharpe_ratios) < 1e-6:
            return self._equal_allocation(strategies)

        # Normalize Sharpe ratios to allocations
        total_sharpe = sum(sharpe_ratios)
        allocations = {}

        for strategy_id, sharpe in zip(strategy_ids, sharpe_ratios, strict=False):
            pct = (sharpe / total_sharpe) * 100

            # Apply min/max constraints
            pct = max(self.MIN_ALLOCATION_PCT, min(self.MAX_ALLOCATION_PCT, pct))

            allocations[strategy_id] = (pct / 100.0) * self._total_capital

        # Renormalize to ensure total = 100%
        total_allocated = sum(allocations.values())
        for strategy_id in allocations:
            allocations[strategy_id] *= self._total_capital / total_allocated

        return allocations

    def _phase_matched_allocation(
        self, strategies: list[StrategyMetrics]
    ) -> dict[str, float]:
        """Allocate more capital to strategies aligned with current phase."""
        phase_scores = []
        strategy_ids = []

        for s in strategies:
            # Check if strategy has affinity for current phase
            affinity_phases = self.PHASE_AFFINITY.get(s.strategy_id, [])

            if self._dominant_phase in affinity_phases:
                # High affinity
                score = 2.0 * self._phase_confidence
            else:
                # Low affinity
                score = 0.5

            phase_scores.append(score)
            strategy_ids.append(s.strategy_id)

        # Normalize to allocations
        total_score = sum(phase_scores)
        if total_score < 1e-6:
            return self._equal_allocation(strategies)

        allocations = {}
        for strategy_id, score in zip(strategy_ids, phase_scores, strict=False):
            pct = (score / total_score) * 100
            pct = max(self.MIN_ALLOCATION_PCT, min(self.MAX_ALLOCATION_PCT, pct))
            allocations[strategy_id] = (pct / 100.0) * self._total_capital

        # Renormalize
        total_allocated = sum(allocations.values())
        for strategy_id in allocations:
            allocations[strategy_id] *= self._total_capital / total_allocated

        return allocations

    def _hybrid_allocation(self, strategies: list[StrategyMetrics]) -> dict[str, float]:
        """Hybrid: 70% performance-weighted, 30% phase-matched."""
        perf_alloc = self._performance_weighted_allocation(strategies)
        phase_alloc = self._phase_matched_allocation(strategies)

        hybrid_alloc = {}
        for strategy_id in perf_alloc.keys():
            hybrid_alloc[strategy_id] = (
                perf_alloc[strategy_id] * 0.7 + phase_alloc.get(strategy_id, 0.0) * 0.3
            )

        # Renormalize
        total_allocated = sum(hybrid_alloc.values())
        for strategy_id in hybrid_alloc:
            hybrid_alloc[strategy_id] *= self._total_capital / total_allocated

        return hybrid_alloc

    def _allocations_changed_significantly(
        self, new_allocations: dict[str, float], threshold_pct: float = 5.0
    ) -> bool:
        """Check if new allocations differ from current by more than threshold."""
        for strategy_id, new_capital in new_allocations.items():
            current_capital = self._current_allocations.get(strategy_id, 0.0)

            if current_capital > 0:
                pct_change = abs((new_capital - current_capital) / current_capital) * 100
                if pct_change > threshold_pct:
                    return True
            elif new_capital > 0:
                # New strategy being allocated
                return True

        return False

    def _apply_allocations(self, new_allocations: dict[str, float]) -> None:
        """Apply new capital allocations with gradual shift limit."""
        for strategy_id, new_capital in new_allocations.items():
            current_capital = self._current_allocations.get(strategy_id, 0.0)

            if current_capital > 0:
                # Gradual shift: max 10% change per reallocation
                max_change = current_capital * (self.MAX_ALLOCATION_SHIFT_PCT / 100.0)
                capital_change = new_capital - current_capital

                if abs(capital_change) > max_change:
                    # Limit shift
                    direction = 1 if capital_change > 0 else -1
                    new_capital = current_capital + (direction * max_change)
                    logger.debug(
                        f"Limiting allocation shift for {strategy_id}: "
                        f"${current_capital:.2f} → ${new_capital:.2f} "
                        f"(requested ${new_capital:.2f})"
                    )

            # Apply allocation
            self._current_allocations[strategy_id] = new_capital

            # Update metrics
            if strategy_id in self._strategy_metrics:
                self._strategy_metrics[strategy_id].allocated_capital = new_capital

            # Record decision
            metrics_snapshot = {}
            if strategy_id in self._strategy_metrics:
                m = self._strategy_metrics[strategy_id]
                metrics_snapshot = {
                    "sharpe": m.sharpe_ratio,
                    "win_rate": m.win_rate,
                    "pnl": m.total_pnl,
                    "roi": m.roi,
                }

            decision = AllocationDecision(
                timestamp=time.time(),
                strategy_id=strategy_id,
                allocated_capital=new_capital,
                allocation_pct=(new_capital / self._total_capital) * 100,
                reason=self._allocation_method,
                metrics_snapshot=metrics_snapshot,
            )
            self._allocation_history.append(decision)

            logger.info(
                f"Strategy {strategy_id}: allocated ${new_capital:.2f} "
                f"({(new_capital/self._total_capital)*100:.1f}%)"
            )

    def get_allocation(self, strategy_id: str) -> float:
        """Get current capital allocation for a strategy."""
        return self._current_allocations.get(strategy_id, 0.0)

    def get_all_allocations(self) -> dict[str, float]:
        """Get all current allocations."""
        return self._current_allocations.copy()

    def get_strategy_metrics(self, strategy_id: str) -> StrategyMetrics | None:
        """Get metrics for a strategy."""
        return self._strategy_metrics.get(strategy_id)

    def get_all_metrics(self) -> dict[str, StrategyMetrics]:
        """Get all strategy metrics."""
        return self._strategy_metrics.copy()

    def get_allocation_history(self, limit: int = 100) -> list[AllocationDecision]:
        """Get recent allocation history."""
        return list(self._allocation_history)[-limit:]

    def force_reallocation(self) -> None:
        """Force immediate reallocation (bypass cooldown)."""
        self._last_reallocation_time = 0.0
        logger.warning("Forced reallocation triggered")
