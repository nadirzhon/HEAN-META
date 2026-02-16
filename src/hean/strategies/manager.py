"""Strategy Manager with dynamic capital allocation based on performance.

Tracks per-strategy metrics and dynamically allocates capital to top performers.
Integrates with Physics engine to adjust allocations based on market phase.
"""

import asyncio
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class StrategyPerformance:
    """Tracks performance metrics for a single strategy."""

    def __init__(self, strategy_id: str, lookback_trades: int = 50):
        self.strategy_id = strategy_id
        self.lookback_trades = lookback_trades

        # Trade history
        self.trades: deque = deque(maxlen=lookback_trades)

        # Cumulative metrics
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

        # Peak tracking for drawdown
        self.peak_pnl = 0.0
        self.current_pnl = 0.0

    def add_trade(self, pnl: float, symbol: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a completed trade."""
        is_win = pnl > 0

        self.trades.append({
            "pnl": pnl,
            "symbol": symbol,
            "is_win": is_win,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {},
        })

        self.total_pnl += pnl
        self.current_pnl += pnl
        self.total_trades += 1

        if is_win:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(
                self.max_consecutive_losses, self.consecutive_losses
            )

        if self.current_pnl > self.peak_pnl:
            self.peak_pnl = self.current_pnl

    def get_win_rate(self) -> float:
        """Calculate win rate."""
        if len(self.trades) == 0:
            return 0.5  # Neutral default
        wins = sum(1 for t in self.trades if t["is_win"])
        return wins / len(self.trades)

    def get_profit_factor(self) -> float:
        """Calculate profit factor (gross wins / gross losses)."""
        if len(self.trades) == 0:
            return 1.0  # Neutral default

        gross_wins = sum(t["pnl"] for t in self.trades if t["pnl"] > 0)
        gross_losses = abs(sum(t["pnl"] for t in self.trades if t["pnl"] < 0))

        if gross_losses == 0:
            return 10.0 if gross_wins > 0 else 1.0

        return gross_wins / gross_losses

    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(self.trades) < 2:
            return 0.0

        returns = [t["pnl"] for t in self.trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        return (mean_return - risk_free_rate) / std_return

    def get_drawdown(self) -> float:
        """Calculate current drawdown percentage."""
        if self.peak_pnl == 0:
            return 0.0
        return ((self.peak_pnl - self.current_pnl) / self.peak_pnl) * 100

    def get_score(self) -> float:
        """Calculate composite performance score for ranking.

        Components:
        - Sharpe ratio (40%)
        - Profit factor (30%)
        - Win rate (20%)
        - Drawdown penalty (10%)
        """
        sharpe = self.get_sharpe_ratio()
        pf = self.get_profit_factor()
        wr = self.get_win_rate()
        dd = self.get_drawdown()

        # Normalize components to 0-1 range
        sharpe_norm = np.clip(sharpe / 3.0, 0.0, 1.0)  # Sharpe > 3.0 = excellent
        pf_norm = np.clip(pf / 2.0, 0.0, 1.0)  # PF > 2.0 = excellent
        wr_norm = wr  # Already 0-1
        dd_penalty = np.clip(dd / 20.0, 0.0, 1.0)  # 20% DD = max penalty

        score = (
            sharpe_norm * 0.4
            + pf_norm * 0.3
            + wr_norm * 0.2
            - dd_penalty * 0.1
        )

        return max(0.0, score)  # Ensure non-negative


class StrategyAllocator:
    """Dynamically allocates capital to strategies based on performance and market conditions.

    Features:
    - Performance-based allocation (Sharpe, PF, win rate)
    - Market phase awareness (allocate to strategies suited for current phase)
    - Automatic rebalancing
    - Min/max allocation limits per strategy
    """

    def __init__(
        self,
        bus: EventBus,
        initial_capital: float,
        rebalance_interval: int = 300,  # 5 minutes
        min_allocation_pct: float = 0.05,  # 5% minimum
        max_allocation_pct: float = 0.40,  # 40% maximum
    ):
        self._bus = bus
        self._initial_capital = initial_capital
        self._current_capital = initial_capital
        self._rebalance_interval = rebalance_interval
        self._min_allocation_pct = min_allocation_pct
        self._max_allocation_pct = max_allocation_pct

        # Strategy performance tracking
        self._strategies: dict[str, StrategyPerformance] = {}

        # Current capital allocations
        self._allocations: dict[str, float] = {}

        # Market state (from Physics)
        self._market_phase: dict[str, str] = {}

        # Strategy phase preferences (which strategies work best in each phase)
        self._phase_preferences = {
            "accumulation": [
                "impulse_engine",
                "funding_harvester",
                "basis_arbitrage",
            ],
            "markup": [
                "momentum_trader",
                "sentiment_strategy",
                "correlation_arb",
            ],
            "distribution": [
                "liquidity_sweep",
                "rebate_farmer",
                "inventory_neutral_mm",
            ],
            "markdown": [
                "funding_harvester",
                "basis_arbitrage",
                "liquidity_sweep",
            ],
        }

        self._running = False
        self._rebalance_task: asyncio.Task | None = None
        self._last_rebalance = datetime.utcnow()

    async def start(self) -> None:
        """Start strategy allocator."""
        self._running = True

        # Subscribe to events
        self._bus.subscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        self._bus.subscribe(EventType.EQUITY_UPDATE, self._handle_equity_update)
        self._bus.subscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)

        # Start rebalancing task
        self._rebalance_task = asyncio.create_task(self._rebalance_loop())

        logger.info("StrategyAllocator started")

    async def stop(self) -> None:
        """Stop strategy allocator."""
        self._running = False

        self._bus.unsubscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        self._bus.unsubscribe(EventType.EQUITY_UPDATE, self._handle_equity_update)
        self._bus.unsubscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)

        if self._rebalance_task:
            self._rebalance_task.cancel()
            try:
                await self._rebalance_task
            except asyncio.CancelledError:
                pass

        logger.info("StrategyAllocator stopped")

    def register_strategy(self, strategy_id: str) -> None:
        """Register a new strategy for tracking."""
        if strategy_id not in self._strategies:
            self._strategies[strategy_id] = StrategyPerformance(strategy_id)
            # Initial equal allocation
            num_strategies = len(self._strategies)
            initial_alloc = self._current_capital / num_strategies
            self._allocations[strategy_id] = initial_alloc
            logger.info(f"Registered strategy: {strategy_id}")

    async def _handle_position_closed(self, event: Event) -> None:
        """Track completed trades for performance calculation."""
        position = event.data.get("position")
        if not position:
            return

        strategy_id = position.strategy_id
        if strategy_id not in self._strategies:
            self.register_strategy(strategy_id)

        pnl = position.realized_pnl
        self._strategies[strategy_id].add_trade(
            pnl=pnl,
            symbol=position.symbol,
            metadata={"position_id": position.position_id},
        )

    async def _handle_equity_update(self, event: Event) -> None:
        """Track total capital changes."""
        equity = event.data.get("equity", 0.0)
        self._current_capital = equity

    async def _handle_physics_update(self, event: Event) -> None:
        """Track market phase for phase-aware allocation."""
        data = event.data
        symbol = data.get("symbol")
        if symbol:
            self._market_phase[symbol] = data.get("phase", "unknown")

    async def _rebalance_loop(self) -> None:
        """Periodically rebalance capital allocations."""
        while self._running:
            try:
                await asyncio.sleep(self._rebalance_interval)
                await self._rebalance_allocations()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rebalance loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _rebalance_allocations(self) -> None:
        """Rebalance capital allocations based on performance and market phase."""
        if not self._strategies:
            return

        # Calculate performance scores for each strategy
        scores = {}
        for strategy_id, perf in self._strategies.items():
            base_score = perf.get_score()

            # Adjust score based on market phase alignment
            phase_bonus = self._calculate_phase_bonus(strategy_id)
            adjusted_score = base_score * (1.0 + phase_bonus)

            scores[strategy_id] = max(0.01, adjusted_score)  # Minimum score

        # Calculate allocations based on scores
        total_score = sum(scores.values())
        if total_score == 0:
            # Equal allocation if no data
            equal_alloc = self._current_capital / len(self._strategies)
            self._allocations = dict.fromkeys(self._strategies, equal_alloc)
            return

        # Proportional allocation with min/max limits
        new_allocations = {}
        for strategy_id, score in scores.items():
            allocation_pct = score / total_score

            # Apply min/max limits
            allocation_pct = np.clip(
                allocation_pct,
                self._min_allocation_pct,
                self._max_allocation_pct,
            )

            new_allocations[strategy_id] = self._current_capital * allocation_pct

        # Normalize to ensure total = current_capital
        total_allocated = sum(new_allocations.values())
        if total_allocated > 0:
            scale_factor = self._current_capital / total_allocated
            new_allocations = {
                k: v * scale_factor for k, v in new_allocations.items()
            }

        # Check for significant changes
        old_allocations = self._allocations.copy()
        max_change = 0.0
        for strategy_id in new_allocations:
            old = old_allocations.get(strategy_id, 0.0)
            new = new_allocations[strategy_id]
            change_pct = abs(new - old) / old if old > 0 else 0.0
            max_change = max(max_change, change_pct)

        # Update allocations
        self._allocations = new_allocations
        self._last_rebalance = datetime.utcnow()

        # Log if significant change
        if max_change > 0.10:  # 10% change threshold
            logger.info("Capital allocations rebalanced:")
            for strategy_id, alloc in sorted(
                self._allocations.items(), key=lambda x: x[1], reverse=True
            ):
                pct = (alloc / self._current_capital) * 100
                perf = self._strategies[strategy_id]
                logger.info(
                    f"  {strategy_id}: ${alloc:.2f} ({pct:.1f}%) - "
                    f"Score={scores[strategy_id]:.3f} WR={perf.get_win_rate():.2f} "
                    f"PF={perf.get_profit_factor():.2f}"
                )

    def _calculate_phase_bonus(self, strategy_id: str) -> float:
        """Calculate phase alignment bonus for a strategy.

        Returns:
            Bonus multiplier (0.0 to 0.5) based on how well strategy aligns with current phase
        """
        if not self._market_phase:
            return 0.0

        # Get dominant phase
        phases = list(self._market_phase.values())
        phase_counts = {
            "accumulation": phases.count("accumulation"),
            "markup": phases.count("markup"),
            "distribution": phases.count("distribution"),
            "markdown": phases.count("markdown"),
        }
        dominant_phase = max(phase_counts, key=phase_counts.get) if phases else "unknown"

        # Check if strategy is preferred for this phase
        preferred_strategies = self._phase_preferences.get(dominant_phase, [])
        if strategy_id in preferred_strategies:
            return 0.3  # 30% bonus for phase alignment

        return 0.0

    def get_allocation(self, strategy_id: str) -> float:
        """Get current capital allocation for a strategy."""
        return self._allocations.get(strategy_id, 0.0)

    def get_all_allocations(self) -> dict[str, float]:
        """Get all current allocations."""
        return dict(self._allocations)

    def get_performance(self, strategy_id: str) -> StrategyPerformance | None:
        """Get performance tracker for a strategy."""
        return self._strategies.get(strategy_id)

    def get_all_performances(self) -> dict[str, StrategyPerformance]:
        """Get all strategy performances."""
        return dict(self._strategies)
