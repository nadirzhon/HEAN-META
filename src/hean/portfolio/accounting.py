"""Portfolio accounting and PnL tracking."""

from collections import defaultdict
from datetime import datetime, timedelta

from hean.core.types import EquitySnapshot, Order, Position
from hean.logging import get_logger

logger = get_logger(__name__)


class PortfolioAccounting:
    """Tracks portfolio equity, PnL, and drawdown."""

    def __init__(self, initial_capital: float) -> None:
        """Initialize portfolio accounting."""
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._positions: dict[str, Position] = {}
        self._equity_history: list[EquitySnapshot] = []
        self._peak_equity = initial_capital
        self._daily_start_equity: float | None = None
        self._daily_start_date = datetime.utcnow().date()
        self._realized_pnl = 0.0
        self._total_fees = 0.0

        # Per-strategy tracking
        self._strategy_pnl: dict[str, float] = defaultdict(float)
        self._strategy_trades: dict[str, int] = defaultdict(int)
        self._strategy_wins: dict[str, int] = defaultdict(int)
        self._strategy_losses: dict[str, int] = defaultdict(int)
        self._strategy_equity_history: dict[str, list[float]] = defaultdict(list)
        self._strategy_peak_equity: dict[str, float] = {}
        self._strategy_initial_capital: dict[str, float] = {}
        # Per-strategy per-regime tracking
        self._strategy_regime_pnl: dict[tuple[str, str], float] = defaultdict(float)
        self._strategy_regime_trades: dict[tuple[str, str], int] = defaultdict(int)

        # Caching for performance optimization
        self._metrics_cache: dict[str, dict[str, dict[str, float]]] | None = None
        self._metrics_cache_timestamp: datetime | None = None
        self._metrics_cache_ttl = timedelta(seconds=5)  # Cache for 5 seconds

    def update_cash(self, amount: float) -> None:
        """Update cash balance."""
        self._cash += amount

    def get_cash_balance(self) -> float:
        """Return current cash balance."""
        return self._cash

    def get_realized_pnl_total(self) -> float:
        """Return cumulative realized PnL."""
        return self._realized_pnl

    def get_total_fees(self) -> float:
        """Return total fees paid."""
        return self._total_fees

    def add_position(self, position: Position) -> None:
        """Add a new position."""
        self._positions[position.position_id] = position
        # Note: Cash is already updated in record_fill

    def remove_position(self, position_id: str) -> None:
        """Remove a closed position."""
        self._positions.pop(position_id, None)

    def update_position_price(self, position_id: str, price: float) -> None:
        """Update the current price of a position."""
        if position_id in self._positions:
            pos = self._positions[position_id]
            pos.current_price = price
            # Calculate unrealized PnL
            if pos.side == "long":
                pos.unrealized_pnl = (price - pos.entry_price) * pos.size
            else:  # short
                pos.unrealized_pnl = (pos.entry_price - price) * pos.size

    def record_fill(self, order: Order, fill_price: float, fee: float) -> None:
        """Record an order fill and update cash."""
        cost = order.filled_size * fill_price
        if order.side == "buy":
            self._cash -= cost + fee
        else:  # sell
            self._cash += cost - fee

        self._total_fees += fee

        # Track per-strategy
        if order.strategy_id:
            self._strategy_trades[order.strategy_id] += 1

        # Invalidate cache when data changes
        self._invalidate_metrics_cache()

    def record_realized_pnl(
        self, amount: float, strategy_id: str | None = None, regime: str | None = None
    ) -> None:
        """Record realized PnL from a closed position."""
        self._realized_pnl += amount
        self._cash += amount

        if strategy_id:
            self._strategy_pnl[strategy_id] += amount
            if amount > 0:
                self._strategy_wins[strategy_id] += 1
            else:
                self._strategy_losses[strategy_id] += 1

            # Track per-regime
            if regime:
                key = (strategy_id, regime)
                self._strategy_regime_pnl[key] += amount
                self._strategy_regime_trades[key] += 1

        # Invalidate cache when data changes
        self._invalidate_metrics_cache()

    def get_equity(self, current_prices: dict[str, float] | None = None) -> float:
        """Calculate current equity."""
        # Update position prices if provided
        if current_prices:
            for pos_id, pos in self._positions.items():
                if pos.symbol in current_prices:
                    self.update_position_price(pos_id, current_prices[pos.symbol])

        # Calculate unrealized PnL
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self._positions.values())

        # Equity = cash + unrealized PnL
        # (positions are already accounted for in cash when opened)
        equity = self._cash + unrealized_pnl
        return equity

    def get_drawdown(self, equity: float) -> tuple[float, float]:
        """Calculate drawdown from peak equity.

        Returns:
            (drawdown_amount, drawdown_pct) tuple
        """
        if equity > self._peak_equity:
            self._peak_equity = equity

        drawdown = self._peak_equity - equity
        drawdown_pct = (drawdown / self._peak_equity * 100) if self._peak_equity > 0 else 0.0

        return drawdown, drawdown_pct

    def get_daily_pnl(self, equity: float) -> float:
        """Calculate daily PnL."""
        now = datetime.utcnow().date()
        if now != self._daily_start_date:
            self._daily_start_equity = equity
            self._daily_start_date = now

        if self._daily_start_equity is None:
            self._daily_start_equity = equity
            return 0.0

        return equity - self._daily_start_equity

    def snapshot(self, current_prices: dict[str, float] | None = None) -> EquitySnapshot:
        """Create an equity snapshot."""
        equity = self.get_equity(current_prices)
        positions_value = sum(pos.current_price * pos.size for pos in self._positions.values())
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self._positions.values())
        daily_pnl = self.get_daily_pnl(equity)
        drawdown, drawdown_pct = self.get_drawdown(equity)

        snapshot = EquitySnapshot(
            timestamp=datetime.utcnow(),
            equity=equity,
            cash=self._cash,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self._realized_pnl,
            daily_pnl=daily_pnl,
            drawdown=drawdown,
            drawdown_pct=drawdown_pct,
        )

        self._equity_history.append(snapshot)
        return snapshot

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self._positions.values())

    def get_unrealized_pnl_total(self) -> float:
        """Get aggregate unrealized PnL across all open positions."""
        return sum(pos.unrealized_pnl for pos in self._positions.values())

    def update_strategy_equity(self, strategy_id: str, equity: float) -> None:
        """Update equity history for a strategy."""
        self._strategy_equity_history[strategy_id].append(equity)

        # Track peak equity per strategy
        if strategy_id not in self._strategy_peak_equity:
            self._strategy_peak_equity[strategy_id] = equity
            self._strategy_initial_capital[strategy_id] = equity
        else:
            if equity > self._strategy_peak_equity[strategy_id]:
                self._strategy_peak_equity[strategy_id] = equity

    def get_strategy_metrics(self) -> dict[str, dict[str, float]]:
        """Get per-strategy metrics with caching for performance."""
        # Check cache
        now = datetime.utcnow()
        if (
            self._metrics_cache is not None
            and self._metrics_cache_timestamp is not None
            and (now - self._metrics_cache_timestamp) < self._metrics_cache_ttl
        ):
            return self._metrics_cache

        metrics: dict[str, dict[str, float]] = {}

        # Get all strategy IDs that have activity (optimized: use set union)
        strategy_ids = set(self._strategy_pnl.keys()) | set(self._strategy_trades.keys())

        for strategy_id in strategy_ids:
            # Use initial capital if set, otherwise use a default based on total capital
            initial = self._strategy_initial_capital.get(strategy_id, 0.0)
            if initial == 0.0:
                # Default to equal split if not set
                num_strategies = max(1, len(strategy_ids))
                initial = self._initial_capital / num_strategies
                self._strategy_initial_capital[strategy_id] = initial
                if strategy_id not in self._strategy_peak_equity:
                    self._strategy_peak_equity[strategy_id] = initial

            pnl = self._strategy_pnl.get(strategy_id, 0.0)
            trades = self._strategy_trades.get(strategy_id, 0)
            wins = self._strategy_wins.get(strategy_id, 0)
            losses = self._strategy_losses.get(strategy_id, 0)

            # Calculate return
            return_pct = (pnl / initial * 100) if initial > 0 else 0.0

            # Calculate win rate
            total_closed = wins + losses
            win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0

            # Calculate profit factor (sum of wins / sum of losses)
            # For now, use wins/losses ratio as proxy
            profit_factor = (wins / losses) if losses > 0 else (wins if wins > 0 else 1.0)

            # Calculate max drawdown (optimized: single pass)
            equity_history = self._strategy_equity_history.get(strategy_id, [])
            max_dd_pct = 0.0
            if equity_history:
                peak = initial
                for equity in equity_history:
                    if equity > peak:
                        peak = equity
                    elif peak > 0:  # Only calculate if we have a valid peak
                        dd_pct = (peak - equity) / peak * 100
                        if dd_pct > max_dd_pct:
                            max_dd_pct = dd_pct

            metrics[strategy_id] = {
                "return_pct": return_pct,
                "trades": float(trades),
                "win_rate_pct": win_rate,
                "profit_factor": profit_factor,
                "max_drawdown_pct": max_dd_pct,
                "pnl": pnl,
                "wins": float(wins),
                "losses": float(losses),
            }

        # Cache results
        self._metrics_cache = metrics
        self._metrics_cache_timestamp = now

        return metrics

    def _invalidate_metrics_cache(self) -> None:
        """Invalidate metrics cache when data changes."""
        self._metrics_cache = None
        self._metrics_cache_timestamp = None

    @property
    def cash(self) -> float:
        """Get current cash balance."""
        return self._cash

    @property
    def initial_capital(self) -> float:
        """Get initial capital."""
        return self._initial_capital
