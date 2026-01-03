"""Strategy memory for tracking performance and applying penalties."""

from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from typing import NamedTuple

from hean.config import settings
from hean.core.regime import Regime
from hean.logging import get_logger

logger = get_logger(__name__)


class TradeRecord(NamedTuple):
    """Record of a single trade."""

    pnl: float
    regime: str
    timestamp: datetime


class StrategyMemory:
    """Tracks strategy performance and applies penalties for poor performance."""

    def __init__(self) -> None:
        """Initialize strategy memory."""
        # Rolling trade history per strategy (last N trades)
        self._trade_history: dict[str, deque[TradeRecord]] = defaultdict(
            lambda: deque(maxlen=settings.memory_window_trades)
        )

        # Rolling drawdown per strategy
        self._strategy_peak_equity: dict[str, float] = {}
        self._strategy_current_equity: dict[str, float] = {}
        self._strategy_drawdown: dict[str, float] = {}  # Current drawdown %

        # Regime-specific loss tracking
        self._regime_losses: dict[tuple[str, str], int] = defaultdict(
            int
        )  # (strategy_id, regime) -> loss count
        self._regime_blacklist: dict[
            tuple[str, str], date
        ] = {}  # (strategy_id, regime) -> blacklist until date

        # Cooldown tracking (for drawdown penalties)
        self._cooldown_until: dict[str, date] = {}  # strategy_id -> cooldown until date

    def record_trade(
        self, strategy_id: str, pnl: float, regime: Regime, timestamp: datetime | None = None
    ) -> None:
        """Record a completed trade.

        Args:
            strategy_id: Strategy identifier
            pnl: Realized PnL from the trade
            regime: Market regime when trade was executed
            timestamp: Trade timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        regime_str = regime.value if isinstance(regime, Regime) else str(regime)
        record = TradeRecord(pnl=pnl, regime=regime_str, timestamp=timestamp)
        self._trade_history[strategy_id].append(record)

        # Track regime-specific losses
        if pnl < 0:
            key = (strategy_id, regime_str)
            self._regime_losses[key] += 1

            # Check if we should blacklist this regime
            if self._regime_losses[key] >= 3:  # 3 consecutive losses in same regime
                blacklist_until = date.today() + timedelta(
                    days=settings.regime_blacklist_duration_days
                )
                self._regime_blacklist[key] = blacklist_until
                logger.warning(
                    f"Regime {regime_str} blacklisted for strategy {strategy_id} "
                    f"until {blacklist_until}"
                )
        else:
            # Reset loss count on win
            key = (strategy_id, regime_str)
            if key in self._regime_losses:
                self._regime_losses[key] = 0

    def update_equity(self, strategy_id: str, equity: float) -> None:
        """Update equity for a strategy to track drawdown.

        Args:
            strategy_id: Strategy identifier
            equity: Current equity for the strategy
        """
        # Initialize peak if first time
        if strategy_id not in self._strategy_peak_equity:
            self._strategy_peak_equity[strategy_id] = equity
            self._strategy_current_equity[strategy_id] = equity
            self._strategy_drawdown[strategy_id] = 0.0
            return

        # Update peak equity
        if equity > self._strategy_peak_equity[strategy_id]:
            self._strategy_peak_equity[strategy_id] = equity

        self._strategy_current_equity[strategy_id] = equity

        # Calculate drawdown
        peak = self._strategy_peak_equity[strategy_id]
        if peak > 0:
            drawdown_pct = ((peak - equity) / peak) * 100.0
            self._strategy_drawdown[strategy_id] = drawdown_pct

            # Check if drawdown exceeds threshold -> trigger cooldown
            if drawdown_pct > settings.drawdown_threshold_pct:
                cooldown_until = date.today() + timedelta(
                    days=settings.regime_blacklist_duration_days
                )
                self._cooldown_until[strategy_id] = cooldown_until
                logger.warning(
                    f"Strategy {strategy_id} in cooldown until {cooldown_until} "
                    f"due to drawdown {drawdown_pct:.2f}%"
                )

    def get_rolling_profit_factor(self, strategy_id: str) -> float:
        """Calculate rolling profit factor over last N trades.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Profit factor (wins / losses), or 1.0 if no trades
        """
        trades = list(self._trade_history.get(strategy_id, deque()))
        if not trades:
            return 1.0

        wins = sum(t.pnl for t in trades if t.pnl > 0)
        losses = abs(sum(t.pnl for t in trades if t.pnl < 0))

        if losses == 0:
            return wins if wins > 0 else 1.0

        return wins / losses

    def get_rolling_drawdown(self, strategy_id: str) -> float:
        """Get current rolling drawdown for a strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Drawdown percentage
        """
        return self._strategy_drawdown.get(strategy_id, 0.0)

    def is_regime_blacklisted(self, strategy_id: str, regime: Regime) -> bool:
        """Check if a regime is blacklisted for a strategy.

        Args:
            strategy_id: Strategy identifier
            regime: Market regime to check

        Returns:
            True if regime is blacklisted
        """
        regime_str = regime.value if isinstance(regime, Regime) else str(regime)
        key = (strategy_id, regime_str)

        if key not in self._regime_blacklist:
            return False

        blacklist_until = self._regime_blacklist[key]
        if date.today() >= blacklist_until:
            # Blacklist expired, remove it
            del self._regime_blacklist[key]
            return False

        return True

    def is_in_cooldown(self, strategy_id: str) -> bool:
        """Check if a strategy is in cooldown due to drawdown.

        Args:
            strategy_id: Strategy identifier

        Returns:
            True if strategy is in cooldown
        """
        if strategy_id not in self._cooldown_until:
            return False

        cooldown_until = self._cooldown_until[strategy_id]
        if date.today() >= cooldown_until:
            # Cooldown expired, remove it
            del self._cooldown_until[strategy_id]
            return False

        return True

    def should_reduce_weight(self, strategy_id: str, regime: Regime) -> bool:
        """Determine if weight should be reduced for a strategy.

        Args:
            strategy_id: Strategy identifier
            regime: Current market regime

        Returns:
            True if weight should be reduced
        """
        # Check regime blacklist
        if self.is_regime_blacklisted(strategy_id, regime):
            logger.debug(f"Reducing weight for {strategy_id}: regime {regime.value} is blacklisted")
            return True

        # Check cooldown
        if self.is_in_cooldown(strategy_id):
            logger.debug(f"Reducing weight for {strategy_id}: in cooldown due to drawdown")
            return True

        # Check rolling profit factor
        pf = self.get_rolling_profit_factor(strategy_id)
        if pf < settings.pf_penalty_threshold:
            logger.debug(
                f"Reducing weight for {strategy_id}: PF {pf:.2f} < {settings.pf_penalty_threshold}"
            )
            return True

        # Check drawdown threshold
        drawdown = self.get_rolling_drawdown(strategy_id)
        if drawdown > settings.drawdown_threshold_pct:
            logger.debug(
                f"Reducing weight for {strategy_id}: drawdown {drawdown:.2f}% > "
                f"{settings.drawdown_threshold_pct}%"
            )
            return True

        return False

    def get_penalty_multiplier(self, strategy_id: str, regime: Regime) -> float:
        """Get penalty multiplier for weight adjustment.

        Args:
            strategy_id: Strategy identifier
            regime: Current market regime

        Returns:
            Multiplier (0.0 to 1.0), where 1.0 = no penalty, < 1.0 = penalty
        """
        if not self.should_reduce_weight(strategy_id, regime):
            return 1.0

        # Apply cumulative penalties
        multiplier = 1.0

        # PF penalty
        pf = self.get_rolling_profit_factor(strategy_id)
        if pf < settings.pf_penalty_threshold:
            # Reduce by 20% for PF < threshold
            multiplier *= 0.8

        # Drawdown penalty
        drawdown = self.get_rolling_drawdown(strategy_id)
        if drawdown > settings.drawdown_threshold_pct:
            # Additional 10% reduction for high drawdown
            multiplier *= 0.9

        # Regime blacklist penalty (stronger)
        if self.is_regime_blacklisted(strategy_id, regime):
            # Reduce by 50% for blacklisted regime
            multiplier *= 0.5

        # Cooldown penalty (stronger)
        if self.is_in_cooldown(strategy_id):
            # Reduce by 50% during cooldown
            multiplier *= 0.5

        return multiplier
