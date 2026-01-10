"""Risk limits and constraints."""

from collections import defaultdict
from datetime import datetime

from hean.config import settings
from hean.core.types import OrderRequest, Position
from hean.logging import get_logger
from hean.paper_trade_assist import (
    get_cooldown_multiplier,
    get_daily_attempts_multiplier,
    get_max_open_positions_override,
    is_paper_assist_enabled,
    log_allow_reason,
    log_block_reason,
)

logger = get_logger(__name__)


class RiskLimits:
    """Enforces risk limits and constraints."""

    def __init__(self) -> None:
        """Initialize risk limits."""
        self._open_positions: dict[str, Position] = {}
        self._daily_attempts: dict[str, int] = defaultdict(int)
        self._last_reset_date = datetime.utcnow().date()
        self._consecutive_losses: dict[str, int] = defaultdict(int)

    def check_order_request(self, order_request: OrderRequest, equity: float) -> tuple[bool, str]:
        """Check if an order request passes risk limits.

        Returns:
            (allowed, reason) tuple
        """
        # Reset daily counters if needed
        self._reset_daily_counters_if_needed()

        # Check max open positions (with paper assist override)
        max_positions = settings.max_open_positions
        override = get_max_open_positions_override()
        if override is not None:
            max_positions = override
        
        if len(self._open_positions) >= max_positions:
            reason = f"Max open positions ({max_positions}) reached"
            log_block_reason(
                "max_positions",
                measured_value=len(self._open_positions),
                threshold=max_positions,
                symbol=order_request.symbol,
                strategy_id=order_request.strategy_id,
            )
            return False, reason

        # Check if position already exists for symbol
        existing_pos = self._get_position_for_symbol(order_request.symbol)
        if existing_pos:
            reason = f"Position already exists for {order_request.symbol}"
            log_block_reason(
                "position_exists",
                symbol=order_request.symbol,
                strategy_id=order_request.strategy_id,
            )
            return False, reason

        # Check leverage (would need position value calculation)
        # This is simplified - in production, calculate total position value
        # and compare to equity * max_leverage

        log_allow_reason("risk_limits_ok", symbol=order_request.symbol, strategy_id=order_request.strategy_id)
        return True, ""

    def check_daily_attempts(self, strategy_id: str, regime=None) -> tuple[bool, str]:
        """Check if strategy has exceeded daily attempt limit."""
        self._reset_daily_counters_if_needed()

        # This is strategy-specific, so we check per strategy
        # For impulse engine, use its specific limit
        if strategy_id == "impulse_engine":
            max_attempts = settings.impulse_max_attempts_per_day
            # Stricter limit in IMPULSE regime
            if regime and regime.value == "impulse":
                max_attempts = int(max_attempts * 0.8)  # 20% reduction
            
            # Apply paper assist multiplier
            multiplier = get_daily_attempts_multiplier()
            max_attempts = int(max_attempts * multiplier)
            
            if self._daily_attempts[strategy_id] >= max_attempts:
                reason = f"Daily attempt limit ({max_attempts}) reached for {strategy_id}"
                log_block_reason(
                    "daily_attempts",
                    measured_value=self._daily_attempts[strategy_id],
                    threshold=max_attempts,
                    strategy_id=strategy_id,
                )
                return False, reason
        else:
            # Other strategies have no specific limit (or use a default)
            pass

        log_allow_reason("daily_attempts_ok", strategy_id=strategy_id)
        return True, ""

    def check_cooldown(self, strategy_id: str) -> tuple[bool, str]:
        """Check if strategy is in cooldown period."""
        # Aggressive Mode: Bypass cooldowns completely when DEBUG_MODE=True
        if settings.debug_mode:
            log_allow_reason("cooldown_bypassed_debug", strategy_id=strategy_id, note="DEBUG_MODE bypass")
            return True, ""
        
        if strategy_id == "impulse_engine":
            cooldown_threshold = settings.impulse_cooldown_after_losses
            # Apply paper assist multiplier (shorter cooldown)
            multiplier = get_cooldown_multiplier()
            # If multiplier is 0, bypass cooldown
            if multiplier == 0.0:
                log_allow_reason("cooldown_bypassed_aggressive", strategy_id=strategy_id, note="Aggressive Mode")
                return True, ""
            effective_threshold = int(cooldown_threshold / multiplier) if multiplier > 0 else cooldown_threshold
            
            if self._consecutive_losses[strategy_id] >= effective_threshold:
                reason = f"Cooldown active: {effective_threshold} consecutive losses"
                log_block_reason(
                    "cooldown",
                    measured_value=self._consecutive_losses[strategy_id],
                    threshold=effective_threshold,
                    strategy_id=strategy_id,
                )
                return False, reason
        
        log_allow_reason("cooldown_ok", strategy_id=strategy_id)
        return True, ""

    def record_attempt(self, strategy_id: str) -> None:
        """Record an attempt for a strategy."""
        self._daily_attempts[strategy_id] += 1

    def record_loss(self, strategy_id: str) -> None:
        """Record a loss for a strategy."""
        self._consecutive_losses[strategy_id] += 1

    def record_win(self, strategy_id: str) -> None:
        """Record a win for a strategy (resets consecutive losses)."""
        self._consecutive_losses[strategy_id] = 0

    def register_position(self, position: Position) -> None:
        """Register an open position."""
        self._open_positions[position.position_id] = position

    def unregister_position(self, position_id: str) -> None:
        """Unregister a closed position."""
        self._open_positions.pop(position_id, None)

    def _get_position_for_symbol(self, symbol: str) -> Position | None:
        """Get existing position for a symbol."""
        for pos in self._open_positions.values():
            if pos.symbol == symbol:
                return pos
        return None

    def get_consecutive_losses(self, strategy_id: str) -> int:
        """Get number of consecutive losses for a strategy."""
        return self._consecutive_losses.get(strategy_id, 0)

    def _reset_daily_counters_if_needed(self) -> None:
        """Reset daily counters if a new day has started."""
        today = datetime.utcnow().date()
        if today > self._last_reset_date:
            self._daily_attempts.clear()
            self._last_reset_date = today
            logger.debug("Daily risk counters reset")
