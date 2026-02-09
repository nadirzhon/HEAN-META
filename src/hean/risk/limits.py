"""Risk limits and constraints."""

from collections import defaultdict
from datetime import datetime

from hean.config import settings
from hean.core.types import OrderRequest, Position
from hean.logging import get_logger
from hean.observability.signal_rejection_telemetry import signal_rejection_telemetry
from hean.paper_trade_assist import (
    get_cooldown_multiplier,
    get_daily_attempts_multiplier,
    get_max_open_positions_override,
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
            signal_rejection_telemetry.record_rejection(
                reason="risk_limits_reject",
                symbol=order_request.symbol,
                strategy_id=order_request.strategy_id,
                details={"current": len(self._open_positions), "max": max_positions},
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
            signal_rejection_telemetry.record_rejection(
                reason="position_size_reject",
                symbol=order_request.symbol,
                strategy_id=order_request.strategy_id,
                details={"existing_position": existing_pos.position_id},
            )
            return False, reason

        # HEAN v2 Iron Rule #5: Minimum R:R = 1:2
        if order_request.stop_loss is not None and order_request.take_profit is not None:
            entry_price = order_request.price or 0.0
            if entry_price > 0:
                risk = abs(entry_price - order_request.stop_loss)
                reward = abs(order_request.take_profit - entry_price)
                if risk > 0:
                    rr_ratio = reward / risk
                    min_rr = settings.min_risk_reward_ratio
                    if rr_ratio < min_rr:
                        reason = f"R:R ratio {rr_ratio:.2f} below minimum {min_rr:.1f}"
                        log_block_reason(
                            "rr_ratio_too_low",
                            measured_value=rr_ratio,
                            threshold=min_rr,
                            symbol=order_request.symbol,
                            strategy_id=order_request.strategy_id,
                        )
                        signal_rejection_telemetry.record_rejection(
                            reason="rr_ratio_reject",
                            symbol=order_request.symbol,
                            strategy_id=order_request.strategy_id,
                            details={"rr_ratio": rr_ratio, "min_rr": min_rr},
                        )
                        return False, reason

        log_allow_reason("risk_limits_ok", symbol=order_request.symbol, strategy_id=order_request.strategy_id)
        return True, ""

    def check_daily_attempts(self, strategy_id: str, regime=None) -> tuple[bool, str]:
        """Check if strategy has exceeded daily attempt limit."""
        self._reset_daily_counters_if_needed()

        # Universal daily attempts limit for all strategies
        attempts_map = {
            "impulse_engine": settings.impulse_max_attempts_per_day,
            "hf_scalping": 60,
            "momentum_trader": 40,
            "paper_assist_micro": 100,
        }

        max_attempts = attempts_map.get(strategy_id, 50)  # default: 50

        # Stricter limit in IMPULSE regime (only for impulse_engine)
        if strategy_id == "impulse_engine" and regime and regime.value == "impulse":
            max_attempts = int(max_attempts * 0.8)  # 20% reduction

        if self._daily_attempts[strategy_id] >= max_attempts:
            reason = f"Daily attempt limit ({max_attempts}) reached for {strategy_id}"
            log_block_reason(
                "daily_attempts",
                measured_value=self._daily_attempts[strategy_id],
                threshold=max_attempts,
                strategy_id=strategy_id,
            )
            signal_rejection_telemetry.record_rejection(
                reason="daily_attempts_reject",
                strategy_id=strategy_id,
                details={"current": self._daily_attempts[strategy_id], "max": max_attempts},
            )
            return False, reason

        log_allow_reason("daily_attempts_ok", strategy_id=strategy_id)
        return True, ""

    def check_cooldown(self, strategy_id: str) -> tuple[bool, str]:
        """Check if strategy is in cooldown period."""
        # Aggressive Mode: Bypass cooldowns completely when DEBUG_MODE=True
        if settings.debug_mode:
            log_allow_reason("cooldown_bypassed_debug", strategy_id=strategy_id, note="DEBUG_MODE bypass")
            return True, ""

        # Universal cooldown for all strategies
        cooldown_map = {
            "impulse_engine": settings.impulse_cooldown_after_losses,
            "hf_scalping": 2,
            "momentum_trader": 3,
            "paper_assist_micro": 5,
        }

        threshold = cooldown_map.get(strategy_id, 3)  # default: 3

        if self._consecutive_losses[strategy_id] >= threshold:
            reason = f"Cooldown: {threshold} consecutive losses for {strategy_id}"
            log_block_reason(
                "cooldown",
                measured_value=self._consecutive_losses[strategy_id],
                threshold=threshold,
                strategy_id=strategy_id,
            )
            signal_rejection_telemetry.record_rejection(
                reason="cooldown_reject",
                strategy_id=strategy_id,
                details={
                    "consecutive_losses": self._consecutive_losses[strategy_id],
                    "threshold": threshold,
                },
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
