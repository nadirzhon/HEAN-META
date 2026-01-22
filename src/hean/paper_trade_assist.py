"""Paper Trade Assist - softens filters and provides fallback micro-trades for paper mode.

This module provides helper functions to adjust thresholds and limits when
PAPER_TRADE_ASSIST is enabled, making it easier to see trades in paper/dry_run mode.
"""

from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)


def is_paper_assist_enabled() -> bool:
    """Check if paper trade assist is enabled and safe to use."""
    if not settings.paper_trade_assist:
        return False
    
    # Double-check safety (should be validated in config, but extra safety)
    is_paper_safe = settings.dry_run or settings.bybit_testnet
    is_live_unsafe = not settings.dry_run and settings.is_live
    
    if is_live_unsafe:
        logger.error("PAPER_TRADE_ASSIST enabled in live mode - this should not happen!")
        return False
    
    return is_paper_safe


def get_spread_threshold_multiplier() -> float:
    """Get multiplier for spread threshold (higher = more lenient).
    
    Returns:
        Multiplier (default 2.5x for paper assist, 2.0x for debug mode, 1.0x otherwise)
        In Aggressive Mode (both enabled), returns 2.0x effectively doubling MAX_ALLOWED_SPREAD
    """
    # Aggressive Mode: Double the spread tolerance when both enabled
    if settings.debug_mode:
        if is_paper_assist_enabled():
            return 2.0  # Double tolerance (100% increase)
        return 2.0  # Still double in debug mode alone
    if is_paper_assist_enabled():
        return 2.5  # Original paper assist multiplier
    return 1.0


def get_volatility_gate_relaxation() -> tuple[float, float]:
    """Get volatility gate relaxation factors.
    
    Returns:
        (min_multiplier, max_multiplier) - multipliers for min/max volatility thresholds
        Lower min = more lenient, higher max = more lenient
    """
    if is_paper_assist_enabled():
        return (0.5, 1.5)  # 50% lower min, 50% higher max
    return (1.0, 1.0)


def get_edge_threshold_reduction_pct() -> float:
    """Get percentage reduction for edge thresholds.
    
    Returns:
        Reduction percentage (e.g., 40.0 means reduce threshold by 40%)
        In Aggressive Mode (DEBUG_MODE=True), reduces to near-zero (95% reduction)
    """
    # Aggressive Mode: Reduce edge requirement to near-zero
    if settings.debug_mode:
        return 95.0  # 95% reduction = near-zero threshold
    if is_paper_assist_enabled():
        return 40.0  # Reduce threshold by 40%
    return 0.0


def get_max_open_positions_override() -> int | None:
    """Get override for max_open_positions (guarantee at least 1-2 positions).
    
    Returns:
        Override value or None to use default
    """
    if is_paper_assist_enabled():
        return max(2, settings.max_open_positions)  # At least 2
    return None


def get_min_notional_override() -> float | None:
    """Get override for minimum notional (guarantee trades can pass).
    
    Returns:
        Override value in USD or None to use default
    """
    if is_paper_assist_enabled():
        return 10.0  # 10 USD minimum
    return None


def get_cooldown_multiplier() -> float:
    """Get multiplier for cooldown periods (lower = shorter cooldown).
    
    Returns:
        Multiplier (default 0.33x = 3x shorter for paper assist)
        In Aggressive Mode (DEBUG_MODE=True), returns 0.0 to bypass cooldowns
    """
    # Aggressive Mode: Bypass cooldowns completely
    if settings.debug_mode:
        return 0.0  # Effectively bypass cooldowns (0 duration)
    if is_paper_assist_enabled():
        return 0.33  # 3x shorter
    return 1.0


def get_daily_attempts_multiplier() -> float:
    """Get multiplier for daily attempt limits (higher = more attempts).
    
    Returns:
        Multiplier (default 2.0x for paper assist)
    """
    if is_paper_assist_enabled():
        return 2.0  # 2x more attempts
    return 1.0


def should_allow_regime(regime_value: str) -> bool:
    """Check if regime should be allowed (paper assist allows neutral/chop).
    
    Args:
        regime_value: Regime value (e.g., "normal", "impulse", "range")
    
    Returns:
        True if regime should be allowed
    """
    if not is_paper_assist_enabled():
        return True  # Use default logic
    
    # In paper assist, allow all regimes (including neutral/chop)
    return True


def log_block_reason(
    reason_code: str,
    measured_value: float | None = None,
    threshold: float | None = None,
    symbol: str = "",
    strategy_id: str = "",
    agent_name: str = "",
    reasons: list[str] | None = None,
    suggested_fix: list[str] | None = None,
) -> None:
    """Log a block reason with diagnostic information.
    
    Args:
        reason_code: Reason code (e.g., "spread_reject", "edge_reject")
        measured_value: Measured value that caused block
        threshold: Threshold that was exceeded
        symbol: Trading symbol
        strategy_id: Strategy ID
        agent_name: Agent/strategy name (for compatibility with warden-style logging)
    """
    # Use agent_name if provided, otherwise use strategy_id
    agent_display = agent_name or strategy_id or "unknown"
    
    # Format: [SIGNAL REJECTED] Agent: {name} | Reason: {reason} | Current Value: {value} | Threshold: {limit}
    msg_parts = [f"[SIGNAL REJECTED] Agent: {agent_display} | Reason: {reason_code}"]
    if measured_value is not None:
        msg_parts.append(f"| Current Value: {measured_value:.6f}")
    if threshold is not None:
        msg_parts.append(f"| Threshold: {threshold:.6f}")
    if symbol:
        msg_parts.append(f"| Symbol: {symbol}")
    if reasons:
        msg_parts.append(f"| Reasons: {', '.join(reasons)}")
    if suggested_fix:
        msg_parts.append(f"| Fix: {', '.join(suggested_fix)}")
    
    # Always log at INFO level for visibility in docker-compose logs
    logger.info(" ".join(msg_parts))
    
    # Also log detailed debug info
    debug_parts = [f"[BLOCK] {reason_code}"]
    if symbol:
        debug_parts.append(f"symbol={symbol}")
    if strategy_id:
        debug_parts.append(f"strategy={strategy_id}")
    if measured_value is not None:
        debug_parts.append(f"measured={measured_value:.6f}")
    if threshold is not None:
        debug_parts.append(f"threshold={threshold:.6f}")
    if reasons:
        debug_parts.append(f"reasons={reasons}")
    if suggested_fix:
        debug_parts.append(f"fix={suggested_fix}")
    logger.debug(" ".join(debug_parts))


def log_allow_reason(
    reason_code: str,
    symbol: str = "",
    strategy_id: str = "",
    note: str = "",
) -> None:
    """Log an allow reason.
    
    Args:
        reason_code: Reason code (e.g., "ALLOW", "PAPER_ASSIST_OVERRIDE")
        symbol: Trading symbol
        strategy_id: Strategy ID
        note: Additional note
    """
    msg_parts = [f"[ALLOW] {reason_code}"]
    if symbol:
        msg_parts.append(f"symbol={symbol}")
    if strategy_id:
        msg_parts.append(f"strategy={strategy_id}")
    if note:
        msg_parts.append(f"note={note}")
    
    logger.info(" ".join(msg_parts))
