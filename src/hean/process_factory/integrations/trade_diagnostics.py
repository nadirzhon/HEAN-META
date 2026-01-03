"""Trade blocking diagnostics for Process Factory.

Emits structured log events when trades are blocked with reasons and suggested fixes.
"""

from typing import Any

from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)


def log_trade_blocked(
    symbol: str,
    strategy_id: str,
    reasons: list[str],
    suggested_fix: list[str] | None = None,
) -> None:
    """Log a trade blocked event with structured information.

    Args:
        symbol: Trading symbol
        strategy_id: Strategy ID
        reasons: List of reason strings why trade was blocked
        suggested_fix: Optional list of suggested fixes
    """
    if suggested_fix is None:
        suggested_fix = []
    
    logger.warning(
        f"event=trade_blocked symbol={symbol} strategy={strategy_id} "
        f"reasons={reasons} suggested_fix={suggested_fix}"
    )
    
    # Also print to console for visibility
    print(f"\n⚠ Trade blocked: {symbol} ({strategy_id})")
    print(f"  Reasons: {', '.join(reasons)}")
    if suggested_fix:
        print(f"  Suggested fixes:")
        for fix in suggested_fix:
            print(f"    - {fix}")


def check_live_enabled() -> tuple[bool, list[str], list[str]]:
    """Check if live trading is enabled.

    Returns:
        Tuple of (is_enabled, reasons, suggested_fix)
    """
    reasons = []
    fixes = []
    
    if not settings.is_live:
        reasons.append("live_disabled")
        fixes.append("Set LIVE_CONFIRM=YES and trading_mode=live")
    
    return len(reasons) == 0, reasons, fixes


def check_dry_run() -> tuple[bool, list[str], list[str]]:
    """Check if dry run mode is blocking trades.

    Returns:
        Tuple of (is_allowed, reasons, suggested_fix)
    """
    reasons = []
    fixes = []
    
    if settings.dry_run:
        reasons.append("dry_run")
        fixes.append("Set DRY_RUN=false to allow real orders")
    
    return len(reasons) == 0, reasons, fixes


def check_process_factory_actions() -> tuple[bool, list[str], list[str]]:
    """Check if Process Factory actions are enabled.

    Returns:
        Tuple of (is_enabled, reasons, suggested_fix)
    """
    reasons = []
    fixes = []
    
    if not settings.process_factory_allow_actions:
        reasons.append("process_factory_allow_actions_false")
        fixes.append("Set PROCESS_FACTORY_ALLOW_ACTIONS=true")
    
    return len(reasons) == 0, reasons, fixes

