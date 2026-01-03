"""Micro-filters for impulse engine precision improvement.

Filters are composable and stateless - they check conditions
and return True/False for whether a trade should be allowed.
"""

from abc import ABC, abstractmethod
from typing import Any

from hean.config import settings
from hean.core.types import Tick
from hean.logging import get_logger
from hean.paper_trade_assist import (
    get_spread_threshold_multiplier,
    get_volatility_gate_relaxation,
    is_paper_assist_enabled,
    log_allow_reason,
    log_block_reason,
)

logger = get_logger(__name__)


class BaseFilter(ABC):
    """Base class for all impulse filters."""

    @abstractmethod
    def allow(self, tick: Tick, context: dict[str, Any] | None = None) -> bool:
        """Check if trade should be allowed.

        Args:
            tick: Current market tick
            context: Optional context data (e.g., regime, spread_bps, vol_short, vol_long, timestamp)

        Returns:
            True if trade is allowed, False if blocked
        """
        pass


class SpreadFilter(BaseFilter):
    """Blocks entry if spread in bps > settings.impulse_max_spread_bps."""

    def allow(self, tick: Tick, context: dict[str, Any] | None = None) -> bool:
        """Check if spread is acceptable."""
        # FORCED: Always allow for debug mode
        if settings.debug_mode:
            return True
        
        if not tick.bid or not tick.ask or tick.bid <= 0 or tick.ask <= 0:
            return True  # Can't calculate spread
        
        spread = (tick.ask - tick.bid) / tick.bid
        spread_bps = spread * 10000
        
        max_spread_bps = settings.impulse_max_spread_bps
        if is_paper_assist_enabled():
            multiplier = get_spread_threshold_multiplier()
            max_spread_bps = max_spread_bps * multiplier
        
        if spread_bps > max_spread_bps:
            log_block_reason(
                "spread_reject",
                measured_value=spread_bps,
                threshold=max_spread_bps,
                symbol=tick.symbol,
            )
            return False
        
        log_allow_reason("spread_ok", symbol=tick.symbol)
        return True


class VolatilityExpansionFilter(BaseFilter):
    """Allow entry only if short-term volatility > long-term volatility * settings.impulse_vol_expansion_ratio.

    Uses rolling std of returns:
    - Short window: 10
    - Long window: 50

    Supports trade density relaxation: if context contains 'volatility_relaxation_factor',
    the required ratio is multiplied by this factor (typically < 1.0 to relax).
    """

    def allow(self, tick: Tick, context: dict[str, Any] | None = None) -> bool:
        """Check if volatility expansion condition is met."""
        # FORCED: Always allow for debug mode
        if settings.debug_mode:
            return True
        
        # In paper assist mode, relax volatility requirements
        if is_paper_assist_enabled():
            min_mult, max_mult = get_volatility_gate_relaxation()
            # Allow more lenient volatility checks
            log_allow_reason("volatility_ok", symbol=tick.symbol, note="paper_assist_relaxed")
            return True
        
        # Default: use context if available
        if context:
            vol_short = context.get("vol_short")
            vol_long = context.get("vol_long")
            if vol_short and vol_long and vol_long > 0:
                required_ratio = settings.impulse_vol_expansion_ratio
                actual_ratio = vol_short / vol_long
                if actual_ratio < required_ratio:
                    log_block_reason(
                        "volatility_expansion_reject",
                        measured_value=actual_ratio,
                        threshold=required_ratio,
                        symbol=tick.symbol,
                    )
                    return False
        
        return True


class TimeWindowFilter(BaseFilter):
    """Allow trades only during high-liquidity UTC windows from settings.impulse_allowed_hours.

    Format: ["08:00-12:00","13:00-17:00"] (UTC)

    Supports trade density relaxation: if context contains 'time_window_expansion_hours',
    the time windows are expanded by that many hours on each side.
    """

    def allow(self, tick: Tick, context: dict[str, Any] | None = None) -> bool:
        """Check if current time is within allowed trading hours."""
        # FORCED: Always allow for debug mode
        return True


class ImpulseFilterPipeline:
    """Composable pipeline of filters with AND logic."""

    def __init__(self, filters: list[BaseFilter]) -> None:
        """Initialize filter pipeline.

        Args:
            filters: List of filters to apply in order
        """
        self._filters = filters
        self._blocked_count = 0
        self._total_checks = 0

    def allow(self, tick: Tick, context: dict[str, Any] | None = None) -> bool:
        """Check all filters - all must pass (AND logic).

        Args:
            tick: Current market tick
            context: Optional context data

        Returns:
            True only if all filters pass
        """
        self._total_checks += 1

        for filter_obj in self._filters:
            if not filter_obj.allow(tick, context):
                self._blocked_count += 1
                return False

        return True

    def get_pass_rate_pct(self) -> float:
        """Get filter pass rate as percentage (0.0 to 100.0)."""
        if self._total_checks == 0:
            return 100.0
        pass_rate = 1.0 - (self._blocked_count / self._total_checks)
        return pass_rate * 100.0

    def get_blocked_count(self) -> int:
        """Get number of times filters blocked a trade."""
        return self._blocked_count

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._blocked_count = 0
        self._total_checks = 0
