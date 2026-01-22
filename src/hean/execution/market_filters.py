"""Market regime filters for small capital mode.

Filters detect bad market conditions:
- Stale market data
- Low liquidity (wide spreads, low volume)
- Choppy/uncertain conditions
"""

from datetime import datetime, timedelta, timezone

from hean.config import settings
from hean.core.types import Tick
from hean.logging import get_logger

logger = get_logger(__name__)


class MarketFilters:
    """Market condition filters for small capital profit mode."""

    def __init__(self) -> None:
        """Initialize market filters."""
        self._last_tick_time: dict[str, datetime] = {}
        self._blocks_by_reason: dict[str, int] = {}

    def check_tick_staleness(
        self,
        symbol: str,
        tick: Tick | None,
        current_time: datetime | None = None,
    ) -> tuple[bool, str]:
        """Check if market tick is stale.

        Returns:
            (is_stale, reason_code)
        """
        if not settings.small_capital_mode:
            return False, ""

        if not tick:
            return True, "NO_TICK_DATA"

        current_time = current_time or datetime.now(timezone.utc)

        # Check if tick has timestamp
        if hasattr(tick, "timestamp") and tick.timestamp:
            tick_age = (current_time - tick.timestamp).total_seconds()
            if tick_age > settings.stale_tick_max_age_sec:
                logger.debug(
                    f"Stale tick for {symbol}: age={tick_age:.1f}s > "
                    f"{settings.stale_tick_max_age_sec}s"
                )
                self._increment_block("STALE_MARKET_DATA")
                return True, "STALE_MARKET_DATA"

        # Update last seen time
        self._last_tick_time[symbol] = current_time

        return False, ""

    def check_liquidity(
        self,
        symbol: str,
        tick: Tick | None,
    ) -> tuple[bool, str]:
        """Check if market has sufficient liquidity.

        Detects:
        - Wide spreads (low liquidity)
        - Missing bid/ask data

        Returns:
            (should_block, reason_code)
        """
        if not settings.small_capital_mode:
            return False, ""

        if not tick:
            self._increment_block("NO_TICK_DATA")
            return True, "NO_TICK_DATA"

        # Check bid/ask presence
        if not tick.bid or not tick.ask or tick.price <= 0:
            self._increment_block("MISSING_BID_ASK")
            return True, "MISSING_BID_ASK"

        # Check spread width
        spread_bps = ((tick.ask - tick.bid) / tick.price) * 10000

        if spread_bps > settings.max_spread_bps:
            logger.debug(
                f"Wide spread for {symbol}: {spread_bps:.1f} bps > "
                f"{settings.max_spread_bps} bps"
            )
            self._increment_block("SPREAD_TOO_WIDE")
            return True, "SPREAD_TOO_WIDE"

        # Check for abnormally narrow spread (possible data error)
        if spread_bps < 0.1:  # Less than 0.1 bps is suspicious
            logger.warning(
                f"Suspiciously narrow spread for {symbol}: {spread_bps:.1f} bps"
            )
            self._increment_block("INVALID_SPREAD")
            return True, "INVALID_SPREAD"

        return False, ""

    def check_all_filters(
        self,
        symbol: str,
        tick: Tick | None,
        current_time: datetime | None = None,
    ) -> tuple[bool, list[str]]:
        """Run all market filters and return block status.

        Returns:
            (should_block, [reason_codes])
        """
        if not settings.small_capital_mode:
            return False, []

        reasons = []

        # Check staleness
        is_stale, stale_reason = self.check_tick_staleness(symbol, tick, current_time)
        if is_stale:
            reasons.append(stale_reason)

        # Check liquidity (only if not stale)
        if not is_stale:
            should_block, liq_reason = self.check_liquidity(symbol, tick)
            if should_block:
                reasons.append(liq_reason)

        should_block = len(reasons) > 0

        return should_block, reasons

    def _increment_block(self, reason: str) -> None:
        """Track block reason counts."""
        self._blocks_by_reason[reason] = self._blocks_by_reason.get(reason, 0) + 1

    def get_block_reasons_summary(self, top_n: int = 5) -> list[dict[str, any]]:
        """Get top N block reasons.

        Returns:
            List of {reason, count} sorted by count descending
        """
        items = [
            {"reason": reason, "count": count}
            for reason, count in self._blocks_by_reason.items()
        ]
        items.sort(key=lambda x: x["count"], reverse=True)
        return items[:top_n]

    def get_metrics(self) -> dict[str, any]:
        """Return market filter metrics."""
        total_blocks = sum(self._blocks_by_reason.values())
        return {
            "total_market_blocks": total_blocks,
            "blocks_by_reason": dict(self._blocks_by_reason),
            "top_block_reasons": self.get_block_reasons_summary(top_n=5),
        }

    def reset_metrics(self) -> None:
        """Reset block counters."""
        self._blocks_by_reason.clear()
