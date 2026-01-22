"""Multi-symbol scanner for AFO-Director feature."""

from datetime import datetime
from typing import Any

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Tick
from hean.logging import get_logger

logger = get_logger(__name__)


class MultiSymbolScanner:
    """Scans multiple symbols and classifies market regime per symbol.

    Minimal implementation that:
    - Scans symbols sequentially
    - Classifies regime: TREND|RANGE|LOW_LIQ|STALE_DATA
    - Tracks scan state for /trading/why endpoint
    """

    def __init__(self, bus: EventBus) -> None:
        """Initialize multi-symbol scanner.

        Args:
            bus: Event bus for publishing events
        """
        self._bus = bus
        self._enabled = settings.multi_symbol_enabled
        self._symbols = settings.symbols if hasattr(settings, "symbols") else settings.trading_symbols
        self._scan_cursor = 0
        self._last_scanned_symbol: str | None = None
        self._scan_cycle_ts: datetime | None = None
        self._last_tick_at: dict[str, datetime] = {}
        self._last_prices: dict[str, float] = {}
        self._regimes: dict[str, str] = {}  # symbol -> regime

    def get_state(self) -> dict[str, Any]:
        """Get current scanner state for /trading/why endpoint."""
        return {
            "enabled": self._enabled,
            "symbols_count": len(self._symbols),
            "last_scanned_symbol": self._last_scanned_symbol,
            "scan_cursor": self._scan_cursor,
            "scan_cycle_ts": self._scan_cycle_ts.isoformat() if self._scan_cycle_ts else None,
        }

    async def scan_symbol(self, symbol: str, current_price: float | None = None) -> dict[str, Any]:
        """Scan a single symbol and return market analysis.

        Args:
            symbol: Symbol to scan
            current_price: Current price (if available)

        Returns:
            Dictionary with market_regime, market_metrics_short, last_tick_age_sec
        """
        now = datetime.utcnow()

        # Get last tick age
        last_tick_age_sec = None
        if symbol in self._last_tick_at:
            last_tick_age_sec = (now - self._last_tick_at[symbol]).total_seconds()

        # Classify regime
        market_regime = "RANGE"  # Default
        market_metrics_short = {}

        # Check for stale data
        if last_tick_age_sec is not None and last_tick_age_sec > 30:
            market_regime = "STALE_DATA"
        else:
            # Simple regime classification based on price movement
            if symbol in self._last_prices and current_price:
                price_change = abs((current_price - self._last_prices[symbol]) / self._last_prices[symbol])
                if price_change > 0.01:  # >1% change suggests trend
                    market_regime = "TREND"
                else:
                    market_regime = "RANGE"

            # Update last price
            if current_price:
                self._last_prices[symbol] = current_price

        # Store regime
        self._regimes[symbol] = market_regime

        return {
            "market_regime": market_regime,
            "market_metrics_short": market_metrics_short,
            "last_tick_age_sec": last_tick_age_sec,
        }

    async def handle_tick(self, event: Event) -> None:
        """Handle tick event to update last_tick_at."""
        tick: Tick = event.data["tick"]
        self._last_tick_at[tick.symbol] = datetime.utcnow()

        # Update last price
        if tick.price:
            self._last_prices[tick.symbol] = tick.price

    async def start(self) -> None:
        """Start the scanner."""
        if not self._enabled:
            return

        # Subscribe to tick events
        self._bus.subscribe(EventType.TICK, self.handle_tick)
        logger.info(f"Multi-symbol scanner started: {len(self._symbols)} symbols")

    async def stop(self) -> None:
        """Stop the scanner."""
        if not self._enabled:
            return

        self._bus.unsubscribe(EventType.TICK, self.handle_tick)
        logger.info("Multi-symbol scanner stopped")

    def get_regime(self, symbol: str) -> str | None:
        """Get current regime for a symbol."""
        return self._regimes.get(symbol)
