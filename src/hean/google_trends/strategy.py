"""
Google Trends Trading Strategy

Integrates with HEAN trading system
"""

import asyncio
import logging

from hean.core.bus import EventBus
from hean.core.types import Signal
from hean.strategies.base import BaseStrategy

from .analyzer import GoogleTrendsAnalyzer
from .models import TrendsSignal

logger = logging.getLogger(__name__)


class GoogleTrendsStrategy(BaseStrategy):
    """
    Trading strategy based on Google Trends search interest

    Monitors Google search interest for crypto assets and generates
    trading signals based on trend direction and momentum.

    Research shows search interest is a leading indicator:
    - Rising searches → Price increase (24-48h lead)
    - Spikes → High volatility incoming
    - Declining interest → Bearish

    Usage:
        strategy = GoogleTrendsStrategy(
            bus=event_bus,
            symbols=["BTCUSDT", "ETHUSDT"],
            enabled=True,
            timeframe="now 7-d",
            min_interest=40,
            min_confidence=0.7
        )
    """

    def __init__(
        self,
        bus: EventBus,
        symbols: list[str] | None = None,
        enabled: bool = True,
        timeframe: str = "now 7-d",
        min_interest: int = 40,
        min_momentum: float = 0.2,
        min_confidence: float = 0.7,
        check_interval_seconds: int = 3600,  # 1 hour (don't query too often!)
        cooldown_hours: int = 24,  # Don't retrade same symbol for 24h
        stop_loss_pct: float = 2.0,  # Stop loss percentage (default 2%)
        take_profit_pct: float = 6.0,  # Take profit percentage (default 6%, 1:3 R:R)
        use_dynamic_risk: bool = True  # Adjust risk based on signal confidence
    ):
        """
        Initialize Google Trends strategy

        Args:
            bus: Event bus
            symbols: Symbols to monitor
            enabled: Enable strategy
            timeframe: Google Trends timeframe
            min_interest: Minimum interest score (0-100)
            min_momentum: Minimum momentum (-1 to +1)
            min_confidence: Minimum signal confidence (0-1)
            check_interval_seconds: How often to check trends
            cooldown_hours: Hours before retrading same symbol
        """
        super().__init__("google_trends", bus)

        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.enabled = enabled
        self.timeframe = timeframe
        self.min_interest = min_interest
        self.min_momentum = min_momentum
        self.min_confidence = min_confidence
        self.check_interval_seconds = check_interval_seconds
        self.cooldown_hours = cooldown_hours

        # Risk management parameters
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_dynamic_risk = use_dynamic_risk

        # Initialize analyzer
        self.analyzer = GoogleTrendsAnalyzer(
            timeframe=timeframe,
            min_interest=min_interest,
            min_momentum=min_momentum
        )

        # Track recent signals (for cooldown)
        self._recent_signals = {}  # symbol -> last_signal_time

        # Stats
        self.signals_generated = 0
        self.trades_executed = 0

        # Price cache updated from EventBus TICK events
        self._last_prices: dict[str, float] = {}

        self._initialized = False

    async def initialize(self):
        """Initialize strategy"""
        if self._initialized:
            return

        await self.analyzer.initialize()
        self._initialized = True

        logger.info(
            f"Google Trends strategy initialized: "
            f"symbols={self.symbols}, "
            f"timeframe={self.timeframe}, "
            f"min_interest={self.min_interest}"
        )

    async def on_tick(self, event):
        """Handle tick events to update price cache."""
        tick = event.data.get("tick")
        if tick and hasattr(tick, "symbol") and hasattr(tick, "price") and tick.price > 0:
            self._last_prices[tick.symbol] = tick.price

    async def on_funding(self, event):
        """Handle funding events - not used"""
        pass

    async def run(self):
        """
        Main strategy loop

        Periodically checks Google Trends and generates signals
        """
        if not self._initialized:
            await self.initialize()

        if not self.enabled:
            logger.info("Google Trends strategy is disabled")
            return

        logger.info("Starting Google Trends strategy")

        while self.enabled:
            try:
                await self._check_trends()
                await asyncio.sleep(self.check_interval_seconds)

            except Exception as e:
                logger.error(f"Error in Google Trends loop: {e}")
                await asyncio.sleep(self.check_interval_seconds)

    async def _check_trends(self):
        """Check trends for all symbols"""
        # Analyze all symbols
        signals = await self.analyzer.analyze_comparative(
            self.symbols,
            timeframe=self.timeframe
        )

        # Rank by opportunity
        ranked = self.analyzer.compare_signals(signals)

        # Process top signals
        for symbol, signal in ranked:
            if self._should_trade_signal(symbol, signal):
                await self._execute_signal(symbol, signal)

    def _should_trade_signal(self, symbol: str, signal: TrendsSignal) -> bool:
        """Check if signal meets our criteria"""
        # Basic checks
        if not signal.should_trade:
            return False

        if signal.confidence < self.min_confidence:
            return False

        # Check cooldown
        if symbol in self._recent_signals:
            from datetime import datetime, timedelta
            last_signal_time = self._recent_signals[symbol]
            cooldown_end = last_signal_time + timedelta(hours=self.cooldown_hours)

            if datetime.utcnow() < cooldown_end:
                logger.debug(f"Symbol {symbol} in cooldown period")
                return False

        return True

    def _calculate_risk_levels(
        self,
        entry_price: float,
        side: str,
        confidence: float
    ) -> tuple[float | None, float | None]:
        """
        Calculate stop loss and take profit levels

        Args:
            entry_price: Entry price for the trade
            side: Trade side ("buy" or "sell")
            confidence: Signal confidence (0-1)

        Returns:
            Tuple of (stop_loss, take_profit) prices
        """
        if entry_price <= 0:
            logger.warning("Invalid entry price, cannot calculate risk levels")
            return None, None

        # Adjust risk based on confidence if enabled
        stop_pct = self.stop_loss_pct
        profit_pct = self.take_profit_pct

        if self.use_dynamic_risk:
            # Higher confidence = wider stops, larger targets
            # Lower confidence = tighter stops, smaller targets
            confidence_multiplier = 0.5 + (confidence * 0.5)  # Range: 0.5x to 1.0x
            stop_pct *= confidence_multiplier
            profit_pct *= confidence_multiplier

        # Calculate levels based on side
        if side == "buy":
            stop_loss = entry_price * (1 - stop_pct / 100)
            take_profit = entry_price * (1 + profit_pct / 100)
        else:  # sell
            stop_loss = entry_price * (1 + stop_pct / 100)
            take_profit = entry_price * (1 - profit_pct / 100)

        logger.debug(
            f"Risk levels calculated: SL={stop_loss:.2f}, "
            f"TP={take_profit:.2f}, confidence={confidence:.0%}"
        )

        return stop_loss, take_profit

    async def _execute_signal(self, symbol: str, signal: TrendsSignal):
        """Execute trading signal"""
        try:
            logger.info(
                f"Executing Google Trends signal: {symbol} {signal.action} "
                f"(confidence: {signal.confidence:.0%}, "
                f"interest: {signal.interest_score}, "
                f"direction: {signal.trend_direction.value})"
            )

            # Get current price from bus state
            entry_price = await self._get_current_price(symbol)
            if entry_price is None:
                # Skip signal if no price data available
                logger.debug(f"Skipping signal for {symbol} - no price data")
                continue

            # Calculate risk management levels
            side = "buy" if signal.action == "BUY" else "sell"
            stop_loss, take_profit = self._calculate_risk_levels(
                entry_price=entry_price,
                side=side,
                confidence=signal.confidence
            )

            # Convert to HEAN signal
            hean_signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    "source": "google_trends",
                    "interest_score": signal.interest_score,
                    "interest_level": signal.interest_level.value,
                    "trend_direction": signal.trend_direction.value,
                    "momentum": signal.momentum,
                    "confidence": signal.confidence,
                    "reason": signal.reason,
                    "timeframe": self.timeframe,
                    "risk_pct": self.stop_loss_pct,
                    "reward_pct": self.take_profit_pct,
                    "dynamic_risk_applied": self.use_dynamic_risk
                }
            )

            # Publish signal
            await self._publish_signal(hean_signal)

            # Update tracking
            from datetime import datetime
            self._recent_signals[symbol] = datetime.utcnow()
            self.signals_generated += 1
            self.trades_executed += 1

            logger.info(
                f"Google Trends signal published for {symbol}: "
                f"Entry={entry_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}"
            )

        except Exception as e:
            logger.error(f"Error executing signal: {e}", exc_info=True)

    async def _get_current_price(self, symbol: str) -> float | None:
        """Get current market price for symbol from price cache.

        The cache is updated via on_tick events from the EventBus.
        Returns None if no price data available.
        """
        if symbol in self._last_prices:
            price = self._last_prices[symbol]
            if price > 0:
                return price

        # No price data available - return None to skip signal
        logger.warning(
            f"No price data available for {symbol}, cannot generate signal. "
            f"Waiting for TICK events."
        )
        return None

    def get_stats(self) -> dict:
        """Get strategy statistics"""
        return {
            "strategy": "google_trends",
            "enabled": self.enabled,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "signals_generated": self.signals_generated,
            "trades_executed": self.trades_executed,
            "min_interest": self.min_interest,
            "min_confidence": self.min_confidence,
            "cooldown_hours": self.cooldown_hours,
            "active_symbols": list(self._recent_signals.keys())
        }


# Example usage
async def main():
    """Example usage"""
    from hean.core.bus import EventBus

    bus = EventBus()

    strategy = GoogleTrendsStrategy(
        bus=bus,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        enabled=True,
        timeframe="now 7-d",
        min_interest=40,
        min_confidence=0.7,
        check_interval_seconds=3600  # Check every hour
    )

    await strategy.initialize()

    # Run one check
    await strategy._check_trends()

    # Print stats
    stats = strategy.get_stats()
    print("\nStrategy Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
