"""
Multi-Exchange Funding Arbitrage Strategy

Integrates with HEAN trading system
"""

import logging

from hean.core.bus import EventBus
from hean.core.types import Signal
from hean.strategies.base import BaseStrategy

from .aggregator import FundingArbitrageAggregator
from .models import ArbitrageSignal, FundingOpportunity

logger = logging.getLogger(__name__)


class FundingArbitrageStrategy(BaseStrategy):
    """
    Multi-exchange funding arbitrage strategy

    Monitors funding rates across Bybit, Binance, and OKX.
    Opens hedged positions when significant spreads are detected.

    Example:
        If Binance funding is +0.05% and Bybit is -0.02%:
        - Long on Bybit (receive 0.02%)
        - Short on Binance (receive 0.05%)
        - Net profit: 0.07% per funding period

    Usage:
        strategy = FundingArbitrageStrategy(
            bus=event_bus,
            symbols=["BTCUSDT", "ETHUSDT"],
            enabled=True,
            min_spread_pct=0.02,  # Minimum 0.02% spread
            position_size_usd=1000
        )
    """

    def __init__(
        self,
        bus: EventBus,
        symbols: list[str] | None = None,
        enabled: bool = True,
        testnet: bool = True,
        min_spread_pct: float = 0.02,  # 0.02% minimum spread
        min_confidence: float = 0.7,  # 70% minimum confidence
        position_size_usd: float = 1000.0,
        max_positions: int = 3,
        check_interval_seconds: int = 300  # 5 minutes
    ):
        """
        Initialize funding arbitrage strategy

        Args:
            bus: Event bus
            symbols: Symbols to monitor (default: BTC, ETH)
            enabled: Enable strategy
            testnet: Use testnet exchanges
            min_spread_pct: Minimum spread to trade (%)
            min_confidence: Minimum confidence (0-1)
            position_size_usd: Position size per arbitrage
            max_positions: Maximum concurrent positions
            check_interval_seconds: How often to check for opportunities
        """
        super().__init__("funding_arbitrage", bus)

        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.enabled = enabled
        self.testnet = testnet
        self.min_spread = min_spread_pct / 100  # Convert to decimal
        self.min_confidence = min_confidence
        self.position_size_usd = position_size_usd
        self.max_positions = max_positions
        self.check_interval_seconds = check_interval_seconds

        # Initialize aggregator
        self.aggregator = FundingArbitrageAggregator(testnet=testnet)

        # Track active positions
        self.active_positions: dict[str, ArbitrageSignal] = {}

        # Stats
        self.opportunities_found = 0
        self.trades_executed = 0
        self.total_profit = 0.0

        self._initialized = False

    async def initialize(self):
        """Initialize strategy"""
        if self._initialized:
            return

        await self.aggregator.initialize()
        self._initialized = True

        logger.info(
            f"Funding arbitrage strategy initialized: "
            f"symbols={self.symbols}, "
            f"min_spread={self.min_spread * 100:.3f}%, "
            f"min_confidence={self.min_confidence:.0%}"
        )

    async def on_tick(self, event):
        """Handle tick events - not used for this strategy"""
        pass

    async def on_funding(self, event):
        """Handle funding rate events - not used (we poll exchanges directly)"""
        pass

    async def run(self):
        """
        Main strategy loop

        Continuously checks for arbitrage opportunities
        """
        if not self._initialized:
            await self.initialize()

        if not self.enabled:
            logger.info("Funding arbitrage strategy is disabled")
            return

        logger.info("Starting funding arbitrage strategy")

        import asyncio

        while self.enabled:
            try:
                await self._check_opportunities()
                await asyncio.sleep(self.check_interval_seconds)

            except Exception as e:
                logger.error(f"Error in funding arbitrage loop: {e}")
                await asyncio.sleep(self.check_interval_seconds)

    async def _check_opportunities(self):
        """Check for arbitrage opportunities"""
        # Check if we have capacity for more positions
        if len(self.active_positions) >= self.max_positions:
            logger.debug("Max positions reached, skipping opportunity check")
            return

        all_opportunities = []

        # Check each symbol
        for symbol in self.symbols:
            opportunities = await self.aggregator.find_opportunities(symbol)
            all_opportunities.extend(opportunities)

        # Filter opportunities
        tradeable = [
            opp for opp in all_opportunities
            if self._is_tradeable(opp)
        ]

        self.opportunities_found += len(tradeable)

        if tradeable:
            logger.info(f"Found {len(tradeable)} tradeable arbitrage opportunities")

            # Take best opportunity
            best_opp = tradeable[0]  # Already sorted by profit potential
            await self._execute_arbitrage(best_opp)

    def _is_tradeable(self, opportunity: FundingOpportunity) -> bool:
        """Check if opportunity meets our criteria"""
        # Basic checks from opportunity
        if not opportunity.should_trade:
            return False

        # Our strategy-specific criteria
        if abs(opportunity.funding_spread) < self.min_spread:
            return False

        if opportunity.confidence < self.min_confidence:
            return False

        # Check if we already have a position in this symbol
        if opportunity.symbol in self.active_positions:
            return False

        return True

    async def _execute_arbitrage(self, opportunity: FundingOpportunity):
        """
        Execute arbitrage trade

        Opens hedged positions on both exchanges
        """
        try:
            # Generate signal
            signal = await self.aggregator.generate_signal(
                opportunity,
                position_size_usd=self.position_size_usd
            )

            logger.info(
                f"Executing arbitrage: {opportunity.symbol} "
                f"Long {opportunity.long_exchange.value} / "
                f"Short {opportunity.short_exchange.value} "
                f"Spread: {opportunity.funding_spread * 100:.4f}% "
                f"Expected: ${signal.expected_profit_usd:.2f}"
            )

            # Convert to HEAN signals (one for each exchange)
            # NOTE: This requires multi-exchange support in HEAN execution layer

            # Long signal
            long_signal = Signal(
                strategy_id=self.strategy_id,
                symbol=opportunity.symbol,
                side="buy",
                entry_price=signal.entry_prices.get(opportunity.long_exchange, 0.0),
                stop_loss=None,  # Hedged, no stop needed
                take_profit=None,  # Close based on time/funding
                metadata={
                    "arbitrage_type": "funding",
                    "exchange": opportunity.long_exchange.value,
                    "funding_rate": opportunity.long_rate,
                    "target_hold_hours": signal.target_hold_hours,
                    "pair_exchange": opportunity.short_exchange.value,
                    "expected_profit_usd": signal.expected_profit_usd
                }
            )

            # Short signal
            short_signal = Signal(
                strategy_id=self.strategy_id,
                symbol=opportunity.symbol,
                side="sell",
                entry_price=signal.entry_prices.get(opportunity.short_exchange, 0.0),
                stop_loss=None,  # Hedged, no stop needed
                take_profit=None,  # Close based on time/funding
                metadata={
                    "arbitrage_type": "funding",
                    "exchange": opportunity.short_exchange.value,
                    "funding_rate": opportunity.short_rate,
                    "target_hold_hours": signal.target_hold_hours,
                    "pair_exchange": opportunity.long_exchange.value,
                    "expected_profit_usd": signal.expected_profit_usd
                }
            )

            # Publish signals
            await self._publish_signal(long_signal)
            await self._publish_signal(short_signal)

            # Track position
            self.active_positions[opportunity.symbol] = signal
            self.trades_executed += 1

            logger.info(f"Arbitrage signals published for {opportunity.symbol}")

        except Exception as e:
            logger.error(f"Error executing arbitrage: {e}")

    async def close_position(self, symbol: str, profit_usd: float):
        """
        Close arbitrage position

        Args:
            symbol: Symbol to close
            profit_usd: Realized profit
        """
        if symbol in self.active_positions:
            del self.active_positions[symbol]

            self.total_profit += profit_usd

            logger.info(
                f"Closed arbitrage position: {symbol} "
                f"Profit: ${profit_usd:.2f} "
                f"Total profit: ${self.total_profit:.2f}"
            )

    def get_stats(self) -> dict:
        """Get strategy statistics"""
        return {
            "strategy": "funding_arbitrage",
            "enabled": self.enabled,
            "opportunities_found": self.opportunities_found,
            "trades_executed": self.trades_executed,
            "active_positions": len(self.active_positions),
            "total_profit_usd": self.total_profit,
            "avg_profit_per_trade": (
                self.total_profit / self.trades_executed
                if self.trades_executed > 0 else 0
            ),
            "symbols": self.symbols,
            "min_spread_pct": self.min_spread * 100,
            "min_confidence_pct": self.min_confidence * 100,
        }


# Example usage
async def main():
    """Example usage"""
    from hean.core.bus import EventBus

    bus = EventBus()

    strategy = FundingArbitrageStrategy(
        bus=bus,
        symbols=["BTCUSDT", "ETHUSDT"],
        enabled=True,
        testnet=True,
        min_spread_pct=0.02,
        position_size_usd=1000
    )

    await strategy.initialize()

    # Check once
    await strategy._check_opportunities()

    # Print stats
    stats = strategy.get_stats()
    print("\nStrategy Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
