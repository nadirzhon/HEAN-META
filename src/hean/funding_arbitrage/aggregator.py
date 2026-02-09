"""
Funding arbitrage aggregator - compares rates across exchanges
"""

import asyncio
import logging
from datetime import datetime

from .binance_funding import BinanceFundingClient
from .bybit_funding import BybitFundingClient
from .models import (
    ArbitrageSignal,
    ExchangeFundingRate,
    ExchangeName,
    FundingOpportunity,
    FundingRateSpread,
)
from .okx_funding import OKXFundingClient

logger = logging.getLogger(__name__)


class FundingArbitrageAggregator:
    """
    Aggregates funding rates from multiple exchanges and finds arbitrage opportunities

    Usage:
        aggregator = FundingArbitrageAggregator()
        await aggregator.initialize()

        opportunities = await aggregator.find_opportunities("BTCUSDT")
        for opp in opportunities:
            if opp.should_trade:
                print(f"Arbitrage opportunity: {opp}")
    """

    def __init__(
        self,
        testnet: bool = True,
        enabled_exchanges: list[ExchangeName] | None = None
    ):
        """
        Initialize aggregator

        Args:
            testnet: Use testnet APIs (default True for safety)
            enabled_exchanges: List of exchanges to monitor (default: all)
        """
        self.testnet = testnet
        self.enabled_exchanges = enabled_exchanges or [
            ExchangeName.BYBIT,
            ExchangeName.BINANCE,
            ExchangeName.OKX
        ]

        # Initialize clients
        self.clients = {}
        if ExchangeName.BYBIT in self.enabled_exchanges:
            self.clients[ExchangeName.BYBIT] = BybitFundingClient(testnet=testnet)
        if ExchangeName.BINANCE in self.enabled_exchanges:
            self.clients[ExchangeName.BINANCE] = BinanceFundingClient(testnet=testnet)
        if ExchangeName.OKX in self.enabled_exchanges:
            self.clients[ExchangeName.OKX] = OKXFundingClient(testnet=testnet)

        self._initialized = False

    async def initialize(self):
        """Initialize all exchange clients"""
        if self._initialized:
            return

        await asyncio.gather(*[
            client.initialize()
            for client in self.clients.values()
        ], return_exceptions=True)

        self._initialized = True
        logger.info(f"Funding arbitrage aggregator initialized with {len(self.clients)} exchanges")

    async def close(self):
        """Close all exchange clients"""
        await asyncio.gather(*[
            client.close()
            for client in self.clients.values()
        ], return_exceptions=True)

    async def get_all_rates(self, symbol: str) -> dict[ExchangeName, ExchangeFundingRate]:
        """
        Get funding rates from all exchanges

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            Dictionary of exchange -> funding rate
        """
        if not self._initialized:
            await self.initialize()

        # Fetch from all exchanges in parallel
        results = await asyncio.gather(*[
            client.get_funding_rate(symbol)
            for client in self.clients.values()
        ], return_exceptions=True)

        # Filter successful results
        rates = {}
        for exchange, result in zip(self.clients.keys(), results, strict=False):
            if isinstance(result, ExchangeFundingRate):
                rates[exchange] = result
            elif isinstance(result, Exception):
                logger.error(f"Error fetching from {exchange}: {result}")

        return rates

    def find_spreads(self, rates: dict[ExchangeName, ExchangeFundingRate]) -> list[FundingRateSpread]:
        """
        Find funding rate spreads between exchanges

        Args:
            rates: Dictionary of exchange -> funding rate

        Returns:
            List of spreads, sorted by spread size (largest first)
        """
        spreads = []

        exchanges = list(rates.keys())
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                ex1, ex2 = exchanges[i], exchanges[j]
                rate1 = rates[ex1]
                rate2 = rates[ex2]

                # Calculate spread
                spread = rate1.rate - rate2.rate

                # Determine high/low exchange
                if spread > 0:
                    high_ex, low_ex = ex1, ex2
                    high_rate, low_rate = rate1.rate, rate2.rate
                else:
                    high_ex, low_ex = ex2, ex1
                    high_rate, low_rate = rate2.rate, rate1.rate
                    spread = abs(spread)

                spread_obj = FundingRateSpread(
                    symbol=rate1.symbol,
                    high_exchange=high_ex,
                    low_exchange=low_ex,
                    high_rate=high_rate,
                    low_rate=low_rate,
                    spread=spread,
                    spread_percent=spread * 100
                )

                spreads.append(spread_obj)

        # Sort by spread size (largest first)
        spreads.sort(key=lambda s: abs(s.spread), reverse=True)

        return spreads

    def create_opportunity(
        self,
        spread: FundingRateSpread,
        rates: dict[ExchangeName, ExchangeFundingRate]
    ) -> FundingOpportunity:
        """
        Create arbitrage opportunity from spread

        Args:
            spread: Funding rate spread
            rates: All exchange rates

        Returns:
            FundingOpportunity
        """
        # Long on low funding exchange (receive funding)
        # Short on high funding exchange (receive funding)
        long_exchange = spread.low_exchange
        short_exchange = spread.high_exchange

        long_rate_obj = rates[long_exchange]
        short_rate_obj = rates[short_exchange]

        # Calculate profit per funding period
        # If we hold $1000 worth:
        # Long side: receive (or pay) low_rate * $1000
        # Short side: receive (or pay) high_rate * $1000
        # Net: (high_rate - low_rate) * $1000 = spread * $1000
        profit_per_funding = spread.spread  # As fraction of position size

        # Calculate hours until next funding
        # Use the earlier funding time (conservative)
        next_funding_time = min(long_rate_obj.next_funding_time, short_rate_obj.next_funding_time)
        hours_until_funding = (next_funding_time - datetime.utcnow()).total_seconds() / 3600

        # Calculate confidence
        confidence = self._calculate_confidence(spread, long_rate_obj, short_rate_obj)

        # Price spread (if available)
        price_spread = None
        if long_rate_obj.mark_price and short_rate_obj.mark_price:
            price_spread = abs(long_rate_obj.mark_price - short_rate_obj.mark_price)

        return FundingOpportunity(
            symbol=spread.symbol,
            long_exchange=long_exchange,
            short_exchange=short_exchange,
            funding_spread=spread.spread,
            profit_per_funding=profit_per_funding,
            next_funding_time=next_funding_time,
            hours_until_funding=max(0, hours_until_funding),
            long_rate=spread.low_rate,
            short_rate=spread.high_rate,
            confidence=confidence,
            long_mark_price=long_rate_obj.mark_price,
            short_mark_price=short_rate_obj.mark_price,
            price_spread=price_spread
        )

    def _calculate_confidence(
        self,
        spread: FundingRateSpread,
        long_rate: ExchangeFundingRate,
        short_rate: ExchangeFundingRate
    ) -> float:
        """
        Calculate confidence in arbitrage opportunity

        Higher confidence when:
        - Large spread
        - Predicted rates confirm the spread
        - Low price spread between exchanges
        - Consistent funding direction
        """
        confidence = 0.5  # Base confidence

        # 1. Spread size factor (larger = more confident)
        if abs(spread.spread) > 0.0005:  # 0.05%
            confidence += 0.2
        elif abs(spread.spread) > 0.0003:  # 0.03%
            confidence += 0.1

        # 2. Predicted rates confirm spread
        if long_rate.predicted_rate is not None and short_rate.predicted_rate is not None:
            predicted_spread = short_rate.predicted_rate - long_rate.predicted_rate
            # Check if predicted spread has same sign as current spread
            if predicted_spread * spread.spread > 0:
                confidence += 0.2
            else:
                confidence -= 0.1  # Predicted reversal, less confident

        # 3. Price spread (smaller = more confident)
        if long_rate.mark_price and short_rate.mark_price:
            price_diff_pct = abs(long_rate.mark_price - short_rate.mark_price) / long_rate.mark_price
            if price_diff_pct < 0.001:  # Less than 0.1% price difference
                confidence += 0.1
            elif price_diff_pct > 0.005:  # More than 0.5% price difference (risky)
                confidence -= 0.2

        return min(1.0, max(0.0, confidence))

    async def find_opportunities(self, symbol: str) -> list[FundingOpportunity]:
        """
        Find arbitrage opportunities for symbol

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            List of opportunities, sorted by profit potential
        """
        # Get rates from all exchanges
        rates = await self.get_all_rates(symbol)

        if len(rates) < 2:
            logger.warning(f"Not enough exchanges with data for {symbol}")
            return []

        # Find spreads
        spreads = self.find_spreads(rates)

        # Create opportunities
        opportunities = []
        for spread in spreads:
            if spread.is_significant:
                opp = self.create_opportunity(spread, rates)
                opportunities.append(opp)

        # Sort by profit potential
        opportunities.sort(key=lambda o: o.profit_per_funding, reverse=True)

        return opportunities

    async def generate_signal(
        self,
        opportunity: FundingOpportunity,
        position_size_usd: float = 1000.0
    ) -> ArbitrageSignal:
        """
        Generate trading signal from opportunity

        Args:
            opportunity: Arbitrage opportunity
            position_size_usd: Position size in USD

        Returns:
            ArbitrageSignal
        """
        # Calculate expected profit and max loss
        expected_profit_usd = opportunity.profit_per_funding * position_size_usd

        # Max loss: assume 1% adverse price movement + fees
        max_loss_usd = position_size_usd * 0.01 + position_size_usd * 0.001  # 1% + 0.1% fees

        # Entry prices
        entry_prices = {}
        if opportunity.long_mark_price:
            entry_prices[opportunity.long_exchange] = opportunity.long_mark_price
        if opportunity.short_mark_price:
            entry_prices[opportunity.short_exchange] = opportunity.short_mark_price

        # Target hold time: until next funding + a bit of buffer
        target_hold_hours = opportunity.hours_until_funding + 0.5  # 30 min buffer

        # Generate reason
        reason = self._generate_reason(opportunity)

        return ArbitrageSignal(
            opportunity=opportunity,
            action="OPEN",
            position_size_usd=position_size_usd,
            entry_prices=entry_prices,
            target_hold_hours=target_hold_hours,
            max_loss_usd=max_loss_usd,
            expected_profit_usd=expected_profit_usd,
            reason=reason
        )

    def _generate_reason(self, opportunity: FundingOpportunity) -> str:
        """Generate human-readable reason"""
        parts = [
            f"Funding arbitrage: {opportunity.symbol}",
            f"Long {opportunity.long_exchange.value} ({opportunity.long_rate * 100:.4f}%)",
            f"Short {opportunity.short_exchange.value} ({opportunity.short_rate * 100:.4f}%)",
            f"Spread: {opportunity.funding_spread * 100:.4f}%",
            f"Confidence: {opportunity.confidence:.0%}",
            f"Annual profit: {opportunity.annual_profit_rate:.1%}"
        ]
        return " | ".join(parts)

    async def monitor_continuous(
        self,
        symbols: list[str],
        callback: callable,
        interval_seconds: int = 300  # 5 minutes
    ):
        """
        Continuously monitor for arbitrage opportunities

        Args:
            symbols: List of symbols to monitor
            callback: async function called with opportunities
            interval_seconds: check interval
        """
        logger.info(f"Starting continuous funding arbitrage monitoring for {symbols}")

        while True:
            try:
                all_opportunities = []

                # Check each symbol
                for symbol in symbols:
                    opportunities = await self.find_opportunities(symbol)
                    all_opportunities.extend(opportunities)

                # Filter tradeable opportunities
                tradeable = [opp for opp in all_opportunities if opp.should_trade]

                if tradeable:
                    await callback(tradeable)

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(interval_seconds)


# Example usage
async def main():
    """Example usage"""
    aggregator = FundingArbitrageAggregator(testnet=True)
    await aggregator.initialize()

    # Find opportunities
    opportunities = await aggregator.find_opportunities("BTCUSDT")

    print(f"\nFound {len(opportunities)} arbitrage opportunities for BTC:")
    for i, opp in enumerate(opportunities, 1):
        print(f"\n{i}. {opp.long_exchange.value} vs {opp.short_exchange.value}")
        print(f"   Funding Spread: {opp.funding_spread * 100:.4f}%")
        print(f"   Profit/Funding: {opp.profit_per_funding * 100:.4f}%")
        print(f"   Annual Profit: {opp.annual_profit_rate:.1%}")
        print(f"   Confidence: {opp.confidence:.0%}")
        print(f"   Risk: {opp.risk_level}")
        print(f"   Should Trade: {opp.should_trade}")

        if opp.should_trade:
            signal = await aggregator.generate_signal(opp, position_size_usd=1000)
            print(f"   Signal: {signal.action}")
            print(f"   Expected Profit: ${signal.expected_profit_usd:.2f}")
            print(f"   P/R Ratio: {signal.profit_to_risk_ratio:.2f}")

    await aggregator.close()


if __name__ == "__main__":
    asyncio.run(main())
