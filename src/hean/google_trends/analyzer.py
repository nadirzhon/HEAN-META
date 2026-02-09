"""
Google Trends analyzer - generates trading signals
"""

import logging

from .client import GoogleTrendsClient
from .models import InterestLevel, TrendDirection, TrendsData, TrendsSignal

logger = logging.getLogger(__name__)


class GoogleTrendsAnalyzer:
    """
    Analyze Google Trends data and generate trading signals

    Research shows that Google search interest correlates with price:
    - Increased searches → Price increase (24-48h lead time)
    - Spike in interest → High volatility incoming
    - Declining interest → Bearish signal

    Usage:
        analyzer = GoogleTrendsAnalyzer()
        await analyzer.initialize()

        signal = await analyzer.get_signal("BTC")
        if signal.should_trade:
            execute_trade(signal)
    """

    # Keyword mappings for crypto symbols
    CRYPTO_KEYWORDS = {
        "BTC": ["bitcoin", "BTC"],
        "ETH": ["ethereum", "ETH"],
        "SOL": ["solana", "SOL"],
        "BNB": ["binance coin", "BNB"],
        "ADA": ["cardano", "ADA"],
        "XRP": ["ripple", "XRP"],
        "DOT": ["polkadot", "DOT"],
        "DOGE": ["dogecoin", "DOGE"],
        "AVAX": ["avalanche", "AVAX"],
        "MATIC": ["polygon", "MATIC"],
    }

    def __init__(
        self,
        timeframe: str = "now 7-d",
        min_interest: int = 40,  # Minimum interest to generate signals
        min_momentum: float = 0.2,  # Minimum momentum
        language: str = "en-US"
    ):
        """
        Initialize analyzer

        Args:
            timeframe: Default timeframe for queries
            min_interest: Minimum interest score to trade
            min_momentum: Minimum momentum to trade
            language: Language for queries
        """
        self.timeframe = timeframe
        self.min_interest = min_interest
        self.min_momentum = min_momentum

        # Initialize client
        self.client = GoogleTrendsClient(language=language)

        self._initialized = False

    async def initialize(self):
        """Initialize client"""
        if self._initialized:
            return

        await self.client.initialize()
        self._initialized = True

        logger.info("Google Trends analyzer initialized")

    def _get_search_keywords(self, symbol: str) -> list[str]:
        """
        Get search keywords for crypto symbol

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")

        Returns:
            List of keywords to search
        """
        # Extract base symbol (remove USDT, PERP, etc.)
        base = symbol.replace("USDT", "").replace("PERP", "").replace("USD", "")

        # Get keywords from mapping
        return self.CRYPTO_KEYWORDS.get(base, [base.lower()])

    async def get_signal(
        self,
        symbol: str,
        timeframe: str | None = None
    ) -> TrendsSignal | None:
        """
        Get trading signal based on Google Trends

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Time period (default: use instance timeframe)

        Returns:
            TrendsSignal or None
        """
        if not self._initialized:
            await self.initialize()

        timeframe = timeframe or self.timeframe

        # Get search keywords
        keywords = self._get_search_keywords(symbol)

        # Try each keyword until we get data
        trends_data = None
        for keyword in keywords:
            trends_data = await self.client.get_interest_over_time(
                keyword,
                timeframe=timeframe
            )
            if trends_data:
                break

        if not trends_data:
            logger.warning(f"No Google Trends data for {symbol}")
            return None

        # Analyze trends
        return self._analyze_trends(symbol, trends_data)

    def _analyze_trends(
        self,
        symbol: str,
        trends: TrendsData
    ) -> TrendsSignal:
        """
        Analyze trends data and generate signal

        Args:
            symbol: Trading symbol
            trends: Trends data

        Returns:
            TrendsSignal
        """
        # Get key metrics
        interest_score = trends.current_interest
        interest_level = trends.interest_level
        trend_direction = trends.get_trend_direction()
        momentum = trends.calculate_momentum()

        # Determine action
        action, confidence = self._determine_action(
            interest_score,
            interest_level,
            trend_direction,
            momentum
        )

        # Generate reason
        reason = self._generate_reason(
            interest_score,
            interest_level,
            trend_direction,
            momentum
        )

        return TrendsSignal(
            keyword=trends.keyword,
            interest_score=interest_score,
            interest_level=interest_level,
            trend_direction=trend_direction,
            momentum=momentum,
            action=action,
            confidence=confidence,
            reason=reason,
            related_queries=trends.related_queries,
            rising_queries=trends.rising_queries
        )

    def _determine_action(
        self,
        interest: int,
        level: InterestLevel,
        direction: TrendDirection,
        momentum: float
    ) -> tuple[str, float]:
        """
        Determine trading action and confidence

        Returns:
            (action, confidence) tuple
        """
        confidence = 0.5  # Base confidence

        # Check minimum requirements
        if interest < self.min_interest:
            return ("HOLD", confidence)

        # Analyze direction and momentum
        if direction == TrendDirection.SPIKE:
            # Sudden spike - bullish but risky
            action = "BUY"
            confidence = 0.7
        elif direction == TrendDirection.RISING and momentum > self.min_momentum:
            # Steady rise - bullish
            action = "BUY"
            confidence = 0.75
        elif direction == TrendDirection.CRASH:
            # Sudden drop - bearish
            action = "SELL"
            confidence = 0.7
        elif direction == TrendDirection.FALLING and momentum < -self.min_momentum:
            # Steady decline - bearish
            action = "SELL"
            confidence = 0.75
        else:
            # Stable or unclear
            action = "HOLD"
            confidence = 0.5

        # Adjust confidence based on interest level
        if level == InterestLevel.VERY_HIGH:
            confidence += 0.1
        elif level == InterestLevel.HIGH:
            confidence += 0.05
        elif level == InterestLevel.VERY_LOW:
            confidence -= 0.1

        # Adjust for momentum strength
        confidence += min(0.1, abs(momentum) * 0.2)

        return (action, min(1.0, max(0.0, confidence)))

    def _generate_reason(
        self,
        interest: int,
        level: InterestLevel,
        direction: TrendDirection,
        momentum: float
    ) -> str:
        """Generate human-readable reason"""
        parts = []

        # Interest level
        parts.append(f"Search interest: {interest}/100 ({level.value})")

        # Trend direction
        if direction == TrendDirection.SPIKE:
            parts.append("Sudden spike in searches (high volatility expected)")
        elif direction == TrendDirection.RISING:
            parts.append("Rising search interest (bullish)")
        elif direction == TrendDirection.CRASH:
            parts.append("Sudden drop in searches (bearish)")
        elif direction == TrendDirection.FALLING:
            parts.append("Declining search interest (bearish)")
        else:
            parts.append("Stable search interest")

        # Momentum
        if abs(momentum) > 0.3:
            parts.append(f"Strong momentum ({momentum:+.1%})")
        elif abs(momentum) > 0.1:
            parts.append(f"Moderate momentum ({momentum:+.1%})")

        return " | ".join(parts)

    async def analyze_comparative(
        self,
        symbols: list[str],
        timeframe: str | None = None
    ) -> dict[str, TrendsSignal | None]:
        """
        Analyze multiple symbols and compare

        Args:
            symbols: List of trading symbols
            timeframe: Time period

        Returns:
            Dictionary of symbol -> signal
        """
        if not self._initialized:
            await self.initialize()

        timeframe = timeframe or self.timeframe

        # Get signals for each symbol
        signals = {}
        for symbol in symbols:
            signal = await self.get_signal(symbol, timeframe=timeframe)
            signals[symbol] = signal

        return signals

    def compare_signals(
        self,
        signals: dict[str, TrendsSignal | None]
    ) -> list[tuple[str, TrendsSignal]]:
        """
        Compare signals and rank by opportunity

        Args:
            signals: Dictionary of symbol -> signal

        Returns:
            List of (symbol, signal) sorted by opportunity (best first)
        """
        # Filter out None signals
        valid_signals = [
            (symbol, signal)
            for symbol, signal in signals.items()
            if signal is not None
        ]

        # Sort by interest * confidence * momentum
        def score(item):
            symbol, signal = item
            return signal.interest_score * signal.confidence * abs(signal.momentum)

        valid_signals.sort(key=score, reverse=True)

        return valid_signals


# Example usage
async def main():
    """Example usage"""
    analyzer = GoogleTrendsAnalyzer(
        timeframe="now 7-d",
        min_interest=40,
        min_momentum=0.2
    )

    await analyzer.initialize()

    # Analyze BTC
    signal = await analyzer.get_signal("BTCUSDT")

    if signal:
        print("\nGoogle Trends Signal for BTC:")
        print(f"  Action: {signal.action}")
        print(f"  Confidence: {signal.confidence:.0%}")
        print(f"  Interest: {signal.interest_score}/100 ({signal.interest_level.value})")
        print(f"  Direction: {signal.trend_direction.value}")
        print(f"  Momentum: {signal.momentum:+.1%}")
        print(f"  Risk: {signal.risk_level}")
        print(f"  Should Trade: {signal.should_trade}")
        print(f"\n  Reason: {signal.reason}")

        if signal.rising_queries:
            print("\n  Rising queries:")
            for query in signal.rising_queries[:3]:
                print(f"    - {query}")

    # Compare multiple symbols
    print("\n\nComparing BTC, ETH, SOL:")
    signals = await analyzer.analyze_comparative(
        ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        timeframe="now 7-d"
    )

    ranked = analyzer.compare_signals(signals)

    for i, (symbol, signal) in enumerate(ranked, 1):
        print(f"\n{i}. {symbol}")
        print(f"   Action: {signal.action} (confidence: {signal.confidence:.0%})")
        print(f"   Interest: {signal.interest_score}")
        print(f"   Direction: {signal.trend_direction.value}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
