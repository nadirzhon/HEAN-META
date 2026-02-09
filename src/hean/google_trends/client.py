"""
Google Trends API client

Uses pytrends (unofficial Google Trends API)
"""

import asyncio
import logging
from datetime import UTC, datetime

from .models import ComparativeTrendsData, TrendsData, TrendsHistory

logger = logging.getLogger(__name__)


class GoogleTrendsClient:
    """
    Fetch search interest data from Google Trends

    Uses pytrends library (unofficial API)

    Note: Google Trends has rate limiting. Don't query too frequently!
    Recommended: Max 1 request per minute per keyword
    """

    def __init__(self, language: str = "en-US", timezone_offset: int = 360):
        """
        Initialize Google Trends client

        Args:
            language: Language code (default: en-US)
            timezone_offset: Timezone offset in minutes (default: 360 = UTC-6)
        """
        self.language = language
        self.timezone_offset = timezone_offset
        self._pytrends = None
        self._initialized = False

        # Rate limiting
        self._last_request_time = None
        self._min_request_interval = 2.0  # seconds between requests

    async def initialize(self):
        """Initialize pytrends"""
        if self._initialized:
            return

        try:
            # Import here to avoid requiring it if not used
            from pytrends.request import TrendReq

            # Run in executor (blocking operation)
            loop = asyncio.get_event_loop()
            self._pytrends = await loop.run_in_executor(
                None,
                lambda: TrendReq(hl=self.language, tz=self.timezone_offset)
            )

            self._initialized = True
            logger.info("Google Trends client initialized")

        except ImportError:
            logger.error(
                "pytrends not installed. Install with: "
                "pip install pytrends --break-system-packages"
            )
            raise

    async def _rate_limit(self):
        """Enforce rate limiting"""
        if self._last_request_time is not None:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self._min_request_interval:
                wait_time = self._min_request_interval - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        self._last_request_time = datetime.now()

    async def get_interest_over_time(
        self,
        keyword: str,
        timeframe: str = "now 7-d",
        category: int = 0,
        geo: str = ""
    ) -> TrendsData | None:
        """
        Get search interest over time for keyword

        Args:
            keyword: Search keyword (e.g., "bitcoin", "BTC")
            timeframe: Time period (default: "now 7-d")
                - "now 1-H" - Last hour
                - "now 4-H" - Last 4 hours
                - "now 1-d" - Last day
                - "now 7-d" - Last 7 days
                - "today 1-m" - Last month
                - "today 3-m" - Last 3 months
                - "today 12-m" - Last year
            category: Google category (0 = all categories)
            geo: Geographic location (e.g., "US", "" = worldwide)

        Returns:
            TrendsData or None if error
        """
        if not self._initialized:
            await self.initialize()

        await self._rate_limit()

        try:
            # Build payload
            loop = asyncio.get_event_loop()

            # Run blocking operation in executor
            await loop.run_in_executor(
                None,
                lambda: self._pytrends.build_payload(
                    kw_list=[keyword],
                    cat=category,
                    timeframe=timeframe,
                    geo=geo,
                    gprop=""
                )
            )

            # Get interest over time
            df = await loop.run_in_executor(
                None,
                self._pytrends.interest_over_time
            )

            if df.empty or keyword not in df.columns:
                logger.warning(f"No data for keyword: {keyword}")
                return None

            # Extract data
            interest_values = df[keyword].tolist()
            timestamps = [
                datetime.fromtimestamp(ts.timestamp(), tz=UTC)
                for ts in df.index
            ]

            # Get related queries
            related = await self._get_related_queries(keyword)

            return TrendsData(
                keyword=keyword,
                timeframe=timeframe,
                interest_over_time=interest_values,
                timestamps=timestamps,
                related_queries=related.get("top", []),
                rising_queries=related.get("rising", [])
            )

        except Exception as e:
            logger.error(f"Error fetching trends for {keyword}: {e}")
            return None

    async def _get_related_queries(self, keyword: str) -> dict:
        """Get related queries for keyword"""
        try:
            loop = asyncio.get_event_loop()
            related_dict = await loop.run_in_executor(
                None,
                self._pytrends.related_queries
            )

            if keyword not in related_dict:
                return {"top": [], "rising": []}

            result = related_dict[keyword]

            # Extract top queries
            top_queries = []
            if result["top"] is not None and not result["top"].empty:
                top_queries = result["top"]["query"].head(5).tolist()

            # Extract rising queries
            rising_queries = []
            if result["rising"] is not None and not result["rising"].empty:
                rising_queries = result["rising"]["query"].head(5).tolist()

            return {
                "top": top_queries,
                "rising": rising_queries
            }

        except Exception as e:
            logger.warning(f"Could not get related queries: {e}")
            return {"top": [], "rising": []}

    async def compare_keywords(
        self,
        keywords: list[str],
        timeframe: str = "now 7-d",
        geo: str = ""
    ) -> ComparativeTrendsData | None:
        """
        Compare multiple keywords

        Args:
            keywords: List of keywords to compare (max 5)
            timeframe: Time period
            geo: Geographic location

        Returns:
            ComparativeTrendsData or None
        """
        if not self._initialized:
            await self.initialize()

        if len(keywords) > 5:
            logger.warning("Google Trends allows max 5 keywords. Taking first 5.")
            keywords = keywords[:5]

        await self._rate_limit()

        try:
            loop = asyncio.get_event_loop()

            # Build payload
            await loop.run_in_executor(
                None,
                lambda: self._pytrends.build_payload(
                    kw_list=keywords,
                    cat=0,
                    timeframe=timeframe,
                    geo=geo,
                    gprop=""
                )
            )

            # Get interest over time
            df = await loop.run_in_executor(
                None,
                self._pytrends.interest_over_time
            )

            if df.empty:
                logger.warning(f"No comparative data for keywords: {keywords}")
                return None

            # Extract data for each keyword
            interest_data = {}
            for kw in keywords:
                if kw not in df.columns:
                    continue

                interest_values = df[kw].tolist()
                timestamps = [
                    datetime.fromtimestamp(ts.timestamp(), tz=UTC)
                    for ts in df.index
                ]

                interest_data[kw] = TrendsData(
                    keyword=kw,
                    timeframe=timeframe,
                    interest_over_time=interest_values,
                    timestamps=timestamps
                )

            # Find winner (highest current interest)
            winner = max(
                interest_data.items(),
                key=lambda x: x[1].current_interest
            )

            winner_kw = winner[0]
            winner_interest = winner[1].current_interest

            # Calculate advantage
            other_interests = [
                data.current_interest
                for kw, data in interest_data.items()
                if kw != winner_kw
            ]
            avg_other = sum(other_interests) / len(other_interests) if other_interests else 0

            if avg_other > 0:
                leader_advantage = (winner_interest - avg_other) / avg_other
            else:
                leader_advantage = 1.0

            return ComparativeTrendsData(
                keywords=keywords,
                interest_data=interest_data,
                winner=winner_kw,
                leader_advantage=leader_advantage
            )

        except Exception as e:
            logger.error(f"Error comparing keywords: {e}")
            return None

    async def get_history(
        self,
        keyword: str,
        timeframe: str = "today 3-m"
    ) -> TrendsHistory | None:
        """
        Get historical trends data

        Args:
            keyword: Search keyword
            timeframe: Time period (default: last 3 months)

        Returns:
            TrendsHistory or None
        """
        trends = await self.get_interest_over_time(keyword, timeframe=timeframe)

        if trends is None:
            return None

        return TrendsHistory(
            keyword=keyword,
            interest_values=trends.interest_over_time,
            timestamps=trends.timestamps
        )


# Example usage
async def main():
    """Example usage"""
    client = GoogleTrendsClient()
    await client.initialize()

    # Get trends for Bitcoin
    trends = await client.get_interest_over_time("bitcoin", timeframe="now 7-d")

    if trends:
        print("\nGoogle Trends for 'bitcoin' (last 7 days):")
        print(f"  Current interest: {trends.current_interest}")
        print(f"  Average interest: {trends.average_interest:.1f}")
        print(f"  Interest level: {trends.interest_level.value}")
        print(f"  Trend direction: {trends.get_trend_direction().value}")
        print(f"  Momentum: {trends.calculate_momentum():.2f}")
        print(f"  Volatility: {trends.volatility:.2f}")

        if trends.rising_queries:
            print("\n  Rising queries:")
            for query in trends.rising_queries[:3]:
                print(f"    - {query}")

    # Compare BTC vs ETH
    print("\n\nComparing BTC vs ETH vs SOL:")
    comparison = await client.compare_keywords(
        ["bitcoin", "ethereum", "solana"],
        timeframe="now 7-d"
    )

    if comparison:
        print(f"  Winner: {comparison.winner}")
        print(f"  Leader advantage: {comparison.leader_advantage:.1%}")
        print("\n  Relative strengths:")
        for kw, strength in comparison.relative_strengths.items():
            print(f"    {kw}: {strength:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
