"""Crypto Fear & Greed Index collector.

Fetches the Fear & Greed Index from alternative.me API.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import redis
import requests

logger = logging.getLogger(__name__)


@dataclass
class FearGreedData:
    """Fear & Greed Index data point."""

    value: int  # 0-100 (0 = Extreme Fear, 100 = Extreme Greed)
    classification: str  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    timestamp: datetime


class FearGreedIndexCollector:
    """Collects Crypto Fear & Greed Index data."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        cache_ttl: int = 3600,  # 1 hour (index updates daily)
    ):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
        )
        self.cache_ttl = cache_ttl
        
        self.api_url = "https://api.alternative.me/fng/"
        
        logger.info("FearGreedIndexCollector initialized")

    def _get_cache_key(self, metric: str = "current") -> str:
        """Generate Redis cache key."""
        return f"fear_greed:{metric}"

    def _get_cached_data(self, metric: str = "current") -> Optional[Dict]:
        """Get cached Fear & Greed data."""
        cache_key = self._get_cache_key(metric)
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                return eval(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval error for {cache_key}: {e}")
        return None

    def _cache_data(self, metric: str, data: Dict):
        """Cache Fear & Greed data in Redis."""
        cache_key = self._get_cache_key(metric)
        try:
            self.redis_client.setex(cache_key, self.cache_ttl, str(data))
        except Exception as e:
            logger.warning(f"Cache write error for {cache_key}: {e}")

    def _classify_value(self, value: int) -> str:
        """Classify Fear & Greed value.

        0-24: Extreme Fear
        25-49: Fear
        50: Neutral
        51-74: Greed
        75-100: Extreme Greed

        Args:
            value: Index value (0-100)

        Returns:
            Classification string
        """
        if value <= 24:
            return "Extreme Fear"
        elif value <= 49:
            return "Fear"
        elif value == 50:
            return "Neutral"
        elif value <= 74:
            return "Greed"
        else:
            return "Extreme Greed"

    def get_current_index(self, use_cache: bool = True) -> Optional[FearGreedData]:
        """Get current Fear & Greed Index.

        Args:
            use_cache: Whether to use cached data

        Returns:
            FearGreedData or None
        """
        metric = "current"
        
        if use_cache:
            cached = self._get_cached_data(metric)
            if cached:
                logger.debug("Cache hit for current Fear & Greed Index")
                return FearGreedData(
                    value=cached["value"],
                    classification=cached["classification"],
                    timestamp=datetime.fromisoformat(cached["timestamp"]),
                )

        try:
            response = requests.get(self.api_url, params={"limit": 1}, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data["data"]:
                index_data = data["data"][0]
                value = int(index_data["value"])
                classification = index_data["value_classification"]
                timestamp = datetime.fromtimestamp(int(index_data["timestamp"]))

                result = FearGreedData(
                    value=value,
                    classification=classification,
                    timestamp=timestamp,
                )

                # Cache the result
                cache_data = {
                    "value": value,
                    "classification": classification,
                    "timestamp": timestamp.isoformat(),
                }
                self._cache_data(metric, cache_data)

                logger.info(f"Fear & Greed Index: {value} ({classification})")
                return result

        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed Index: {e}")
            return None

    def get_historical_index(
        self, days: int = 30, use_cache: bool = True
    ) -> List[FearGreedData]:
        """Get historical Fear & Greed Index data.

        Args:
            days: Number of days of historical data
            use_cache: Whether to use cached data

        Returns:
            List of FearGreedData points
        """
        metric = f"historical_{days}d"
        
        if use_cache:
            cached = self._get_cached_data(metric)
            if cached:
                logger.debug(f"Cache hit for {days}d historical Fear & Greed Index")
                return [
                    FearGreedData(
                        value=item["value"],
                        classification=item["classification"],
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                    )
                    for item in cached
                ]

        try:
            response = requests.get(
                self.api_url,
                params={"limit": days},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            historical_data = []
            for index_data in data["data"]:
                value = int(index_data["value"])
                historical_data.append(FearGreedData(
                    value=value,
                    classification=index_data["value_classification"],
                    timestamp=datetime.fromtimestamp(int(index_data["timestamp"])),
                ))

            # Cache the result
            cache_data = [
                {
                    "value": item.value,
                    "classification": item.classification,
                    "timestamp": item.timestamp.isoformat(),
                }
                for item in historical_data
            ]
            self._cache_data(metric, cache_data)

            logger.info(f"Fetched {len(historical_data)} days of Fear & Greed Index")
            return historical_data

        except Exception as e:
            logger.error(f"Failed to fetch historical Fear & Greed Index: {e}")
            return []

    def get_trend_analysis(self, days: int = 7) -> Dict:
        """Analyze Fear & Greed Index trend.

        Args:
            days: Number of days to analyze

        Returns:
            Trend analysis dict
        """
        historical = self.get_historical_index(days)
        
        if not historical:
            return {
                "trend": "unknown",
                "current_value": 0,
                "average": 0,
                "change": 0,
            }

        current = historical[0]  # Most recent
        values = [h.value for h in historical]
        
        average = sum(values) / len(values)
        change = current.value - historical[-1].value  # Change from oldest to newest

        # Determine trend
        if change > 10:
            trend = "increasing"
        elif change < -10:
            trend = "decreasing"
        else:
            trend = "stable"

        result = {
            "trend": trend,
            "current_value": current.value,
            "current_classification": current.classification,
            "average": average,
            "change": change,
            "change_percent": (change / historical[-1].value * 100) if historical[-1].value > 0 else 0,
            "min": min(values),
            "max": max(values),
        }

        logger.info(
            f"Fear & Greed trend ({days}d): {trend}, "
            f"current={current.value}, avg={average:.1f}, change={change:+.0f}"
        )

        return result

    def get_trading_signal(self) -> Dict:
        """Get trading signal based on Fear & Greed Index.

        Contrarian strategy:
        - Extreme Fear (0-24): Buy signal
        - Fear (25-49): Weak buy signal
        - Neutral (50): Hold
        - Greed (51-74): Weak sell signal
        - Extreme Greed (75-100): Sell signal

        Returns:
            Trading signal dict
        """
        current = self.get_current_index()
        
        if not current:
            return {
                "signal": "hold",
                "strength": 0.0,
                "reasoning": "Unable to fetch Fear & Greed Index",
            }

        value = current.value
        
        # Contrarian signals
        if value <= 24:
            signal = "strong_buy"
            strength = 1.0 - (value / 24)
            reasoning = "Extreme Fear - contrarian buy opportunity"
        elif value <= 49:
            signal = "buy"
            strength = 0.5 - (value - 25) / 48
            reasoning = "Fear in market - potential buy opportunity"
        elif value == 50:
            signal = "hold"
            strength = 0.0
            reasoning = "Neutral market sentiment"
        elif value <= 74:
            signal = "sell"
            strength = (value - 51) / 48
            reasoning = "Greed in market - consider taking profits"
        else:
            signal = "strong_sell"
            strength = (value - 75) / 25
            reasoning = "Extreme Greed - contrarian sell opportunity"

        result = {
            "signal": signal,
            "strength": strength,
            "value": value,
            "classification": current.classification,
            "reasoning": reasoning,
            "timestamp": current.timestamp.isoformat(),
        }

        logger.info(f"Fear & Greed signal: {signal} (strength: {strength:.2f}, value: {value})")
        
        return result

    def health_check(self) -> Dict[str, bool]:
        """Check if components are healthy."""
        health = {
            "redis_connected": False,
            "api_accessible": False,
        }

        try:
            self.redis_client.ping()
            health["redis_connected"] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")

        # Test API access
        try:
            response = requests.get(self.api_url, params={"limit": 1}, timeout=5)
            if response.status_code == 200:
                health["api_accessible"] = True
        except Exception as e:
            logger.error(f"API health check failed: {e}")

        return health
