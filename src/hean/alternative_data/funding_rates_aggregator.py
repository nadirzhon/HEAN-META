"""Funding rates aggregator for Bybit, Binance, and OKX.

Real-time funding rate collection and analysis across multiple exchanges.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import redis
import requests

logger = logging.getLogger(__name__)


@dataclass
class FundingRateData:
    """Funding rate data for a symbol on an exchange."""

    exchange: str
    symbol: str
    funding_rate: float  # Positive = longs pay shorts, Negative = shorts pay longs
    next_funding_time: datetime
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class FundingRatesAggregator:
    """Aggregates funding rates from Bybit, Binance, and OKX."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        cache_ttl: int = 300,  # 5 minutes
    ):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
        )
        self.cache_ttl = cache_ttl

        # Exchange API endpoints
        self.endpoints = {
            "binance": "https://fapi.binance.com/fapi/v1/premiumIndex",
            "bybit": "https://api.bybit.com/v5/market/tickers",
            "okx": "https://www.okx.com/api/v5/public/funding-rate",
        }

        logger.info("FundingRatesAggregator initialized")

    def _get_cache_key(self, exchange: str, symbol: str) -> str:
        """Generate Redis cache key."""
        return f"funding:{exchange}:{symbol}"

    def _get_cached_rate(self, exchange: str, symbol: str) -> Optional[FundingRateData]:
        """Get cached funding rate."""
        cache_key = self._get_cache_key(exchange, symbol)
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                data = eval(cached)
                return FundingRateData(**data)
        except Exception as e:
            logger.warning(f"Cache retrieval error for {cache_key}: {e}")
        return None

    def _cache_rate(self, rate_data: FundingRateData):
        """Cache funding rate in Redis."""
        cache_key = self._get_cache_key(rate_data.exchange, rate_data.symbol)
        try:
            data = {
                "exchange": rate_data.exchange,
                "symbol": rate_data.symbol,
                "funding_rate": rate_data.funding_rate,
                "next_funding_time": rate_data.next_funding_time.isoformat(),
                "timestamp": rate_data.timestamp.isoformat(),
            }
            self.redis_client.setex(cache_key, self.cache_ttl, str(data))
        except Exception as e:
            logger.warning(f"Cache write error for {cache_key}: {e}")

    def _normalize_symbol(self, symbol: str, exchange: str) -> str:
        """Normalize symbol format for each exchange.

        Args:
            symbol: Base symbol (e.g., "BTC", "ETH")
            exchange: Exchange name

        Returns:
            Exchange-specific symbol format
        """
        symbol = symbol.upper()
        
        if exchange == "binance":
            return f"{symbol}USDT"
        elif exchange == "bybit":
            return f"{symbol}USDT"
        elif exchange == "okx":
            return f"{symbol}-USDT-SWAP"
        
        return symbol

    def get_binance_funding_rate(
        self, symbol: str, use_cache: bool = True
    ) -> Optional[FundingRateData]:
        """Get funding rate from Binance.

        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH")
            use_cache: Whether to use cached data

        Returns:
            FundingRateData or None
        """
        if use_cache:
            cached = self._get_cached_rate("binance", symbol)
            if cached:
                logger.debug(f"Cache hit for Binance {symbol}")
                return cached

        normalized_symbol = self._normalize_symbol(symbol, "binance")
        
        try:
            response = requests.get(
                self.endpoints["binance"],
                params={"symbol": normalized_symbol},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            funding_rate = float(data["lastFundingRate"])
            next_funding_time = datetime.fromtimestamp(data["nextFundingTime"] / 1000)

            rate_data = FundingRateData(
                exchange="binance",
                symbol=symbol,
                funding_rate=funding_rate,
                next_funding_time=next_funding_time,
            )

            self._cache_rate(rate_data)
            logger.info(f"Binance {symbol} funding rate: {funding_rate:.6f}")
            return rate_data

        except Exception as e:
            logger.error(f"Binance funding rate error for {symbol}: {e}")
            return None

    def get_bybit_funding_rate(
        self, symbol: str, use_cache: bool = True
    ) -> Optional[FundingRateData]:
        """Get funding rate from Bybit.

        Args:
            symbol: Trading symbol
            use_cache: Whether to use cached data

        Returns:
            FundingRateData or None
        """
        if use_cache:
            cached = self._get_cached_rate("bybit", symbol)
            if cached:
                logger.debug(f"Cache hit for Bybit {symbol}")
                return cached

        normalized_symbol = self._normalize_symbol(symbol, "bybit")
        
        try:
            response = requests.get(
                self.endpoints["bybit"],
                params={"category": "linear", "symbol": normalized_symbol},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if data["retCode"] == 0 and data["result"]["list"]:
                ticker = data["result"]["list"][0]
                funding_rate = float(ticker["fundingRate"])
                next_funding_time = datetime.fromtimestamp(
                    int(ticker["nextFundingTime"]) / 1000
                )

                rate_data = FundingRateData(
                    exchange="bybit",
                    symbol=symbol,
                    funding_rate=funding_rate,
                    next_funding_time=next_funding_time,
                )

                self._cache_rate(rate_data)
                logger.info(f"Bybit {symbol} funding rate: {funding_rate:.6f}")
                return rate_data

        except Exception as e:
            logger.error(f"Bybit funding rate error for {symbol}: {e}")
            return None

    def get_okx_funding_rate(
        self, symbol: str, use_cache: bool = True
    ) -> Optional[FundingRateData]:
        """Get funding rate from OKX.

        Args:
            symbol: Trading symbol
            use_cache: Whether to use cached data

        Returns:
            FundingRateData or None
        """
        if use_cache:
            cached = self._get_cached_rate("okx", symbol)
            if cached:
                logger.debug(f"Cache hit for OKX {symbol}")
                return cached

        normalized_symbol = self._normalize_symbol(symbol, "okx")
        
        try:
            response = requests.get(
                self.endpoints["okx"],
                params={"instId": normalized_symbol},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if data["code"] == "0" and data["data"]:
                rate_info = data["data"][0]
                funding_rate = float(rate_info["fundingRate"])
                next_funding_time = datetime.fromtimestamp(
                    int(rate_info["nextFundingTime"]) / 1000
                )

                rate_data = FundingRateData(
                    exchange="okx",
                    symbol=symbol,
                    funding_rate=funding_rate,
                    next_funding_time=next_funding_time,
                )

                self._cache_rate(rate_data)
                logger.info(f"OKX {symbol} funding rate: {funding_rate:.6f}")
                return rate_data

        except Exception as e:
            logger.error(f"OKX funding rate error for {symbol}: {e}")
            return None

    def get_all_funding_rates(
        self, symbol: str, use_cache: bool = True
    ) -> Dict[str, FundingRateData]:
        """Get funding rates from all exchanges.

        Args:
            symbol: Trading symbol
            use_cache: Whether to use cached data

        Returns:
            Dict mapping exchange names to FundingRateData
        """
        rates = {}

        binance_rate = self.get_binance_funding_rate(symbol, use_cache)
        if binance_rate:
            rates["binance"] = binance_rate

        bybit_rate = self.get_bybit_funding_rate(symbol, use_cache)
        if bybit_rate:
            rates["bybit"] = bybit_rate

        okx_rate = self.get_okx_funding_rate(symbol, use_cache)
        if okx_rate:
            rates["okx"] = okx_rate

        return rates

    def get_average_funding_rate(
        self, symbol: str, use_cache: bool = True
    ) -> Optional[float]:
        """Get average funding rate across all exchanges.

        Args:
            symbol: Trading symbol
            use_cache: Whether to use cached data

        Returns:
            Average funding rate or None
        """
        rates = self.get_all_funding_rates(symbol, use_cache)
        
        if not rates:
            logger.warning(f"No funding rates available for {symbol}")
            return None

        avg_rate = sum(r.funding_rate for r in rates.values()) / len(rates)
        
        logger.info(
            f"{symbol} average funding rate: {avg_rate:.6f} "
            f"(from {len(rates)} exchanges)"
        )
        
        return avg_rate

    def get_funding_rate_spread(
        self, symbol: str, use_cache: bool = True
    ) -> Dict[str, float]:
        """Get funding rate spread analysis.

        Args:
            symbol: Trading symbol
            use_cache: Whether to use cached data

        Returns:
            Dict with spread statistics
        """
        rates = self.get_all_funding_rates(symbol, use_cache)
        
        if not rates:
            return {
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "spread": 0.0,
                "std_dev": 0.0,
            }

        rate_values = [r.funding_rate for r in rates.values()]
        avg_rate = sum(rate_values) / len(rate_values)
        min_rate = min(rate_values)
        max_rate = max(rate_values)
        spread = max_rate - min_rate
        
        # Standard deviation
        variance = sum((r - avg_rate) ** 2 for r in rate_values) / len(rate_values)
        std_dev = variance ** 0.5

        result = {
            "average": avg_rate,
            "min": min_rate,
            "max": max_rate,
            "spread": spread,
            "std_dev": std_dev,
            "exchanges": {name: rate.funding_rate for name, rate in rates.items()},
        }

        logger.info(
            f"{symbol} funding spread: avg={avg_rate:.6f}, "
            f"spread={spread:.6f}, std={std_dev:.6f}"
        )

        return result

    def analyze_funding_signal(self, symbol: str, use_cache: bool = True) -> Dict:
        """Analyze funding rates for trading signals.

        Interpretation:
        - High positive funding (>0.01%): Longs overheated, potential reversal down
        - High negative funding (<-0.01%): Shorts overheated, potential reversal up
        - Near zero: Balanced market

        Args:
            symbol: Trading symbol
            use_cache: Whether to use cached data

        Returns:
            Analysis with signal interpretation
        """
        spread = self.get_funding_rate_spread(symbol, use_cache)
        avg_rate = spread["average"]

        # Determine signal
        signal = "neutral"
        strength = abs(avg_rate) / 0.01  # Normalize to 0.01% threshold
        
        if avg_rate > 0.01:
            signal = "bearish"  # Longs overheated
        elif avg_rate > 0.005:
            signal = "slightly_bearish"
        elif avg_rate < -0.01:
            signal = "bullish"  # Shorts overheated
        elif avg_rate < -0.005:
            signal = "slightly_bullish"

        analysis = {
            "symbol": symbol,
            "signal": signal,
            "strength": min(strength, 1.0),
            "avg_funding_rate": avg_rate,
            "spread_analysis": spread,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(f"{symbol} funding signal: {signal} (strength: {strength:.2f})")
        
        return analysis

    def health_check(self) -> Dict[str, bool]:
        """Check if all components are healthy."""
        health = {
            "redis_connected": False,
            "binance_api": False,
            "bybit_api": False,
            "okx_api": False,
        }

        try:
            self.redis_client.ping()
            health["redis_connected"] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")

        # Test each exchange API
        test_symbol = "BTC"
        
        if self.get_binance_funding_rate(test_symbol):
            health["binance_api"] = True
        
        if self.get_bybit_funding_rate(test_symbol):
            health["bybit_api"] = True
        
        if self.get_okx_funding_rate(test_symbol):
            health["okx_api"] = True

        return health
