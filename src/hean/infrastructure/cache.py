"""
Redis Caching Layer

Ultra-low latency caching for:
- Price data (<1ms access)
- Technical features (indicators, patterns)
- ML predictions
- Order book snapshots
- Model metadata

Expected Performance:
- Latency: <1ms vs 10-100ms from database
- Throughput: 100,000+ ops/sec
- Feature calculation: 10-100x faster with cache

Author: HEAN Team
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Install with: pip install redis")


@dataclass
class CacheConfig:
    """Cache configuration."""

    # Redis connection
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None

    # TTL (time to live) settings
    default_ttl: int = 300  # 5 minutes
    price_ttl: int = 60     # 1 minute for prices
    feature_ttl: int = 300  # 5 minutes for features
    prediction_ttl: int = 60  # 1 minute for predictions
    orderbook_ttl: int = 10   # 10 seconds for orderbook

    # Serialization
    use_compression: bool = True
    compression_threshold: int = 1024  # Compress if > 1KB

    # Namespace
    namespace: str = "hean"


class FeatureCache:
    """
    Redis-based feature caching system.

    Usage:
        # Initialize
        config = CacheConfig(host="localhost")
        cache = FeatureCache(config)

        # Cache features
        features_df = generate_features(ohlcv)
        cache.set_features("BTCUSDT", "5m", features_df)

        # Get features
        cached = cache.get_features("BTCUSDT", "5m")

        # Cache ML prediction
        prediction = model.predict(features)
        cache.set_prediction("BTCUSDT", prediction)

        # Get prediction
        pred = cache.get_prediction("BTCUSDT")
    """

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        """Initialize cache."""
        if not REDIS_AVAILABLE:
            raise ImportError("Redis required. Install with: pip install redis")

        self.config = config or CacheConfig()

        # Connect to Redis
        self.client = redis.Redis(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            decode_responses=False,  # We handle serialization
        )

        # Test connection
        try:
            self.client.ping()
            logger.info("Redis connected", config=self.config)
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            raise

        self._stats = {"hits": 0, "misses": 0, "sets": 0}

    def set_features(
        self,
        symbol: str,
        timeframe: str,
        features: pd.DataFrame,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache feature DataFrame.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., "5m", "1h")
            features: Features DataFrame
            ttl: Time to live (seconds)

        Returns:
            Success boolean
        """
        key = self._make_key("features", symbol, timeframe)
        ttl = ttl or self.config.feature_ttl

        # Serialize
        data = self._serialize(features)

        # Store
        try:
            self.client.setex(key, ttl, data)
            self._stats["sets"] += 1
            logger.debug(f"Cached features: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache features: {e}")
            return False

    def get_features(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """
        Get cached features.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Features DataFrame or None if not cached
        """
        key = self._make_key("features", symbol, timeframe)

        try:
            data = self.client.get(key)
            if data is None:
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return self._deserialize(data)

        except Exception as e:
            logger.error(f"Failed to get features: {e}")
            return None

    def set_prediction(
        self,
        symbol: str,
        prediction: Any,
        model_name: str = "ensemble",
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache ML prediction."""
        key = self._make_key("prediction", symbol, model_name)
        ttl = ttl or self.config.prediction_ttl

        # Add timestamp
        data = {
            "prediction": prediction,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            serialized = self._serialize(data)
            self.client.setex(key, ttl, serialized)
            self._stats["sets"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to cache prediction: {e}")
            return False

    def get_prediction(
        self,
        symbol: str,
        model_name: str = "ensemble",
    ) -> Optional[Dict[str, Any]]:
        """Get cached prediction."""
        key = self._make_key("prediction", symbol, model_name)

        try:
            data = self.client.get(key)
            if data is None:
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return self._deserialize(data)

        except Exception as e:
            logger.error(f"Failed to get prediction: {e}")
            return None

    def set_price(
        self,
        symbol: str,
        price: float,
        timeframe: str = "1m",
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache current price."""
        key = self._make_key("price", symbol, timeframe)
        ttl = ttl or self.config.price_ttl

        data = {
            "price": price,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            self.client.setex(key, ttl, json.dumps(data))
            self._stats["sets"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to cache price: {e}")
            return False

    def get_price(
        self,
        symbol: str,
        timeframe: str = "1m",
    ) -> Optional[Dict[str, Any]]:
        """Get cached price."""
        key = self._make_key("price", symbol, timeframe)

        try:
            data = self.client.get(key)
            if data is None:
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return json.loads(data)

        except Exception as e:
            logger.error(f"Failed to get price: {e}")
            return None

    def set_orderbook(
        self,
        symbol: str,
        orderbook: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache order book snapshot."""
        key = self._make_key("orderbook", symbol)
        ttl = ttl or self.config.orderbook_ttl

        try:
            data = self._serialize(orderbook)
            self.client.setex(key, ttl, data)
            self._stats["sets"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to cache orderbook: {e}")
            return False

    def get_orderbook(
        self,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        """Get cached order book."""
        key = self._make_key("orderbook", symbol)

        try:
            data = self.client.get(key)
            if data is None:
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return self._deserialize(data)

        except Exception as e:
            logger.error(f"Failed to get orderbook: {e}")
            return None

    def invalidate(
        self,
        pattern: str,
    ) -> int:
        """
        Invalidate cache keys matching pattern.

        Args:
            pattern: Pattern (e.g., "features:BTCUSDT:*")

        Returns:
            Number of keys deleted
        """
        full_pattern = f"{self.config.namespace}:{pattern}"

        try:
            keys = self.client.keys(full_pattern)
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Invalidated {deleted} keys matching {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Failed to invalidate: {e}")
            return 0

    def clear_all(self) -> bool:
        """Clear all cache (WARNING: destructive!)."""
        try:
            pattern = f"{self.config.namespace}:*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
                logger.warning(f"Cleared all cache ({len(keys)} keys)")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            self._stats["hits"] / total
            if total > 0 else 0.0
        )

        return {
            **self._stats,
            "hit_rate": hit_rate,
            "total_requests": total,
        }

    def health_check(self) -> Dict[str, Any]:
        """Check cache health."""
        try:
            # Ping
            latency_start = datetime.now()
            self.client.ping()
            latency = (datetime.now() - latency_start).total_seconds() * 1000

            # Get info
            info = self.client.info()

            return {
                "status": "healthy",
                "latency_ms": latency,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_mb": info.get("used_memory", 0) / 1024 / 1024,
                "total_keys": self.client.dbsize(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    def _make_key(self, *parts: str) -> str:
        """Create cache key with namespace."""
        return f"{self.config.namespace}:{':'.join(parts)}"

    def _serialize(self, data: Any) -> bytes:
        """Serialize data for storage."""
        # Use pickle for complex objects (DataFrames, etc.)
        serialized = pickle.dumps(data)

        # Compress if large
        if self.config.use_compression and len(serialized) > self.config.compression_threshold:
            import zlib
            serialized = zlib.compress(serialized)

        return serialized

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from storage."""
        # Try decompression first
        if self.config.use_compression:
            try:
                import zlib
                data = zlib.decompress(data)
            except Exception:
                pass  # Not compressed

        return pickle.loads(data)


class CacheWarmer:
    """
    Pre-populate cache with frequently accessed data.

    Usage:
        warmer = CacheWarmer(cache)
        await warmer.warm_features(["BTCUSDT", "ETHUSDT"], ["5m", "1h"])
    """

    def __init__(self, cache: FeatureCache) -> None:
        """Initialize cache warmer."""
        self.cache = cache

    async def warm_features(
        self,
        symbols: List[str],
        timeframes: List[str],
        data_source: Any,
    ) -> int:
        """
        Pre-calculate and cache features for symbols.

        Args:
            symbols: List of symbols
            timeframes: List of timeframes
            data_source: Data source with fetch_ohlcv method

        Returns:
            Number of entries warmed
        """
        from hean.features import TALibFeatures

        ta = TALibFeatures()
        warmed = 0

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # Fetch data
                    ohlcv = await data_source.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=1000,
                    )

                    # Generate features
                    features = ta.generate_features(ohlcv)

                    # Cache
                    self.cache.set_features(symbol, timeframe, features)
                    warmed += 1

                    logger.info(f"Warmed {symbol} {timeframe}")

                except Exception as e:
                    logger.error(f"Failed to warm {symbol} {timeframe}: {e}")

        logger.info(f"Cache warming complete. {warmed} entries cached")
        return warmed
