"""On-chain metrics collector for Glassnode and CryptoQuant APIs.

Collects exchange flows, MVRV, active addresses, and other on-chain metrics.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import redis
import requests

logger = logging.getLogger(__name__)


@dataclass
class OnChainMetrics:
    """On-chain metrics data."""

    symbol: str
    exchange_inflow: float  # BTC flowing into exchanges (bearish)
    exchange_outflow: float  # BTC flowing out of exchanges (bullish)
    exchange_netflow: float  # net flow (positive = inflow)
    mvrv_ratio: float  # Market Value to Realized Value ratio
    active_addresses: int
    transaction_count: int
    hash_rate: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class OnChainDataCollector:
    """Collects on-chain metrics from Glassnode and CryptoQuant."""

    def __init__(
        self,
        glassnode_api_key: Optional[str] = None,
        cryptoquant_api_key: Optional[str] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        cache_ttl: int = 600,  # 10 minutes
    ):
        self.glassnode_api_key = glassnode_api_key
        self.cryptoquant_api_key = cryptoquant_api_key
        self.cache_ttl = cache_ttl

        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
        )

        self.glassnode_base_url = "https://api.glassnode.com/v1/metrics"
        self.cryptoquant_base_url = "https://api.cryptoquant.com/v1"

        logger.info("OnChainDataCollector initialized")

    def _get_cache_key(self, symbol: str, metric: str) -> str:
        """Generate Redis cache key."""
        return f"onchain:{metric}:{symbol}"

    def _get_cached_data(self, symbol: str, metric: str) -> Optional[Dict]:
        """Get cached on-chain data."""
        cache_key = self._get_cache_key(symbol, metric)
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                return eval(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval error for {cache_key}: {e}")
        return None

    def _cache_data(self, symbol: str, metric: str, data: Dict):
        """Cache on-chain data in Redis."""
        cache_key = self._get_cache_key(symbol, metric)
        try:
            self.redis_client.setex(cache_key, self.cache_ttl, str(data))
        except Exception as e:
            logger.warning(f"Cache write error for {cache_key}: {e}")

    def _fetch_glassnode_metric(
        self, symbol: str, metric: str, since: Optional[datetime] = None
    ) -> Optional[List[Dict]]:
        """Fetch metric from Glassnode API.

        Args:
            symbol: Asset symbol (e.g., "BTC", "ETH")
            metric: Metric path (e.g., "addresses/active_count")
            since: Start timestamp

        Returns:
            List of metric data points
        """
        if not self.glassnode_api_key:
            logger.warning("Glassnode API key not configured")
            return None

        url = f"{self.glassnode_base_url}/{metric}"
        params = {
            "a": symbol,
            "api_key": self.glassnode_api_key,
        }

        if since:
            params["s"] = int(since.timestamp())

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Glassnode API error for {metric}: {e}")
            return None

    def _fetch_cryptoquant_metric(
        self, symbol: str, endpoint: str
    ) -> Optional[Dict]:
        """Fetch metric from CryptoQuant API.

        Args:
            symbol: Asset symbol
            endpoint: API endpoint

        Returns:
            Metric data
        """
        if not self.cryptoquant_api_key:
            logger.warning("CryptoQuant API key not configured")
            return None

        url = f"{self.cryptoquant_base_url}/{symbol.lower()}/{endpoint}"
        headers = {"Authorization": f"Bearer {self.cryptoquant_api_key}"}

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"CryptoQuant API error for {endpoint}: {e}")
            return None

    def get_exchange_flows(self, symbol: str = "BTC", use_cache: bool = True) -> Dict:
        """Get exchange inflow and outflow data.

        Args:
            symbol: Asset symbol
            use_cache: Whether to use cached data

        Returns:
            Exchange flow metrics
        """
        metric = "exchange_flows"
        if use_cache:
            cached = self._get_cached_data(symbol, metric)
            if cached:
                logger.debug(f"Cache hit for {symbol} {metric}")
                return cached

        # Try Glassnode first
        since = datetime.utcnow() - timedelta(hours=24)
        
        inflow_data = self._fetch_glassnode_metric(
            symbol, "transactions/transfers_volume_exchanges_in", since
        )
        outflow_data = self._fetch_glassnode_metric(
            symbol, "transactions/transfers_volume_exchanges_out", since
        )

        if inflow_data and outflow_data:
            latest_inflow = inflow_data[-1]["v"] if inflow_data else 0
            latest_outflow = outflow_data[-1]["v"] if outflow_data else 0
            
            result = {
                "inflow": latest_inflow,
                "outflow": latest_outflow,
                "netflow": latest_inflow - latest_outflow,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "glassnode",
            }
        else:
            # Fallback to CryptoQuant
            cq_data = self._fetch_cryptoquant_metric(symbol, "exchange-flows")
            if cq_data and "result" in cq_data:
                data = cq_data["result"]
                result = {
                    "inflow": data.get("inflow", 0),
                    "outflow": data.get("outflow", 0),
                    "netflow": data.get("netflow", 0),
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "cryptoquant",
                }
            else:
                logger.warning(f"Failed to fetch exchange flows for {symbol}")
                result = {
                    "inflow": 0,
                    "outflow": 0,
                    "netflow": 0,
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": "none",
                }

        self._cache_data(symbol, metric, result)
        logger.info(f"{symbol} exchange netflow: {result['netflow']:.2f} ({result['source']})")
        return result

    def get_mvrv_ratio(self, symbol: str = "BTC", use_cache: bool = True) -> float:
        """Get MVRV (Market Value to Realized Value) ratio.

        MVRV > 3.7: Overvalued (sell signal)
        MVRV < 1.0: Undervalued (buy signal)

        Args:
            symbol: Asset symbol
            use_cache: Whether to use cached data

        Returns:
            MVRV ratio
        """
        metric = "mvrv"
        if use_cache:
            cached = self._get_cached_data(symbol, metric)
            if cached:
                logger.debug(f"Cache hit for {symbol} {metric}")
                return cached["mvrv"]

        # Glassnode MVRV
        data = self._fetch_glassnode_metric(symbol, "indicators/mvrv")
        
        if data:
            mvrv = data[-1]["v"] if data else 1.0
        else:
            logger.warning(f"Failed to fetch MVRV for {symbol}, using default 1.0")
            mvrv = 1.0

        result = {"mvrv": mvrv, "timestamp": datetime.utcnow().isoformat()}
        self._cache_data(symbol, metric, result)
        
        logger.info(f"{symbol} MVRV ratio: {mvrv:.2f}")
        return mvrv

    def get_active_addresses(self, symbol: str = "BTC", use_cache: bool = True) -> int:
        """Get number of active addresses (24h).

        Args:
            symbol: Asset symbol
            use_cache: Whether to use cached data

        Returns:
            Active address count
        """
        metric = "active_addresses"
        if use_cache:
            cached = self._get_cached_data(symbol, metric)
            if cached:
                logger.debug(f"Cache hit for {symbol} {metric}")
                return cached["count"]

        data = self._fetch_glassnode_metric(symbol, "addresses/active_count")
        
        if data:
            count = int(data[-1]["v"]) if data else 0
        else:
            logger.warning(f"Failed to fetch active addresses for {symbol}")
            count = 0

        result = {"count": count, "timestamp": datetime.utcnow().isoformat()}
        self._cache_data(symbol, metric, result)
        
        logger.info(f"{symbol} active addresses: {count:,}")
        return count

    def get_metrics(self, symbol: str = "BTC", use_cache: bool = True) -> OnChainMetrics:
        """Get all on-chain metrics for a symbol.

        Args:
            symbol: Asset symbol
            use_cache: Whether to use cached data

        Returns:
            OnChainMetrics object with all metrics
        """
        flows = self.get_exchange_flows(symbol, use_cache)
        mvrv = self.get_mvrv_ratio(symbol, use_cache)
        active_addrs = self.get_active_addresses(symbol, use_cache)

        # Get transaction count
        tx_data = self._fetch_glassnode_metric(symbol, "transactions/count")
        tx_count = int(tx_data[-1]["v"]) if tx_data else 0

        # Get hash rate for BTC
        hash_rate = None
        if symbol == "BTC":
            hr_data = self._fetch_glassnode_metric(symbol, "mining/hash_rate_mean")
            hash_rate = hr_data[-1]["v"] if hr_data else None

        metrics = OnChainMetrics(
            symbol=symbol,
            exchange_inflow=flows["inflow"],
            exchange_outflow=flows["outflow"],
            exchange_netflow=flows["netflow"],
            mvrv_ratio=mvrv,
            active_addresses=active_addrs,
            transaction_count=tx_count,
            hash_rate=hash_rate,
            timestamp=datetime.utcnow(),
        )

        logger.info(
            f"{symbol} on-chain metrics: "
            f"netflow={metrics.exchange_netflow:.2f}, "
            f"MVRV={metrics.mvrv_ratio:.2f}, "
            f"active_addrs={metrics.active_addresses:,}"
        )

        return metrics

    def health_check(self) -> Dict[str, bool]:
        """Check if APIs and Redis are accessible."""
        health = {
            "redis_connected": False,
            "glassnode_configured": self.glassnode_api_key is not None,
            "cryptoquant_configured": self.cryptoquant_api_key is not None,
        }

        try:
            self.redis_client.ping()
            health["redis_connected"] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")

        return health
