"""Price Anomaly Detector - Guards against gaps and spikes.

Detects abnormal price movements that could indicate:
- Market gaps (sudden large price jumps, often during low liquidity)
- Flash crashes (extreme downward spikes)
- Flash pumps (extreme upward spikes)
- Manipulation (artificial price moves)

These events can trigger massive slippage and should pause trading.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from hean.logging import get_logger

logger = get_logger(__name__)


class AnomalyType(Enum):
    """Types of price anomalies."""
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    SPIKE_UP = "spike_up"
    SPIKE_DOWN = "spike_down"
    FLASH_CRASH = "flash_crash"
    STALE_PRICE = "stale_price"


@dataclass
class PriceAnomaly:
    """Detected price anomaly."""
    anomaly_type: AnomalyType
    symbol: str
    price: float
    previous_price: float
    change_pct: float
    detected_at: datetime
    severity: str  # "warning", "critical"
    should_block_trading: bool


class PriceAnomalyDetector:
    """Detects abnormal price movements.

    Uses multiple detection methods:
    1. Percentage change threshold (gaps)
    2. Z-score deviation (statistical anomalies)
    3. Price staleness (no updates for extended period)
    """

    def __init__(
        self,
        gap_threshold_pct: float = 2.0,  # 2% gap threshold
        spike_threshold_pct: float = 5.0,  # 5% spike threshold (critical)
        z_score_threshold: float = 4.0,  # 4 standard deviations
        stale_threshold_seconds: int = 60,  # Price stale after 60s
        history_size: int = 100,  # Keep last 100 prices for stats
    ) -> None:
        """Initialize the anomaly detector.

        Args:
            gap_threshold_pct: Price change % to trigger gap warning
            spike_threshold_pct: Price change % to trigger critical spike
            z_score_threshold: Z-score for statistical anomaly detection
            stale_threshold_seconds: Seconds without update to trigger stale warning
            history_size: Number of prices to keep for statistics
        """
        self._gap_threshold = gap_threshold_pct / 100
        self._spike_threshold = spike_threshold_pct / 100
        self._z_score_threshold = z_score_threshold
        self._stale_threshold = timedelta(seconds=stale_threshold_seconds)
        self._history_size = history_size

        # Price history per symbol
        self._price_history: dict[str, deque[float]] = {}
        self._last_prices: dict[str, float] = {}
        self._last_update_time: dict[str, datetime] = {}

        # Statistics
        self._anomaly_count: dict[str, int] = {}
        self._blocked_until: dict[str, datetime] = {}

    def check_price(self, symbol: str, price: float) -> PriceAnomaly | None:
        """Check if a price update is anomalous.

        Args:
            symbol: Trading symbol
            price: New price

        Returns:
            PriceAnomaly if detected, None otherwise
        """
        now = datetime.utcnow()

        # Initialize history if needed
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self._history_size)
            self._anomaly_count[symbol] = 0

        # Check for stale price before update
        if symbol in self._last_update_time:
            time_since_update = now - self._last_update_time[symbol]
            if time_since_update > self._stale_threshold:
                logger.warning(
                    f"[ANOMALY] Stale price detected for {symbol}: "
                    f"no update for {time_since_update.total_seconds():.0f}s"
                )
                return PriceAnomaly(
                    anomaly_type=AnomalyType.STALE_PRICE,
                    symbol=symbol,
                    price=price,
                    previous_price=self._last_prices.get(symbol, price),
                    change_pct=0,
                    detected_at=now,
                    severity="warning",
                    should_block_trading=False,  # Don't block, just warn
                )

        # Get previous price
        previous_price = self._last_prices.get(symbol)

        # Update state
        self._price_history[symbol].append(price)
        self._last_prices[symbol] = price
        self._last_update_time[symbol] = now

        # Skip anomaly check if no previous price
        if previous_price is None or previous_price <= 0:
            return None

        # Calculate price change
        change_pct = (price - previous_price) / previous_price
        abs_change = abs(change_pct)

        # Check for critical spike (block trading)
        if abs_change >= self._spike_threshold:
            anomaly_type = AnomalyType.SPIKE_UP if change_pct > 0 else AnomalyType.SPIKE_DOWN

            if abs_change >= self._spike_threshold * 2:
                # Flash crash/pump detection (>10% in single tick)
                anomaly_type = AnomalyType.FLASH_CRASH if change_pct < 0 else AnomalyType.SPIKE_UP

            logger.critical(
                f"[ANOMALY] CRITICAL price spike detected for {symbol}: "
                f"{change_pct * 100:+.2f}% ({previous_price:.2f} -> {price:.2f})"
            )

            self._anomaly_count[symbol] += 1
            self._blocked_until[symbol] = now + timedelta(minutes=5)  # Block for 5 min

            return PriceAnomaly(
                anomaly_type=anomaly_type,
                symbol=symbol,
                price=price,
                previous_price=previous_price,
                change_pct=change_pct * 100,
                detected_at=now,
                severity="critical",
                should_block_trading=True,
            )

        # Check for gap (warning, reduced sizing)
        if abs_change >= self._gap_threshold:
            anomaly_type = AnomalyType.GAP_UP if change_pct > 0 else AnomalyType.GAP_DOWN

            logger.warning(
                f"[ANOMALY] Price gap detected for {symbol}: "
                f"{change_pct * 100:+.2f}% ({previous_price:.2f} -> {price:.2f})"
            )

            self._anomaly_count[symbol] += 1

            return PriceAnomaly(
                anomaly_type=anomaly_type,
                symbol=symbol,
                price=price,
                previous_price=previous_price,
                change_pct=change_pct * 100,
                detected_at=now,
                severity="warning",
                should_block_trading=False,  # Don't block, but reduce size
            )

        # Check for statistical anomaly using z-score
        if len(self._price_history[symbol]) >= 20:
            returns = []
            prices = list(self._price_history[symbol])
            for i in range(1, len(prices)):
                if prices[i - 1] > 0:
                    returns.append((prices[i] - prices[i - 1]) / prices[i - 1])

            if returns:
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                std_return = variance ** 0.5

                if std_return > 0:
                    z_score = abs(change_pct - mean_return) / std_return

                    if z_score >= self._z_score_threshold:
                        anomaly_type = AnomalyType.GAP_UP if change_pct > 0 else AnomalyType.GAP_DOWN

                        logger.warning(
                            f"[ANOMALY] Statistical anomaly detected for {symbol}: "
                            f"z-score={z_score:.2f}, change={change_pct * 100:+.2f}%"
                        )

                        return PriceAnomaly(
                            anomaly_type=anomaly_type,
                            symbol=symbol,
                            price=price,
                            previous_price=previous_price,
                            change_pct=change_pct * 100,
                            detected_at=now,
                            severity="warning",
                            should_block_trading=False,
                        )

        return None

    def is_blocked(self, symbol: str) -> bool:
        """Check if trading is blocked for a symbol due to anomaly.

        Args:
            symbol: Trading symbol

        Returns:
            True if blocked, False otherwise
        """
        if symbol not in self._blocked_until:
            return False

        if datetime.utcnow() >= self._blocked_until[symbol]:
            del self._blocked_until[symbol]
            logger.info(f"[ANOMALY] Trading unblocked for {symbol} after cooldown")
            return False

        return True

    def get_size_multiplier(self, symbol: str) -> float:
        """Get position size multiplier based on recent anomalies.

        Returns reduced multiplier if recent anomalies detected.

        Args:
            symbol: Trading symbol

        Returns:
            Size multiplier (0.0 to 1.0)
        """
        if self.is_blocked(symbol):
            return 0.0  # Blocked

        # Reduce size based on recent anomaly count
        anomaly_count = self._anomaly_count.get(symbol, 0)

        if anomaly_count >= 5:
            return 0.25  # 75% reduction
        elif anomaly_count >= 3:
            return 0.5  # 50% reduction
        elif anomaly_count >= 1:
            return 0.75  # 25% reduction

        return 1.0  # Normal

    def reset_anomaly_count(self, symbol: str) -> None:
        """Reset anomaly count for a symbol (call on successful trades)."""
        if symbol in self._anomaly_count:
            self._anomaly_count[symbol] = 0

    def get_status(self) -> dict:
        """Get current detector status."""
        return {
            "symbols_tracked": len(self._price_history),
            "blocked_symbols": list(self._blocked_until.keys()),
            "anomaly_counts": dict(self._anomaly_count),
            "thresholds": {
                "gap_pct": self._gap_threshold * 100,
                "spike_pct": self._spike_threshold * 100,
                "z_score": self._z_score_threshold,
                "stale_seconds": self._stale_threshold.total_seconds(),
            },
        }


# Global instance for easy access
price_anomaly_detector = PriceAnomalyDetector()
