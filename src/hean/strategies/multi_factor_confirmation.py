"""Multi-Factor Signal Confirmation System.

Combines multiple independent factors to confirm trading signals,
reducing false positives and improving win rate.

Factors:
1. Momentum - price momentum alignment with signal direction
2. Volume - volume confirmation (spikes on breakouts)
3. OFI - Order Flow Imbalance alignment
4. Regime - market regime compatibility
5. Volatility - volatility regime compatibility
6. Time - favorable trading hours

Each factor provides a score (0.0 to 1.0) and the combined score
determines signal confidence.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from hean.core.regime import Regime
from hean.core.types import Signal
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FactorScore:
    """Score from a single confirmation factor."""
    factor_name: str
    score: float  # 0.0 to 1.0
    weight: float  # Factor weight
    details: dict[str, Any] | None = None
    passed: bool = True  # Whether factor passes minimum threshold


@dataclass
class ConfirmationResult:
    """Result of multi-factor confirmation."""
    signal: Signal
    total_score: float  # Combined weighted score (0.0 to 1.0)
    factor_scores: list[FactorScore]
    confidence: float  # Final confidence for position sizing
    confirmed: bool  # Whether signal passed all required factors
    reason: str | None = None  # Reason if not confirmed


class ConfirmationFactor(ABC):
    """Base class for confirmation factors."""

    def __init__(self, name: str, weight: float = 1.0, required: bool = False):
        """Initialize confirmation factor.

        Args:
            name: Factor name
            weight: Weight in combined score (0.0 to 1.0)
            required: If True, signal is rejected if this factor fails
        """
        self.name = name
        self.weight = weight
        self.required = required

    @abstractmethod
    def score(
        self, signal: Signal, context: dict[str, Any]
    ) -> FactorScore:
        """Calculate factor score for signal.

        Args:
            signal: Signal to evaluate
            context: Context with market data, regime, etc.

        Returns:
            FactorScore with score and details
        """
        pass


class MomentumFactor(ConfirmationFactor):
    """Confirms signal aligns with price momentum.

    For BUY: Price should be above short-term average
    For SELL: Price should be below short-term average
    """

    def __init__(self, weight: float = 1.0, required: bool = False):
        super().__init__("momentum", weight, required)
        self._price_history: dict[str, deque] = {}
        self._short_window = 10
        self._long_window = 50

    def update_price(self, symbol: str, price: float) -> None:
        """Update price history for symbol."""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self._long_window)
        self._price_history[symbol].append(price)

    def score(
        self, signal: Signal, context: dict[str, Any]
    ) -> FactorScore:
        """Calculate momentum alignment score."""
        prices = context.get("price_history", [])
        if not prices:
            prices = list(self._price_history.get(signal.symbol, []))

        if len(prices) < self._short_window:
            # Insufficient data - neutral score
            return FactorScore(
                factor_name=self.name,
                score=0.5,
                weight=self.weight,
                details={"reason": "insufficient_data"},
                passed=True,
            )

        # Calculate short and long term averages
        short_avg = sum(prices[-self._short_window:]) / self._short_window
        long_avg = sum(prices[-self._long_window:]) / len(prices[-self._long_window:])

        # Momentum direction
        momentum_direction = 1 if short_avg > long_avg else -1
        signal_direction = 1 if signal.side == "buy" else -1

        # Alignment check
        aligned = momentum_direction == signal_direction

        # Score based on alignment strength
        if long_avg > 0:
            momentum_strength = abs(short_avg - long_avg) / long_avg
        else:
            momentum_strength = 0.0

        if aligned:
            # Stronger momentum = higher score
            score = min(1.0, 0.6 + momentum_strength * 4)
        else:
            # Counter-trend trade - lower score but not zero
            score = max(0.2, 0.5 - momentum_strength * 2)

        return FactorScore(
            factor_name=self.name,
            score=score,
            weight=self.weight,
            details={
                "aligned": aligned,
                "short_avg": short_avg,
                "long_avg": long_avg,
                "momentum_strength": momentum_strength,
            },
            passed=score >= 0.4,  # Minimum threshold
        )


class VolumeFactor(ConfirmationFactor):
    """Confirms volume supports the signal.

    Higher volume on breakouts = stronger confirmation.
    """

    def __init__(self, weight: float = 0.8, required: bool = False):
        super().__init__("volume", weight, required)
        self._volume_history: dict[str, deque] = {}
        self._window = 20

    def update_volume(self, symbol: str, volume: float) -> None:
        """Update volume history for symbol."""
        if symbol not in self._volume_history:
            self._volume_history[symbol] = deque(maxlen=self._window)
        self._volume_history[symbol].append(volume)

    def score(
        self, signal: Signal, context: dict[str, Any]
    ) -> FactorScore:
        """Calculate volume confirmation score."""
        volumes = context.get("volume_history", [])
        if not volumes:
            volumes = list(self._volume_history.get(signal.symbol, []))

        current_volume = context.get("current_volume", 0.0)

        if len(volumes) < 5 or sum(volumes) == 0:
            # Insufficient data - neutral score
            return FactorScore(
                factor_name=self.name,
                score=0.5,
                weight=self.weight,
                details={"reason": "insufficient_data"},
                passed=True,
            )

        avg_volume = sum(volumes) / len(volumes)

        if avg_volume <= 0:
            return FactorScore(
                factor_name=self.name,
                score=0.5,
                weight=self.weight,
                details={"reason": "zero_avg_volume"},
                passed=True,
            )

        volume_ratio = current_volume / avg_volume if current_volume > 0 else 1.0

        # Score based on volume ratio
        # 1.0x average = 0.5 score
        # 2.0x average = 0.8 score
        # 0.5x average = 0.3 score
        if volume_ratio >= 1.0:
            score = min(1.0, 0.5 + (volume_ratio - 1.0) * 0.3)
        else:
            score = max(0.2, 0.5 - (1.0 - volume_ratio) * 0.4)

        return FactorScore(
            factor_name=self.name,
            score=score,
            weight=self.weight,
            details={
                "volume_ratio": volume_ratio,
                "avg_volume": avg_volume,
                "current_volume": current_volume,
            },
            passed=score >= 0.35,
        )


class OFIFactor(ConfirmationFactor):
    """Order Flow Imbalance confirmation.

    Confirms that order flow supports the signal direction.
    """

    def __init__(self, weight: float = 1.2, required: bool = False):
        super().__init__("ofi", weight, required)

    def score(
        self, signal: Signal, context: dict[str, Any]
    ) -> FactorScore:
        """Calculate OFI alignment score."""
        ofi_value = context.get("ofi_value")
        ofi_signal = context.get("ofi_signal")  # 'buy', 'sell', or None

        if ofi_value is None:
            # No OFI data - neutral score
            return FactorScore(
                factor_name=self.name,
                score=0.5,
                weight=self.weight,
                details={"reason": "no_ofi_data"},
                passed=True,
            )

        # Signal direction
        signal_direction = 1 if signal.side == "buy" else -1

        # OFI alignment
        # Positive OFI = bullish, Negative OFI = bearish
        ofi_direction = 1 if ofi_value > 0 else -1

        aligned = ofi_direction == signal_direction

        # Score based on OFI strength
        ofi_strength = min(1.0, abs(ofi_value))  # Normalize OFI

        if aligned:
            score = min(1.0, 0.6 + ofi_strength * 0.4)
        else:
            # Counter-OFI trade - reduce score
            score = max(0.2, 0.5 - ofi_strength * 0.3)

        return FactorScore(
            factor_name=self.name,
            score=score,
            weight=self.weight,
            details={
                "ofi_value": ofi_value,
                "ofi_signal": ofi_signal,
                "aligned": aligned,
                "ofi_strength": ofi_strength,
            },
            passed=score >= 0.4,
        )


class RegimeFactor(ConfirmationFactor):
    """Market regime compatibility confirmation.

    Certain signals are better in certain regimes.
    """

    # Regime compatibility matrix (signal_type -> regime -> score_multiplier)
    REGIME_COMPATIBILITY = {
        "momentum": {
            Regime.IMPULSE: 1.0,
            Regime.NORMAL: 0.8,
            Regime.RANGE: 0.4,
        },
        "mean_reversion": {
            Regime.RANGE: 1.0,
            Regime.NORMAL: 0.7,
            Regime.IMPULSE: 0.3,
        },
        "breakout": {
            Regime.IMPULSE: 1.0,
            Regime.NORMAL: 0.6,
            Regime.RANGE: 0.5,
        },
        "default": {
            Regime.NORMAL: 0.8,
            Regime.IMPULSE: 0.7,
            Regime.RANGE: 0.7,
        },
    }

    def __init__(self, weight: float = 0.9, required: bool = False):
        super().__init__("regime", weight, required)

    def score(
        self, signal: Signal, context: dict[str, Any]
    ) -> FactorScore:
        """Calculate regime compatibility score."""
        regime = context.get("regime", Regime.NORMAL)
        signal_type = context.get("signal_type", "default")

        if signal_type not in self.REGIME_COMPATIBILITY:
            signal_type = "default"

        compatibility = self.REGIME_COMPATIBILITY[signal_type]
        regime_score = compatibility.get(regime, 0.7)

        return FactorScore(
            factor_name=self.name,
            score=regime_score,
            weight=self.weight,
            details={
                "regime": regime.value if hasattr(regime, 'value') else str(regime),
                "signal_type": signal_type,
            },
            passed=regime_score >= 0.4,
        )


class VolatilityFactor(ConfirmationFactor):
    """Volatility regime confirmation.

    Adjusts score based on current volatility level.
    """

    def __init__(self, weight: float = 0.7, required: bool = False):
        super().__init__("volatility", weight, required)

    def score(
        self, signal: Signal, context: dict[str, Any]
    ) -> FactorScore:
        """Calculate volatility factor score."""
        volatility_percentile = context.get("volatility_percentile", 50.0)

        # Ideal volatility is moderate (30-70 percentile)
        # Too low = no edge, too high = risky
        if 30 <= volatility_percentile <= 70:
            score = 0.9  # Ideal conditions
        elif 20 <= volatility_percentile < 30 or 70 < volatility_percentile <= 80:
            score = 0.7  # Acceptable
        elif volatility_percentile < 20:
            score = 0.5  # Low volatility - might be quiet market
        else:
            score = 0.5  # High volatility - risky

        return FactorScore(
            factor_name=self.name,
            score=score,
            weight=self.weight,
            details={
                "volatility_percentile": volatility_percentile,
            },
            passed=True,  # Volatility factor never blocks
        )


class TimeWindowFactor(ConfirmationFactor):
    """Trading hours confirmation.

    Higher scores during high-liquidity hours.
    """

    # High liquidity hours (UTC)
    HIGH_LIQUIDITY_HOURS = [
        (8, 12),   # London open
        (13, 17),  # London/NY overlap
        (20, 23),  # Asia open
    ]

    def __init__(self, weight: float = 0.5, required: bool = False):
        super().__init__("time_window", weight, required)

    def score(
        self, signal: Signal, context: dict[str, Any]
    ) -> FactorScore:
        """Calculate time window score."""
        current_hour = context.get("current_hour")
        if current_hour is None:
            current_hour = datetime.utcnow().hour

        # Check if in high liquidity window
        in_high_liquidity = any(
            start <= current_hour <= end
            for start, end in self.HIGH_LIQUIDITY_HOURS
        )

        if in_high_liquidity:
            score = 0.9
        else:
            score = 0.6  # Still tradeable, just less ideal

        return FactorScore(
            factor_name=self.name,
            score=score,
            weight=self.weight,
            details={
                "current_hour": current_hour,
                "in_high_liquidity": in_high_liquidity,
            },
            passed=True,  # Time factor never blocks
        )


class MultiFactorConfirmation:
    """Multi-factor signal confirmation system.

    Combines multiple independent factors to calculate signal confidence.
    """

    # Confidence thresholds
    MIN_CONFIRMATION_SCORE = 0.5  # Below this = reject signal
    HIGH_CONFIDENCE_THRESHOLD = 0.75  # Above this = high confidence

    def __init__(self, factors: list[ConfirmationFactor] | None = None):
        """Initialize multi-factor confirmation.

        Args:
            factors: List of confirmation factors (uses defaults if None)
        """
        if factors is None:
            # Default factor set
            factors = [
                MomentumFactor(weight=1.0, required=False),
                VolumeFactor(weight=0.8, required=False),
                OFIFactor(weight=1.2, required=False),
                RegimeFactor(weight=0.9, required=False),
                VolatilityFactor(weight=0.7, required=False),
                TimeWindowFactor(weight=0.5, required=False),
            ]

        self._factors = factors
        self._total_weight = sum(f.weight for f in factors)

        # Statistics
        self._signals_checked = 0
        self._signals_confirmed = 0
        self._factor_pass_rates: dict[str, float] = {}

        logger.info(
            f"MultiFactorConfirmation initialized with {len(factors)} factors, "
            f"total_weight={self._total_weight:.2f}"
        )

    def confirm(self, signal: Signal, context: dict[str, Any]) -> ConfirmationResult:
        """Confirm a signal using all factors.

        Args:
            signal: Signal to confirm
            context: Context with market data, regime, etc.

        Returns:
            ConfirmationResult with combined score and details
        """
        self._signals_checked += 1

        factor_scores: list[FactorScore] = []
        weighted_sum = 0.0
        required_failed = False
        failed_factor: str | None = None

        for factor in self._factors:
            factor_score = factor.score(signal, context)
            factor_scores.append(factor_score)

            weighted_sum += factor_score.score * factor_score.weight

            # Check required factors
            if factor.required and not factor_score.passed:
                required_failed = True
                failed_factor = factor.name

        # Calculate total score
        total_score = weighted_sum / self._total_weight if self._total_weight > 0 else 0.5

        # Determine if confirmed
        confirmed = (
            not required_failed
            and total_score >= self.MIN_CONFIRMATION_SCORE
        )

        # Calculate confidence (0.0 to 1.0)
        if confirmed:
            # Map score to confidence
            # 0.5 = 0.5 confidence, 0.75 = 0.75 confidence, 1.0 = 1.0 confidence
            confidence = total_score
        else:
            confidence = 0.0

        # Determine reason if not confirmed
        reason = None
        if not confirmed:
            if required_failed:
                reason = f"required_factor_failed:{failed_factor}"
            else:
                reason = f"low_score:{total_score:.2f}"

        if confirmed:
            self._signals_confirmed += 1

        # Update factor pass rates
        for fs in factor_scores:
            if fs.factor_name not in self._factor_pass_rates:
                self._factor_pass_rates[fs.factor_name] = 0.0
            # Running average
            current_rate = self._factor_pass_rates[fs.factor_name]
            new_rate = current_rate + (1.0 if fs.passed else 0.0 - current_rate) / self._signals_checked
            self._factor_pass_rates[fs.factor_name] = new_rate

        result = ConfirmationResult(
            signal=signal,
            total_score=total_score,
            factor_scores=factor_scores,
            confidence=confidence,
            confirmed=confirmed,
            reason=reason,
        )

        if not confirmed:
            logger.debug(
                f"[MULTI-FACTOR] Signal NOT confirmed: {signal.symbol} {signal.side}, "
                f"score={total_score:.2f}, reason={reason}"
            )
        else:
            logger.debug(
                f"[MULTI-FACTOR] Signal confirmed: {signal.symbol} {signal.side}, "
                f"score={total_score:.2f}, confidence={confidence:.2f}"
            )

        return result

    def update_market_data(
        self,
        symbol: str,
        price: float,
        volume: float | None = None,
    ) -> None:
        """Update market data for factors that track history.

        Args:
            symbol: Trading symbol
            price: Current price
            volume: Optional current volume
        """
        for factor in self._factors:
            if isinstance(factor, MomentumFactor):
                factor.update_price(symbol, price)
            elif isinstance(factor, VolumeFactor) and volume is not None:
                factor.update_volume(symbol, volume)

    def get_statistics(self) -> dict[str, Any]:
        """Get confirmation statistics."""
        confirmation_rate = (
            self._signals_confirmed / self._signals_checked
            if self._signals_checked > 0 else 0.0
        )

        return {
            "signals_checked": self._signals_checked,
            "signals_confirmed": self._signals_confirmed,
            "confirmation_rate": confirmation_rate,
            "factor_pass_rates": self._factor_pass_rates,
            "factors": [f.name for f in self._factors],
        }

    def set_min_score(self, min_score: float) -> None:
        """Set minimum confirmation score.

        Args:
            min_score: Minimum score for confirmation (0.0 to 1.0)
        """
        self.MIN_CONFIRMATION_SCORE = max(0.0, min(1.0, min_score))
        logger.info(f"Multi-factor min score set to {self.MIN_CONFIRMATION_SCORE:.2f}")

    def set_factor_weight(self, factor_name: str, weight: float) -> None:
        """Set weight for a specific factor.

        Args:
            factor_name: Name of the factor
            weight: New weight (0.0 to 2.0)
        """
        for factor in self._factors:
            if factor.name == factor_name:
                factor.weight = max(0.0, min(2.0, weight))
                self._total_weight = sum(f.weight for f in self._factors)
                logger.info(f"Factor {factor_name} weight set to {factor.weight:.2f}")
                return

        logger.warning(f"Factor {factor_name} not found")

    def set_factor_required(self, factor_name: str, required: bool) -> None:
        """Set whether a factor is required.

        Args:
            factor_name: Name of the factor
            required: Whether factor is required
        """
        for factor in self._factors:
            if factor.name == factor_name:
                factor.required = required
                logger.info(f"Factor {factor_name} required={required}")
                return

        logger.warning(f"Factor {factor_name} not found")
