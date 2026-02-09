"""
Data models for Google Trends analysis
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TrendDirection(str, Enum):
    """Trend direction"""
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    SPIKE = "spike"  # Sudden increase
    CRASH = "crash"  # Sudden decrease


class InterestLevel(str, Enum):
    """Search interest level"""
    VERY_LOW = "very_low"  # 0-20
    LOW = "low"  # 20-40
    MEDIUM = "medium"  # 40-60
    HIGH = "high"  # 60-80
    VERY_HIGH = "very_high"  # 80-100


@dataclass
class TrendsData:
    """Google Trends search interest data"""

    keyword: str
    timeframe: str  # e.g., "now 7-d", "today 3-m"
    interest_over_time: list[int]  # 0-100 values
    timestamps: list[datetime]
    related_queries: list[str] = field(default_factory=list)
    rising_queries: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def current_interest(self) -> int:
        """Current interest score (0-100)"""
        return self.interest_over_time[-1] if self.interest_over_time else 0

    @property
    def average_interest(self) -> float:
        """Average interest over period"""
        return sum(self.interest_over_time) / len(self.interest_over_time) if self.interest_over_time else 0.0

    @property
    def max_interest(self) -> int:
        """Maximum interest in period"""
        return max(self.interest_over_time) if self.interest_over_time else 0

    @property
    def min_interest(self) -> int:
        """Minimum interest in period"""
        return min(self.interest_over_time) if self.interest_over_time else 0

    @property
    def interest_level(self) -> InterestLevel:
        """Classify current interest level"""
        current = self.current_interest
        if current >= 80:
            return InterestLevel.VERY_HIGH
        elif current >= 60:
            return InterestLevel.HIGH
        elif current >= 40:
            return InterestLevel.MEDIUM
        elif current >= 20:
            return InterestLevel.LOW
        else:
            return InterestLevel.VERY_LOW

    @property
    def volatility(self) -> float:
        """Standard deviation of interest"""
        if len(self.interest_over_time) < 2:
            return 0.0
        mean = self.average_interest
        variance = sum((x - mean) ** 2 for x in self.interest_over_time) / len(self.interest_over_time)
        return variance ** 0.5

    def get_trend_direction(self, lookback_periods: int = 7) -> TrendDirection:
        """
        Determine trend direction

        Args:
            lookback_periods: Number of periods to analyze

        Returns:
            TrendDirection
        """
        if len(self.interest_over_time) < lookback_periods:
            return TrendDirection.STABLE

        recent = self.interest_over_time[-lookback_periods:]
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        change_pct = (avg_second - avg_first) / avg_first if avg_first > 0 else 0

        # Detect spikes/crashes (>50% change)
        if change_pct > 0.5:
            return TrendDirection.SPIKE
        elif change_pct < -0.5:
            return TrendDirection.CRASH
        # Regular trends (>10% change)
        elif change_pct > 0.1:
            return TrendDirection.RISING
        elif change_pct < -0.1:
            return TrendDirection.FALLING
        else:
            return TrendDirection.STABLE

    def calculate_momentum(self) -> float:
        """
        Calculate momentum (rate of change)

        Returns:
            Momentum score (-1 to +1)
        """
        if len(self.interest_over_time) < 2:
            return 0.0

        # Compare last value to average
        current = self.current_interest
        avg = self.average_interest

        if avg == 0:
            return 0.0

        momentum = (current - avg) / avg
        # Normalize to -1 to +1
        return max(-1.0, min(1.0, momentum))


@dataclass
class TrendsSignal:
    """Trading signal from Google Trends analysis"""

    keyword: str
    interest_score: int  # 0-100
    interest_level: InterestLevel
    trend_direction: TrendDirection
    momentum: float  # -1 to +1
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0-1
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Supporting data
    related_queries: list[str] = field(default_factory=list)
    rising_queries: list[str] = field(default_factory=list)

    @property
    def should_trade(self) -> bool:
        """Should we trade based on this signal?"""
        # Requirements:
        # 1. Strong interest (high or very high)
        # 2. Clear direction (not stable)
        # 3. High confidence (>70%)
        return (
            self.interest_level in [InterestLevel.HIGH, InterestLevel.VERY_HIGH] and
            self.trend_direction != TrendDirection.STABLE and
            self.confidence > 0.7
        )

    @property
    def is_bullish(self) -> bool:
        """Is this a bullish signal?"""
        return self.trend_direction in [TrendDirection.RISING, TrendDirection.SPIKE]

    @property
    def is_bearish(self) -> bool:
        """Is this a bearish signal?"""
        return self.trend_direction in [TrendDirection.FALLING, TrendDirection.CRASH]

    @property
    def risk_level(self) -> str:
        """Assess risk level"""
        # Spikes and crashes are riskier
        if self.trend_direction in [TrendDirection.SPIKE, TrendDirection.CRASH]:
            return "HIGH"
        elif self.confidence > 0.8:
            return "LOW"
        else:
            return "MEDIUM"


@dataclass
class ComparativeTrendsData:
    """Compare multiple keywords"""

    keywords: list[str]
    interest_data: dict[str, TrendsData]
    winner: str  # Keyword with highest interest
    leader_advantage: float  # How much leader is ahead (%)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def relative_strengths(self) -> dict[str, float]:
        """Calculate relative strength of each keyword"""
        total_interest = sum(
            data.current_interest
            for data in self.interest_data.values()
        )

        if total_interest == 0:
            return dict.fromkeys(self.keywords, 0.0)

        return {
            kw: data.current_interest / total_interest
            for kw, data in self.interest_data.items()
        }


@dataclass
class TrendsHistory:
    """Historical trends data for prediction"""

    keyword: str
    interest_values: list[int]
    timestamps: list[datetime]

    def predict_next(self, periods_ahead: int = 1) -> list[float]:
        """
        Simple moving average prediction

        Args:
            periods_ahead: How many periods to predict

        Returns:
            List of predicted values
        """
        if len(self.interest_values) < 3:
            # Not enough data, return last value
            return [float(self.interest_values[-1])] * periods_ahead if self.interest_values else [0.0] * periods_ahead

        # Use last 7 values for MA
        window = min(7, len(self.interest_values))
        recent = self.interest_values[-window:]
        ma = sum(recent) / len(recent)

        # Simple prediction: assume continuation of trend
        return [ma] * periods_ahead

    def calculate_correlation(self, price_changes: list[float]) -> float:
        """
        Calculate correlation between interest and price changes

        Args:
            price_changes: List of price change percentages

        Returns:
            Correlation coefficient (-1 to +1)
        """
        if len(self.interest_values) != len(price_changes):
            return 0.0

        # Pearson correlation
        n = len(self.interest_values)
        if n == 0:
            return 0.0

        sum_x = sum(self.interest_values)
        sum_y = sum(price_changes)
        sum_xy = sum(x * y for x, y in zip(self.interest_values, price_changes, strict=False))
        sum_x2 = sum(x ** 2 for x in self.interest_values)
        sum_y2 = sum(y ** 2 for y in price_changes)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator
