"""
Data models for multi-exchange funding arbitrage
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ExchangeName(str, Enum):
    """Supported exchanges"""
    BYBIT = "bybit"
    BINANCE = "binance"
    OKX = "okx"


@dataclass
class ExchangeFundingRate:
    """Funding rate from a specific exchange"""

    exchange: ExchangeName
    symbol: str
    rate: float  # Funding rate (-1 to +1, but typically -0.01 to +0.01)
    next_funding_time: datetime
    predicted_rate: float | None = None  # Predicted next funding rate
    mark_price: float | None = None  # Current mark price
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def rate_percent(self) -> float:
        """Return rate as percentage"""
        return self.rate * 100

    @property
    def annual_rate(self) -> float:
        """Calculate annualized funding rate (3 fundings per day)"""
        return self.rate * 3 * 365

    @property
    def is_positive(self) -> bool:
        """Longs pay shorts"""
        return self.rate > 0

    @property
    def is_negative(self) -> bool:
        """Shorts pay longs"""
        return self.rate < 0


@dataclass
class FundingRateSpread:
    """Spread between two exchanges for same symbol"""

    symbol: str
    high_exchange: ExchangeName
    low_exchange: ExchangeName
    high_rate: float
    low_rate: float
    spread: float  # high_rate - low_rate
    spread_percent: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_significant(self) -> bool:
        """Is spread significant enough to trade?"""
        # Spread must be > 0.02% (0.0002) to cover fees + profit
        return abs(self.spread) > 0.0002

    @property
    def profit_potential(self) -> float:
        """Estimate profit potential (spread - fees)"""
        # Assume 0.05% fee per side (0.1% total round trip)
        fees = 0.001
        return abs(self.spread) - fees


@dataclass
class FundingOpportunity:
    """Cross-exchange funding arbitrage opportunity"""

    symbol: str
    long_exchange: ExchangeName  # Where to open long
    short_exchange: ExchangeName  # Where to open short
    funding_spread: float  # Difference in funding rates
    profit_per_funding: float  # Expected profit per funding period
    next_funding_time: datetime
    hours_until_funding: float
    long_rate: float
    short_rate: float
    confidence: float  # 0-1, higher = more reliable
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Additional metadata
    long_mark_price: float | None = None
    short_mark_price: float | None = None
    price_spread: float | None = None  # Price difference between exchanges

    @property
    def should_trade(self) -> bool:
        """Should we trade this opportunity?"""
        # Requirements:
        # 1. Spread > 0.02% (covers fees + profit)
        # 2. Confidence > 70%
        # 3. At least 2 hours until funding (time to execute + hold)
        return (
            abs(self.funding_spread) > 0.0002 and
            self.confidence > 0.7 and
            self.hours_until_funding >= 2.0
        )

    @property
    def annual_profit_rate(self) -> float:
        """Annualized profit rate (3 fundings per day)"""
        return self.profit_per_funding * 3 * 365

    @property
    def risk_level(self) -> str:
        """Risk assessment"""
        if self.confidence > 0.85 and abs(self.funding_spread) > 0.0005:
            return "LOW"
        elif self.confidence > 0.7:
            return "MEDIUM"
        else:
            return "HIGH"


@dataclass
class ArbitrageSignal:
    """Trading signal for funding arbitrage"""

    opportunity: FundingOpportunity
    action: str  # "OPEN", "CLOSE", "HOLD"
    position_size_usd: float
    entry_prices: dict[ExchangeName, float]  # Entry price per exchange
    target_hold_hours: float  # How long to hold position
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Risk management
    max_loss_usd: float = 0.0
    expected_profit_usd: float = 0.0

    @property
    def profit_to_risk_ratio(self) -> float:
        """Calculate reward/risk ratio"""
        if self.max_loss_usd == 0:
            return float('inf')
        return self.expected_profit_usd / self.max_loss_usd


@dataclass
class FundingHistory:
    """Historical funding rate data"""

    exchange: ExchangeName
    symbol: str
    rates: list[float]  # Historical rates
    timestamps: list[datetime]

    @property
    def average_rate(self) -> float:
        """Average funding rate"""
        return sum(self.rates) / len(self.rates) if self.rates else 0.0

    @property
    def volatility(self) -> float:
        """Standard deviation of rates"""
        if len(self.rates) < 2:
            return 0.0
        mean = self.average_rate
        variance = sum((r - mean) ** 2 for r in self.rates) / len(self.rates)
        return variance ** 0.5

    def predict_next(self) -> float:
        """Simple moving average prediction"""
        if len(self.rates) < 3:
            return self.rates[-1] if self.rates else 0.0
        # Use last 3 rates for prediction
        return sum(self.rates[-3:]) / 3
