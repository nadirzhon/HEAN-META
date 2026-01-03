"""Bybit-specific data models."""

from typing import Any

from pydantic import BaseModel


class BybitTicker(BaseModel):
    """Bybit ticker data."""

    symbol: str
    last_price: str
    bid1_price: str
    ask1_price: str
    volume_24h: str
    timestamp: int


class BybitOrderResponse(BaseModel):
    """Bybit order response."""

    order_id: str
    symbol: str
    side: str
    order_type: str
    qty: str
    price: str | None
    status: str


class BybitEarnProduct(BaseModel):
    """Bybit Earn product information."""

    product_id: str
    category: str  # FlexibleSaving, FixedSaving, OnChain, etc.
    coin: str
    product_name: str
    purchase_currency: str
    interest_rate: float | None = None  # Annual interest rate
    min_purchase_amount: str | None = None
    max_purchase_amount: str | None = None
    redemption_delay_days: int | None = None
    lock_period_days: int | None = None
    auto_compound: bool = False
    extra: dict[str, Any] | None = None  # Additional product data


class BybitEarnHolding(BaseModel):
    """Bybit Earn holding (current investment)."""

    product_id: str
    category: str
    coin: str
    quantity: str
    total_interest: str | None = None
    next_interest_payment_time: str | None = None
    next_redemption_time: str | None = None
    auto_compound: bool = False
    extra: dict[str, Any] | None = None
