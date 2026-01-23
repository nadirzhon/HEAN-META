"""
Event Types

Type definitions for trading system events.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class EventType(str, Enum):
    """Event type enumeration."""
    TRADE = "TRADE"
    SIGNAL = "SIGNAL"
    PREDICTION = "PREDICTION"
    RISK = "RISK"
    SYSTEM = "SYSTEM"
    MARKET_DATA = "MARKET_DATA"
    ORDER = "ORDER"


@dataclass
class Event:
    """Base event class."""
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": self.metadata or {},
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Event:
        """Create event from dictionary."""
        return cls(
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data["data"],
            metadata=data.get("metadata"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> Event:
        """Create event from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class TradeEvent(Event):
    """Trade execution event."""

    def __init__(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        pnl: float,
        strategy: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize trade event."""
        super().__init__(
            event_type=EventType.TRADE,
            timestamp=timestamp or datetime.now(),
            data={
                "symbol": symbol,
                "side": side,
                "size": size,
                "price": price,
                "pnl": pnl,
                "strategy": strategy,
            },
            metadata=metadata,
        )


@dataclass
class SignalEvent(Event):
    """Trading signal event."""

    def __init__(
        self,
        symbol: str,
        direction: str,
        strength: float,
        source: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize signal event."""
        super().__init__(
            event_type=EventType.SIGNAL,
            timestamp=timestamp or datetime.now(),
            data={
                "symbol": symbol,
                "direction": direction,
                "strength": strength,
                "source": source,
            },
            metadata=metadata,
        )


@dataclass
class PredictionEvent(Event):
    """ML prediction event."""

    def __init__(
        self,
        model: str,
        symbol: str,
        prediction: str,
        confidence: float,
        features: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize prediction event."""
        super().__init__(
            event_type=EventType.PREDICTION,
            timestamp=timestamp or datetime.now(),
            data={
                "model": model,
                "symbol": symbol,
                "prediction": prediction,
                "confidence": confidence,
                "features": features or {},
            },
            metadata=metadata,
        )


@dataclass
class RiskEvent(Event):
    """Risk management event."""

    def __init__(
        self,
        event_subtype: str,  # "LIMIT_BREACH", "DRAWDOWN_ALERT", etc.
        severity: str,  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
        message: str,
        current_value: float,
        threshold: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize risk event."""
        super().__init__(
            event_type=EventType.RISK,
            timestamp=timestamp or datetime.now(),
            data={
                "event_subtype": event_subtype,
                "severity": severity,
                "message": message,
                "current_value": current_value,
                "threshold": threshold,
            },
            metadata=metadata,
        )


@dataclass
class SystemEvent(Event):
    """System event."""

    def __init__(
        self,
        component: str,
        event_subtype: str,  # "STARTUP", "SHUTDOWN", "ERROR", "WARNING"
        message: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize system event."""
        super().__init__(
            event_type=EventType.SYSTEM,
            timestamp=timestamp or datetime.now(),
            data={
                "component": component,
                "event_subtype": event_subtype,
                "message": message,
            },
            metadata=metadata,
        )


@dataclass
class MarketDataEvent(Event):
    """Market data update event."""

    def __init__(
        self,
        symbol: str,
        price: float,
        volume: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize market data event."""
        super().__init__(
            event_type=EventType.MARKET_DATA,
            timestamp=timestamp or datetime.now(),
            data={
                "symbol": symbol,
                "price": price,
                "volume": volume,
                "bid": bid,
                "ask": ask,
            },
            metadata=metadata,
        )


@dataclass
class OrderEvent(Event):
    """Order event."""

    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        size: float,
        price: Optional[float] = None,
        status: str = "PENDING",
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize order event."""
        super().__init__(
            event_type=EventType.ORDER,
            timestamp=timestamp or datetime.now(),
            data={
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "size": size,
                "price": price,
                "status": status,
            },
            metadata=metadata,
        )
