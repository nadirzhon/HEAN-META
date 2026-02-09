"""
Event Envelope - Унифицированный формат всех событий в системе

Каждое событие (trade, orderbook update, candle, etc) упаковывается
в единый envelope для обработки остальными модулями
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(Enum):
    """Типы событий в нервной системе"""
    TRADE = "trade"
    ORDERBOOK = "orderbook"
    CANDLE = "candle"
    FUNDING = "funding"
    POSITION_UPDATE = "position"
    ORDER_UPDATE = "order"
    LIQUIDATION = "liquidation"
    SYSTEM = "system"
    HEALTH = "health"


@dataclass
class EventEnvelope:
    """
    Единый формат события - все события системы упакованы в этот формат

    Это нерв, который передаёт сигнал по всей системе
    """

    # Identity
    event_id: str
    event_type: EventType
    source: str  # "bybit_ws", "exchange_api", "internal"

    # Timing (nanosecond precision)
    timestamp_ns: int
    received_at_ns: int
    processing_lag_ms: float = field(init=False)

    # Symbol
    symbol: str

    # Data payload (тип зависит от event_type)
    data: dict[str, Any]

    # Health context
    quality_score: float = 1.0  # 0.0 - 1.0
    data_gaps_detected: bool = False
    latency_warning: bool = False

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Вычисление лага обработки"""
        self.processing_lag_ms = (self.received_at_ns - self.timestamp_ns) / 1_000_000

        # Warning если лаг > 100ms
        if self.processing_lag_ms > 100:
            self.latency_warning = True
            self.quality_score *= 0.9

    @property
    def timestamp(self) -> datetime:
        """Timestamp as datetime"""
        return datetime.fromtimestamp(self.timestamp_ns / 1_000_000_000)

    @property
    def age_ms(self) -> float:
        """Возраст события в миллисекундах"""
        now_ns = time.time_ns()
        return (now_ns - self.timestamp_ns) / 1_000_000

    def is_stale(self, max_age_ms: float = 1000) -> bool:
        """Проверка - не устарело ли событие"""
        return self.age_ms > max_age_ms

    def to_dict(self) -> dict[str, Any]:
        """Сериализация в dict"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'source': self.source,
            'timestamp_ns': self.timestamp_ns,
            'received_at_ns': self.received_at_ns,
            'processing_lag_ms': self.processing_lag_ms,
            'symbol': self.symbol,
            'data': self.data,
            'quality_score': self.quality_score,
            'data_gaps_detected': self.data_gaps_detected,
            'latency_warning': self.latency_warning,
            'metadata': self.metadata,
        }


@dataclass
class TradeEvent:
    """Trade событие - отдельная сделка"""
    price: float
    qty: float
    side: str  # "Buy" or "Sell"
    trade_id: str
    is_block_trade: bool = False


@dataclass
class OrderbookEvent:
    """Orderbook update - изменение стакана"""
    bids: list[tuple[float, float]]  # [(price, qty), ...]
    asks: list[tuple[float, float]]
    update_id: int

    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None

    @property
    def spread(self) -> float | None:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> float | None:
        """Спред в базисных пунктах"""
        if self.spread and self.best_bid:
            return (self.spread / self.best_bid) * 10000
        return None


@dataclass
class CandleEvent:
    """Candle - свеча"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str  # "1m", "5m", "15m", etc
    is_confirmed: bool  # True если свеча закрыта


@dataclass
class FundingEvent:
    """Funding rate update"""
    funding_rate: float
    predicted_rate: float | None
    next_funding_time: datetime


@dataclass
class PositionEvent:
    """Position update - изменение позиции"""
    symbol: str
    side: str  # "Buy" or "Sell"
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: float


def create_trade_envelope(
    trade: TradeEvent,
    symbol: str,
    timestamp_ns: int,
    source: str = "bybit_ws"
) -> EventEnvelope:
    """Создаёт envelope для trade события"""
    return EventEnvelope(
        event_id=f"trade_{trade.trade_id}",
        event_type=EventType.TRADE,
        source=source,
        timestamp_ns=timestamp_ns,
        received_at_ns=time.time_ns(),
        symbol=symbol,
        data={
            'price': trade.price,
            'qty': trade.qty,
            'side': trade.side,
            'trade_id': trade.trade_id,
            'is_block_trade': trade.is_block_trade,
        }
    )


def create_orderbook_envelope(
    orderbook: OrderbookEvent,
    symbol: str,
    timestamp_ns: int,
    source: str = "bybit_ws"
) -> EventEnvelope:
    """Создаёт envelope для orderbook события"""

    # Quality check - spread
    quality = 1.0
    if orderbook.spread_bps and orderbook.spread_bps > 10:  # Spread > 0.1%
        quality = 0.8  # Тонкая ликвидность

    return EventEnvelope(
        event_id=f"orderbook_{symbol}_{orderbook.update_id}",
        event_type=EventType.ORDERBOOK,
        source=source,
        timestamp_ns=timestamp_ns,
        received_at_ns=time.time_ns(),
        symbol=symbol,
        data={
            'bids': orderbook.bids,
            'asks': orderbook.asks,
            'update_id': orderbook.update_id,
            'best_bid': orderbook.best_bid,
            'best_ask': orderbook.best_ask,
            'spread': orderbook.spread,
            'spread_bps': orderbook.spread_bps,
        },
        quality_score=quality
    )


def create_candle_envelope(
    candle: CandleEvent,
    symbol: str,
    timestamp_ns: int,
    source: str = "bybit_ws"
) -> EventEnvelope:
    """Создаёт envelope для candle события"""
    return EventEnvelope(
        event_id=f"candle_{symbol}_{candle.timeframe}_{timestamp_ns}",
        event_type=EventType.CANDLE,
        source=source,
        timestamp_ns=timestamp_ns,
        received_at_ns=time.time_ns(),
        symbol=symbol,
        data={
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume,
            'timeframe': candle.timeframe,
            'is_confirmed': candle.is_confirmed,
        }
    )
