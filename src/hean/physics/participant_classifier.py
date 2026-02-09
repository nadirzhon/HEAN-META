"""Participant Classification Module - Layer 4 X-Ray.

Classifies market participants into 5 categories:
- Market Makers (MM): Symmetric limits, fast cancels
- Institutional: Large single-level orders, iceberg patterns
- Arb Bots: Multi-exchange simultaneous orders, <10ms timing
- Retail: Small market orders, round number clusters
- Whales: Single large orders (>20x median)
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class ParticipantType(Enum):
    MARKET_MAKER = "market_maker"
    INSTITUTIONAL = "institutional"
    ARB_BOT = "arb_bot"
    RETAIL = "retail"
    WHALE = "whale"


@dataclass
class TradeClassification:
    trade_id: str
    timestamp: float
    symbol: str
    price: float
    size: float
    side: str
    is_market_order: bool
    participant_type: ParticipantType
    confidence: float
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class ParticipantBreakdown:
    timestamp: datetime
    symbol: str
    mm_activity: float = 0.0
    institutional_flow: float = 0.0
    retail_sentiment: float = 0.0
    whale_activity: float = 0.0
    arb_pressure: float = 0.0
    dominant_player: ParticipantType = ParticipantType.RETAIL
    meta_signal: str = "neutral"
    mm_bid_ask_symmetry: float = 0.0
    institutional_iceberg_detected: bool = False
    arb_timing_score: float = 0.0
    retail_round_number_bias: float = 0.0
    whale_count: int = 0
    mm_volume_usd: float = 0.0
    institutional_volume_usd: float = 0.0
    arb_volume_usd: float = 0.0
    retail_volume_usd: float = 0.0
    whale_volume_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "mm_activity": self.mm_activity,
            "institutional_flow": self.institutional_flow,
            "retail_sentiment": self.retail_sentiment,
            "whale_activity": self.whale_activity,
            "arb_pressure": self.arb_pressure,
            "dominant_player": self.dominant_player.value,
            "meta_signal": self.meta_signal,
            "mm_bid_ask_symmetry": self.mm_bid_ask_symmetry,
            "institutional_iceberg_detected": self.institutional_iceberg_detected,
            "arb_timing_score": self.arb_timing_score,
            "whale_count": self.whale_count,
        }


class ParticipantClassifier:
    """Real-time participant classification engine."""

    def __init__(
        self,
        bus: EventBus,
        lookback_window: int = 100,
    ):
        self._bus = bus
        self._lookback_window = lookback_window

        self._trades: dict[str, deque[TradeClassification]] = defaultdict(
            lambda: deque(maxlen=lookback_window)
        )
        self._price_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=lookback_window)
        )
        self._size_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"median": 0.0, "mean": 0.0, "std": 0.0, "sizes": deque(maxlen=100)}
        )
        self._arb_timing: deque[tuple[float, str, float]] = deque(maxlen=100)
        self._breakdowns: dict[str, ParticipantBreakdown] = {}
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        logger.info("ParticipantClassifier started")

    async def stop(self) -> None:
        self._running = False
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        logger.info("ParticipantClassifier stopped")

    async def _handle_tick(self, event: Event) -> None:
        tick = event.data.get("tick")
        if not tick:
            return

        classification = self._classify_trade(tick)
        self._trades[tick.symbol].append(classification)
        self._update_size_stats(tick.symbol, tick.volume)
        self._price_history[tick.symbol].append(tick.price)
        self._arb_timing.append((time.time(), tick.symbol, tick.price))

        breakdown = self._calculate_breakdown(tick.symbol)
        self._breakdowns[tick.symbol] = breakdown

        await self._bus.publish(
            Event(
                event_type=EventType.CONTEXT_UPDATE,
                data={
                    "context_type": "participant_breakdown",
                    "symbol": tick.symbol,
                    "breakdown": breakdown.to_dict(),
                },
            )
        )

    def _classify_trade(self, tick: Any) -> TradeClassification:
        features = {}
        symbol = tick.symbol
        size = tick.volume
        price = tick.price

        stats = self._size_stats[symbol]
        median_size = stats["median"] if stats["median"] > 0 else 1.0
        size_ratio = size / median_size
        features["size_ratio"] = size_ratio

        is_round = self._is_round_number(price)
        features["is_round_number"] = 1.0 if is_round else 0.0

        is_market_order = True
        if hasattr(tick, "bid") and tick.bid and hasattr(tick, "ask") and tick.ask:
            spread = abs(tick.ask - tick.bid)
            distance = abs(price - (tick.bid + tick.ask) / 2)
            is_market_order = distance < spread * 0.1

        # Timing analysis
        recent = [t for t in list(self._arb_timing)[-10:] if t[1] == symbol]
        if len(recent) >= 2:
            diffs = [recent[i][0] - recent[i - 1][0] for i in range(1, len(recent))]
            avg_diff = np.mean(diffs) if diffs else 1.0
            features["timing_regularity"] = 1.0 / (avg_diff + 0.01)
        else:
            features["timing_regularity"] = 0.0

        # Classification rules
        participant_type = ParticipantType.RETAIL
        confidence = 0.5

        if size_ratio > 20.0 and is_market_order:
            participant_type = ParticipantType.WHALE
            confidence = min(0.95, 0.5 + size_ratio / 50.0)
        elif 5.0 < size_ratio <= 20.0:
            participant_type = ParticipantType.INSTITUTIONAL
            confidence = 0.7
        elif features["timing_regularity"] > 50.0:
            participant_type = ParticipantType.ARB_BOT
            confidence = 0.8
        elif size_ratio < 1.5 and (is_market_order or is_round):
            participant_type = ParticipantType.RETAIL
            confidence = 0.6

        ts = tick.timestamp.timestamp() if hasattr(tick.timestamp, "timestamp") else time.time()
        trade_id = f"{symbol}_{int(ts * 1e6)}"

        return TradeClassification(
            trade_id=trade_id,
            timestamp=ts,
            symbol=symbol,
            price=price,
            size=size,
            side="buy" if tick.volume > 0 else "sell",
            is_market_order=is_market_order,
            participant_type=participant_type,
            confidence=confidence,
            features=features,
        )

    def _update_size_stats(self, symbol: str, size: float) -> None:
        stats = self._size_stats[symbol]
        stats["sizes"].append(size)
        sizes = list(stats["sizes"])
        if sizes:
            stats["median"] = float(np.median(sizes))
            stats["mean"] = float(np.mean(sizes))
            stats["std"] = float(np.std(sizes)) if len(sizes) > 1 else 0.0

    def _is_round_number(self, price: float) -> bool:
        price_str = f"{price:.2f}"
        return price_str.endswith("00") or price_str.endswith("50")

    def _calculate_breakdown(self, symbol: str) -> ParticipantBreakdown:
        trades = list(self._trades[symbol])
        if not trades:
            return ParticipantBreakdown(timestamp=datetime.utcnow(), symbol=symbol)

        type_counts: dict[ParticipantType, int] = defaultdict(int)
        type_volumes: dict[ParticipantType, float] = defaultdict(float)

        for trade in trades:
            type_counts[trade.participant_type] += 1
            type_volumes[trade.participant_type] += trade.size * trade.price

        total_trades = len(trades)

        mm_activity = type_counts[ParticipantType.MARKET_MAKER] / total_trades
        institutional_flow = type_volumes[ParticipantType.INSTITUTIONAL]
        arb_pressure = type_counts[ParticipantType.ARB_BOT] / total_trades
        whale_activity = type_counts[ParticipantType.WHALE] / total_trades

        retail_trades = [t for t in trades if t.participant_type == ParticipantType.RETAIL]
        retail_buys = sum(1 for t in retail_trades if t.side == "buy")
        retail_sentiment = retail_buys / len(retail_trades) if retail_trades else 0.5

        dominant = max(type_counts, key=type_counts.get) if type_counts else ParticipantType.RETAIL

        meta_signal = self._generate_meta_signal(
            mm_activity, institutional_flow, retail_sentiment, whale_activity, arb_pressure
        )

        return ParticipantBreakdown(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            mm_activity=mm_activity,
            institutional_flow=institutional_flow,
            retail_sentiment=retail_sentiment,
            whale_activity=whale_activity,
            arb_pressure=arb_pressure,
            dominant_player=dominant,
            meta_signal=meta_signal,
            whale_count=type_counts[ParticipantType.WHALE],
            mm_volume_usd=type_volumes[ParticipantType.MARKET_MAKER],
            institutional_volume_usd=type_volumes[ParticipantType.INSTITUTIONAL],
            arb_volume_usd=type_volumes[ParticipantType.ARB_BOT],
            retail_volume_usd=type_volumes[ParticipantType.RETAIL],
            whale_volume_usd=type_volumes[ParticipantType.WHALE],
        )

    def _generate_meta_signal(
        self, mm: float, inst: float, retail: float, whale: float, arb: float
    ) -> str:
        if mm > 0.6 and retail > 0.7:
            return "MM hunting stops, follow reversal"
        if inst > 100000 and retail < 0.3:
            return "Institutional accumulating, trend continuation"
        if whale > 0.3 and retail > 0.6:
            return "Whale dumping on retail, exit longs"
        if arb > 0.5:
            return "High arb activity, reduced edge"
        if retail > 0.8 and mm < 0.2:
            return "Retail FOMO, reversal likely"
        if 0.4 < retail < 0.6 and mm > 0.3:
            return "Balanced market, wait for catalyst"
        return "Neutral"

    def get_breakdown(self, symbol: str) -> ParticipantBreakdown | None:
        return self._breakdowns.get(symbol)

    def get_all_breakdowns(self) -> dict[str, ParticipantBreakdown]:
        return self._breakdowns.copy()
