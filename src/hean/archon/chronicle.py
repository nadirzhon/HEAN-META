"""Chronicle -- audit trail for key trading decisions and events."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChronicleEntry:
    """A single audit trail entry."""

    timestamp: datetime
    event_type: str
    correlation_id: str
    strategy_id: str
    symbol: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for API responses."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


# Events that the Chronicle subscribes to
_CHRONICLED_EVENTS: list[EventType] = [
    EventType.SIGNAL,
    EventType.ORDER_REQUEST,
    EventType.RISK_BLOCKED,
    EventType.ORDER_FILLED,
    EventType.ORDER_REJECTED,
    EventType.KILLSWITCH_TRIGGERED,
    EventType.ARCHON_DIRECTIVE,
]


class Chronicle:
    """Audit trail -- records key trading decisions and events.

    Maintains an in-memory ring buffer of ``ChronicleEntry`` records.
    Subscribes to important EventBus events and stores structured
    records for later query and post-mortem analysis.
    """

    def __init__(
        self,
        bus: EventBus,
        max_memory: int = 10000,
    ) -> None:
        self._bus = bus
        self._max_memory = max_memory
        self._entries: list[ChronicleEntry] = []

    async def start(self) -> None:
        """Subscribe to key events and begin recording."""
        for et in _CHRONICLED_EVENTS:
            self._bus.subscribe(et, self._on_event)
        logger.info(
            "[Chronicle] Started -- auditing %d event types",
            len(_CHRONICLED_EVENTS),
        )

    async def stop(self) -> None:
        """Unsubscribe from all events."""
        for et in _CHRONICLED_EVENTS:
            self._bus.unsubscribe(et, self._on_event)
        logger.info("[Chronicle] Stopped -- %d entries recorded", len(self._entries))

    async def _on_event(self, event: Event) -> None:
        """Handle an incoming event and create a chronicle entry."""
        data = event.data
        entry = ChronicleEntry(
            timestamp=event.timestamp,
            event_type=event.event_type.value,
            correlation_id=data.get("_correlation_id", data.get("correlation_id", "")),
            strategy_id=self._extract_strategy_id(data),
            symbol=self._extract_symbol(data),
            details=self._extract_details(event),
        )
        self._entries.append(entry)

        # Enforce ring buffer limit
        while len(self._entries) > self._max_memory:
            self._entries.pop(0)

    def query(
        self,
        event_type: str | None = None,
        symbol: str | None = None,
        strategy_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Filter and return matching chronicle entries.

        Args:
            event_type: Filter by event type string (e.g. "signal").
            symbol: Filter by trading symbol (e.g. "BTCUSDT").
            strategy_id: Filter by strategy ID.
            limit: Maximum number of entries to return.

        Returns:
            List of entry dicts, most recent first.
        """
        results: list[ChronicleEntry] = []

        for entry in reversed(self._entries):
            if event_type and entry.event_type != event_type:
                continue
            if symbol and entry.symbol != symbol:
                continue
            if strategy_id and entry.strategy_id != strategy_id:
                continue
            results.append(entry)
            if len(results) >= limit:
                break

        return [e.to_dict() for e in results]

    def get_signal_journey(self, correlation_id: str) -> list[dict[str, Any]]:
        """Return all entries for a given correlation_id.

        This traces the full lifecycle of a signal through the system.

        Args:
            correlation_id: The correlation ID to trace.

        Returns:
            List of entry dicts in chronological order.
        """
        journey = [e for e in self._entries if e.correlation_id == correlation_id]
        return [e.to_dict() for e in journey]

    @property
    def size(self) -> int:
        """Number of entries currently stored."""
        return len(self._entries)

    # -- Extraction helpers ----------------------------------------------

    @staticmethod
    def _extract_strategy_id(data: dict[str, Any]) -> str:
        """Extract strategy_id from event data."""
        if "strategy_id" in data:
            return str(data["strategy_id"])
        signal = data.get("signal")
        if signal:
            return str(getattr(signal, "strategy_id", ""))
        order_request = data.get("order_request")
        if order_request:
            return str(getattr(order_request, "strategy_id", ""))
        return ""

    @staticmethod
    def _extract_symbol(data: dict[str, Any]) -> str:
        """Extract symbol from event data."""
        if "symbol" in data:
            return str(data["symbol"])
        signal = data.get("signal")
        if signal:
            return str(getattr(signal, "symbol", ""))
        order_request = data.get("order_request")
        if order_request:
            return str(getattr(order_request, "symbol", ""))
        return ""

    @staticmethod
    def _extract_details(event: Event) -> dict[str, Any]:
        """Extract relevant details from event data.

        Strips out non-serializable objects, keeping only
        primitive values for the audit record.
        """
        data = event.data
        details: dict[str, Any] = {}

        # Common fields
        for key in (
            "side",
            "confidence",
            "reason",
            "risk_state",
            "order_id",
            "fill_price",
            "fill_qty",
            "price",
            "qty",
            "order_type",
            "size",
            "directive_type",
            "target_component",
        ):
            if key in data:
                val = data[key]
                # Only store serializable values
                if isinstance(val, (str, int, float, bool, type(None))):
                    details[key] = val

        # Signal-specific
        signal = data.get("signal")
        if signal:
            details["signal_side"] = getattr(signal, "side", "")
            details["signal_confidence"] = getattr(signal, "confidence", 0.0)
            details["signal_entry_price"] = getattr(signal, "entry_price", 0.0)

        return details
