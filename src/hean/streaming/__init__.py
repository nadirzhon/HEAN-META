"""
Event Streaming

Real-time event streaming with Redis Streams and Kafka.
"""

from .event_streaming import (
    EventPublisher,
    EventConsumer,
    EventStreamConfig,
    StreamBackend,
)
from .event_types import (
    Event,
    TradeEvent,
    SignalEvent,
    PredictionEvent,
    RiskEvent,
    SystemEvent,
)

__all__ = [
    "EventPublisher",
    "EventConsumer",
    "EventStreamConfig",
    "StreamBackend",
    "Event",
    "TradeEvent",
    "SignalEvent",
    "PredictionEvent",
    "RiskEvent",
    "SystemEvent",
]
