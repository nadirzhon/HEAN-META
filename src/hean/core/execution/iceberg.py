"""Iceberg order splitter for stealth execution and UI detection."""

from __future__ import annotations

import asyncio
import math
import random
import time
import uuid
from collections import deque
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, OrderRequest
from hean.logging import get_logger

logger = get_logger(__name__)


class IcebergOrder:
    """Split large orders into micro-batches with staggered timing."""

    def __init__(
        self,
        bus: EventBus,
        min_size_usdt: float = 1000.0,
        max_micro_size_usdt: float = 100.0,
        min_delay_ms: int = 100,
        max_delay_ms: int = 500,
    ) -> None:
        self._bus = bus
        self._min_size_usdt = min_size_usdt
        self._max_micro_size_usdt = max_micro_size_usdt
        self._min_delay_ms = min_delay_ms
        self._max_delay_ms = max_delay_ms
        self._detections: deque[dict[str, Any]] = deque(maxlen=100)

    async def process_order(
        self,
        order_request: OrderRequest,
        current_price: float,
    ) -> list[OrderRequest]:
        """Return micro-batches or original order if below threshold."""
        if current_price <= 0:
            return [order_request]

        metadata = dict(order_request.metadata or {})
        if metadata.get("iceberg") or metadata.get("iceberg_parent_id"):
            return [order_request]

        notional = order_request.size * current_price
        if notional < self._min_size_usdt or self._max_micro_size_usdt <= 0:
            return [order_request]

        micro_size = self._max_micro_size_usdt / current_price
        if micro_size <= 0 or order_request.size <= micro_size:
            return [order_request]

        parent_id = str(uuid.uuid4())
        micro_count = max(1, math.ceil(order_request.size / micro_size))

        self._detections.appendleft(
            {
                "symbol": order_request.symbol,
                "price": current_price,
                "suspectedSize": notional,
                "side": "bid" if order_request.side == "buy" else "ask",
                "detectionTime": int(time.time() * 1000),
                "parent_id": parent_id,
            }
        )

        micro_requests: list[OrderRequest] = []
        remaining = order_request.size
        for idx in range(micro_count):
            size = micro_size if remaining > micro_size else remaining
            remaining -= size
            micro_metadata = dict(metadata)
            micro_metadata.update(
                {
                    "iceberg": True,
                    "iceberg_parent_id": parent_id,
                    "iceberg_index": idx + 1,
                    "iceberg_count": micro_count,
                    "iceberg_notional_usd": notional,
                }
            )
            micro_requests.append(
                OrderRequest(
                    signal_id=order_request.signal_id,
                    strategy_id=order_request.strategy_id,
                    symbol=order_request.symbol,
                    side=order_request.side,
                    size=size,
                    price=order_request.price,
                    order_type=order_request.order_type,
                    stop_loss=order_request.stop_loss,
                    take_profit=order_request.take_profit,
                    metadata=micro_metadata,
                )
            )

        logger.info(
            "Iceberg order detected: %s %s size=%.6f split=%d",
            order_request.symbol,
            order_request.side,
            order_request.size,
            micro_count,
        )

        return micro_requests

    async def schedule_micro_orders(self, micro_requests: list[OrderRequest]) -> None:
        """Publish micro orders with jittered delays."""
        for request in micro_requests:
            delay = random.uniform(self._min_delay_ms, self._max_delay_ms) / 1000.0
            asyncio.create_task(self._publish_after_delay(request, delay))

    def get_recent_detections(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent iceberg detections for dashboards."""
        return list(self._detections)[:limit]

    async def _publish_after_delay(self, request: OrderRequest, delay: float) -> None:
        await asyncio.sleep(delay)
        await self._bus.publish(
            Event(event_type=EventType.ORDER_REQUEST, data={"order_request": request})
        )
