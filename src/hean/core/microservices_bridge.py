"""Redis stream bridge for external microservices.

Bridges microservice streams into the in-process EventBus:
- ``physics:{symbol}`` -> ``EventType.PHYSICS_UPDATE``
- ``brain:analysis`` -> ``EventType.BRAIN_ANALYSIS``
- ``risk:approved`` -> ``EventType.SIGNAL`` (already risk-approved)
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import orjson
import redis.asyncio as aioredis

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Signal
from hean.logging import get_logger

logger = get_logger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _temperature_regime(temperature: float) -> str:
    if temperature >= 800:
        return "HOT"
    if temperature >= 400:
        return "WARM"
    return "COLD"


def _entropy_state(entropy: float) -> str:
    if entropy >= 3.5:
        return "EQUILIBRIUM"
    if entropy < 2.0:
        return "COMPRESSED"
    return "NORMAL"


def normalize_physics_update(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize physics payload from microservice to core physics schema."""
    temperature = _safe_float(raw.get("temperature"), 0.0)
    entropy = _safe_float(raw.get("entropy"), 0.0)
    phase = str(raw.get("phase", "unknown")).lower()

    return {
        "symbol": raw.get("symbol"),
        "temperature": temperature,
        "temperature_regime": raw.get("temperature_regime", _temperature_regime(temperature)),
        "entropy": entropy,
        "entropy_state": raw.get("entropy_state", _entropy_state(entropy)),
        "phase": phase,
        "phase_confidence": _safe_float(raw.get("phase_confidence"), 0.0),
        "szilard_profit": _safe_float(raw.get("szilard_profit"), 0.0),
        "should_trade": bool(raw.get("should_trade", phase in {"water", "ice"})),
        "trade_reason": raw.get("trade_reason", f"microservice_phase={phase}"),
        "size_multiplier": _safe_float(raw.get("size_multiplier"), 1.0),
        "timestamp": raw.get("timestamp"),
    }


def normalize_brain_analysis(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize brain analysis payload from microservice."""
    bias = str(raw.get("bias", "NEUTRAL")).upper()
    sentiment_map = {
        "BULLISH": "bullish",
        "BEARISH": "bearish",
        "NEUTRAL": "neutral",
    }
    sentiment = str(raw.get("sentiment") or sentiment_map.get(bias, "neutral")).lower()

    return {
        "symbol": raw.get("symbol"),
        "sentiment": sentiment,
        "confidence": max(0.0, min(1.0, _safe_float(raw.get("confidence"), 0.5))),
        "forces": raw.get("forces", []),
        "summary": raw.get("summary", ""),
        "market_regime": str(raw.get("market_regime", raw.get("phase", "unknown"))).lower(),
        "risk_level": raw.get("risk_level"),
        "timestamp": raw.get("timestamp"),
    }


def risk_payload_to_signal(raw: dict[str, Any]) -> Signal | None:
    """Convert ``risk:approved`` payload into core ``Signal`` model."""
    action = str(raw.get("signal", "")).upper()
    if action not in {"BUY", "SELL"}:
        return None

    symbol = raw.get("symbol")
    if not symbol:
        return None

    price = _safe_float(raw.get("price"), 0.0)
    if price <= 0:
        return None

    side = "buy" if action == "BUY" else "sell"
    confidence = max(0.0, min(1.0, _safe_float(raw.get("confidence"), 0.5)))

    stop_loss = raw.get("stop_loss")
    take_profit = raw.get("take_profit")
    stop_loss_val = _safe_float(stop_loss) if stop_loss is not None else None
    take_profit_val = _safe_float(take_profit) if take_profit is not None else None

    # Ensure we always have bounded risk for downstream sizing.
    if stop_loss_val is None or stop_loss_val <= 0:
        stop_loss_val = price * (0.995 if side == "buy" else 1.005)
    if take_profit_val is None or take_profit_val <= 0:
        take_profit_val = price * (1.01 if side == "buy" else 0.99)

    metadata = {
        "source": "risk-svc",
        "external_risk_approved": True,
        "risk_reason": raw.get("risk_reason", "APPROVED"),
        "phase": raw.get("phase"),
        "temperature": raw.get("temperature"),
        "entropy": raw.get("entropy"),
    }

    return Signal(
        strategy_id="risk_svc",
        symbol=str(symbol),
        side=side,
        entry_price=price,
        stop_loss=stop_loss_val,
        take_profit=take_profit_val,
        confidence=confidence,
        metadata=metadata,
    )


class MicroservicesBridge:
    """Read external Redis streams and publish normalized EventBus events."""

    def __init__(
        self,
        *,
        bus: EventBus,
        redis_url: str,
        symbols: list[str],
        consume_physics: bool = False,
        consume_brain: bool = False,
        consume_risk: bool = False,
        group_prefix: str = "hean-core",
    ) -> None:
        self._bus = bus
        self._redis_url = redis_url
        self._symbols = symbols
        self._consume_physics = consume_physics
        self._consume_brain = consume_brain
        self._consume_risk = consume_risk

        self._consumer_id = f"{group_prefix}-{uuid.uuid4().hex[:8]}"
        self._physics_group = f"{group_prefix}-physics"
        self._brain_group = f"{group_prefix}-brain"
        self._risk_group = f"{group_prefix}-risk"

        self._redis: aioredis.Redis | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False

    async def start(self) -> None:
        if self._running:
            return

        self._redis = aioredis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=False,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )
        await self._redis.ping()
        self._running = True

        if self._consume_physics:
            for symbol in self._symbols:
                stream = f"physics:{symbol}"
                await self._ensure_group(stream, self._physics_group)
                self._tasks.append(asyncio.create_task(self._physics_loop(symbol)))

        if self._consume_brain:
            await self._ensure_group("brain:analysis", self._brain_group)
            self._tasks.append(asyncio.create_task(self._brain_loop()))

        if self._consume_risk:
            await self._ensure_group("risk:approved", self._risk_group)
            self._tasks.append(asyncio.create_task(self._risk_loop()))

        logger.info(
            "MicroservicesBridge started: physics=%s brain=%s risk=%s consumer=%s",
            self._consume_physics,
            self._consume_brain,
            self._consume_risk,
            self._consumer_id,
        )

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

        if self._redis:
            await self._redis.close()
            self._redis = None
        logger.info("MicroservicesBridge stopped")

    async def _ensure_group(self, stream: str, group: str) -> None:
        if not self._redis:
            return
        try:
            await self._redis.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception:
            # Group usually exists already.
            pass

    @staticmethod
    def _decode_payload(msg_data: dict[Any, Any]) -> dict[str, Any] | None:
        data_bytes = msg_data.get(b"data") or msg_data.get("data")
        if not data_bytes:
            return None
        if isinstance(data_bytes, str):
            data_bytes = data_bytes.encode()
        try:
            return orjson.loads(data_bytes)
        except Exception:
            return None

    async def _physics_loop(self, symbol: str) -> None:
        if not self._redis:
            return
        stream_name = f"physics:{symbol}"
        while self._running:
            try:
                messages = await self._redis.xreadgroup(
                    self._physics_group,
                    self._consumer_id,
                    {stream_name: ">"},
                    count=20,
                    block=1000,
                )
                for stream, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        payload = self._decode_payload(msg_data)
                        if payload:
                            physics = normalize_physics_update(payload)
                            sym = physics.get("symbol") or symbol
                            await self._bus.publish(
                                Event(
                                    event_type=EventType.PHYSICS_UPDATE,
                                    data={"symbol": sym, "physics": physics},
                                )
                            )
                        await self._redis.xack(stream, self._physics_group, msg_id)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Physics bridge loop error (%s): %s", symbol, exc)
                await asyncio.sleep(1)

    async def _brain_loop(self) -> None:
        if not self._redis:
            return
        stream_name = "brain:analysis"
        while self._running:
            try:
                messages = await self._redis.xreadgroup(
                    self._brain_group,
                    self._consumer_id,
                    {stream_name: ">"},
                    count=20,
                    block=1000,
                )
                for stream, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        payload = self._decode_payload(msg_data)
                        if payload:
                            analysis = normalize_brain_analysis(payload)
                            await self._bus.publish(
                                Event(event_type=EventType.BRAIN_ANALYSIS, data=analysis)
                            )
                        await self._redis.xack(stream, self._brain_group, msg_id)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Brain bridge loop error: %s", exc)
                await asyncio.sleep(1)

    async def _risk_loop(self) -> None:
        if not self._redis:
            return
        stream_name = "risk:approved"
        while self._running:
            try:
                messages = await self._redis.xreadgroup(
                    self._risk_group,
                    self._consumer_id,
                    {stream_name: ">"},
                    count=20,
                    block=1000,
                )
                for stream, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        payload = self._decode_payload(msg_data)
                        if payload:
                            signal = risk_payload_to_signal(payload)
                            if signal is not None:
                                await self._bus.publish(
                                    Event(
                                        event_type=EventType.SIGNAL,
                                        data={"signal": signal},
                                    )
                                )
                        await self._redis.xack(stream, self._risk_group, msg_id)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Risk bridge loop error: %s", exc)
                await asyncio.sleep(1)
