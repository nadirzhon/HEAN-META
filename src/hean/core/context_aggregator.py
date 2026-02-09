# src/hean/core/context_aggregator.py
"""
ContextAggregator — центральный хаб интеграции HEAN.

Слушает обновления от ВСЕХ компонентов (Physics, Brain, Oracle, OFI, Causal),
обновляет UnifiedMarketContext для каждого символа,
публикует CONTEXT_READY и CONTEXT_UPDATE (backward compat).

Это клей, который соединяет все подсистемы.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from hean.core.bus import EventBus
from hean.core.market_context import (
    BrainSnapshot,
    CausalSnapshot,
    OrderFlowSnapshot,
    PhysicsSnapshot,
    PredictionSnapshot,
    UnifiedMarketContext,
)
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class ContextAggregator:
    """
    Центральный агрегатор контекста HEAN.

    Подписывается на ВСЕ источники данных,
    обновляет UnifiedMarketContext и публикует готовый контекст.

    Usage:
        agg = ContextAggregator(bus, ["BTCUSDT", "ETHUSDT", ...])
        await agg.start()

        # Стратегия читает:
        ctx = agg.get_context("BTCUSDT")
        if ctx.consensus_direction == "buy" and ctx.should_increase_size:
            ...
    """

    def __init__(self, bus: EventBus, symbols: list[str]) -> None:
        self._bus = bus
        self._symbols = symbols
        self._contexts: dict[str, UnifiedMarketContext] = {
            s: UnifiedMarketContext(symbol=s) for s in symbols
        }
        self._running = False
        self._publish_throttle_ms = 200
        self._last_publish: dict[str, datetime] = {}
        self._tick_count = 0

    def get_context(self, symbol: str) -> UnifiedMarketContext | None:
        """Получить текущий контекст для символа."""
        return self._contexts.get(symbol)

    def get_all_contexts(self) -> dict[str, UnifiedMarketContext]:
        """Получить все контексты."""
        return self._contexts.copy()

    async def start(self) -> None:
        """Подписаться на все источники данных."""
        self._running = True

        # Тики — основной источник
        self._bus.subscribe(EventType.TICK, self._on_tick)

        # Regime
        self._bus.subscribe(EventType.REGIME_UPDATE, self._on_regime)

        # Physics (приходит как CONTEXT_UPDATE с context_type=physics)
        self._bus.subscribe(EventType.CONTEXT_UPDATE, self._on_context_update)

        # Physics отдельный event (если переведём)
        self._bus.subscribe(EventType.PHYSICS_UPDATE, self._on_physics)

        # Brain
        self._bus.subscribe(EventType.BRAIN_ANALYSIS, self._on_brain)

        # Oracle/TCN
        self._bus.subscribe(EventType.ORACLE_PREDICTION, self._on_prediction)

        # OFI
        self._bus.subscribe(EventType.OFI_UPDATE, self._on_ofi)

        # Causal
        self._bus.subscribe(EventType.CAUSAL_SIGNAL, self._on_causal)

        logger.info(
            f"ContextAggregator started — {len(self._symbols)} symbols, "
            f"listening on 7 event types"
        )

    async def stop(self) -> None:
        """Отписаться от всех событий."""
        self._running = False
        for et, handler in [
            (EventType.TICK, self._on_tick),
            (EventType.REGIME_UPDATE, self._on_regime),
            (EventType.CONTEXT_UPDATE, self._on_context_update),
            (EventType.PHYSICS_UPDATE, self._on_physics),
            (EventType.BRAIN_ANALYSIS, self._on_brain),
            (EventType.ORACLE_PREDICTION, self._on_prediction),
            (EventType.OFI_UPDATE, self._on_ofi),
            (EventType.CAUSAL_SIGNAL, self._on_causal),
        ]:
            self._bus.unsubscribe(et, handler)
        logger.info("ContextAggregator stopped")


    # ─── Обработчики обновлений ────────────────────────────────

    async def _on_tick(self, event: Event) -> None:
        """Обновить цену из тика — самый частый event."""
        tick = event.data.get("tick")
        if tick is None:
            return
        ctx = self._contexts.get(tick.symbol)
        if not ctx:
            return

        ctx.price = tick.price
        if tick.bid:
            ctx.bid = tick.bid
        if tick.ask:
            ctx.ask = tick.ask
        ctx.last_tick_at = datetime.utcnow()

        # Рассчитать spread
        if ctx.bid > 0 and ctx.ask > 0 and ctx.price > 0:
            ctx.order_flow.spread_bps = (ctx.ask - ctx.bid) / ctx.price * 10000

        ctx.components_updated["tick"] = datetime.utcnow()
        self._tick_count += 1

        # Публиковать CONTEXT_READY с тротлингом (каждые 200мс)
        await self._maybe_publish(tick.symbol)

    async def _on_regime(self, event: Event) -> None:
        """Обновить режим рынка."""
        symbol = event.data.get("symbol")
        regime = event.data.get("regime")
        ctx = self._contexts.get(symbol) if symbol else None
        if not ctx or regime is None:
            return

        ctx.regime = regime
        ctx.regime_confidence = event.data.get("confidence", 0.0)
        ctx.components_updated["regime"] = datetime.utcnow()

    async def _on_context_update(self, event: Event) -> None:
        """Обработать CONTEXT_UPDATE — backward compat + physics."""
        data = event.data
        context_type = data.get("context_type")

        if context_type == "physics":
            # PhysicsEngine публикует сюда
            physics_data = data.get("physics", {})
            symbol = physics_data.get("symbol")
            ctx = self._contexts.get(symbol) if symbol else None
            if not ctx:
                return

            ctx.physics = PhysicsSnapshot(
                temperature=physics_data.get("temperature", 0.0),
                temperature_regime=physics_data.get("temperature_regime", "COLD"),
                entropy=physics_data.get("entropy", 0.0),
                entropy_state=physics_data.get("entropy_state", "COMPRESSED"),
                phase=physics_data.get("phase", "unknown"),
                phase_confidence=physics_data.get("phase_confidence", 0.0),
                szilard_profit=physics_data.get("szilard_profit", 0.0),
                should_trade=physics_data.get("should_trade", False),
                trade_reason=physics_data.get("trade_reason", ""),
                size_multiplier=physics_data.get("size_multiplier", 1.0),
            )
            ctx.components_updated["physics"] = datetime.utcnow()

    async def _on_physics(self, event: Event) -> None:
        """Обработать PHYSICS_UPDATE (отдельный event type)."""
        data = event.data
        symbol = data.get("symbol")
        ctx = self._contexts.get(symbol) if symbol else None
        if not ctx:
            return

        ctx.physics = PhysicsSnapshot(
            temperature=data.get("temperature", 0.0),
            temperature_regime=data.get("temperature_regime", "COLD"),
            entropy=data.get("entropy", 0.0),
            entropy_state=data.get("entropy_state", "COMPRESSED"),
            phase=data.get("phase", "unknown"),
            phase_confidence=data.get("phase_confidence", 0.0),
            szilard_profit=data.get("szilard_profit", 0.0),
            should_trade=data.get("should_trade", False),
            trade_reason=data.get("trade_reason", ""),
            size_multiplier=data.get("size_multiplier", 1.0),
        )
        ctx.components_updated["physics"] = datetime.utcnow()


    async def _on_brain(self, event: Event) -> None:
        """Обновить анализ Brain (Claude AI)."""
        data = event.data
        # Brain может публиковать для конкретного символа или для "market"
        symbol = data.get("symbol")
        if symbol and symbol in self._contexts:
            self._update_brain_for_symbol(symbol, data)
        else:
            # Общий market analysis — применить ко всем символам
            for sym in self._symbols:
                self._update_brain_for_symbol(sym, data)

    def _update_brain_for_symbol(self, symbol: str, data: dict) -> None:
        ctx = self._contexts.get(symbol)
        if not ctx:
            return
        ctx.brain = BrainSnapshot(
            sentiment=data.get("sentiment", data.get("overall_sentiment", "neutral")),
            confidence=data.get("confidence", 0.0),
            key_forces=data.get("key_forces", data.get("forces", [])),
            recommended_action=data.get("recommended_action", "hold"),
            reasoning=data.get("reasoning", data.get("summary", "")),
            updated_at=datetime.utcnow(),
        )
        ctx.components_updated["brain"] = datetime.utcnow()

    async def _on_prediction(self, event: Event) -> None:
        """Обновить предсказания Oracle/TCN."""
        data = event.data
        symbol = data.get("symbol")
        ctx = self._contexts.get(symbol) if symbol else None
        if not ctx:
            return

        ctx.prediction = PredictionSnapshot(
            tcn_direction=data.get("direction", "neutral"),
            tcn_confidence=data.get("confidence", 0.0),
            tcn_magnitude=data.get("magnitude", 0.0),
            fingerprint_signal=data.get("fingerprint_signal"),
            fingerprint_confidence=data.get("fingerprint_confidence", 0.0),
            price_prediction_5s=data.get("price_5s"),
        )
        ctx.components_updated["prediction"] = datetime.utcnow()

    async def _on_ofi(self, event: Event) -> None:
        """Обновить Order Flow данные."""
        data = event.data
        symbol = data.get("symbol")
        ctx = self._contexts.get(symbol) if symbol else None
        if not ctx:
            return

        ctx.order_flow.ofi_value = data.get("ofi_value", 0.0)
        ctx.order_flow.ofi_trend = data.get("ofi_trend", "neutral")
        ctx.order_flow.aggression_buy = data.get("aggression_buy", 0.0)
        ctx.order_flow.aggression_sell = data.get("aggression_sell", 0.0)
        ctx.order_flow.book_imbalance = data.get("book_imbalance", 0.0)
        ctx.components_updated["ofi"] = datetime.utcnow()

    async def _on_causal(self, event: Event) -> None:
        """Обновить каузальные сигналы."""
        data = event.data
        symbol = data.get("target_symbol") or data.get("symbol")
        ctx = self._contexts.get(symbol) if symbol else None
        if not ctx:
            return

        ctx.causal = CausalSnapshot(
            pre_echo_detected=data.get("pre_echo_detected", True),
            pre_echo_direction=data.get("direction", "neutral"),
            pre_echo_confidence=data.get("confidence", 0.0),
            source_symbol=data.get("source_symbol", ""),
            lag_ms=data.get("lag_ms", 0),
        )
        ctx.components_updated["causal"] = datetime.utcnow()


    # ─── Публикация ────────────────────────────────────────────

    async def _maybe_publish(self, symbol: str) -> None:
        """Опубликовать CONTEXT_READY с тротлингом."""
        now = datetime.utcnow()
        last = self._last_publish.get(symbol)

        if last and (now - last).total_seconds() * 1000 < self._publish_throttle_ms:
            return

        self._last_publish[symbol] = now
        ctx = self._contexts[symbol]

        # CONTEXT_READY — новый event для стратегий
        await self._bus.publish(Event(
            event_type=EventType.CONTEXT_READY,
            data={"context": ctx, "symbol": symbol},
        ))

    # ─── Диагностика ──────────────────────────────────────────

    def get_diagnostics(self) -> dict[str, Any]:
        """Диагностика: какие компоненты обновляются, какие молчат."""
        now = datetime.utcnow()
        result: dict[str, Any] = {
            "tick_count": self._tick_count,
            "symbols": {},
        }
        for symbol, ctx in self._contexts.items():
            components_status: dict[str, str] = {}
            for comp, ts in ctx.components_updated.items():
                age_sec = (now - ts).total_seconds()
                if age_sec < 5:
                    status = "LIVE"
                elif age_sec < 60:
                    status = f"STALE ({age_sec:.0f}s)"
                else:
                    status = f"DEAD ({age_sec:.0f}s)"
                components_status[comp] = status

            # Какие компоненты НИКОГДА не обновлялись
            expected = {"tick", "regime", "physics", "prediction", "ofi", "brain", "causal"}
            never_updated = expected - set(ctx.components_updated.keys())

            result["symbols"][symbol] = {
                "price": ctx.price,
                "regime": ctx.regime.value if hasattr(ctx.regime, "value") else str(ctx.regime),
                "signal_strength": round(ctx.overall_signal_strength, 4),
                "consensus": ctx.consensus_direction,
                "consensus_count": ctx.consensus_count,
                "size_multiplier": round(ctx.size_multiplier, 3),
                "components": components_status,
                "never_updated": list(never_updated),
                "data_fresh": ctx.is_data_fresh,
            }
        return result
