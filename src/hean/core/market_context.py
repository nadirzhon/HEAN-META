# src/hean/core/market_context.py
"""
Unified MarketContext — центральное хранилище состояния рынка.

Расширяет базовый MarketContext из context.py, добавляя данные от ВСЕХ
подсистем HEAN: Physics, Brain, Oracle, OFI, Causal Inference.

Каждый компонент обновляет свой snapshot, стратегии читают полный контекст.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from hean.core.regime import Regime
from hean.logging import get_logger

logger = get_logger(__name__)


# ─── Snapshots от каждой подсистемы ───────────────────────────


@dataclass
class PhysicsSnapshot:
    """Снимок физических метрик рынка (PhysicsEngine)."""
    temperature: float = 0.0
    temperature_regime: str = "COLD"          # COLD / WARM / HOT / CRITICAL
    entropy: float = 0.0
    entropy_state: str = "COMPRESSED"         # COMPRESSED / EXPANDING / PEAK
    phase: str = "unknown"                    # accumulation / markup / distribution / markdown
    phase_confidence: float = 0.0
    szilard_profit: float = 0.0
    should_trade: bool = False
    trade_reason: str = ""
    size_multiplier: float = 1.0



@dataclass
class PredictionSnapshot:
    """Снимок предсказаний Oracle/TCN."""
    tcn_direction: str = "neutral"            # bullish / bearish / neutral
    tcn_confidence: float = 0.0
    tcn_magnitude: float = 0.0
    fingerprint_signal: str | None = None
    fingerprint_confidence: float = 0.0
    price_prediction_5s: float | None = None


@dataclass
class OrderFlowSnapshot:
    """Снимок Order Flow Imbalance."""
    ofi_value: float = 0.0
    ofi_trend: str = "neutral"                # bullish / bearish / neutral
    aggression_buy: float = 0.0
    aggression_sell: float = 0.0
    spread_bps: float = 0.0
    book_imbalance: float = 0.0


@dataclass
class BrainSnapshot:
    """Снимок анализа Brain (Claude AI)."""
    sentiment: str = "neutral"                # bullish / bearish / neutral
    confidence: float = 0.0
    key_forces: list[str] = field(default_factory=list)
    recommended_action: str = "hold"          # buy / sell / hold / reduce
    reasoning: str = ""
    updated_at: datetime | None = None


@dataclass
class CausalSnapshot:
    """Снимок каузальных сигналов (Granger/Transfer Entropy)."""
    pre_echo_detected: bool = False
    pre_echo_direction: str = "neutral"
    pre_echo_confidence: float = 0.0
    source_symbol: str = ""
    lag_ms: int = 0



# ─── Unified MarketContext ─────────────────────────────────────


@dataclass
class UnifiedMarketContext:
    """
    Единый контекст рынка для одного символа.

    Обновляется ContextAggregator, читается всеми стратегиями.
    Это ЦЕНТРАЛЬНАЯ точка интеграции — все подсистемы HEAN
    пишут сюда, все стратегии читают отсюда.
    """
    symbol: str

    # Базовые данные (от тиков)
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume_1m: float = 0.0

    # Режим рынка (от RegimeDetector)
    regime: Regime = Regime.NORMAL
    regime_confidence: float = 0.0

    # Подсистемы
    physics: PhysicsSnapshot = field(default_factory=PhysicsSnapshot)
    prediction: PredictionSnapshot = field(default_factory=PredictionSnapshot)
    order_flow: OrderFlowSnapshot = field(default_factory=OrderFlowSnapshot)
    brain: BrainSnapshot = field(default_factory=BrainSnapshot)
    causal: CausalSnapshot = field(default_factory=CausalSnapshot)

    # Мета
    last_tick_at: datetime | None = None
    components_updated: dict[str, datetime] = field(default_factory=dict)

    # ─── Свойства для принятия решений ───────────────────────

    @property
    def is_data_fresh(self) -> bool:
        """Данные свежие (тик < 5 секунд назад)?"""
        if self.last_tick_at is None:
            return False
        return (datetime.utcnow() - self.last_tick_at).total_seconds() < 5

    @property
    def overall_signal_strength(self) -> float:
        """
        Объединённая сила сигнала от всех компонентов.
        -1.0 = максимально bearish, +1.0 = максимально bullish, 0.0 = neutral.

        Веса подобраны так:
        - Order Flow (0.25) — самый быстрый и реактивный сигнал
        - TCN Prediction (0.25) — ML предсказание направления
        - Brain/Claude (0.20) — стратегический анализ
        - Physics (0.15) — фазовый анализ рынка
        - Causal (0.15) — pre-echo от lead symbols
        """
        signals: list[float] = []
        weights: list[float] = []

        # Physics: phase direction
        phase_map = {
            "markup": +0.6, "accumulation": +0.3,
            "markdown": -0.6, "distribution": -0.3,
        }
        physics_signal = phase_map.get(self.physics.phase, 0.0)
        if self.physics.phase_confidence > 0.5:
            physics_signal *= self.physics.phase_confidence
        else:
            physics_signal *= 0.3  # Низкая уверенность — ослабляем
        signals.append(physics_signal)
        weights.append(0.15)

        # TCN prediction
        if self.prediction.tcn_confidence > 0.5:
            direction = +1.0 if self.prediction.tcn_direction == "bullish" else (
                -1.0 if self.prediction.tcn_direction == "bearish" else 0.0
            )
            signals.append(direction * self.prediction.tcn_confidence)
        else:
            signals.append(0.0)
        weights.append(0.25)

        # Order Flow — прямой сигнал
        net_aggression = self.order_flow.aggression_buy - self.order_flow.aggression_sell
        signals.append(max(-1.0, min(1.0, net_aggression)))
        weights.append(0.25)

        # Brain analysis
        if self.brain.confidence > 0.4:
            brain_dir = {"bullish": 1.0, "bearish": -1.0}.get(self.brain.sentiment, 0.0)
            signals.append(brain_dir * self.brain.confidence)
        else:
            signals.append(0.0)
        weights.append(0.20)

        # Causal pre-echo
        if self.causal.pre_echo_detected and self.causal.pre_echo_confidence > 0.6:
            causal_dir = +1.0 if self.causal.pre_echo_direction == "bullish" else -1.0
            signals.append(causal_dir * self.causal.pre_echo_confidence)
        else:
            signals.append(0.0)
        weights.append(0.15)

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        return sum(s * w for s, w in zip(signals, weights, strict=False)) / total_weight

    @property
    def consensus_direction(self) -> str:
        """Консенсус направления: buy / sell / neutral."""
        strength = self.overall_signal_strength
        if strength > 0.15:
            return "buy"
        elif strength < -0.15:
            return "sell"
        return "neutral"

    @property
    def consensus_count(self) -> int:
        """Сколько компонентов согласны с направлением."""
        direction = self.consensus_direction
        if direction == "neutral":
            return 0

        count = 0
        is_bullish = direction == "buy"

        if is_bullish and self.physics.phase in ("markup", "accumulation"):
            count += 1
        elif not is_bullish and self.physics.phase in ("markdown", "distribution"):
            count += 1

        if self.prediction.tcn_confidence > 0.5:
            if (is_bullish and self.prediction.tcn_direction == "bullish") or \
               (not is_bullish and self.prediction.tcn_direction == "bearish"):
                count += 1

        if self.brain.confidence > 0.4:
            if (is_bullish and self.brain.sentiment == "bullish") or \
               (not is_bullish and self.brain.sentiment == "bearish"):
                count += 1

        ofi_bullish = self.order_flow.aggression_buy > self.order_flow.aggression_sell
        if (is_bullish and ofi_bullish) or (not is_bullish and not ofi_bullish):
            count += 1

        if self.causal.pre_echo_detected and self.causal.pre_echo_confidence > 0.6:
            if (is_bullish and self.causal.pre_echo_direction == "bullish") or \
               (not is_bullish and self.causal.pre_echo_direction == "bearish"):
                count += 1

        return count

    @property
    def should_increase_size(self) -> bool:
        """Все компоненты согласны? Можно увеличить размер."""
        return (
            self.consensus_count >= 4
            and self.physics.should_trade
            and self.order_flow.spread_bps < 15
            and abs(self.overall_signal_strength) > 0.4
        )

    @property
    def should_reduce_size(self) -> bool:
        """Компоненты конфликтуют? Уменьшить размер."""
        if not self.physics.should_trade:
            return True
        # TCN и Brain в разных направлениях
        if self.prediction.tcn_confidence > 0.6 and self.brain.confidence > 0.5:
            if (self.prediction.tcn_direction == "bullish" and self.brain.sentiment == "bearish") or \
               (self.prediction.tcn_direction == "bearish" and self.brain.sentiment == "bullish"):
                return True
        if self.order_flow.spread_bps > 25:
            return True
        return False

    @property
    def size_multiplier(self) -> float:
        """Итоговый множитель размера позиции от контекста (0.3 - 2.0)."""
        mult = 1.0

        # Physics multiplier
        mult *= self.physics.size_multiplier

        # Consensus boost/penalty
        if self.should_increase_size:
            mult *= 1.5
        elif self.should_reduce_size:
            mult *= 0.5

        # Brain conflict penalty
        if self.brain.confidence > 0.6 and self.brain.recommended_action == "reduce":
            mult *= 0.7

        # Spread penalty
        if self.order_flow.spread_bps > 15:
            mult *= max(0.5, 1.0 - (self.order_flow.spread_bps - 15) / 50)

        return max(0.3, min(2.0, mult))

    def to_dict(self) -> dict[str, Any]:
        """Для API, телеметрии, логов."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "bid": self.bid,
            "ask": self.ask,
            "regime": self.regime.value if hasattr(self.regime, "value") else str(self.regime),
            "regime_confidence": self.regime_confidence,
            "physics": {
                "temperature": self.physics.temperature,
                "temperature_regime": self.physics.temperature_regime,
                "entropy": self.physics.entropy,
                "entropy_state": self.physics.entropy_state,
                "phase": self.physics.phase,
                "phase_confidence": self.physics.phase_confidence,
                "should_trade": self.physics.should_trade,
                "size_multiplier": self.physics.size_multiplier,
            },
            "prediction": {
                "direction": self.prediction.tcn_direction,
                "confidence": self.prediction.tcn_confidence,
                "magnitude": self.prediction.tcn_magnitude,
                "fingerprint": self.prediction.fingerprint_signal,
            },
            "order_flow": {
                "ofi": self.order_flow.ofi_value,
                "trend": self.order_flow.ofi_trend,
                "spread_bps": self.order_flow.spread_bps,
                "buy_aggression": self.order_flow.aggression_buy,
                "sell_aggression": self.order_flow.aggression_sell,
            },
            "brain": {
                "sentiment": self.brain.sentiment,
                "confidence": self.brain.confidence,
                "action": self.brain.recommended_action,
                "forces": self.brain.key_forces,
            },
            "causal": {
                "pre_echo": self.causal.pre_echo_detected,
                "direction": self.causal.pre_echo_direction,
                "confidence": self.causal.pre_echo_confidence,
                "source": self.causal.source_symbol,
            },
            "composite": {
                "signal_strength": self.overall_signal_strength,
                "consensus_direction": self.consensus_direction,
                "consensus_count": self.consensus_count,
                "size_multiplier": self.size_multiplier,
                "should_increase": self.should_increase_size,
                "should_reduce": self.should_reduce_size,
            },
            "meta": {
                "data_fresh": self.is_data_fresh,
                "last_tick": self.last_tick_at.isoformat() if self.last_tick_at else None,
                "components": {
                    k: v.isoformat() for k, v in self.components_updated.items()
                },
            },
        }
