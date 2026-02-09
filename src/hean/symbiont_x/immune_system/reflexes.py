"""
Reflex System - Система рефлексов

Мгновенные автоматические реакции на опасности
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ReflexType(Enum):
    """Типы рефлексов"""
    SAFE_MODE = "safe_mode"                  # Безопасный режим (stop all trading)
    REDUCE_EXPOSURE = "reduce_exposure"      # Уменьшить exposure
    CLOSE_POSITIONS = "close_positions"      # Закрыть позиции
    PAUSE_STRATEGY = "pause_strategy"        # Приостановить стратегию
    SWITCH_TO_PASSIVE = "switch_to_passive"  # Только пассивные ордера
    INCREASE_SPREADS = "increase_spreads"    # Увеличить спреды
    REJECT_ORDERS = "reject_orders"          # Отклонять новые ордера


@dataclass
class ReflexTrigger:
    """Триггер для рефлекса"""

    trigger_name: str
    reflex_type: ReflexType
    condition: str  # Описание условия

    # Thresholds
    threshold_value: Any
    check_interval_ms: float = 100  # Как часто проверять

    # State
    triggered: bool = False
    trigger_count: int = 0
    last_trigger_ns: int = 0
    last_check_ns: int = 0

    # Callback
    callback: Callable | None = None


class ReflexSystem:
    """
    Система рефлексов

    Мгновенно реагирует на опасные условия без раздумий
    """

    def __init__(self) -> None:
        # Reflex triggers
        self.triggers: dict[str, ReflexTrigger] = {}

        # Active reflexes
        self.active_reflexes: dict[ReflexType, bool] = dict.fromkeys(ReflexType, False)

        # History
        self.reflex_history: list[dict[str, Any]] = []

        # Statistics
        self.total_reflexes_triggered = 0
        self.reflexes_prevented_losses = 0

    def register_trigger(self, trigger: ReflexTrigger) -> None:
        """Регистрирует новый триггер"""
        self.triggers[trigger.trigger_name] = trigger

    def register_default_triggers(self) -> None:
        """Регистрирует дефолтные триггеры"""

        # Trigger 1: Flash crash detection
        self.register_trigger(ReflexTrigger(
            trigger_name="flash_crash",
            reflex_type=ReflexType.SAFE_MODE,
            condition="Price drops >3% in <10 seconds",
            threshold_value=-3.0,
            check_interval_ms=100,
        ))

        # Trigger 2: Extreme volatility
        self.register_trigger(ReflexTrigger(
            trigger_name="extreme_volatility",
            reflex_type=ReflexType.REDUCE_EXPOSURE,
            condition="Volatility >3x normal",
            threshold_value=3.0,
            check_interval_ms=1000,
        ))

        # Trigger 3: Drawdown limit
        self.register_trigger(ReflexTrigger(
            trigger_name="drawdown_limit",
            reflex_type=ReflexType.CLOSE_POSITIONS,
            condition="Drawdown >20%",
            threshold_value=-20.0,
            check_interval_ms=500,
        ))

        # Trigger 4: Thin liquidity
        self.register_trigger(ReflexTrigger(
            trigger_name="thin_liquidity",
            reflex_type=ReflexType.SWITCH_TO_PASSIVE,
            condition="Orderbook depth <$10K",
            threshold_value=10000,
            check_interval_ms=500,
        ))

        # Trigger 5: Wide spreads
        self.register_trigger(ReflexTrigger(
            trigger_name="wide_spreads",
            reflex_type=ReflexType.REJECT_ORDERS,
            condition="Spread >100 bps",
            threshold_value=100,
            check_interval_ms=200,
        ))

        # Trigger 6: Consecutive losses
        self.register_trigger(ReflexTrigger(
            trigger_name="consecutive_losses",
            reflex_type=ReflexType.PAUSE_STRATEGY,
            condition="5+ consecutive losses",
            threshold_value=5,
            check_interval_ms=1000,
        ))

        # Trigger 7: API errors
        self.register_trigger(ReflexTrigger(
            trigger_name="api_errors",
            reflex_type=ReflexType.SAFE_MODE,
            condition="3+ API errors in 10 seconds",
            threshold_value=3,
            check_interval_ms=500,
        ))

    def check_trigger(
        self,
        trigger_name: str,
        current_value: Any
    ) -> bool:
        """
        Проверяет триггер

        Returns True если сработал
        """

        if trigger_name not in self.triggers:
            return False

        trigger = self.triggers[trigger_name]

        # Check if enough time passed since last check
        now_ns = time.time_ns()
        elapsed_ms = (now_ns - trigger.last_check_ns) / 1_000_000

        if elapsed_ms < trigger.check_interval_ms:
            return False  # Too soon

        trigger.last_check_ns = now_ns

        # Check condition
        triggered = self._evaluate_trigger(trigger, current_value)

        if triggered:
            self._activate_reflex(trigger)
            trigger.triggered = True
            trigger.trigger_count += 1
            trigger.last_trigger_ns = now_ns
            return True

        return False

    def _evaluate_trigger(self, trigger: ReflexTrigger, current_value: Any) -> bool:
        """Оценивает условие триггера"""

        threshold = trigger.threshold_value

        # Numeric comparisons
        if isinstance(threshold, (int, float)) and isinstance(current_value, (int, float)):
            # For negative thresholds (drawdowns, price drops)
            if threshold < 0:
                return current_value <= threshold
            # For positive thresholds
            else:
                return current_value >= threshold

        # Equality check
        return bool(current_value == threshold)

    def _activate_reflex(self, trigger: ReflexTrigger) -> None:
        """Активирует рефлекс"""

        reflex_type = trigger.reflex_type

        # Mark as active
        self.active_reflexes[reflex_type] = True

        # Execute callback if registered
        if trigger.callback:
            trigger.callback(trigger)

        # Record to history
        self.reflex_history.append({
            'timestamp_ns': time.time_ns(),
            'trigger_name': trigger.trigger_name,
            'reflex_type': reflex_type.value,
            'condition': trigger.condition,
        })

        self.total_reflexes_triggered += 1

    def activate_safe_mode(self) -> None:
        """Активирует безопасный режим (emergency stop)"""
        self.active_reflexes[ReflexType.SAFE_MODE] = True

        self.reflex_history.append({
            'timestamp_ns': time.time_ns(),
            'trigger_name': 'manual',
            'reflex_type': ReflexType.SAFE_MODE.value,
            'condition': 'Manual activation',
        })

    def deactivate_reflex(self, reflex_type: ReflexType) -> None:
        """Деактивирует рефлекс"""
        self.active_reflexes[reflex_type] = False

    def deactivate_all_reflexes(self) -> None:
        """Деактивирует все рефлексы"""
        for reflex_type in ReflexType:
            self.active_reflexes[reflex_type] = False

    def is_reflex_active(self, reflex_type: ReflexType) -> bool:
        """Проверяет активен ли рефлекс"""
        return self.active_reflexes.get(reflex_type, False)

    def is_safe_mode_active(self) -> bool:
        """Проверяет активен ли safe mode"""
        return self.is_reflex_active(ReflexType.SAFE_MODE)

    def can_trade(self) -> bool:
        """Проверяет можно ли торговать (не активны критические рефлексы)"""
        critical_reflexes = [
            ReflexType.SAFE_MODE,
            ReflexType.CLOSE_POSITIONS,
        ]

        for reflex in critical_reflexes:
            if self.is_reflex_active(reflex):
                return False

        return True

    def can_open_new_positions(self) -> bool:
        """Проверяет можно ли открывать новые позиции"""
        blocking_reflexes = [
            ReflexType.SAFE_MODE,
            ReflexType.CLOSE_POSITIONS,
            ReflexType.REJECT_ORDERS,
        ]

        for reflex in blocking_reflexes:
            if self.is_reflex_active(reflex):
                return False

        return True

    def should_use_passive_only(self) -> bool:
        """Проверяет нужно ли использовать только пассивные ордера"""
        return self.is_reflex_active(ReflexType.SWITCH_TO_PASSIVE)

    def get_active_reflexes(self) -> list[ReflexType]:
        """Возвращает список активных рефлексов"""
        return [
            reflex for reflex, active in self.active_reflexes.items()
            if active
        ]

    def get_reflex_history(self, last_n: int = 100) -> list[dict[str, Any]]:
        """Возвращает историю рефлексов"""
        return self.reflex_history[-last_n:]

    def get_statistics(self) -> dict[str, Any]:
        """Статистика рефлексов"""

        # Count by reflex type
        reflexes_by_type: dict[str, int] = {}
        for entry in self.reflex_history:
            rtype = entry['reflex_type']
            reflexes_by_type[rtype] = reflexes_by_type.get(rtype, 0) + 1

        # Count by trigger
        reflexes_by_trigger: dict[str, int] = {}
        for entry in self.reflex_history:
            trigger = entry['trigger_name']
            reflexes_by_trigger[trigger] = reflexes_by_trigger.get(trigger, 0) + 1

        return {
            'total_reflexes_triggered': self.total_reflexes_triggered,
            'reflexes_prevented_losses': self.reflexes_prevented_losses,
            'active_reflexes': [r.value for r in self.get_active_reflexes()],
            'safe_mode_active': self.is_safe_mode_active(),
            'can_trade': self.can_trade(),
            'can_open_new_positions': self.can_open_new_positions(),
            'reflexes_by_type': reflexes_by_type,
            'reflexes_by_trigger': reflexes_by_trigger,
            'total_triggers_registered': len(self.triggers),
        }
