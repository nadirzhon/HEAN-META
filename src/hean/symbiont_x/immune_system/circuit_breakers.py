"""
Circuit Breakers - Автоматические выключатели

Последняя линия защиты - аварийное отключение
"""

import time
from dataclasses import dataclass
from enum import Enum


class CircuitBreakerState(Enum):
    """Состояния circuit breaker"""
    CLOSED = "closed"      # Нормальная работа
    OPEN = "open"          # Аварийный стоп
    HALF_OPEN = "half_open"  # Тестовое восстановление


@dataclass
class CircuitBreakerConfig:
    """Конфигурация circuit breaker"""

    name: str

    # Thresholds
    failure_threshold: int  # Сколько failures до срабатывания
    failure_window_seconds: int  # За какое время

    # Recovery
    recovery_timeout_seconds: int  # Сколько ждать до half-open
    success_threshold: int  # Сколько успехов для полного восстановления

    # Current state
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_ns: int = 0
    opened_at_ns: int = 0
    closed_at_ns: int = 0

    def reset(self) -> None:
        """Сброс состояния"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_ns = 0
        self.opened_at_ns = 0


class CircuitBreakerSystem:
    """
    Система circuit breakers

    Автоматически отключает торговлю при критических сбоях
    """

    def __init__(self) -> None:
        # Circuit breakers
        self.breakers: dict[str, CircuitBreakerConfig] = {}

        # Global kill switch
        self.kill_switch_active = False
        self.kill_switch_activated_at_ns: int | None = None

        # Statistics
        self.total_trips = 0  # Сколько раз сработали breakers
        self.total_recoveries = 0

    def register_breaker(self, config: CircuitBreakerConfig) -> None:
        """Регистрирует circuit breaker"""
        self.breakers[config.name] = config

    def register_default_breakers(self) -> None:
        """Регистрирует дефолтные circuit breakers"""

        # Breaker 1: API failures
        self.register_breaker(CircuitBreakerConfig(
            name="api_failures",
            failure_threshold=5,           # 5 failures
            failure_window_seconds=60,     # in 60 seconds
            recovery_timeout_seconds=300,  # Wait 5 minutes
            success_threshold=3,           # 3 successes to recover
        ))

        # Breaker 2: Order rejections
        self.register_breaker(CircuitBreakerConfig(
            name="order_rejections",
            failure_threshold=10,          # 10 rejections
            failure_window_seconds=60,
            recovery_timeout_seconds=180,
            success_threshold=5,
        ))

        # Breaker 3: Execution errors
        self.register_breaker(CircuitBreakerConfig(
            name="execution_errors",
            failure_threshold=3,           # 3 errors
            failure_window_seconds=30,
            recovery_timeout_seconds=600,  # Wait 10 minutes
            success_threshold=5,
        ))

        # Breaker 4: Data quality
        self.register_breaker(CircuitBreakerConfig(
            name="data_quality",
            failure_threshold=5,           # 5 quality issues
            failure_window_seconds=120,
            recovery_timeout_seconds=300,
            success_threshold=10,
        ))

        # Breaker 5: Risk violations
        self.register_breaker(CircuitBreakerConfig(
            name="risk_violations",
            failure_threshold=3,           # 3 violations
            failure_window_seconds=300,
            recovery_timeout_seconds=1800,  # Wait 30 minutes
            success_threshold=5,
        ))

        # Breaker 6: Catastrophic loss
        self.register_breaker(CircuitBreakerConfig(
            name="catastrophic_loss",
            failure_threshold=1,           # 1 occurrence
            failure_window_seconds=86400,  # Per day
            recovery_timeout_seconds=3600,  # Wait 1 hour
            success_threshold=10,
        ))

    def record_failure(self, breaker_name: str) -> None:
        """Записывает failure для breaker"""

        if breaker_name not in self.breakers:
            return

        breaker = self.breakers[breaker_name]
        now_ns = time.time_ns()

        # If breaker is OPEN, ignore
        if breaker.state == CircuitBreakerState.OPEN:
            return

        # Check if within failure window
        window_ns = breaker.failure_window_seconds * 1_000_000_000
        time_since_last_failure = now_ns - breaker.last_failure_ns

        # Reset count if outside window
        if time_since_last_failure > window_ns:
            breaker.failure_count = 0

        # Increment failure count
        breaker.failure_count += 1
        breaker.last_failure_ns = now_ns

        # Check if threshold exceeded
        if breaker.failure_count >= breaker.failure_threshold:
            self._trip_breaker(breaker)

    def record_success(self, breaker_name: str) -> None:
        """Записывает success для breaker"""

        if breaker_name not in self.breakers:
            return

        breaker = self.breakers[breaker_name]

        # Only count successes in HALF_OPEN state
        if breaker.state != CircuitBreakerState.HALF_OPEN:
            return

        breaker.success_count += 1

        # Check if enough successes to close
        if breaker.success_count >= breaker.success_threshold:
            self._close_breaker(breaker)

    def _trip_breaker(self, breaker: CircuitBreakerConfig) -> None:
        """Срабатывает breaker (CLOSED -> OPEN)"""

        breaker.state = CircuitBreakerState.OPEN
        breaker.opened_at_ns = time.time_ns()
        breaker.success_count = 0

        self.total_trips += 1

    def _close_breaker(self, breaker: CircuitBreakerConfig) -> None:
        """Закрывает breaker (HALF_OPEN -> CLOSED)"""

        breaker.state = CircuitBreakerState.CLOSED
        breaker.closed_at_ns = time.time_ns()
        breaker.failure_count = 0
        breaker.success_count = 0

        self.total_recoveries += 1

    def check_recovery(self, breaker_name: str) -> bool:
        """
        Проверяет можно ли перейти в HALF_OPEN

        Returns True если перешли
        """

        if breaker_name not in self.breakers:
            return False

        breaker = self.breakers[breaker_name]

        # Only from OPEN state
        if breaker.state != CircuitBreakerState.OPEN:
            return False

        # Check if enough time passed
        now_ns = time.time_ns()
        elapsed_seconds = (now_ns - breaker.opened_at_ns) / 1_000_000_000

        if elapsed_seconds >= breaker.recovery_timeout_seconds:
            # Try HALF_OPEN
            breaker.state = CircuitBreakerState.HALF_OPEN
            breaker.success_count = 0
            return True

        return False

    def check_all_recoveries(self) -> None:
        """Проверяет восстановление для всех breakers"""
        for breaker_name in self.breakers.keys():
            self.check_recovery(breaker_name)

    def is_breaker_open(self, breaker_name: str) -> bool:
        """Проверяет открыт ли breaker"""
        if breaker_name not in self.breakers:
            return False

        return self.breakers[breaker_name].state == CircuitBreakerState.OPEN

    def any_breaker_open(self) -> bool:
        """Проверяет есть ли открытые breakers"""
        return any(
            breaker.state == CircuitBreakerState.OPEN
            for breaker in self.breakers.values()
        )

    def get_open_breakers(self) -> list[str]:
        """Возвращает список открытых breakers"""
        return [
            name for name, breaker in self.breakers.items()
            if breaker.state == CircuitBreakerState.OPEN
        ]

    def activate_kill_switch(self) -> None:
        """Активирует глобальный kill switch"""
        self.kill_switch_active = True
        self.kill_switch_activated_at_ns = time.time_ns()

        # Trip all breakers
        for breaker in self.breakers.values():
            if breaker.state != CircuitBreakerState.OPEN:
                self._trip_breaker(breaker)

    def deactivate_kill_switch(self) -> None:
        """Деактивирует kill switch (requires manual intervention)"""
        self.kill_switch_active = False

        # Reset all breakers
        for breaker in self.breakers.values():
            breaker.reset()

    def can_trade(self) -> bool:
        """Проверяет можно ли торговать (ни один breaker не сработал)"""
        if self.kill_switch_active:
            return False

        return not self.any_breaker_open()

    def can_execute_order(self) -> bool:
        """Проверяет можно ли исполнять ордера"""
        if self.kill_switch_active:
            return False

        # Check critical breakers
        critical_breakers = ['execution_errors', 'api_failures', 'risk_violations']

        for breaker_name in critical_breakers:
            if breaker_name in self.breakers:
                if self.breakers[breaker_name].state == CircuitBreakerState.OPEN:
                    return False

        return True

    def get_statistics(self) -> dict:
        """Статистика circuit breakers"""

        # Count by state
        states_count = {
            'closed': 0,
            'open': 0,
            'half_open': 0,
        }

        for breaker in self.breakers.values():
            states_count[breaker.state.value] += 1

        # Breaker details
        breaker_details = {}
        for name, breaker in self.breakers.items():
            breaker_details[name] = {
                'state': breaker.state.value,
                'failure_count': breaker.failure_count,
                'success_count': breaker.success_count,
            }

        return {
            'kill_switch_active': self.kill_switch_active,
            'total_breakers': len(self.breakers),
            'states_count': states_count,
            'total_trips': self.total_trips,
            'total_recoveries': self.total_recoveries,
            'open_breakers': self.get_open_breakers(),
            'can_trade': self.can_trade(),
            'can_execute_order': self.can_execute_order(),
            'breaker_details': breaker_details,
        }

    def get_status_summary(self) -> str:
        """Возвращает краткий статус"""
        if self.kill_switch_active:
            return "🔴 KILL SWITCH ACTIVE"

        open_breakers = self.get_open_breakers()
        if open_breakers:
            return f"🟡 BREAKERS OPEN: {', '.join(open_breakers)}"

        half_open = [
            name for name, breaker in self.breakers.items()
            if breaker.state == CircuitBreakerState.HALF_OPEN
        ]
        if half_open:
            return f"🟠 RECOVERING: {', '.join(half_open)}"

        return "🟢 ALL SYSTEMS OPERATIONAL"
