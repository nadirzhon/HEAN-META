"""
Health Sensors - Сенсоры здоровья данных

Отслеживают качество данных, лаги, пропуски, аномалии
Это "иммунная система первого уровня" - раннее обнаружение проблем
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .event_envelope import EventEnvelope, EventType


class HealthLevel(Enum):
    """Уровни здоровья данных"""
    EXCELLENT = "excellent"  # > 95%
    GOOD = "good"            # 80-95%
    DEGRADED = "degraded"    # 50-80%
    POOR = "poor"            # < 50%
    CRITICAL = "critical"    # < 20%


@dataclass
class HealthStatus:
    """Статус здоровья данных"""

    # Overall health
    overall_score: float  # 0.0 - 1.0
    health_level: HealthLevel

    # Individual metrics
    lag_ms: float
    lag_healthy: bool

    gaps_detected: int
    gaps_healthy: bool

    spread_bps: float
    spread_healthy: bool

    message_rate_hz: float
    message_rate_healthy: bool

    quality_score: float
    quality_healthy: bool

    # Warnings
    warnings: list[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        """Проверка - здоровы ли данные"""
        return self.health_level in [HealthLevel.EXCELLENT, HealthLevel.GOOD]

    def is_degraded(self) -> bool:
        """Проверка - деградированы ли данные"""
        return self.health_level == HealthLevel.DEGRADED

    def is_critical(self) -> bool:
        """Проверка - критическое ли состояние"""
        return self.health_level in [HealthLevel.POOR, HealthLevel.CRITICAL]


class HealthSensor:
    """
    Базовый сенсор здоровья

    Каждый сенсор отслеживает один аспект здоровья данных
    """

    def __init__(self, name: str, window_size: int = 100):
        self.name = name
        self.window_size = window_size
        self.measurements = deque(maxlen=window_size)

    def record(self, value: float):
        """Записывает измерение"""
        self.measurements.append({
            'value': value,
            'timestamp': time.time()
        })

    def get_recent_values(self, n: int = 10) -> list[float]:
        """Возвращает последние N значений"""
        return [m['value'] for m in list(self.measurements)[-n:]]

    def get_average(self, n: int | None = None) -> float:
        """Возвращает среднее значение"""
        values = self.get_recent_values(n or self.window_size)
        return sum(values) / len(values) if values else 0.0

    def get_max(self, n: int | None = None) -> float:
        """Возвращает максимальное значение"""
        values = self.get_recent_values(n or self.window_size)
        return max(values) if values else 0.0

    def get_min(self, n: int | None = None) -> float:
        """Возвращает минимальное значение"""
        values = self.get_recent_values(n or self.window_size)
        return min(values) if values else 0.0


class LagSensor(HealthSensor):
    """Сенсор задержки обработки данных"""

    def __init__(self):
        super().__init__("lag", window_size=100)
        self.threshold_ms = 100  # Warning threshold

    def check(self, event: EventEnvelope) -> bool:
        """Проверяет lag события"""
        lag_ms = event.processing_lag_ms
        self.record(lag_ms)

        return lag_ms < self.threshold_ms

    def is_healthy(self) -> bool:
        """Проверка - здоровый ли lag"""
        avg_lag = self.get_average(n=20)  # Last 20 events
        max_lag = self.get_max(n=20)

        return avg_lag < self.threshold_ms and max_lag < self.threshold_ms * 2


class GapSensor(HealthSensor):
    """Сенсор пропусков данных"""

    def __init__(self):
        super().__init__("gaps", window_size=1000)
        self.last_timestamps: dict[str, int] = {}
        self.expected_interval_ms = {
            EventType.TRADE: 1000,      # Expect trade every 1s
            EventType.ORDERBOOK: 100,   # Expect orderbook update every 100ms
            EventType.CANDLE: 60000,    # Expect candle every 1m
        }
        self.gaps_detected = 0

    def check(self, event: EventEnvelope) -> bool:
        """Проверяет пропуски между событиями"""
        key = f"{event.event_type}_{event.symbol}"

        if key in self.last_timestamps:
            last_ts = self.last_timestamps[key]
            interval_ms = (event.timestamp_ns - last_ts) / 1_000_000

            expected = self.expected_interval_ms.get(event.event_type, 1000)

            # Gap detected если interval > 2x expected
            if interval_ms > expected * 2:
                self.gaps_detected += 1
                self.record(interval_ms)
                self.last_timestamps[key] = event.timestamp_ns
                return False

        self.last_timestamps[key] = event.timestamp_ns
        self.record(0)  # No gap
        return True

    def is_healthy(self) -> bool:
        """Проверка - много ли пропусков"""
        recent_gaps = sum(1 for v in self.get_recent_values(100) if v > 0)
        return recent_gaps < 5  # Less than 5 gaps in last 100 events


class SpreadSensor(HealthSensor):
    """Сенсор спреда - индикатор ликвидности"""

    def __init__(self):
        super().__init__("spread", window_size=100)
        self.threshold_bps = 10  # 0.1% - normal spread
        self.thin_liquidity_bps = 50  # 0.5% - thin liquidity

    def check(self, event: EventEnvelope) -> bool:
        """Проверяет спред"""
        if event.event_type != EventType.ORDERBOOK:
            return True

        spread_bps = event.data.get('spread_bps', 0)
        self.record(spread_bps)

        return spread_bps < self.threshold_bps

    def is_healthy(self) -> bool:
        """Проверка - нормальный ли спред"""
        avg_spread = self.get_average(n=20)
        max_spread = self.get_max(n=20)

        return avg_spread < self.threshold_bps and max_spread < self.thin_liquidity_bps

    def is_thin_liquidity(self) -> bool:
        """Проверка - тонкая ли ликвидность"""
        avg_spread = self.get_average(n=20)
        return avg_spread > self.thin_liquidity_bps


class MessageRateSensor(HealthSensor):
    """Сенсор частоты сообщений"""

    def __init__(self):
        super().__init__("message_rate", window_size=60)
        self.message_counts: dict[str, int] = {}
        self.last_check_time = time.time()
        self.min_rate_hz = {
            EventType.TRADE: 0.1,      # At least 1 trade per 10s
            EventType.ORDERBOOK: 1.0,  # At least 1 update per second
        }

    def record_message(self, event: EventEnvelope):
        """Записывает сообщение"""
        key = f"{event.event_type}_{event.symbol}"
        self.message_counts[key] = self.message_counts.get(key, 0) + 1

    def check_rates(self) -> dict[str, float]:
        """Проверяет частоту сообщений (вызывать раз в секунду)"""
        now = time.time()
        elapsed = now - self.last_check_time

        rates = {}
        for key, count in self.message_counts.items():
            rate_hz = count / elapsed if elapsed > 0 else 0
            rates[key] = rate_hz

        # Reset counters
        self.message_counts = {}
        self.last_check_time = now

        return rates

    def is_healthy(self, rates: dict[str, float]) -> bool:
        """Проверка - нормальная ли частота"""
        for key, rate in rates.items():
            event_type_str = key.split('_')[0]
            event_type = EventType(event_type_str)

            min_rate = self.min_rate_hz.get(event_type, 0.1)
            if rate < min_rate:
                return False

        return True


class QualitySensor(HealthSensor):
    """Сенсор общего качества данных"""

    def __init__(self):
        super().__init__("quality", window_size=100)

    def check(self, event: EventEnvelope) -> float:
        """Проверяет качество события"""
        quality = event.quality_score
        self.record(quality)
        return quality

    def is_healthy(self) -> bool:
        """Проверка - хорошее ли качество"""
        avg_quality = self.get_average(n=50)
        return avg_quality > 0.8  # > 80% quality


class HealthSensorArray:
    """
    Массив сенсоров здоровья

    Координирует работу всех сенсоров и выдаёт общий статус
    """

    def __init__(self):
        self.lag_sensor = LagSensor()
        self.gap_sensor = GapSensor()
        self.spread_sensor = SpreadSensor()
        self.message_rate_sensor = MessageRateSensor()
        self.quality_sensor = QualitySensor()

        # Statistics
        self.events_processed = 0
        self.start_time = time.time()

    def process_event(self, event: EventEnvelope):
        """Обрабатывает событие через все сенсоры"""
        self.events_processed += 1

        # Check lag
        self.lag_sensor.check(event)

        # Check gaps
        self.gap_sensor.check(event)

        # Check spread (for orderbook events)
        if event.event_type == EventType.ORDERBOOK:
            self.spread_sensor.check(event)

        # Check quality
        self.quality_sensor.check(event)

        # Record message
        self.message_rate_sensor.record_message(event)

    def get_health_status(self) -> HealthStatus:
        """Возвращает общий статус здоровья"""
        warnings = []

        # Check lag
        lag_healthy = self.lag_sensor.is_healthy()
        avg_lag = self.lag_sensor.get_average(n=20)

        if not lag_healthy:
            warnings.append(f"High latency: {avg_lag:.1f}ms")

        # Check gaps
        gaps_healthy = self.gap_sensor.is_healthy()
        gaps_count = self.gap_sensor.gaps_detected

        if not gaps_healthy:
            warnings.append(f"Data gaps detected: {gaps_count}")

        # Check spread
        spread_healthy = self.spread_sensor.is_healthy()
        avg_spread = self.spread_sensor.get_average(n=20)

        if not spread_healthy:
            warnings.append(f"Wide spread: {avg_spread:.1f} bps")

        if self.spread_sensor.is_thin_liquidity():
            warnings.append("Thin liquidity detected")

        # Check message rate
        rates = self.message_rate_sensor.check_rates()
        message_rate_healthy = self.message_rate_sensor.is_healthy(rates)
        avg_rate = sum(rates.values()) / len(rates) if rates else 0

        if not message_rate_healthy:
            warnings.append(f"Low message rate: {avg_rate:.1f} Hz")

        # Check quality
        quality_healthy = self.quality_sensor.is_healthy()
        avg_quality = self.quality_sensor.get_average(n=50)

        if not quality_healthy:
            warnings.append(f"Low data quality: {avg_quality:.1%}")

        # Calculate overall score
        scores = [
            1.0 if lag_healthy else 0.7,
            1.0 if gaps_healthy else 0.6,
            1.0 if spread_healthy else 0.8,
            1.0 if message_rate_healthy else 0.5,
            avg_quality,
        ]

        overall_score = sum(scores) / len(scores)

        # Determine health level
        if overall_score >= 0.95:
            health_level = HealthLevel.EXCELLENT
        elif overall_score >= 0.80:
            health_level = HealthLevel.GOOD
        elif overall_score >= 0.50:
            health_level = HealthLevel.DEGRADED
        elif overall_score >= 0.20:
            health_level = HealthLevel.POOR
        else:
            health_level = HealthLevel.CRITICAL

        return HealthStatus(
            overall_score=overall_score,
            health_level=health_level,
            lag_ms=avg_lag,
            lag_healthy=lag_healthy,
            gaps_detected=gaps_count,
            gaps_healthy=gaps_healthy,
            spread_bps=avg_spread,
            spread_healthy=spread_healthy,
            message_rate_hz=avg_rate,
            message_rate_healthy=message_rate_healthy,
            quality_score=avg_quality,
            quality_healthy=quality_healthy,
            warnings=warnings,
        )

    def get_statistics(self) -> dict:
        """Возвращает статистику обработки"""
        elapsed = time.time() - self.start_time
        rate = self.events_processed / elapsed if elapsed > 0 else 0

        return {
            'events_processed': self.events_processed,
            'elapsed_seconds': elapsed,
            'processing_rate_hz': rate,
            'uptime_hours': elapsed / 3600,
        }
