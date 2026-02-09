"""
Risk Constitution - Конституция риска

Неизменяемые правила безопасности
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ViolationType(Enum):
    """Типы нарушений"""
    MAX_POSITION_SIZE = "max_position_size"
    MAX_LEVERAGE = "max_leverage"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_DRAWDOWN = "max_drawdown"
    MIN_LIQUIDITY = "min_liquidity"
    MAX_CORRELATION = "max_correlation"
    FORBIDDEN_ACTION = "forbidden_action"
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"


@dataclass
class ConstitutionViolation:
    """Нарушение конституции"""

    violation_type: ViolationType
    severity: str  # "warning", "error", "critical"
    message: str

    # Context
    rule_value: Any
    actual_value: Any
    strategy_id: str | None = None

    # Timestamp
    timestamp_ns: int = 0

    def __post_init__(self):
        if self.timestamp_ns == 0:
            self.timestamp_ns = time.time_ns()


class RiskConstitution:
    """
    Конституция риска

    Неизменяемые правила которые НИ ПРИ КАКИХ УСЛОВИЯХ не могут быть нарушены
    """

    def __init__(self, constitution_config: dict | None = None):
        """
        Args:
            constitution_config: Конфигурация правил
                If None, используются default безопасные правила
        """

        # Default safe rules
        self.rules = {
            # Position limits
            'max_position_size_usd': 10000,      # Max $10K per position
            'max_position_size_pct': 20.0,       # Max 20% of capital
            'max_leverage': 5.0,                 # Max 5x leverage

            # Loss limits
            'max_daily_loss_pct': 5.0,           # Max 5% loss per day
            'max_drawdown_pct': 25.0,            # Max 25% drawdown from peak
            'max_consecutive_losses': 5,         # Max 5 losses in a row

            # Liquidity requirements
            'min_orderbook_depth_usd': 50000,    # Min $50K orderbook depth
            'max_spread_bps': 50,                # Max 50 bps spread

            # Diversification
            'max_correlation': 0.8,              # Max 0.8 correlation between strategies
            'min_strategies': 1,                 # Min 1 strategy (can go to 0 in safe mode)
            'max_strategies': 50,                # Max 50 strategies

            # Emergency
            'enable_kill_switch': True,          # Kill switch enabled
            'enable_safe_mode': True,            # Safe mode enabled

            # Operational
            'max_orders_per_second': 10,         # Max 10 orders/sec (avoid spam)
            'max_api_requests_per_minute': 100,  # Max 100 API calls/min
        }

        # Override with custom config
        if constitution_config:
            self.rules.update(constitution_config)

        # Immutability flag (once set to True, rules cannot change)
        self.immutable = False

        # Violation history
        self.violations: list[ConstitutionViolation] = []

        # Statistics
        self.checks_performed = 0
        self.violations_detected = 0

    def make_immutable(self):
        """
        Делает конституцию immutable

        После этого правила не могут быть изменены
        """
        self.immutable = True

    def check_position_size(
        self,
        position_size_usd: float,
        total_capital: float,
        strategy_id: str | None = None
    ) -> ConstitutionViolation | None:
        """Проверка размера позиции"""

        self.checks_performed += 1

        # Check absolute limit
        if position_size_usd > self.rules['max_position_size_usd']:
            violation = ConstitutionViolation(
                violation_type=ViolationType.MAX_POSITION_SIZE,
                severity="critical",
                message=f"Position size ${position_size_usd:.2f} exceeds max ${self.rules['max_position_size_usd']:.2f}",
                rule_value=self.rules['max_position_size_usd'],
                actual_value=position_size_usd,
                strategy_id=strategy_id,
            )
            self.violations.append(violation)
            self.violations_detected += 1
            return violation

        # Check percentage limit
        position_pct = (position_size_usd / total_capital) * 100 if total_capital > 0 else 0

        if position_pct > self.rules['max_position_size_pct']:
            violation = ConstitutionViolation(
                violation_type=ViolationType.MAX_POSITION_SIZE,
                severity="critical",
                message=f"Position size {position_pct:.1f}% exceeds max {self.rules['max_position_size_pct']}%",
                rule_value=self.rules['max_position_size_pct'],
                actual_value=position_pct,
                strategy_id=strategy_id,
            )
            self.violations.append(violation)
            self.violations_detected += 1
            return violation

        return None

    def check_leverage(
        self,
        leverage: float,
        strategy_id: str | None = None
    ) -> ConstitutionViolation | None:
        """Проверка кредитного плеча"""

        self.checks_performed += 1

        if leverage > self.rules['max_leverage']:
            violation = ConstitutionViolation(
                violation_type=ViolationType.MAX_LEVERAGE,
                severity="critical",
                message=f"Leverage {leverage:.1f}x exceeds max {self.rules['max_leverage']}x",
                rule_value=self.rules['max_leverage'],
                actual_value=leverage,
                strategy_id=strategy_id,
            )
            self.violations.append(violation)
            self.violations_detected += 1
            return violation

        return None

    def check_daily_loss(
        self,
        daily_loss_pct: float,
        strategy_id: str | None = None
    ) -> ConstitutionViolation | None:
        """Проверка дневных потерь"""

        self.checks_performed += 1

        if abs(daily_loss_pct) > self.rules['max_daily_loss_pct']:
            violation = ConstitutionViolation(
                violation_type=ViolationType.MAX_DAILY_LOSS,
                severity="critical",
                message=f"Daily loss {daily_loss_pct:.1f}% exceeds max {self.rules['max_daily_loss_pct']}%",
                rule_value=self.rules['max_daily_loss_pct'],
                actual_value=abs(daily_loss_pct),
                strategy_id=strategy_id,
            )
            self.violations.append(violation)
            self.violations_detected += 1
            return violation

        return None

    def check_drawdown(
        self,
        drawdown_pct: float,
        strategy_id: str | None = None
    ) -> ConstitutionViolation | None:
        """Проверка просадки"""

        self.checks_performed += 1

        if abs(drawdown_pct) > self.rules['max_drawdown_pct']:
            violation = ConstitutionViolation(
                violation_type=ViolationType.MAX_DRAWDOWN,
                severity="critical",
                message=f"Drawdown {drawdown_pct:.1f}% exceeds max {self.rules['max_drawdown_pct']}%",
                rule_value=self.rules['max_drawdown_pct'],
                actual_value=abs(drawdown_pct),
                strategy_id=strategy_id,
            )
            self.violations.append(violation)
            self.violations_detected += 1
            return violation

        return None

    def check_liquidity(
        self,
        orderbook_depth_usd: float,
        spread_bps: float,
        symbol: str
    ) -> ConstitutionViolation | None:
        """Проверка ликвидности"""

        self.checks_performed += 1

        # Check orderbook depth
        if orderbook_depth_usd < self.rules['min_orderbook_depth_usd']:
            violation = ConstitutionViolation(
                violation_type=ViolationType.MIN_LIQUIDITY,
                severity="error",
                message=f"{symbol} orderbook depth ${orderbook_depth_usd:.0f} below min ${self.rules['min_orderbook_depth_usd']:.0f}",
                rule_value=self.rules['min_orderbook_depth_usd'],
                actual_value=orderbook_depth_usd,
            )
            self.violations.append(violation)
            self.violations_detected += 1
            return violation

        # Check spread
        if spread_bps > self.rules['max_spread_bps']:
            violation = ConstitutionViolation(
                violation_type=ViolationType.MIN_LIQUIDITY,
                severity="warning",
                message=f"{symbol} spread {spread_bps:.1f} bps exceeds max {self.rules['max_spread_bps']} bps",
                rule_value=self.rules['max_spread_bps'],
                actual_value=spread_bps,
            )
            self.violations.append(violation)
            self.violations_detected += 1
            return violation

        return None

    def get_violations(
        self,
        severity: str | None = None,
        last_n: int | None = None
    ) -> list[ConstitutionViolation]:
        """Возвращает нарушения"""

        violations = self.violations

        if severity:
            violations = [v for v in violations if v.severity == severity]

        if last_n:
            violations = violations[-last_n:]

        return violations

    def get_critical_violations(self, last_n: int = 10) -> list[ConstitutionViolation]:
        """Возвращает критические нарушения"""
        return self.get_violations(severity="critical", last_n=last_n)

    def has_critical_violations(self, within_seconds: int = 60) -> bool:
        """Проверяет есть ли критические нарушения за последние N секунд"""

        threshold_ns = time.time_ns() - (within_seconds * 1_000_000_000)

        for violation in self.violations:
            if violation.severity == "critical" and violation.timestamp_ns > threshold_ns:
                return True

        return False

    def get_statistics(self) -> dict:
        """Статистика"""

        violation_rate = (
            self.violations_detected / self.checks_performed
            if self.checks_performed > 0 else 0
        )

        # Count by type
        violations_by_type = {}
        for violation in self.violations:
            vtype = violation.violation_type.value
            violations_by_type[vtype] = violations_by_type.get(vtype, 0) + 1

        # Count by severity
        violations_by_severity = {}
        for violation in self.violations:
            sev = violation.severity
            violations_by_severity[sev] = violations_by_severity.get(sev, 0) + 1

        return {
            'immutable': self.immutable,
            'rules': self.rules,
            'checks_performed': self.checks_performed,
            'violations_detected': self.violations_detected,
            'violation_rate': violation_rate,
            'violations_by_type': violations_by_type,
            'violations_by_severity': violations_by_severity,
            'has_recent_critical_violations': self.has_critical_violations(within_seconds=300),
        }
