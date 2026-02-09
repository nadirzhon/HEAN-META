"""
Decision Types - Типы решений

Определения всех типов решений в системе
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DecisionType(Enum):
    """Типы решений"""

    # Trading decisions
    OPEN_POSITION = "open_position"
    CLOSE_POSITION = "close_position"
    MODIFY_POSITION = "modify_position"
    CANCEL_ORDER = "cancel_order"

    # Strategy decisions
    ACTIVATE_STRATEGY = "activate_strategy"
    PAUSE_STRATEGY = "pause_strategy"
    KILL_STRATEGY = "kill_strategy"
    MUTATE_STRATEGY = "mutate_strategy"

    # Capital allocation
    ALLOCATE_CAPITAL = "allocate_capital"
    REBALANCE_PORTFOLIO = "rebalance_portfolio"

    # Risk management
    ACTIVATE_SAFE_MODE = "activate_safe_mode"
    TRIGGER_REFLEX = "trigger_reflex"
    TRIP_CIRCUIT_BREAKER = "trip_circuit_breaker"
    ACTIVATE_KILL_SWITCH = "activate_kill_switch"

    # Regime adaptation
    DETECT_REGIME_CHANGE = "detect_regime_change"
    ADAPT_TO_REGIME = "adapt_to_regime"


class DecisionOutcome(Enum):
    """Результаты решений"""
    PENDING = "pending"        # Ещё не исполнено
    SUCCESS = "success"        # Успешно
    FAILURE = "failure"        # Провал
    PARTIAL = "partial"        # Частичное исполнение
    CANCELLED = "cancelled"    # Отменено
    REJECTED = "rejected"      # Отклонено


@dataclass
class Decision:
    """
    Решение

    Единица записи в ledger
    """

    decision_id: str
    decision_type: DecisionType

    # Context
    strategy_id: str | None = None
    symbol: str | None = None

    # Decision parameters
    parameters: dict[str, Any] = field(default_factory=dict)

    # Reasoning
    reason: str = ""
    market_regime: str | None = None
    confidence: float = 0.0

    # Execution
    outcome: DecisionOutcome = DecisionOutcome.PENDING
    execution_time_ms: float = 0.0

    # Results
    expected_result: Any | None = None
    actual_result: Any | None = None

    # Financial impact
    pnl_impact: float = 0.0
    risk_impact: float = 0.0

    # Timestamps
    decided_at_ns: int = 0
    executed_at_ns: int = 0
    completed_at_ns: int = 0

    # Metadata
    parent_decision_id: str | None = None  # For chained decisions
    related_decisions: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.decided_at_ns == 0:
            self.decided_at_ns = time.time_ns()

    def mark_executed(self, execution_time_ms: float):
        """Отмечает как исполненное"""
        self.executed_at_ns = time.time_ns()
        self.execution_time_ms = execution_time_ms

    def mark_completed(self, outcome: DecisionOutcome, actual_result: Any = None):
        """Отмечает как завершённое"""
        self.completed_at_ns = time.time_ns()
        self.outcome = outcome
        self.actual_result = actual_result

    def is_completed(self) -> bool:
        """Проверяет завершено ли решение"""
        return self.outcome in [
            DecisionOutcome.SUCCESS,
            DecisionOutcome.FAILURE,
            DecisionOutcome.PARTIAL,
            DecisionOutcome.CANCELLED,
            DecisionOutcome.REJECTED
        ]

    def is_successful(self) -> bool:
        """Проверяет успешно ли решение"""
        return self.outcome == DecisionOutcome.SUCCESS

    def get_latency_ms(self) -> float:
        """Вычисляет latency от решения до исполнения"""
        if self.executed_at_ns == 0:
            return 0.0
        return (self.executed_at_ns - self.decided_at_ns) / 1_000_000

    def get_total_duration_ms(self) -> float:
        """Вычисляет полную длительность"""
        if self.completed_at_ns == 0:
            return 0.0
        return (self.completed_at_ns - self.decided_at_ns) / 1_000_000

    def to_dict(self) -> dict:
        """Сериализация"""
        return {
            'decision_id': self.decision_id,
            'decision_type': self.decision_type.value,
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'parameters': self.parameters,
            'reason': self.reason,
            'market_regime': self.market_regime,
            'confidence': self.confidence,
            'outcome': self.outcome.value,
            'execution_time_ms': self.execution_time_ms,
            'expected_result': self.expected_result,
            'actual_result': self.actual_result,
            'pnl_impact': self.pnl_impact,
            'risk_impact': self.risk_impact,
            'decided_at_ns': self.decided_at_ns,
            'executed_at_ns': self.executed_at_ns,
            'completed_at_ns': self.completed_at_ns,
            'parent_decision_id': self.parent_decision_id,
            'related_decisions': self.related_decisions,
            'tags': self.tags,
            'latency_ms': self.get_latency_ms(),
            'total_duration_ms': self.get_total_duration_ms(),
        }

    @staticmethod
    def from_dict(data: dict) -> 'Decision':
        """Десериализация"""
        return Decision(
            decision_id=data['decision_id'],
            decision_type=DecisionType(data['decision_type']),
            strategy_id=data.get('strategy_id'),
            symbol=data.get('symbol'),
            parameters=data.get('parameters', {}),
            reason=data.get('reason', ''),
            market_regime=data.get('market_regime'),
            confidence=data.get('confidence', 0.0),
            outcome=DecisionOutcome(data.get('outcome', 'pending')),
            execution_time_ms=data.get('execution_time_ms', 0.0),
            expected_result=data.get('expected_result'),
            actual_result=data.get('actual_result'),
            pnl_impact=data.get('pnl_impact', 0.0),
            risk_impact=data.get('risk_impact', 0.0),
            decided_at_ns=data.get('decided_at_ns', 0),
            executed_at_ns=data.get('executed_at_ns', 0),
            completed_at_ns=data.get('completed_at_ns', 0),
            parent_decision_id=data.get('parent_decision_id'),
            related_decisions=data.get('related_decisions', []),
            tags=data.get('tags', []),
        )
