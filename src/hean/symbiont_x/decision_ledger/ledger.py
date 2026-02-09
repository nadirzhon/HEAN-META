"""
Decision Ledger - Хранилище решений

Append-only log всех решений
"""

import json
import time
from collections import deque
from pathlib import Path

from .decision_types import Decision, DecisionOutcome, DecisionType


class DecisionLedger:
    """
    Ledger всех решений

    Append-only, immutable log
    """

    def __init__(
        self,
        storage_path: str | None = None,
        in_memory_limit: int = 10000,  # Keep last 10K in memory
        auto_persist: bool = True,
        persist_interval_seconds: int = 60,
    ):
        self.storage_path = Path(storage_path) if storage_path else None
        self.in_memory_limit = in_memory_limit
        self.auto_persist = auto_persist
        self.persist_interval_seconds = persist_interval_seconds

        # In-memory storage (ring buffer)
        self.decisions: deque = deque(maxlen=in_memory_limit)

        # Index by ID for fast lookup
        self.decision_index: dict[str, Decision] = {}

        # Index by strategy for fast filtering
        self.strategy_index: dict[str, list[str]] = {}

        # Statistics
        self.total_decisions = 0
        self.last_persist_ns = time.time_ns()

        # Load from disk if exists
        if self.storage_path and self.storage_path.exists():
            self._load_from_disk()

    def record_decision(self, decision: Decision):
        """
        Записывает решение в ledger

        Append-only
        """

        # Add to deque
        self.decisions.append(decision)

        # Index
        self.decision_index[decision.decision_id] = decision

        # Index by strategy
        if decision.strategy_id:
            if decision.strategy_id not in self.strategy_index:
                self.strategy_index[decision.strategy_id] = []
            self.strategy_index[decision.strategy_id].append(decision.decision_id)

        self.total_decisions += 1

        # Auto-persist if needed
        if self.auto_persist:
            self._check_auto_persist()

    def get_decision(self, decision_id: str) -> Decision | None:
        """Получает решение по ID"""
        return self.decision_index.get(decision_id)

    def get_decisions(
        self,
        decision_type: DecisionType | None = None,
        strategy_id: str | None = None,
        outcome: DecisionOutcome | None = None,
        symbol: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Decision]:
        """
        Получает решения с фильтрацией

        Returns последние N решений matching filters
        """

        # Start with all decisions (or by strategy)
        if strategy_id and strategy_id in self.strategy_index:
            decision_ids = self.strategy_index[strategy_id]
            decisions = [self.decision_index[did] for did in decision_ids if did in self.decision_index]
        else:
            decisions = list(self.decisions)

        # Apply filters
        if decision_type:
            decisions = [d for d in decisions if d.decision_type == decision_type]

        if outcome:
            decisions = [d for d in decisions if d.outcome == outcome]

        if symbol:
            decisions = [d for d in decisions if d.symbol == symbol]

        # Sort by timestamp (newest first)
        decisions = sorted(decisions, key=lambda d: d.decided_at_ns, reverse=True)

        # Pagination
        start = offset
        end = offset + limit

        return decisions[start:end]

    def get_recent_decisions(self, n: int = 100) -> list[Decision]:
        """Получает последние N решений"""
        decisions = list(self.decisions)
        return sorted(decisions, key=lambda d: d.decided_at_ns, reverse=True)[:n]

    def get_decisions_by_time_range(
        self,
        start_time_ns: int,
        end_time_ns: int
    ) -> list[Decision]:
        """Получает решения за период времени"""
        decisions = [
            d for d in self.decisions
            if start_time_ns <= d.decided_at_ns <= end_time_ns
        ]
        return sorted(decisions, key=lambda d: d.decided_at_ns)

    def get_related_decisions(self, decision_id: str) -> list[Decision]:
        """Получает связанные решения"""
        decision = self.get_decision(decision_id)
        if not decision:
            return []

        related = []

        # Get parent
        if decision.parent_decision_id:
            parent = self.get_decision(decision.parent_decision_id)
            if parent:
                related.append(parent)

        # Get related
        for related_id in decision.related_decisions:
            related_decision = self.get_decision(related_id)
            if related_decision:
                related.append(related_decision)

        return related

    def get_decision_chain(self, decision_id: str) -> list[Decision]:
        """Получает всю цепочку решений (parent → child)"""
        chain = []

        decision = self.get_decision(decision_id)
        if not decision:
            return chain

        # Walk up to root
        current = decision
        while current:
            chain.insert(0, current)  # Add to beginning
            if current.parent_decision_id:
                current = self.get_decision(current.parent_decision_id)
            else:
                break

        return chain

    def persist_to_disk(self):
        """Сохраняет ledger на диск"""
        if not self.storage_path:
            return

        # Create directory if needed
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file (append mode)
        with open(self.storage_path, 'a') as f:
            for decision in self.decisions:
                # Check if already persisted (simple check)
                # In production, would use proper persistence tracking
                decision_json = json.dumps(decision.to_dict())
                f.write(decision_json + '\n')

        self.last_persist_ns = time.time_ns()

    def _load_from_disk(self):
        """Загружает ledger с диска"""
        if not self.storage_path or not self.storage_path.exists():
            return

        with open(self.storage_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    decision = Decision.from_dict(data)

                    # Add to memory (respecting limit)
                    self.decisions.append(decision)
                    self.decision_index[decision.decision_id] = decision

                    if decision.strategy_id:
                        if decision.strategy_id not in self.strategy_index:
                            self.strategy_index[decision.strategy_id] = []
                        self.strategy_index[decision.strategy_id].append(decision.decision_id)

                    self.total_decisions += 1

    def _check_auto_persist(self):
        """Проверяет нужна ли auto-persist"""
        elapsed_seconds = (time.time_ns() - self.last_persist_ns) / 1_000_000_000

        if elapsed_seconds >= self.persist_interval_seconds:
            self.persist_to_disk()

    def get_statistics(self) -> dict:
        """Статистика ledger"""

        if not self.decisions:
            return {
                'total_decisions': self.total_decisions,
                'in_memory_decisions': 0,
            }

        decisions = list(self.decisions)

        # Count by type
        by_type = {}
        for decision in decisions:
            dtype = decision.decision_type.value
            by_type[dtype] = by_type.get(dtype, 0) + 1

        # Count by outcome
        by_outcome = {}
        for decision in decisions:
            outcome = decision.outcome.value
            by_outcome[outcome] = by_outcome.get(outcome, 0) + 1

        # Success rate
        total_completed = sum(
            count for outcome, count in by_outcome.items()
            if outcome != 'pending'
        )
        successful = by_outcome.get('success', 0)
        success_rate = (successful / total_completed) if total_completed > 0 else 0

        # Average latency
        completed_decisions = [d for d in decisions if d.is_completed()]
        if completed_decisions:
            avg_latency = sum(d.get_latency_ms() for d in completed_decisions) / len(completed_decisions)
            avg_duration = sum(d.get_total_duration_ms() for d in completed_decisions) / len(completed_decisions)
        else:
            avg_latency = 0
            avg_duration = 0

        # PnL impact
        total_pnl = sum(d.pnl_impact for d in decisions)

        return {
            'total_decisions': self.total_decisions,
            'in_memory_decisions': len(self.decisions),
            'decisions_by_type': by_type,
            'decisions_by_outcome': by_outcome,
            'success_rate': success_rate,
            'avg_latency_ms': avg_latency,
            'avg_duration_ms': avg_duration,
            'total_pnl_impact': total_pnl,
            'total_strategies': len(self.strategy_index),
        }
