"""
Decision Replayer - Воспроизведение решений

Позволяет replay решений для анализа и debugging
"""

import time
from collections.abc import Callable

from .decision_types import Decision
from .ledger import DecisionLedger


class DecisionReplayer:
    """
    Воспроизведение решений

    Позволяет проиграть решения с той же скоростью или ускоренно
    """

    def __init__(self, ledger: DecisionLedger):
        self.ledger = ledger

        # Replay state
        self.is_playing = False
        self.current_index = 0

    def replay(
        self,
        decisions: list[Decision],
        speed_multiplier: float = 1.0,
        on_decision: Callable[[Decision], None] | None = None,
        real_time: bool = False,
    ):
        """
        Воспроизводит решения

        Args:
            decisions: Список решений для replay
            speed_multiplier: Множитель скорости (1.0 = real-time, 2.0 = 2x faster)
            on_decision: Callback для каждого решения
            real_time: Воспроизводить с учётом временных интервалов
        """

        self.is_playing = True
        self.current_index = 0

        for i, decision in enumerate(decisions):
            if not self.is_playing:
                break

            self.current_index = i

            # Call callback
            if on_decision:
                on_decision(decision)

            # Wait if real-time mode
            if real_time and i < len(decisions) - 1:
                next_decision = decisions[i + 1]
                time_diff_ns = next_decision.decided_at_ns - decision.decided_at_ns
                time_diff_seconds = time_diff_ns / 1_000_000_000

                # Apply speed multiplier
                wait_seconds = time_diff_seconds / speed_multiplier

                if wait_seconds > 0:
                    time.sleep(wait_seconds)

    def replay_time_range(
        self,
        start_time_ns: int,
        end_time_ns: int,
        speed_multiplier: float = 1.0,
        on_decision: Callable[[Decision], None] | None = None,
    ):
        """Воспроизводит решения за период времени"""

        decisions = self.ledger.get_decisions_by_time_range(start_time_ns, end_time_ns)

        self.replay(
            decisions,
            speed_multiplier=speed_multiplier,
            on_decision=on_decision,
            real_time=True
        )

    def replay_strategy(
        self,
        strategy_id: str,
        speed_multiplier: float = 1.0,
        on_decision: Callable[[Decision], None] | None = None,
    ):
        """Воспроизводит решения конкретной стратегии"""

        decisions = self.ledger.get_decisions(
            strategy_id=strategy_id,
            limit=10000
        )

        self.replay(
            decisions,
            speed_multiplier=speed_multiplier,
            on_decision=on_decision,
            real_time=True
        )

    def stop(self):
        """Останавливает replay"""
        self.is_playing = False

    def pause(self):
        """Ставит на паузу"""
        self.is_playing = False

    def resume(self):
        """Возобновляет"""
        self.is_playing = True
