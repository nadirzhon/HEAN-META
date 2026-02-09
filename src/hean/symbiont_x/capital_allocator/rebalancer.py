"""
Portfolio Rebalancer - Ребалансировщик портфеля

Автоматически ребалансирует портфель на основе performance и market conditions
"""

import time
from enum import Enum

from ..adversarial_twin.survival_score import SurvivalScore
from .allocator import CapitalAllocator
from .portfolio import Portfolio


class RebalanceReason(Enum):
    """Причины ребалансировки"""
    SCHEDULED = "scheduled"                    # Запланированная
    DRIFT_THRESHOLD = "drift_threshold"        # Превышен drift
    STRATEGY_FAILURE = "strategy_failure"      # Провал стратегии
    NEW_STRATEGY = "new_strategy"              # Новая стратегия добавлена
    MARKET_REGIME_CHANGE = "market_regime"     # Смена режима рынка
    PERFORMANCE_DEGRADATION = "performance"    # Деградация performance
    MANUAL = "manual"                          # Ручная ребалансировка


class PortfolioRebalancer:
    """
    Ребалансировщик портфеля

    Автоматически поддерживает оптимальную аллокацию
    """

    def __init__(
        self,
        allocator: CapitalAllocator,
        rebalance_interval_hours: int = 24,    # Ребалансировка раз в 24 часа
        min_time_between_rebalances_hours: int = 6,  # Минимум 6 часов между ребалансировками
        enable_auto_rebalance: bool = True,
    ):
        self.allocator = allocator
        self.rebalance_interval_hours = rebalance_interval_hours
        self.min_time_between_rebalances_hours = min_time_between_rebalances_hours
        self.enable_auto_rebalance = enable_auto_rebalance

        # State
        self.last_rebalance_ns: int | None = None
        self.rebalance_history: list[dict] = []

    def should_rebalance(
        self,
        portfolio: Portfolio,
        survival_scores: list[SurvivalScore],
        force_reason: RebalanceReason | None = None
    ) -> tuple[bool, RebalanceReason | None]:
        """
        Определяет нужна ли ребалансировка

        Returns (should_rebalance, reason)
        """

        # If forced, rebalance
        if force_reason:
            return True, force_reason

        # Check minimum time between rebalances
        if not self._can_rebalance_now():
            return False, None

        # Check scheduled rebalance
        if self._is_scheduled_rebalance_due():
            return True, RebalanceReason.SCHEDULED

        # Check drift threshold
        if self.allocator.should_rebalance(portfolio, survival_scores):
            return True, RebalanceReason.DRIFT_THRESHOLD

        # Check strategy failures
        if self._has_strategy_failures(portfolio, survival_scores):
            return True, RebalanceReason.STRATEGY_FAILURE

        # Check performance degradation
        if self._has_performance_degradation(portfolio):
            return True, RebalanceReason.PERFORMANCE_DEGRADATION

        return False, None

    def rebalance(
        self,
        portfolio: Portfolio,
        survival_scores: list[SurvivalScore],
        reason: RebalanceReason = RebalanceReason.MANUAL
    ) -> dict:
        """
        Выполняет ребалансировку портфеля

        Returns отчёт о ребалансировке
        """

        if not self.enable_auto_rebalance and reason != RebalanceReason.MANUAL:
            return {
                'success': False,
                'reason': 'Auto rebalance is disabled'
            }

        start_time_ns = time.time_ns()

        # Snapshot before
        before_stats = portfolio.get_statistics()

        # Kill underperformers first
        killed_strategies = self.allocator.kill_underperformers(
            portfolio,
            min_survival_score=self.allocator.min_survival_score
        )

        # Calculate new allocations
        survival_scores_dict = {s.strategy_id: s for s in survival_scores}

        new_allocations = self.allocator.allocate_capital(
            portfolio,
            survival_scores,
            portfolio.total_capital
        )

        # Apply new allocations
        self.allocator.update_portfolio_allocations(
            portfolio,
            new_allocations,
            survival_scores_dict
        )

        # Boost top performers (if capital available)
        if portfolio.available_capital > portfolio.total_capital * 0.05:  # >5% available
            self.allocator.boost_top_performers(portfolio, boost_pct=0.10)

        # Update state
        self.last_rebalance_ns = time.time_ns()
        portfolio.last_rebalance_ns = self.last_rebalance_ns

        # Snapshot after
        after_stats = portfolio.get_statistics()

        # Duration
        duration_ms = (time.time_ns() - start_time_ns) / 1_000_000

        # Record history
        rebalance_record = {
            'timestamp_ns': self.last_rebalance_ns,
            'reason': reason.value,
            'duration_ms': duration_ms,
            'killed_strategies': killed_strategies,
            'before': before_stats,
            'after': after_stats,
            'changes': {
                'capital_utilization_change': after_stats['capital_utilization_pct'] - before_stats['capital_utilization_pct'],
                'strategy_count_change': after_stats['total_strategies'] - before_stats['total_strategies'],
                'diversification_change': after_stats['diversification_score'] - before_stats['diversification_score'],
            }
        }

        self.rebalance_history.append(rebalance_record)

        return {
            'success': True,
            'reason': reason.value,
            'duration_ms': duration_ms,
            'killed_strategies': len(killed_strategies),
            'new_strategies': after_stats['total_strategies'] - before_stats['total_strategies'] + len(killed_strategies),
            'capital_utilization_pct': after_stats['capital_utilization_pct'],
            'diversification_score': after_stats['diversification_score'],
        }

    def auto_rebalance(
        self,
        portfolio: Portfolio,
        survival_scores: list[SurvivalScore]
    ) -> dict | None:
        """
        Автоматическая ребалансировка (если нужна)

        Returns отчёт о ребалансировке или None если не было
        """

        should_rebal, reason = self.should_rebalance(portfolio, survival_scores)

        if should_rebal:
            return self.rebalance(portfolio, survival_scores, reason=reason)

        return None

    def _can_rebalance_now(self) -> bool:
        """Проверяет можно ли ребалансировать сейчас (минимальное время прошло)"""

        if self.last_rebalance_ns is None:
            return True

        elapsed_hours = (time.time_ns() - self.last_rebalance_ns) / (3600 * 1e9)

        return elapsed_hours >= self.min_time_between_rebalances_hours

    def _is_scheduled_rebalance_due(self) -> bool:
        """Проверяет пора ли для запланированной ребалансировки"""

        if self.last_rebalance_ns is None:
            return True

        elapsed_hours = (time.time_ns() - self.last_rebalance_ns) / (3600 * 1e9)

        return elapsed_hours >= self.rebalance_interval_hours

    def _has_strategy_failures(
        self,
        portfolio: Portfolio,
        survival_scores: list[SurvivalScore]
    ) -> bool:
        """Проверяет есть ли провалившиеся стратегии"""

        survival_scores_dict = {s.strategy_id: s for s in survival_scores}

        for strategy_id, _allocation in portfolio.allocations.items():
            survival_score = survival_scores_dict.get(strategy_id)

            if survival_score and survival_score.overall_score < self.allocator.min_survival_score:
                return True

        return False

    def _has_performance_degradation(self, portfolio: Portfolio) -> bool:
        """
        Проверяет деградацию performance

        Если portfolio PnL падает >10% от peak
        """

        # NOTE: Proper performance tracking not yet implemented
        # This function uses a basic drawdown check as a proxy
        #
        # LIMITATION: Without historical peak tracking, we can only detect
        # current drawdown state, not degradation from a previous peak.
        # A full implementation would track:
        # - Rolling peak equity values
        # - Time-weighted performance metrics
        # - Performance relative to recent moving average
        # - Regime-adjusted performance expectations

        if portfolio.portfolio_max_drawdown_pct < -15.0:  # >15% drawdown
            return True

        return False

    def get_rebalance_history(self, last_n: int = 10) -> list[dict]:
        """Возвращает историю ребалансировок"""
        return self.rebalance_history[-last_n:]

    def get_statistics(self) -> dict:
        """Статистика ребалансировок"""

        if not self.rebalance_history:
            return {
                'total_rebalances': 0,
                'enable_auto_rebalance': self.enable_auto_rebalance,
            }

        total_rebalances = len(self.rebalance_history)

        # Count by reason
        reasons_count = {}
        for record in self.rebalance_history:
            reason = record['reason']
            reasons_count[reason] = reasons_count.get(reason, 0) + 1

        # Average duration
        avg_duration_ms = sum(r['duration_ms'] for r in self.rebalance_history) / total_rebalances

        # Total strategies killed
        total_killed = sum(len(r['killed_strategies']) for r in self.rebalance_history)

        # Last rebalance
        last_rebalance = self.rebalance_history[-1]

        return {
            'total_rebalances': total_rebalances,
            'enable_auto_rebalance': self.enable_auto_rebalance,
            'rebalance_interval_hours': self.rebalance_interval_hours,
            'min_time_between_rebalances_hours': self.min_time_between_rebalances_hours,
            'last_rebalance_timestamp_ns': last_rebalance['timestamp_ns'],
            'last_rebalance_reason': last_rebalance['reason'],
            'avg_duration_ms': avg_duration_ms,
            'total_strategies_killed': total_killed,
            'rebalance_reasons': reasons_count,
        }
