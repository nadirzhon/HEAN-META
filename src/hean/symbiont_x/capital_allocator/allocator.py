"""
Capital Allocator - Дарвиновский распределитель капитала

Аллоцирует капитал на основе survival scores с учётом корреляций
"""

import time

import numpy as np

from ..adversarial_twin.survival_score import SurvivalScore
from .portfolio import Portfolio, StrategyAllocation


class CapitalAllocator:
    """
    Распределитель капитала

    Использует Дарвиновский подход: сильные получают больше, слабые - меньше
    """

    def __init__(
        self,
        allocation_method: str = 'survival_weighted',
        rebalance_threshold: float = 0.10,  # Rebalance if allocation drifts >10%
        min_survival_score: float = 60.0,   # Minimum score to get capital
    ):
        self.allocation_method = allocation_method
        self.rebalance_threshold = rebalance_threshold
        self.min_survival_score = min_survival_score

        # Statistics
        self.allocations_performed = 0
        self.rebalances_performed = 0

    def allocate_capital(
        self,
        portfolio: Portfolio,
        survival_scores: list[SurvivalScore],
        total_capital: float,
    ) -> dict[str, float]:
        """
        Распределяет капитал между стратегиями

        Args:
            portfolio: Портфель
            survival_scores: Список survival scores
            total_capital: Общий капитал для распределения

        Returns:
            Dict {strategy_id: allocated_capital}
        """

        # Filter strategies that pass minimum score
        qualified_strategies = [
            s for s in survival_scores
            if s.overall_score >= self.min_survival_score
        ]

        if not qualified_strategies:
            return {}  # No strategies qualify

        # Calculate allocations based on method
        if self.allocation_method == 'survival_weighted':
            allocations = self._survival_weighted_allocation(
                qualified_strategies, total_capital
            )
        elif self.allocation_method == 'equal_weight':
            allocations = self._equal_weight_allocation(
                qualified_strategies, total_capital
            )
        elif self.allocation_method == 'risk_parity':
            allocations = self._risk_parity_allocation(
                qualified_strategies, total_capital
            )
        else:
            allocations = self._survival_weighted_allocation(
                qualified_strategies, total_capital
            )

        # Apply portfolio constraints
        allocations = self._apply_constraints(
            allocations,
            portfolio,
            total_capital
        )

        self.allocations_performed += 1

        return allocations

    def _survival_weighted_allocation(
        self,
        strategies: list[SurvivalScore],
        total_capital: float
    ) -> dict[str, float]:
        """
        Survival-weighted аллокация

        Больше капитала стратегиям с выше survival score
        """

        # Calculate weights based on survival scores
        scores = np.array([s.overall_score for s in strategies])

        # Use exponential weighting to amplify differences
        # Better strategies get exponentially more capital
        weights = np.exp(scores / 50.0)  # Scale factor 50
        weights = weights / weights.sum()  # Normalize

        # Allocate
        allocations = {}
        for strategy, weight in zip(strategies, weights, strict=False):
            allocated = total_capital * weight
            allocations[strategy.strategy_id] = allocated

        return allocations

    def _equal_weight_allocation(
        self,
        strategies: list[SurvivalScore],
        total_capital: float
    ) -> dict[str, float]:
        """
        Equal-weight аллокация

        Все квалифицированные стратегии получают одинаково
        """

        n_strategies = len(strategies)
        per_strategy = total_capital / n_strategies

        allocations = {}
        for strategy in strategies:
            allocations[strategy.strategy_id] = per_strategy

        return allocations

    def _risk_parity_allocation(
        self,
        strategies: list[SurvivalScore],
        total_capital: float
    ) -> dict[str, float]:
        """
        Risk parity аллокация

        Аллоцирует на основе inverse volatility
        (больше капитала менее волатильным стратегиям)
        """

        # Use max drawdown as risk proxy
        risks = np.array([
            abs(s.micro_real_result.max_drawdown_pct)
            for s in strategies
        ])

        # Inverse volatility weights
        inv_risks = 1.0 / (risks + 0.01)  # Add small epsilon
        weights = inv_risks / inv_risks.sum()

        # Allocate
        allocations = {}
        for strategy, weight in zip(strategies, weights, strict=False):
            allocated = total_capital * weight
            allocations[strategy.strategy_id] = allocated

        return allocations

    def _apply_constraints(
        self,
        allocations: dict[str, float],
        portfolio: Portfolio,
        total_capital: float
    ) -> dict[str, float]:
        """
        Применяет ограничения портфеля

        - Min/max allocation per strategy
        - Max number of strategies
        """

        # Apply min/max limits
        constrained = {}

        for strategy_id, capital in allocations.items():
            allocation_pct = (capital / total_capital) * 100

            # Clamp to min/max
            allocation_pct = max(portfolio.min_allocation_pct, allocation_pct)
            allocation_pct = min(portfolio.max_allocation_pct, allocation_pct)

            constrained[strategy_id] = (allocation_pct / 100) * total_capital

        # Limit number of strategies
        if len(constrained) > portfolio.max_strategies:
            # Keep top N by allocation
            sorted_strategies = sorted(
                constrained.items(),
                key=lambda x: x[1],
                reverse=True
            )
            constrained = dict(sorted_strategies[:portfolio.max_strategies])

        # Renormalize to sum to total_capital
        total_allocated = sum(constrained.values())
        if total_allocated > 0:
            scale_factor = total_capital / total_allocated
            constrained = {
                sid: capital * scale_factor
                for sid, capital in constrained.items()
            }

        return constrained

    def update_portfolio_allocations(
        self,
        portfolio: Portfolio,
        new_allocations: dict[str, float],
        survival_scores: dict[str, SurvivalScore]
    ):
        """
        Обновляет аллокации в портфеле

        Добавляет новые, удаляет старые, обновляет существующие
        """

        current_strategy_ids = set(portfolio.allocations.keys())
        new_strategy_ids = set(new_allocations.keys())

        # Remove strategies not in new allocation
        to_remove = current_strategy_ids - new_strategy_ids
        for strategy_id in to_remove:
            portfolio.remove_strategy(strategy_id)

        # Add or update strategies
        for strategy_id, new_capital in new_allocations.items():
            survival_score = survival_scores.get(strategy_id)

            if not survival_score:
                continue

            if strategy_id in portfolio.allocations:
                # Update existing
                portfolio.update_allocation(strategy_id, new_capital)
            else:
                # Add new
                allocation = StrategyAllocation(
                    strategy_id=strategy_id,
                    strategy_name=survival_score.strategy_name,
                    allocated_capital=new_capital,
                    allocation_pct=(new_capital / portfolio.total_capital) * 100,
                    survival_score=survival_score.overall_score,
                    sharpe_ratio=survival_score.micro_real_result.sharpe_ratio,
                    max_drawdown_pct=survival_score.micro_real_result.max_drawdown_pct,
                    win_rate=survival_score.micro_real_result.win_rate,
                    allocated_at_ns=time.time_ns(),
                )

                portfolio.add_strategy(allocation)

        portfolio.last_rebalance_ns = time.time_ns()

    def should_rebalance(
        self,
        portfolio: Portfolio,
        new_survival_scores: list[SurvivalScore]
    ) -> bool:
        """
        Проверяет нужен ли rebalance

        Returns True если drift превышает threshold
        """

        # Calculate what allocations should be
        new_allocations = self.allocate_capital(
            portfolio,
            new_survival_scores,
            portfolio.total_capital
        )

        # Check drift
        for strategy_id, new_capital in new_allocations.items():
            if strategy_id in portfolio.allocations:
                current_capital = portfolio.allocations[strategy_id].allocated_capital

                if current_capital > 0:
                    drift = abs(new_capital - current_capital) / current_capital

                    if drift > self.rebalance_threshold:
                        return True

        # Check if strategies changed significantly
        current_ids = set(portfolio.allocations.keys())
        new_ids = set(new_allocations.keys())

        # If >20% of strategies changed, rebalance
        changed_pct = len(current_ids.symmetric_difference(new_ids)) / max(len(current_ids), 1)
        if changed_pct > 0.2:
            return True

        return False

    def kill_underperformers(
        self,
        portfolio: Portfolio,
        min_survival_score: float | None = None
    ) -> list[str]:
        """
        Убивает underperforming стратегии

        Returns список убитых strategy_ids
        """

        if min_survival_score is None:
            min_survival_score = self.min_survival_score

        killed = []

        for strategy_id, allocation in list(portfolio.allocations.items()):
            if allocation.survival_score < min_survival_score:
                portfolio.remove_strategy(strategy_id)
                killed.append(strategy_id)

        return killed

    def boost_top_performers(
        self,
        portfolio: Portfolio,
        boost_pct: float = 0.20  # Increase allocation by 20%
    ):
        """
        Увеличивает аллокацию топ исполнителей

        Дарвиновская награда за успех
        """

        top_performers = portfolio.get_top_performers(n=3)

        for allocation in top_performers:
            current_capital = allocation.allocated_capital
            boosted_capital = current_capital * (1 + boost_pct)

            # Check if we have available capital
            capital_needed = boosted_capital - current_capital

            if capital_needed <= portfolio.available_capital:
                portfolio.update_allocation(allocation.strategy_id, boosted_capital)

    def get_statistics(self) -> dict:
        """Статистика аллокатора"""
        return {
            'allocation_method': self.allocation_method,
            'rebalance_threshold': self.rebalance_threshold,
            'min_survival_score': self.min_survival_score,
            'allocations_performed': self.allocations_performed,
            'rebalances_performed': self.rebalances_performed,
        }
