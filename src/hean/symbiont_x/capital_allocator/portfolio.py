"""
Portfolio - Портфель стратегий

Содержит все активные стратегии и их аллокации
"""

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StrategyAllocation:
    """
    Аллокация капитала для одной стратегии
    """

    strategy_id: str
    strategy_name: str

    # Allocation
    allocated_capital: float
    allocation_pct: float  # % от total capital

    # Performance
    survival_score: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float

    # Activity
    is_active: bool = True
    is_paused: bool = False

    # Tracking
    allocated_at_ns: int = 0
    last_rebalance_ns: int = 0

    # PnL tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def get_total_pnl(self) -> float:
        """Общий PnL"""
        return self.realized_pnl + self.unrealized_pnl

    def get_roi(self) -> float:
        """ROI в процентах"""
        if self.allocated_capital == 0:
            return 0.0
        return (self.get_total_pnl() / self.allocated_capital) * 100


@dataclass
class Portfolio:
    """
    Портфель стратегий

    Управляет капиталом между множеством стратегий
    """

    portfolio_id: str
    name: str

    # Capital
    total_capital: float
    allocated_capital: float = 0.0
    available_capital: float = 0.0

    # Allocations
    allocations: dict[str, StrategyAllocation] = field(default_factory=dict)

    # Constraints
    min_allocation_pct: float = 1.0    # Minimum 1% per strategy
    max_allocation_pct: float = 30.0   # Maximum 30% per strategy
    max_strategies: int = 20           # Maximum number of strategies

    # Performance tracking
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    portfolio_sharpe: float = 0.0
    portfolio_max_drawdown_pct: float = 0.0

    # Timestamps
    created_at_ns: int = 0
    last_rebalance_ns: int = 0

    def __post_init__(self):
        if self.created_at_ns == 0:
            self.created_at_ns = time.time_ns()
        self.available_capital = self.total_capital

    def add_strategy(self, allocation: StrategyAllocation) -> bool:
        """
        Добавляет стратегию в портфель

        Returns True if successful
        """

        # Check limits
        if len(self.allocations) >= self.max_strategies:
            return False

        # Check if enough capital
        if allocation.allocated_capital > self.available_capital:
            return False

        # Check allocation percentage
        allocation_pct = (allocation.allocated_capital / self.total_capital) * 100
        if allocation_pct < self.min_allocation_pct or allocation_pct > self.max_allocation_pct:
            return False

        # Add strategy
        self.allocations[allocation.strategy_id] = allocation
        self.allocated_capital += allocation.allocated_capital
        self.available_capital = self.total_capital - self.allocated_capital

        return True

    def remove_strategy(self, strategy_id: str) -> bool:
        """Удаляет стратегию из портфеля"""

        if strategy_id not in self.allocations:
            return False

        allocation = self.allocations[strategy_id]

        # Return capital
        self.allocated_capital -= allocation.allocated_capital
        self.available_capital = self.total_capital - self.allocated_capital

        # Remove
        del self.allocations[strategy_id]

        return True

    def update_allocation(self, strategy_id: str, new_capital: float) -> bool:
        """
        Обновляет аллокацию для стратегии

        Returns True if successful
        """

        if strategy_id not in self.allocations:
            return False

        allocation = self.allocations[strategy_id]
        old_capital = allocation.allocated_capital
        capital_diff = new_capital - old_capital

        # Check if enough available capital
        if capital_diff > 0 and capital_diff > self.available_capital:
            return False

        # Check percentage limits
        new_pct = (new_capital / self.total_capital) * 100
        if new_pct < self.min_allocation_pct or new_pct > self.max_allocation_pct:
            return False

        # Update
        allocation.allocated_capital = new_capital
        allocation.allocation_pct = new_pct
        allocation.last_rebalance_ns = time.time_ns()

        self.allocated_capital += capital_diff
        self.available_capital = self.total_capital - self.allocated_capital

        return True

    def pause_strategy(self, strategy_id: str):
        """Приостанавливает стратегию (не удаляет, но не торгует)"""
        if strategy_id in self.allocations:
            self.allocations[strategy_id].is_paused = True
            self.allocations[strategy_id].is_active = False

    def resume_strategy(self, strategy_id: str):
        """Возобновляет стратегию"""
        if strategy_id in self.allocations:
            self.allocations[strategy_id].is_paused = False
            self.allocations[strategy_id].is_active = True

    def get_active_strategies(self) -> list[StrategyAllocation]:
        """Возвращает список активных стратегий"""
        return [
            alloc for alloc in self.allocations.values()
            if alloc.is_active and not alloc.is_paused
        ]

    def get_top_performers(self, n: int = 5) -> list[StrategyAllocation]:
        """Возвращает топ N стратегий по survival score"""
        sorted_allocations = sorted(
            self.allocations.values(),
            key=lambda a: a.survival_score,
            reverse=True
        )
        return sorted_allocations[:n]

    def get_worst_performers(self, n: int = 5) -> list[StrategyAllocation]:
        """Возвращает худшие N стратегий"""
        sorted_allocations = sorted(
            self.allocations.values(),
            key=lambda a: a.survival_score
        )
        return sorted_allocations[:n]

    def calculate_portfolio_metrics(self):
        """Вычисляет метрики портфеля"""

        if not self.allocations:
            return

        # Total PnL
        self.total_realized_pnl = sum(a.realized_pnl for a in self.allocations.values())
        self.total_unrealized_pnl = sum(a.unrealized_pnl for a in self.allocations.values())

        # Weighted Sharpe ratio
        total_weight = sum(a.allocation_pct for a in self.allocations.values())
        if total_weight > 0:
            weighted_sharpe = sum(
                a.sharpe_ratio * a.allocation_pct
                for a in self.allocations.values()
            )
            self.portfolio_sharpe = weighted_sharpe / total_weight

        # Worst drawdown
        drawdowns = [a.max_drawdown_pct for a in self.allocations.values()]
        self.portfolio_max_drawdown_pct = min(drawdowns) if drawdowns else 0.0

    def get_correlation_matrix(self) -> np.ndarray:
        """
        Вычисляет корреляционную матрицу между стратегиями

        TODO: Требует historical PnL data
        """

        n_strategies = len(self.allocations)

        if n_strategies == 0:
            return np.array([[]])

        # For now, return identity matrix (no correlation assumed)
        # In real implementation, would use historical PnL data
        return np.eye(n_strategies)

    def get_diversification_score(self) -> float:
        """
        Оценка диверсификации (0-1)

        1.0 = идеальная диверсификация
        0.0 = нет диверсификации
        """

        if len(self.allocations) <= 1:
            return 0.0

        # Component 1: Number of strategies (normalized)
        strategy_score = min(len(self.allocations) / self.max_strategies, 1.0)

        # Component 2: Allocation balance (Herfindahl index)
        allocations_pct = [a.allocation_pct for a in self.allocations.values()]
        total_pct = sum(allocations_pct)

        if total_pct > 0:
            normalized_pcts = [p / total_pct for p in allocations_pct]
            herfindahl = sum(p ** 2 for p in normalized_pcts)
            balance_score = 1.0 - (herfindahl - (1.0 / len(self.allocations))) / (1.0 - (1.0 / len(self.allocations)))
        else:
            balance_score = 0.0

        # Weighted
        diversification = strategy_score * 0.4 + balance_score * 0.6

        return diversification

    def get_statistics(self) -> dict:
        """Статистика портфеля"""

        self.calculate_portfolio_metrics()

        active_strategies = self.get_active_strategies()
        total_pnl = self.total_realized_pnl + self.total_unrealized_pnl
        roi = (total_pnl / self.total_capital) * 100 if self.total_capital > 0 else 0

        return {
            'portfolio_id': self.portfolio_id,
            'name': self.name,
            'total_capital': self.total_capital,
            'allocated_capital': self.allocated_capital,
            'available_capital': self.available_capital,
            'capital_utilization_pct': (self.allocated_capital / self.total_capital) * 100,
            'total_strategies': len(self.allocations),
            'active_strategies': len(active_strategies),
            'paused_strategies': len([a for a in self.allocations.values() if a.is_paused]),
            'total_realized_pnl': self.total_realized_pnl,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'total_pnl': total_pnl,
            'roi_pct': roi,
            'portfolio_sharpe': self.portfolio_sharpe,
            'portfolio_max_drawdown_pct': self.portfolio_max_drawdown_pct,
            'diversification_score': self.get_diversification_score(),
            'top_performer': self.get_top_performers(1)[0].strategy_name if self.allocations else None,
            'worst_performer': self.get_worst_performers(1)[0].strategy_name if self.allocations else None,
        }

    def to_dict(self) -> dict:
        """Сериализация"""
        return {
            'portfolio_id': self.portfolio_id,
            'name': self.name,
            'total_capital': self.total_capital,
            'allocated_capital': self.allocated_capital,
            'available_capital': self.available_capital,
            'allocations': [
                {
                    'strategy_id': a.strategy_id,
                    'strategy_name': a.strategy_name,
                    'allocated_capital': a.allocated_capital,
                    'allocation_pct': a.allocation_pct,
                    'survival_score': a.survival_score,
                    'is_active': a.is_active,
                    'is_paused': a.is_paused,
                    'realized_pnl': a.realized_pnl,
                    'unrealized_pnl': a.unrealized_pnl,
                    'roi_pct': a.get_roi(),
                }
                for a in self.allocations.values()
            ],
            'statistics': self.get_statistics(),
        }
