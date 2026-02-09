"""
KPI System - 6 основных показателей здоровья организма

Всегда видимые метрики состояния SYMBIONT X
"""

import time
from dataclasses import dataclass
from enum import Enum


class KPIStatus(Enum):
    """Статус KPI"""
    EXCELLENT = "excellent"    # > 90%
    GOOD = "good"              # 70-90%
    ACCEPTABLE = "acceptable"  # 50-70%
    POOR = "poor"              # 30-50%
    CRITICAL = "critical"      # < 30%


@dataclass
class KPI:
    """Один KPI показатель"""

    name: str
    value: float  # 0-100
    status: KPIStatus
    description: str
    timestamp_ns: int

    def get_emoji(self) -> str:
        """Возвращает emoji для статуса"""
        if self.status == KPIStatus.EXCELLENT:
            return "🟢"
        elif self.status == KPIStatus.GOOD:
            return "🟡"
        elif self.status == KPIStatus.ACCEPTABLE:
            return "🟠"
        elif self.status == KPIStatus.POOR:
            return "🔴"
        else:
            return "💀"

    def __str__(self) -> str:
        return f"{self.get_emoji()} {self.name}: {self.value:.1f}%"


class KPISystem:
    """
    Система 6 ключевых показателей

    1. Survival Score - общая живучесть системы
    2. Execution Edge - качество исполнения
    3. Immunity Saves - сколько раз защита спасла
    4. Alpha Production - генерация прибыли
    5. Truth Mode - качество данных и классификации
    6. Autonomy Level - уровень автономности
    """

    def __init__(self) -> None:
        # KPIs
        self.survival_score: KPI | None = None
        self.execution_edge: KPI | None = None
        self.immunity_saves: KPI | None = None
        self.alpha_production: KPI | None = None
        self.truth_mode: KPI | None = None
        self.autonomy_level: KPI | None = None

        # History - stores historical values (floats), not KPI objects
        self.kpi_history: dict[str, list[float]] = {
            'survival_score': [],
            'execution_edge': [],
            'immunity_saves': [],
            'alpha_production': [],
            'truth_mode': [],
            'autonomy_level': [],
        }

    def update_survival_score(self, score: float, portfolio_stats: dict[str, float]) -> None:
        """
        Обновляет Survival Score (0-100)

        Измеряет общую живучесть системы
        """

        # Determine status
        if score >= 90:
            status = KPIStatus.EXCELLENT
        elif score >= 70:
            status = KPIStatus.GOOD
        elif score >= 50:
            status = KPIStatus.ACCEPTABLE
        elif score >= 30:
            status = KPIStatus.POOR
        else:
            status = KPIStatus.CRITICAL

        self.survival_score = KPI(
            name="Survival Score",
            value=score,
            status=status,
            description=f"Portfolio Sharpe: {portfolio_stats.get('portfolio_sharpe', 0):.2f}",
            timestamp_ns=time.time_ns(),
        )

        self.kpi_history['survival_score'].append(score)

    def update_execution_edge(self, avg_slippage_bps: float, avg_latency_ms: float):
        """
        Обновляет Execution Edge (0-100)

        Измеряет качество исполнения ордеров
        """

        # Score based on slippage and latency
        # Lower is better
        slippage_score = max(0, 100 - (avg_slippage_bps * 10))  # 10 bps = -100
        latency_score = max(0, 100 - avg_latency_ms)  # 100ms = 0

        score = (slippage_score + latency_score) / 2

        if score >= 90:
            status = KPIStatus.EXCELLENT
        elif score >= 70:
            status = KPIStatus.GOOD
        elif score >= 50:
            status = KPIStatus.ACCEPTABLE
        elif score >= 30:
            status = KPIStatus.POOR
        else:
            status = KPIStatus.CRITICAL

        self.execution_edge = KPI(
            name="Execution Edge",
            value=score,
            status=status,
            description=f"Slippage: {avg_slippage_bps:.1f}bps, Latency: {avg_latency_ms:.1f}ms",
            timestamp_ns=time.time_ns(),
        )

        self.kpi_history['execution_edge'].append(score)

    def update_immunity_saves(self, total_reflexes: int, total_breakers: int):
        """
        Обновляет Immunity Saves (0-100)

        Сколько раз иммунная система спасла от потерь
        """

        # More saves = better (up to a point)
        # Too many saves might mean system is fragile

        total_saves = total_reflexes + total_breakers

        if total_saves == 0:
            score = 100  # No threats = good
        elif total_saves < 5:
            score = 95  # Some threats handled = excellent
        elif total_saves < 20:
            score = 80  # Moderate activity = good
        elif total_saves < 50:
            score = 60  # High activity = concerning
        else:
            score = 30  # Too many threats = poor

        if score >= 90:
            status = KPIStatus.EXCELLENT
        elif score >= 70:
            status = KPIStatus.GOOD
        elif score >= 50:
            status = KPIStatus.ACCEPTABLE
        elif score >= 30:
            status = KPIStatus.POOR
        else:
            status = KPIStatus.CRITICAL

        self.immunity_saves = KPI(
            name="Immunity Saves",
            value=score,
            status=status,
            description=f"Reflexes: {total_reflexes}, Breakers: {total_breakers}",
            timestamp_ns=time.time_ns(),
        )

        self.kpi_history['immunity_saves'].append(score)

    def update_alpha_production(self, total_pnl: float, roi_pct: float):
        """
        Обновляет Alpha Production (0-100)

        Генерация прибыли
        """

        # Score based on ROI
        # > 50% ROI = excellent
        # > 20% ROI = good
        # > 5% ROI = acceptable
        # > 0% ROI = poor
        # < 0% ROI = critical

        score: float
        if roi_pct >= 50:
            score = 100.0
        elif roi_pct >= 20:
            score = 85.0
        elif roi_pct >= 5:
            score = 65.0
        elif roi_pct >= 0:
            score = 40.0
        else:
            # Negative ROI
            score = max(0.0, 40.0 + roi_pct)  # -40% = 0 score

        if score >= 90:
            status = KPIStatus.EXCELLENT
        elif score >= 70:
            status = KPIStatus.GOOD
        elif score >= 50:
            status = KPIStatus.ACCEPTABLE
        elif score >= 30:
            status = KPIStatus.POOR
        else:
            status = KPIStatus.CRITICAL

        self.alpha_production = KPI(
            name="Alpha Production",
            value=score,
            status=status,
            description=f"PnL: ${total_pnl:.2f}, ROI: {roi_pct:.1f}%",
            timestamp_ns=time.time_ns(),
        )

        self.kpi_history['alpha_production'].append(score)

    def update_truth_mode(self, data_health_score: float, regime_confidence: float):
        """
        Обновляет Truth Mode (0-100)

        Качество данных и классификации
        """

        # Combine data health and regime confidence
        score = (data_health_score + regime_confidence) / 2

        if score >= 90:
            status = KPIStatus.EXCELLENT
        elif score >= 70:
            status = KPIStatus.GOOD
        elif score >= 50:
            status = KPIStatus.ACCEPTABLE
        elif score >= 30:
            status = KPIStatus.POOR
        else:
            status = KPIStatus.CRITICAL

        self.truth_mode = KPI(
            name="Truth Mode",
            value=score,
            status=status,
            description=f"Data: {data_health_score:.0f}%, Regime: {regime_confidence:.0f}%",
            timestamp_ns=time.time_ns(),
        )

        self.kpi_history['truth_mode'].append(score)

    def update_autonomy_level(
        self,
        strategies_active: int,
        decisions_per_hour: int,
        manual_interventions: int
    ):
        """
        Обновляет Autonomy Level (0-100)

        Уровень автономности системы
        """

        # More active strategies = higher autonomy
        strategy_score = min(strategies_active * 10, 50)  # Max 50 points

        # More decisions = higher autonomy
        decision_score = min(decisions_per_hour / 2, 30)  # Max 30 points

        # Fewer interventions = higher autonomy
        intervention_penalty = manual_interventions * 5
        intervention_score = max(0, 20 - intervention_penalty)  # Max 20 points

        score = strategy_score + decision_score + intervention_score

        if score >= 90:
            status = KPIStatus.EXCELLENT
        elif score >= 70:
            status = KPIStatus.GOOD
        elif score >= 50:
            status = KPIStatus.ACCEPTABLE
        elif score >= 30:
            status = KPIStatus.POOR
        else:
            status = KPIStatus.CRITICAL

        self.autonomy_level = KPI(
            name="Autonomy Level",
            value=score,
            status=status,
            description=f"Strategies: {strategies_active}, Decisions/hr: {decisions_per_hour}",
            timestamp_ns=time.time_ns(),
        )

        self.kpi_history['autonomy_level'].append(score)

    def get_all_kpis(self) -> dict[str, KPI | None]:
        """Возвращает все KPI"""
        return {
            'survival_score': self.survival_score,
            'execution_edge': self.execution_edge,
            'immunity_saves': self.immunity_saves,
            'alpha_production': self.alpha_production,
            'truth_mode': self.truth_mode,
            'autonomy_level': self.autonomy_level,
        }

    def get_overall_health(self) -> float:
        """Вычисляет общее здоровье системы (0-100)"""

        kpis = self.get_all_kpis()
        values = [kpi.value for kpi in kpis.values() if kpi is not None]

        if not values:
            return 0.0

        return sum(values) / len(values)

    def get_dashboard_view(self) -> str:
        """Возвращает текстовое представление для dashboard"""

        lines = []
        lines.append("=" * 60)
        lines.append("HEAN SYMBIONT X - VITAL SIGNS")
        lines.append("=" * 60)

        kpis = self.get_all_kpis()

        for kpi in kpis.values():
            if kpi:
                lines.append(f"{str(kpi):<50} {kpi.description}")

        lines.append("=" * 60)
        lines.append(f"Overall Health: {self.get_overall_health():.1f}%")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Сериализация"""

        kpis = self.get_all_kpis()

        return {
            'kpis': {
                name: {
                    'value': kpi.value,
                    'status': kpi.status.value,
                    'description': kpi.description,
                } if kpi else None
                for name, kpi in kpis.items()
            },
            'overall_health': self.get_overall_health(),
        }
