"""Automatic report generation using LLM."""

from datetime import datetime
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Автоматическая генерация отчетов о производительности."""

    def __init__(self) -> None:
        """Инициализация генератора отчетов."""
        self._reports: list[dict[str, Any]] = []

    def generate_daily_report(
        self,
        performance_data: dict[str, Any],
        strategy_metrics: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Генерировать ежедневный отчет.

        Args:
            performance_data: Данные о производительности системы
            strategy_metrics: Метрики по стратегиям

        Returns:
            Словарь с отчетом
        """
        try:
            equity = performance_data.get("equity", 0.0)
            drawdown_pct = performance_data.get("drawdown_pct", 0.0)
            total_trades = performance_data.get("total_trades", 0)

            # Найти лучшую и худшую стратегии
            best_strategy = None
            worst_strategy = None
            best_pf = 0.0
            worst_pf = float("inf")

            for strategy_id, metrics in strategy_metrics.items():
                pf = metrics.get("profit_factor", 1.0)
                if pf > best_pf:
                    best_pf = pf
                    best_strategy = strategy_id
                if pf < worst_pf:
                    worst_pf = pf
                    worst_strategy = strategy_id

            # Рассчитать общий PnL
            total_pnl = sum(m.get("pnl", 0.0) for m in strategy_metrics.values())

            # Создать отчет
            report = {
                "date": datetime.utcnow().date().isoformat(),
                "summary": {
                    "equity": equity,
                    "total_pnl": total_pnl,
                    "drawdown_pct": drawdown_pct,
                    "total_trades": total_trades,
                },
                "best_strategy": {
                    "id": best_strategy,
                    "profit_factor": best_pf,
                    "metrics": strategy_metrics.get(best_strategy, {}) if best_strategy else {},
                },
                "worst_strategy": {
                    "id": worst_strategy,
                    "profit_factor": worst_pf,
                    "metrics": strategy_metrics.get(worst_strategy, {}) if worst_strategy else {},
                },
                "strategy_details": strategy_metrics,
                "recommendations": self._generate_recommendations(
                    performance_data, strategy_metrics
                ),
            }

            self._reports.append(report)

            return report

        except Exception as e:
            logger.error(f"Error generating daily report: {e}", exc_info=True)
            return {}

    def _generate_recommendations(
        self,
        performance_data: dict[str, Any],
        strategy_metrics: dict[str, dict[str, float]],
    ) -> list[str]:
        """Генерировать рекомендации на основе данных."""
        recommendations = []

        # Анализ drawdown
        drawdown_pct = performance_data.get("drawdown_pct", 0.0)
        if drawdown_pct > 10.0:
            recommendations.append(
                f"Внимание: Drawdown достиг {drawdown_pct:.1f}%. "
                "Рекомендуется активировать консервативный режим."
            )

        # Анализ стратегий
        for strategy_id, metrics in strategy_metrics.items():
            pf = metrics.get("profit_factor", 1.0)
            trades = metrics.get("trades", 0)

            if pf < 1.0 and trades >= 5:
                recommendations.append(
                    f"Стратегия {strategy_id} убыточна (PF={pf:.2f}). "
                    "Рекомендуется отключить или оптимизировать параметры."
                )
            elif pf > 2.0 and trades >= 10:
                recommendations.append(
                    f"Стратегия {strategy_id} показывает отличные результаты (PF={pf:.2f}). "
                    "Рекомендуется увеличить выделенный капитал."
                )

        if not recommendations:
            recommendations.append("Все системы работают в норме. Продолжайте мониторинг.")

        return recommendations

    def generate_weekly_report(
        self,
        daily_reports: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Генерировать еженедельный отчет."""
        try:
            if not daily_reports:
                return {}

            # Агрегировать данные
            total_pnl = sum(r["summary"]["total_pnl"] for r in daily_reports)
            avg_drawdown = sum(r["summary"]["drawdown_pct"] for r in daily_reports) / len(
                daily_reports
            )
            total_trades = sum(r["summary"]["total_trades"] for r in daily_reports)

            # Найти лучшую стратегию за неделю
            strategy_performance = {}
            for report in daily_reports:
                for strategy_id, metrics in report.get("strategy_details", {}).items():
                    if strategy_id not in strategy_performance:
                        strategy_performance[strategy_id] = {
                            "total_pnl": 0.0,
                            "total_trades": 0,
                            "avg_pf": 0.0,
                            "count": 0,
                        }
                    strategy_performance[strategy_id]["total_pnl"] += metrics.get("pnl", 0.0)
                    strategy_performance[strategy_id]["total_trades"] += metrics.get("trades", 0)
                    strategy_performance[strategy_id]["avg_pf"] += metrics.get("profit_factor", 1.0)
                    strategy_performance[strategy_id]["count"] += 1

            # Рассчитать средние
            for strategy_id in strategy_performance:
                perf = strategy_performance[strategy_id]
                if perf["count"] > 0:
                    perf["avg_pf"] /= perf["count"]

            # Найти лучшую
            best_strategy = (
                max(strategy_performance.items(), key=lambda x: x[1]["total_pnl"])
                if strategy_performance
                else None
            )

            report = {
                "period": "weekly",
                "start_date": daily_reports[0]["date"],
                "end_date": daily_reports[-1]["date"],
                "summary": {
                    "total_pnl": total_pnl,
                    "avg_drawdown": avg_drawdown,
                    "total_trades": total_trades,
                },
                "best_strategy": {
                    "id": best_strategy[0] if best_strategy else None,
                    "performance": best_strategy[1] if best_strategy else {},
                },
                "strategy_performance": strategy_performance,
            }

            return report

        except Exception as e:
            logger.error(f"Error generating weekly report: {e}", exc_info=True)
            return {}

    def get_reports(self) -> list[dict[str, Any]]:
        """Получить все отчеты."""
        return self._reports.copy()
