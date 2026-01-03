"""Automatic parameter optimization using LLM analysis."""

import json
from datetime import datetime
from typing import Any

from hean.agent_generation.generator import AgentGenerator
from hean.logging import get_logger

logger = get_logger(__name__)


class ParameterOptimizer:
    """Автоматическая оптимизация параметров стратегий через LLM."""

    def __init__(self) -> None:
        """Инициализация оптимизатора."""
        self._generator = AgentGenerator()
        self._optimization_history: list[dict[str, Any]] = []

    async def optimize_strategy_parameters(
        self,
        strategy_id: str,
        current_metrics: dict[str, float],
        current_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Оптимизировать параметры стратегии.

        Args:
            strategy_id: ID стратегии
            current_metrics: Текущие метрики (PF, WR, trades, etc.)
            current_params: Текущие параметры стратегии

        Returns:
            Словарь с рекомендациями по оптимизации
        """
        try:
            pf = current_metrics.get("profit_factor", 1.0)
            wr = current_metrics.get("win_rate", 0.0)
            trades = current_metrics.get("trades", 0)
            pnl = current_metrics.get("pnl", 0.0)
            dd = current_metrics.get("max_drawdown_pct", 0.0)

            # Создать промпт для оптимизации
            f"""
Ты - эксперт по оптимизации торговых стратегий.

АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ:
Стратегия: {strategy_id}
Profit Factor: {pf:.2f}
Win Rate: {wr:.1f}%
Количество сделок: {trades}
Total PnL: ${pnl:.2f}
Max Drawdown: {dd:.1f}%

ТЕКУЩИЕ ПАРАМЕТРЫ:
{json.dumps(current_params or {}, indent=2)}

ЗАДАЧА:
Проанализируй производительность и предложи оптимизацию параметров для увеличения Profit Factor.

ОБЛАСТИ ДЛЯ ОПТИМИЗАЦИИ:
1. Stop Loss - расстояние стоп-лосса
2. Take Profit - расстояние тейк-профита
3. Размер позиции - риск на сделку
4. Фильтры входа - условия для входа в позицию
5. Условия выхода - когда закрывать позицию

ФОРМАТ ОТВЕТА (JSON):
{{
    "analysis": "Анализ текущих проблем",
    "recommendations": [
        {{
            "parameter": "stop_loss_pct",
            "current_value": 0.5,
            "recommended_value": 0.7,
            "reason": "Объяснение почему"
        }}
    ],
    "expected_improvement": {{
        "profit_factor": 1.8,
        "win_rate": 55.0,
        "reasoning": "Объяснение ожидаемого улучшения"
    }}
}}
"""

            # Использовать LLM для генерации рекомендаций
            # Пока что возвращаем базовые рекомендации
            recommendations = self._generate_basic_recommendations(strategy_id, pf, wr, trades, dd)

            # Сохранить в историю
            self._optimization_history.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "strategy_id": strategy_id,
                    "current_metrics": current_metrics,
                    "recommendations": recommendations,
                }
            )

            return recommendations

        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}", exc_info=True)
            return {}

    def _generate_basic_recommendations(
        self,
        strategy_id: str,
        pf: float,
        wr: float,
        trades: int,
        dd: float,
    ) -> dict[str, Any]:
        """Генерировать базовые рекомендации без LLM."""
        recommendations = {
            "analysis": "",
            "recommendations": [],
            "expected_improvement": {},
        }

        # Анализ проблем
        issues = []
        if pf < 1.0:
            issues.append("Стратегия убыточна")
        if pf < 1.3:
            issues.append("Низкий Profit Factor")
        if wr < 45.0:
            issues.append("Низкий Win Rate")
        if dd > 10.0:
            issues.append("Высокий Drawdown")

        recommendations["analysis"] = "; ".join(issues) if issues else "Производительность в норме"

        # Рекомендации
        if pf < 1.2:
            # Увеличить stop_loss, чтобы меньше сделок закрывалось по стопу
            recommendations["recommendations"].append(
                {
                    "parameter": "stop_loss_pct",
                    "current_value": 0.5,
                    "recommended_value": 0.7,
                    "reason": "Увеличение stop_loss снизит количество преждевременных выходов",
                }
            )

            # Увеличить take_profit для лучшего соотношения риск/прибыль
            recommendations["recommendations"].append(
                {
                    "parameter": "take_profit_pct",
                    "current_value": 1.0,
                    "recommended_value": 2.0,
                    "reason": "Увеличение take_profit улучшит соотношение риск/прибыль",
                }
            )

        if wr < 45.0:
            # Улучшить фильтры входа
            recommendations["recommendations"].append(
                {
                    "parameter": "entry_filter_strength",
                    "current_value": "medium",
                    "recommended_value": "high",
                    "reason": "Более строгие фильтры входа увеличат Win Rate",
                }
            )

        if dd > 10.0:
            # Уменьшить размер позиции
            recommendations["recommendations"].append(
                {
                    "parameter": "position_size_multiplier",
                    "current_value": 1.0,
                    "recommended_value": 0.7,
                    "reason": "Уменьшение размера позиции снизит Drawdown",
                }
            )

        # Ожидаемое улучшение
        if recommendations["recommendations"]:
            recommendations["expected_improvement"] = {
                "profit_factor": min(pf * 1.3, 2.5),
                "win_rate": min(wr * 1.1, 65.0),
                "reasoning": "Применение рекомендаций должно улучшить метрики",
            }

        return recommendations

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Получить историю оптимизаций."""
        return self._optimization_history.copy()
