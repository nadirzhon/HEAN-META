"""Smart capital allocation optimization using LLM."""

from datetime import datetime
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)


class CapitalOptimizer:
    """Умное распределение капитала между стратегиями."""

    def __init__(self) -> None:
        """Инициализация оптимизатора капитала."""
        self._allocation_history: list[dict[str, Any]] = []

    def optimize_allocation(
        self,
        strategy_metrics: dict[str, dict[str, float]],
        current_weights: dict[str, float],
    ) -> dict[str, float]:
        """Оптимизировать распределение капитала.

        Args:
            strategy_metrics: Метрики по каждой стратегии
            current_weights: Текущие веса стратегий

        Returns:
            Оптимизированные веса
        """
        try:
            if not strategy_metrics or len(strategy_metrics) < 2:
                return current_weights

            # Рассчитать оптимальные веса на основе PF
            optimal_weights = {}
            total_score = 0.0

            for strategy_id, metrics in strategy_metrics.items():
                pf = metrics.get("profit_factor", 1.0)
                trades = metrics.get("trades", 0)
                dd = metrics.get("max_drawdown_pct", 0.0)

                # Минимум 5 сделок для учета
                if trades < 5:
                    # Использовать текущий вес
                    optimal_weights[strategy_id] = current_weights.get(strategy_id, 0.0)
                    continue

                # Оценка стратегии: PF * (1 - normalized_dd)
                # Нормализуем DD: 0% = 1.0, 20% = 0.0
                dd_penalty = max(0, 1.0 - (dd / 20.0))
                score = pf * dd_penalty

                optimal_weights[strategy_id] = score
                total_score += score

            # Нормализовать веса
            if total_score > 0:
                for strategy_id in optimal_weights:
                    optimal_weights[strategy_id] /= total_score
            else:
                # Если все стратегии плохие, использовать равные веса
                equal_weight = 1.0 / len(strategy_metrics)
                for strategy_id in strategy_metrics:
                    optimal_weights[strategy_id] = equal_weight

            # Ограничить максимальное изменение веса (макс 20% за раз)
            final_weights = {}
            for strategy_id in optimal_weights:
                current_weight = current_weights.get(strategy_id, 0.0)
                new_weight = optimal_weights[strategy_id]

                # Ограничить изменение
                max_change = 0.2
                if abs(new_weight - current_weight) > max_change:
                    if new_weight > current_weight:
                        new_weight = current_weight + max_change
                    else:
                        new_weight = current_weight - max_change

                final_weights[strategy_id] = max(0.0, min(1.0, new_weight))

            # Пере-нормализовать
            total = sum(final_weights.values())
            if total > 0:
                for strategy_id in final_weights:
                    final_weights[strategy_id] /= total

            # Сохранить в историю
            self._allocation_history.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "old_weights": current_weights.copy(),
                    "new_weights": final_weights.copy(),
                    "strategy_metrics": strategy_metrics.copy(),
                }
            )

            logger.info(f"Capital optimizer: Updated weights: {final_weights}")

            return final_weights

        except Exception as e:
            logger.error(f"Error optimizing capital allocation: {e}", exc_info=True)
            return current_weights

    def get_allocation_history(self) -> list[dict[str, Any]]:
        """Получить историю распределения капитала."""
        return self._allocation_history.copy()
