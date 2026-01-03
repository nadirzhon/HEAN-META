"""Auto-improvement catalyst system using LLM for continuous optimization."""

import asyncio
from datetime import datetime, timedelta
from typing import Any

from hean.agent_generation.generator import AgentGenerator
from hean.logging import get_logger
from hean.observability.metrics import metrics
from hean.portfolio.accounting import PortfolioAccounting

logger = get_logger(__name__)


class ImprovementCatalyst:
    """Автономная система улучшения проекта через LLM."""

    def __init__(
        self,
        accounting: PortfolioAccounting,
        strategies: dict[str, Any],
        check_interval_minutes: int = 30,
        min_trades_for_analysis: int = 10,
    ) -> None:
        """Инициализация катализатора.

        Args:
            accounting: Система учета портфеля
            strategies: Словарь стратегий {strategy_id: strategy_instance}
            check_interval_minutes: Интервал проверки в минутах
            min_trades_for_analysis: Минимум сделок для анализа
        """
        self._accounting = accounting
        self._strategies = strategies
        self._check_interval = timedelta(minutes=check_interval_minutes)
        self._min_trades = min_trades_for_analysis
        self._generator = AgentGenerator()
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._improvement_history: list[dict[str, Any]] = []
        self._last_check: datetime | None = None
        self._optimization_results: dict[str, dict[str, Any]] = {}

    async def start(self) -> None:
        """Запустить катализатор."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Improvement Catalyst started")

    async def stop(self) -> None:
        """Остановить катализатор."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Improvement Catalyst stopped")

    async def _monitor_loop(self) -> None:
        """Основной цикл мониторинга."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval.total_seconds())
                await self._analyze_and_improve()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in catalyst loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retry

    async def _analyze_and_improve(self) -> None:
        """Анализ и улучшение."""
        logger.info("Catalyst: Starting analysis cycle")

        # Собрать метрики
        performance_data = self._collect_performance_data()

        if not self._should_analyze(performance_data):
            logger.debug("Catalyst: Not enough data for analysis")
            return

        # Выявить проблемы
        problems = self._identify_problems(performance_data)

        if not problems:
            logger.debug("Catalyst: No problems identified")
            # Но все равно можем оптимизировать
            await self._optimize_parameters(performance_data)
            return

        # Генерировать улучшения для каждой проблемы
        for problem in problems:
            await self._generate_improvement(problem, performance_data)

        # Оптимизация параметров
        await self._optimize_parameters(performance_data)

        # Оптимизация распределения капитала
        await self._optimize_capital_allocation(performance_data)

    def _collect_performance_data(self) -> dict[str, Any]:
        """Собрать данные о производительности."""
        strategy_metrics = self._accounting.get_strategy_metrics()
        system_metrics = metrics.get_summary()
        equity = self._accounting.get_equity()
        drawdown, drawdown_pct = self._accounting.get_drawdown(equity)

        # Собрать данные по каждой стратегии
        strategy_details = {}
        for strategy_id, strategy_obj in self._strategies.items():
            if hasattr(strategy_obj, "get_metrics"):
                try:
                    strategy_details[strategy_id] = strategy_obj.get_metrics()
                except Exception as e:
                    logger.debug(f"Could not get metrics for {strategy_id}: {e}")

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "equity": equity,
            "drawdown": drawdown,
            "drawdown_pct": drawdown_pct,
            "strategy_metrics": strategy_metrics or {},
            "strategy_details": strategy_details,
            "system_metrics": system_metrics,
            "total_trades": sum(m.get("trades", 0) for m in (strategy_metrics or {}).values()),
        }

    def _should_analyze(self, data: dict[str, Any]) -> bool:
        """Проверить, достаточно ли данных для анализа."""
        total_trades = data.get("total_trades", 0)
        return total_trades >= self._min_trades

    def _identify_problems(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Выявить проблемы в производительности."""
        problems = []
        strategy_metrics = data.get("strategy_metrics", {})

        # Анализ каждой стратегии
        for strategy_id, metrics_data in strategy_metrics.items():
            pf = metrics_data.get("profit_factor", 1.0)
            wr = metrics_data.get("win_rate", 0.0)
            trades = metrics_data.get("trades", 0)
            pnl = metrics_data.get("pnl", 0.0)
            dd = metrics_data.get("max_drawdown_pct", 0.0)

            # Проблема: низкий Profit Factor
            if pf < 1.2 and trades >= 5:
                problems.append(
                    {
                        "type": "low_profit_factor",
                        "strategy_id": strategy_id,
                        "severity": "high" if pf < 1.0 else "medium",
                        "current_pf": pf,
                        "trades": trades,
                        "description": f"Strategy {strategy_id} has low profit factor {pf:.2f}",
                    }
                )

            # Проблема: низкий Win Rate
            if wr < 45.0 and trades >= 10:
                problems.append(
                    {
                        "type": "low_win_rate",
                        "strategy_id": strategy_id,
                        "severity": "medium",
                        "current_wr": wr,
                        "trades": trades,
                        "description": f"Strategy {strategy_id} has low win rate {wr:.1f}%",
                    }
                )

            # Проблема: высокий Drawdown
            if dd > 10.0:
                problems.append(
                    {
                        "type": "high_drawdown",
                        "strategy_id": strategy_id,
                        "severity": "high" if dd > 15.0 else "medium",
                        "current_dd": dd,
                        "description": f"Strategy {strategy_id} has high drawdown {dd:.1f}%",
                    }
                )

            # Проблема: убыточность
            if pnl < 0 and trades >= 5:
                problems.append(
                    {
                        "type": "losing_strategy",
                        "strategy_id": strategy_id,
                        "severity": "high",
                        "current_pnl": pnl,
                        "trades": trades,
                        "description": f"Strategy {strategy_id} is losing money: ${pnl:.2f}",
                    }
                )

        # Системные проблемы
        drawdown_pct = data.get("drawdown_pct", 0.0)
        if drawdown_pct > 10.0:
            problems.append(
                {
                    "type": "system_high_drawdown",
                    "severity": "high",
                    "current_dd": drawdown_pct,
                    "description": f"System-wide drawdown is high: {drawdown_pct:.1f}%",
                }
            )

        return problems

    async def _generate_improvement(
        self, problem: dict[str, Any], performance_data: dict[str, Any]
    ) -> None:
        """Генерировать улучшение для проблемы."""
        try:
            logger.info(f"Catalyst: Generating improvement for {problem['type']}")

            # Получить код стратегии, если возможно
            strategy_id = problem.get("strategy_id")
            if strategy_id and strategy_id in self._strategies:
                # Попытаться получить исходный код стратегии
                strategy_obj = self._strategies[strategy_id]
                if hasattr(strategy_obj, "__file__"):
                    try:
                        with open(strategy_obj.__file__) as f:
                            _ = f.read()  # Read but don't use for now
                    except Exception:
                        pass

            # Собрать контекст для промпта
            _ = {
                "problem": problem,
                "performance_data": performance_data,
                "strategy_metrics": performance_data.get("strategy_metrics", {}).get(
                    strategy_id, {}
                )
                if strategy_id
                else {},
            }

            # Использовать LLM для генерации улучшения
            # Пока что логируем, в будущем можно генерировать код
            logger.info(f"Catalyst: Problem identified - {problem['description']}")
            logger.info("Catalyst: Would generate improvement using LLM")

            # Сохранить в историю
            self._improvement_history.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "problem": problem,
                    "status": "identified",
                }
            )

        except Exception as e:
            logger.error(f"Error generating improvement: {e}", exc_info=True)

    async def _optimize_parameters(self, performance_data: dict[str, Any]) -> None:
        """Оптимизировать параметры стратегий."""
        try:
            strategy_metrics = performance_data.get("strategy_metrics", {})

            for strategy_id, metrics_data in strategy_metrics.items():
                if strategy_id not in self._strategies:
                    continue

                pf = metrics_data.get("profit_factor", 1.0)
                wr = metrics_data.get("win_rate", 0.0)
                trades = metrics_data.get("trades", 0)

                # Оптимизировать только если есть достаточно данных
                if trades < 5:
                    continue

                # Если PF низкий, предложить оптимизацию
                if pf < 1.5:
                    logger.info(f"Catalyst: Optimizing parameters for {strategy_id} (PF={pf:.2f})")

                    # Создать промпт для оптимизации
                    f"""
Проанализируй производительность стратегии и предложи оптимизацию параметров.

СТРАТЕГИЯ: {strategy_id}
ТЕКУЩИЕ МЕТРИКИ:
- Profit Factor: {pf:.2f}
- Win Rate: {wr:.1f}%
- Количество сделок: {trades}
- PnL: ${metrics_data.get("pnl", 0):.2f}
- Max Drawdown: {metrics_data.get("max_drawdown_pct", 0):.1f}%

ПРОБЛЕМА: Profit Factor ниже оптимального (< 1.5)

ЗАДАЧА: Предложи конкретные изменения параметров (stop_loss, take_profit, размер позиции, фильтры входа), которые могут улучшить Profit Factor.

ФОРМАТ ОТВЕТА:
1. Текущие проблемы в параметрах
2. Конкретные предложения по изменению параметров
3. Ожидаемое улучшение метрик
"""

                    # Здесь можно использовать LLM для генерации рекомендаций
                    logger.info(f"Catalyst: Would optimize {strategy_id} parameters using LLM")

                    # Сохранить результат
                    self._optimization_results[strategy_id] = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "current_pf": pf,
                        "current_wr": wr,
                        "status": "optimization_pending",
                    }

        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}", exc_info=True)

    async def _optimize_capital_allocation(self, performance_data: dict[str, Any]) -> None:
        """Оптимизировать распределение капитала между стратегиями."""
        try:
            strategy_metrics = performance_data.get("strategy_metrics", {})

            if len(strategy_metrics) < 2:
                return  # Нужно минимум 2 стратегии для оптимизации распределения

            # Найти лучшие и худшие стратегии
            strategy_performance = []
            for strategy_id, metrics_data in strategy_metrics.items():
                pf = metrics_data.get("profit_factor", 1.0)
                trades = metrics_data.get("trades", 0)
                if trades >= 5:  # Минимум данных
                    strategy_performance.append(
                        {
                            "strategy_id": strategy_id,
                            "pf": pf,
                            "pnl": metrics_data.get("pnl", 0),
                            "trades": trades,
                        }
                    )

            if len(strategy_performance) < 2:
                return

            # Сортировать по PF
            strategy_performance.sort(key=lambda x: x["pf"], reverse=True)

            best = strategy_performance[0]
            worst = strategy_performance[-1]

            # Если разница значительная, предложить перераспределение
            if best["pf"] > worst["pf"] * 1.5:
                logger.info(
                    f"Catalyst: Capital allocation optimization: "
                    f"Best={best['strategy_id']} (PF={best['pf']:.2f}), "
                    f"Worst={worst['strategy_id']} (PF={worst['pf']:.2f})"
                )

                # Создать рекомендацию
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "rebalance",
                    "increase": best["strategy_id"],
                    "decrease": worst["strategy_id"],
                    "reason": f"PF difference: {best['pf']:.2f} vs {worst['pf']:.2f}",
                }

                logger.info(
                    f"Catalyst: Recommendation - increase {best['strategy_id']} allocation, decrease {worst['strategy_id']}"
                )

        except Exception as e:
            logger.error(f"Error optimizing capital allocation: {e}", exc_info=True)

    def get_improvement_history(self) -> list[dict[str, Any]]:
        """Получить историю улучшений."""
        return self._improvement_history.copy()

    def get_optimization_results(self) -> dict[str, dict[str, Any]]:
        """Получить результаты оптимизации."""
        return self._optimization_results.copy()
