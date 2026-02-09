"""
Survival Score Calculator - Калькулятор финального survival score

Объединяет результаты всех тестовых миров и стресс-тестов
"""

from dataclasses import dataclass

from .stress_tests import StressTestResult
from .test_worlds import TestResult


@dataclass
class SurvivalScore:
    """
    Финальный Survival Score стратегии

    0-100, где:
    - 90-100: Excellent - готов к production
    - 75-89: Good - почти готов, требует минорных улучшений
    - 60-74: Acceptable - можно запустить с низким risk
    - 40-59: Poor - требует доработки
    - 0-39: Fail - не готов к deployment
    """

    strategy_id: str
    strategy_name: str

    # Overall score (0-100)
    overall_score: float

    # Component scores
    replay_score: float       # Replay world performance
    paper_score: float        # Paper trading performance
    micro_real_score: float   # Micro-real performance
    robustness_score: float   # Stress tests

    # Test results
    replay_result: TestResult
    paper_result: TestResult
    micro_real_result: TestResult
    stress_test_results: list[StressTestResult]

    # Flags
    ready_for_production: bool
    requires_improvement: bool
    critical_failures: list[str]

    def get_level(self) -> str:
        """Возвращает уровень готовности"""
        if self.overall_score >= 90:
            return "EXCELLENT"
        elif self.overall_score >= 75:
            return "GOOD"
        elif self.overall_score >= 60:
            return "ACCEPTABLE"
        elif self.overall_score >= 40:
            return "POOR"
        else:
            return "FAIL"

    def to_dict(self) -> dict:
        """Сериализация"""
        return {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'overall_score': self.overall_score,
            'level': self.get_level(),
            'component_scores': {
                'replay': self.replay_score,
                'paper': self.paper_score,
                'micro_real': self.micro_real_score,
                'robustness': self.robustness_score,
            },
            'ready_for_production': self.ready_for_production,
            'requires_improvement': self.requires_improvement,
            'critical_failures': self.critical_failures,
        }


class SurvivalScoreCalculator:
    """
    Калькулятор Survival Score

    Принимает результаты всех тестов, вычисляет финальный score
    """

    def __init__(self):
        # Weights for different test types
        self.weights = {
            'replay': 0.20,       # 20% - базовая валидация
            'paper': 0.30,        # 30% - реалистичное тестирование
            'micro_real': 0.35,   # 35% - реальные деньги (самое важное)
            'robustness': 0.15,   # 15% - стресс-тесты
        }

        # Minimum scores to pass
        self.min_scores = {
            'replay': 60,
            'paper': 55,
            'micro_real': 50,
            'robustness': 40,
        }

    def calculate(
        self,
        replay_result: TestResult,
        paper_result: TestResult,
        micro_real_result: TestResult,
        stress_test_results: list[StressTestResult],
    ) -> SurvivalScore:
        """
        Вычисляет финальный Survival Score

        Args:
            replay_result: Результат Replay World
            paper_result: Результат Paper World
            micro_real_result: Результат Micro-Real World
            stress_test_results: Результаты стресс-тестов

        Returns:
            SurvivalScore
        """

        # Calculate component scores (0-100)
        replay_score = self._score_test_result(replay_result) * 100
        paper_score = self._score_test_result(paper_result) * 100
        micro_real_score = self._score_test_result(micro_real_result) * 100
        robustness_score = self._score_stress_tests(stress_test_results) * 100

        # Weighted overall score
        overall_score = (
            replay_score * self.weights['replay'] +
            paper_score * self.weights['paper'] +
            micro_real_score * self.weights['micro_real'] +
            robustness_score * self.weights['robustness']
        )

        # Check if ready for production
        ready_for_production = self._check_ready_for_production(
            replay_score, paper_score, micro_real_score, robustness_score
        )

        # Check if requires improvement
        requires_improvement = overall_score < 75

        # Identify critical failures
        critical_failures = self._identify_critical_failures(
            replay_result, paper_result, micro_real_result, stress_test_results
        )

        return SurvivalScore(
            strategy_id=micro_real_result.strategy_id,
            strategy_name=micro_real_result.strategy_name,
            overall_score=overall_score,
            replay_score=replay_score,
            paper_score=paper_score,
            micro_real_score=micro_real_score,
            robustness_score=robustness_score,
            replay_result=replay_result,
            paper_result=paper_result,
            micro_real_result=micro_real_result,
            stress_test_results=stress_test_results,
            ready_for_production=ready_for_production,
            requires_improvement=requires_improvement,
            critical_failures=critical_failures,
        )

    def _score_test_result(self, result: TestResult) -> float:
        """
        Оценивает TestResult (0-1)

        Учитывает win rate, profit factor, drawdown, risk violations
        """

        if not result.passed:
            return 0.0

        # Use the get_survival_score method from TestResult
        return result.get_survival_score()

    def _score_stress_tests(self, stress_results: list[StressTestResult]) -> float:
        """
        Оценивает стресс-тесты (0-1)

        Среднее по всем тестам с акцентом на самый слабый
        """

        if not stress_results:
            return 0.0

        # Get individual scores
        scores = [r.get_robustness_score() for r in stress_results]

        if not scores:
            return 0.0

        # Average score
        avg_score = sum(scores) / len(scores)

        # Worst score (penalty for weak points)
        worst_score = min(scores)

        # Weighted: 70% average, 30% worst
        final_score = avg_score * 0.7 + worst_score * 0.3

        return final_score

    def _check_ready_for_production(
        self,
        replay_score: float,
        paper_score: float,
        micro_real_score: float,
        robustness_score: float
    ) -> bool:
        """
        Проверяет готовность к production

        Все минимумы должны быть достигнуты
        """

        return (
            replay_score >= self.min_scores['replay'] and
            paper_score >= self.min_scores['paper'] and
            micro_real_score >= self.min_scores['micro_real'] and
            robustness_score >= self.min_scores['robustness']
        )

    def _identify_critical_failures(
        self,
        replay_result: TestResult,
        paper_result: TestResult,
        micro_real_result: TestResult,
        stress_results: list[StressTestResult]
    ) -> list[str]:
        """Идентифицирует критические провалы"""

        failures = []

        # Check test world failures
        if not replay_result.passed:
            failures.append(f"Replay World: {replay_result.failure_reason}")

        if not paper_result.passed:
            failures.append(f"Paper World: {paper_result.failure_reason}")

        if not micro_real_result.passed:
            failures.append(f"Micro-Real World: {micro_real_result.failure_reason}")

        # Check stress test failures
        failed_stress = [r for r in stress_results if not r.survived]
        for failed in failed_stress:
            failures.append(f"Stress Test {failed.test_type.value}: {failed.failure_reason}")

        # Check specific red flags
        if micro_real_result.risk_violations > 0:
            failures.append(f"Risk violations: {micro_real_result.risk_violations}")

        if micro_real_result.max_drawdown_pct < -25:
            failures.append(f"Excessive drawdown: {micro_real_result.max_drawdown_pct:.1f}%")

        if micro_real_result.profit_factor < 1.0:
            failures.append(f"Poor profit factor: {micro_real_result.profit_factor:.2f}")

        return failures

    def batch_calculate(
        self,
        test_results: list[dict]
    ) -> list[SurvivalScore]:
        """
        Вычисляет scores для нескольких стратегий

        Args:
            test_results: Список результатов тестов
                [{
                    'replay': TestResult,
                    'paper': TestResult,
                    'micro_real': TestResult,
                    'stress_tests': List[StressTestResult]
                }]

        Returns:
            Список SurvivalScore
        """

        survival_scores = []

        for results in test_results:
            score = self.calculate(
                replay_result=results['replay'],
                paper_result=results['paper'],
                micro_real_result=results['micro_real'],
                stress_test_results=results['stress_tests'],
            )
            survival_scores.append(score)

        return survival_scores

    def rank_strategies(
        self,
        survival_scores: list[SurvivalScore]
    ) -> list[SurvivalScore]:
        """
        Ранжирует стратегии по survival score

        Returns отсортированный список (лучшие первые)
        """

        return sorted(
            survival_scores,
            key=lambda s: s.overall_score,
            reverse=True
        )

    def get_statistics(
        self,
        survival_scores: list[SurvivalScore]
    ) -> dict:
        """Статистика по группе стратегий"""

        if not survival_scores:
            return {}

        overall_scores = [s.overall_score for s in survival_scores]

        import statistics

        production_ready = len([s for s in survival_scores if s.ready_for_production])
        needs_improvement = len([s for s in survival_scores if s.requires_improvement])

        return {
            'total_strategies': len(survival_scores),
            'production_ready': production_ready,
            'production_ready_pct': (production_ready / len(survival_scores)) * 100,
            'needs_improvement': needs_improvement,
            'avg_score': statistics.mean(overall_scores),
            'median_score': statistics.median(overall_scores),
            'max_score': max(overall_scores),
            'min_score': min(overall_scores),
            'score_stdev': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
            'level_distribution': {
                'EXCELLENT': len([s for s in survival_scores if s.get_level() == 'EXCELLENT']),
                'GOOD': len([s for s in survival_scores if s.get_level() == 'GOOD']),
                'ACCEPTABLE': len([s for s in survival_scores if s.get_level() == 'ACCEPTABLE']),
                'POOR': len([s for s in survival_scores if s.get_level() == 'POOR']),
                'FAIL': len([s for s in survival_scores if s.get_level() == 'FAIL']),
            }
        }
