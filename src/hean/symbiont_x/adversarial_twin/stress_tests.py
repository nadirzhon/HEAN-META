"""
Stress Tests - Стресс-тесты для стратегий

Проверяют робастность в экстремальных условиях
"""

from dataclasses import dataclass
from enum import Enum


class StressTestType(Enum):
    """Типы стресс-тестов"""
    FLASH_CRASH = "flash_crash"          # Внезапный обвал
    FLASH_PUMP = "flash_pump"            # Внезапный памп
    THIN_LIQUIDITY = "thin_liquidity"    # Тонкая ликвидность
    HIGH_VOLATILITY = "high_volatility"  # Высокая волатильность
    NEWS_SHOCK = "news_shock"            # Новостной шок
    TREND_REVERSAL = "trend_reversal"    # Разворот тренда
    CHOPPY_MARKET = "choppy_market"      # Рубленый рынок
    LOW_VOLUME = "low_volume"            # Низкий объём
    EXCHANGE_OUTAGE = "exchange_outage"  # Сбой биржи
    API_LATENCY = "api_latency"          # Задержки API


@dataclass
class StressTestResult:
    """Результат стресс-теста"""

    test_type: StressTestType
    strategy_id: str

    # Did it survive?
    survived: bool
    failure_reason: str = ""

    # Impact metrics
    max_loss_during_stress: float = 0.0
    recovery_time_seconds: float = 0.0
    trades_during_stress: int = 0
    bad_trades_during_stress: int = 0

    # Response
    triggered_stop_loss: bool = False
    triggered_safe_mode: bool = False
    circuit_breaker_activated: bool = False

    # Flag for simulated results
    is_simulated: bool = False

    def get_robustness_score(self) -> float:
        """
        Оценка робастности (0-1)

        1.0 = идеально пережил стресс
        0.0 = полный провал
        """

        if not self.survived:
            return 0.0

        # If simulated, return 0 to not promote untested strategies
        if self.is_simulated:
            return 0.0

        # Component scores
        loss_score = 1.0 - min(abs(self.max_loss_during_stress) / 20.0, 1.0)  # Max 20% loss acceptable

        recovery_score = 1.0
        if self.recovery_time_seconds > 0:
            recovery_score = max(0.0, 1.0 - self.recovery_time_seconds / 3600)  # 1 hour max

        trade_quality_score = 1.0
        if self.trades_during_stress > 0:
            trade_quality_score = 1.0 - (self.bad_trades_during_stress / self.trades_during_stress)

        safety_score = 1.0 if (self.triggered_safe_mode or self.triggered_stop_loss) else 0.8

        # Weighted
        robustness = (
            loss_score * 0.4 +
            recovery_score * 0.2 +
            trade_quality_score * 0.3 +
            safety_score * 0.1
        )

        return robustness


class StressTestSuite:
    """
    Набор стресс-тестов

    Проверяет стратегию во всех экстремальных сценариях
    """

    def __init__(self):
        self.test_results: list[StressTestResult] = []

    def run_all_tests(self, strategy_config: dict) -> list[StressTestResult]:
        """Запускает все стресс-тесты"""

        strategy_id = strategy_config.get('strategy_id', 'unknown')

        results = []

        # Run each test
        for test_type in StressTestType:
            result = self._run_stress_test(strategy_config, test_type)
            results.append(result)

        self.test_results.extend(results)

        return results

    def _run_stress_test(
        self,
        strategy_config: dict,
        test_type: StressTestType
    ) -> StressTestResult:
        """Run a single stress test against strategy configuration."""

        strategy_id = strategy_config.get('strategy_id', 'unknown')
        stop_loss_pct = strategy_config.get('stop_loss_pct', 2.0)
        max_leverage = strategy_config.get('max_leverage', 5)
        max_position_pct = strategy_config.get('max_position_pct', 10.0)

        # Define stress magnitude for each test type
        stress_magnitudes = {
            StressTestType.FLASH_CRASH: -15.0,       # 15% crash
            StressTestType.FLASH_PUMP: 15.0,          # 15% pump
            StressTestType.THIN_LIQUIDITY: -5.0,      # 5% with slippage
            StressTestType.HIGH_VOLATILITY: -8.0,     # 8% swing
            StressTestType.NEWS_SHOCK: -10.0,         # 10% shock
            StressTestType.TREND_REVERSAL: -7.0,      # 7% reversal
            StressTestType.CHOPPY_MARKET: -3.0,       # 3% chop (repeated)
            StressTestType.LOW_VOLUME: -4.0,          # 4% with poor fills
            StressTestType.EXCHANGE_OUTAGE: -5.0,     # 5% during outage
            StressTestType.API_LATENCY: -3.0,         # 3% with delayed execution
        }

        magnitude = stress_magnitudes.get(test_type, -10.0)
        max_loss = abs(magnitude) * max_leverage * (max_position_pct / 100.0)

        # Check if stop loss would protect
        triggered_stop_loss = abs(magnitude) >= stop_loss_pct
        survived = max_loss < 20.0  # Survive if loss < 20% of capital

        # Exchange outage: can't execute stop loss
        if test_type == StressTestType.EXCHANGE_OUTAGE:
            triggered_stop_loss = False
            survived = max_loss < 30.0

        # Choppy market: repeated small losses
        if test_type == StressTestType.CHOPPY_MARKET:
            max_loss = abs(magnitude) * max_leverage * (max_position_pct / 100.0) * 5  # 5 rounds
            survived = max_loss < 20.0

        return StressTestResult(
            test_type=test_type,
            strategy_id=strategy_id,
            survived=survived,
            failure_reason="" if survived else f"Max loss {max_loss:.1f}% exceeds 20% threshold",
            max_loss_during_stress=max_loss,
            recovery_time_seconds=300.0 if survived else 0.0,
            trades_during_stress=3 if test_type == StressTestType.CHOPPY_MARKET else 1,
            bad_trades_during_stress=0 if survived else 1,
            triggered_stop_loss=triggered_stop_loss,
            triggered_safe_mode=max_loss > 10.0,
            circuit_breaker_activated=max_loss > 15.0,
            is_simulated=False,
        )

    def get_overall_robustness(self) -> float:
        """
        Общая оценка робастности

        Среднее по всем стресс-тестам
        """

        if not self.test_results:
            return 0.0

        scores = [r.get_robustness_score() for r in self.test_results]
        return sum(scores) / len(scores)

    def get_failed_tests(self) -> list[StressTestResult]:
        """Возвращает проваленные тесты"""
        return [r for r in self.test_results if not r.survived]

    def get_statistics(self) -> dict:
        """Статистика по всем тестам"""

        if not self.test_results:
            return {}

        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.survived])
        failed_tests = total_tests - passed_tests

        pass_rate = passed_tests / total_tests if total_tests > 0 else 0

        avg_max_loss = sum(r.max_loss_during_stress for r in self.test_results) / total_tests
        avg_recovery_time = sum(r.recovery_time_seconds for r in self.test_results) / total_tests

        # Count simulated tests
        simulated_tests = len([r for r in self.test_results if r.is_simulated])

        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'simulated_tests': simulated_tests,
            'pass_rate': pass_rate,
            'overall_robustness': self.get_overall_robustness(),
            'avg_max_loss': avg_max_loss,
            'avg_recovery_time_seconds': avg_recovery_time,
            'implementation_status': 'stub' if simulated_tests == total_tests else 'partial',
            'worst_test': min(self.test_results, key=lambda r: r.get_robustness_score()).test_type.value if self.test_results else None,
            'best_test': max(self.test_results, key=lambda r: r.get_robustness_score()).test_type.value if self.test_results else None,
        }
