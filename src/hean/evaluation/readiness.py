"""Readiness evaluation gate for trading system."""

from dataclasses import dataclass
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result of readiness evaluation."""

    passed: bool
    criteria: dict[str, Any]
    recommendations: list[str]
    regime_results: dict[str, dict[str, float]]


class ReadinessEvaluator:
    """Evaluates system readiness for live trading."""

    def __init__(
        self,
        min_profit_factor: float = 1.3,
        max_drawdown_pct: float = 25.0,
        min_positive_regimes: int = 3,
        total_regimes: int = 4,
    ) -> None:
        """Initialize the readiness evaluator.

        Args:
            min_profit_factor: Minimum profit factor required
            max_drawdown_pct: Maximum drawdown percentage allowed
            min_positive_regimes: Minimum number of regimes with positive returns
            total_regimes: Total number of regimes to test
        """
        self._min_profit_factor = min_profit_factor
        self._max_drawdown_pct = max_drawdown_pct
        self._min_positive_regimes = min_positive_regimes
        self._total_regimes = total_regimes

    def evaluate(self, metrics: dict[str, Any]) -> EvaluationResult:
        """Evaluate system readiness based on backtest metrics.

        Args:
            metrics: Backtest metrics dictionary from BacktestMetrics.calculate()

        Returns:
            EvaluationResult with PASS/FAIL status and recommendations
        """
        criteria: dict[str, Any] = {}
        recommendations: list[str] = []
        regime_results: dict[str, dict[str, float]] = {}

        # Extract overall metrics
        profit_factor = metrics.get("profit_factor", 0.0)
        max_drawdown_pct = metrics.get("max_drawdown_pct", 100.0)
        metrics.get("total_return_pct", 0.0)

        # Check profit factor
        pf_passed = profit_factor >= self._min_profit_factor
        criteria["profit_factor"] = {
            "value": profit_factor,
            "threshold": self._min_profit_factor,
            "passed": pf_passed,
        }
        if not pf_passed:
            recommendations.append(
                f"Profit Factor {profit_factor:.2f} below threshold {self._min_profit_factor:.2f}. "
                "Review strategy parameters and risk management."
            )

        # Check max drawdown
        dd_passed = max_drawdown_pct <= self._max_drawdown_pct
        criteria["max_drawdown"] = {
            "value": max_drawdown_pct,
            "threshold": self._max_drawdown_pct,
            "passed": dd_passed,
        }
        if not dd_passed:
            recommendations.append(
                f"Max Drawdown {max_drawdown_pct:.2f}% exceeds threshold {self._max_drawdown_pct:.2f}%. "
                "Reduce position sizes or tighten stop losses."
            )

        # Check regime performance
        regime_returns: dict[str, float] = {}
        if "strategies" in metrics:
            # Aggregate returns across all strategies per regime
            for _strategy_id, strat_metrics in metrics["strategies"].items():
                if "regime_pnl" in strat_metrics and isinstance(strat_metrics["regime_pnl"], dict):
                    for regime, pnl in strat_metrics["regime_pnl"].items():
                        if regime not in regime_returns:
                            regime_returns[regime] = 0.0
                        regime_returns[regime] += pnl

        # Also check overall regime performance if available
        # For now, we'll use strategy-level regime PnL
        positive_regimes = sum(1 for ret in regime_returns.values() if ret > 0)
        regimes_passed = positive_regimes >= self._min_positive_regimes

        criteria["regime_performance"] = {
            "positive_regimes": positive_regimes,
            "total_regimes": len(regime_returns) or self._total_regimes,
            "threshold": self._min_positive_regimes,
            "passed": regimes_passed,
            "regime_returns": regime_returns,
        }

        if not regimes_passed:
            negative_regimes = [regime for regime, ret in regime_returns.items() if ret <= 0]
            recommendations.append(
                f"Only {positive_regimes}/{len(regime_returns) or self._total_regimes} regimes show positive returns. "
                f"Negative regimes: {', '.join(negative_regimes) if negative_regimes else 'unknown'}. "
                "Consider regime-specific strategy adjustments."
            )

        # Store regime results
        regime_results = regime_returns.copy()

        # Overall result
        passed = pf_passed and dd_passed and regimes_passed

        if passed:
            recommendations.append("System ready for live trading.")

        return EvaluationResult(
            passed=passed,
            criteria=criteria,
            recommendations=recommendations,
            regime_results=regime_results,
        )

    def print_report(self, result: EvaluationResult) -> None:
        """Print evaluation report."""
        print("\n" + "=" * 80)
        print("READINESS EVALUATION")
        print("=" * 80)

        status = "PASS" if result.passed else "FAIL"
        print(f"\nStatus: {status}")

        print("\nCriteria:")
        for criterion, data in result.criteria.items():
            status_icon = "✓" if data.get("passed", False) else "✗"
            if criterion == "profit_factor":
                print(
                    f"  {status_icon} Profit Factor: {data['value']:.2f} "
                    f"(threshold: {data['threshold']:.2f})"
                )
            elif criterion == "max_drawdown":
                print(
                    f"  {status_icon} Max Drawdown: {data['value']:.2f}% "
                    f"(threshold: {data['threshold']:.2f}%)"
                )
            elif criterion == "regime_performance":
                print(
                    f"  {status_icon} Regime Performance: {data['positive_regimes']}/{data['total_regimes']} "
                    f"regimes positive (threshold: {data['threshold']})"
                )
                if data.get("regime_returns"):
                    print("    Regime Returns:")
                    for regime, ret in data["regime_returns"].items():
                        print(f"      {regime}: ${ret:,.2f}")

        if result.recommendations:
            print("\nRecommendations:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"  {i}. {rec}")

        print("=" * 80 + "\n")
