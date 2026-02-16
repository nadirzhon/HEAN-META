"""Tests for truth_layer diagnosis logic."""

from hean.evaluation.readiness import EvaluationResult
from hean.evaluation.truth_layer import TruthDiagnosis, analyze_truth
from hean.observability.no_trade_report import NoTradeSummary


def _make_no_trade(
    totals: dict[str, int] | None = None,
) -> NoTradeSummary:
    return NoTradeSummary(
        totals=totals or {},
        per_strategy={},
        per_symbol={},
        pipeline_counters={},
        pipeline_per_strategy={},
    )


def _make_readiness(
    passed: bool,
    positive_regimes: int = 0,
    total_regimes: int = 4,
    regime_returns: dict[str, float] | None = None,
) -> EvaluationResult:
    criteria = {
        "regime_performance": {
            "positive_regimes": positive_regimes,
            "total_regimes": total_regimes,
            "threshold": 1,
            "passed": positive_regimes >= 1,
            "regime_returns": regime_returns or {},
        }
    }
    return EvaluationResult(
        passed=passed,
        criteria=criteria,
        recommendations=[],
        regime_results=regime_returns or {},
    )


def test_no_edge_present_primary_reason() -> None:
    """Flat PF ~1 and near-zero returns should be classified as 'No Edge Present'."""
    metrics = {
        "total_trades": 50,
        "profit_factor": 1.02,
        "total_return_pct": 1.0,
    }
    readiness = _make_readiness(passed=False)
    no_trade = _make_no_trade()

    diagnosis: TruthDiagnosis = analyze_truth(metrics, no_trade, readiness)

    assert diagnosis.primary_reason == "No Edge Present"
    assert "No Edge Present" in diagnosis.signals["hypothesis_scores"]
    assert diagnosis.signals["hypothesis_scores"]["No Edge Present"] > 0.5


def test_over_filtering_primary_reason() -> None:
    """Low filter pass rate with many filter rejects should yield 'Over-Filtering'."""
    metrics = {
        "strategies": {
            "impulse_engine": {
                "impulse_engine_metrics": {
                    "filter_pass_rate_pct": 10.0,
                    "filter_blocked_count": 100.0,
                }
            }
        }
    }
    no_trade = _make_no_trade(
        totals={
            "filter_reject": 50,
        }
    )
    readiness = _make_readiness(passed=False)

    diagnosis = analyze_truth(metrics, no_trade, readiness)

    assert diagnosis.primary_reason == "Over-Filtering"
    assert diagnosis.signals["hypothesis_scores"]["Over-Filtering"] > 0.5


def test_bad_regime_mix_primary_reason() -> None:
    """Concentrated losses in a single regime should be flagged as 'Bad Regime Mix'."""
    metrics = {
        "initial_equity": 100_000.0,
        "strategies": {
            "s1": {
                "regime_pnl": {
                    "NORMAL": 5_000.0,
                    "TREND": -15_000.0,
                }
            }
        },
    }
    readiness = _make_readiness(
        passed=False,
        positive_regimes=1,
        total_regimes=4,
        regime_returns={"NORMAL": 5_000.0, "TREND": -15_000.0},
    )
    no_trade = _make_no_trade()

    diagnosis = analyze_truth(metrics, no_trade, readiness)

    assert diagnosis.primary_reason == "Bad Regime Mix"
    assert diagnosis.signals["hypothesis_scores"]["Bad Regime Mix"] > 0.5


def test_execution_inefficiency_primary_reason() -> None:
    """Poor maker fill rate and many taker fills should be 'Execution Inefficiency'."""
    metrics = {
        "total_return_pct": 3.0,
        "execution": {
            "maker_fill_rate_pct": 10.0,
            "maker_fills": 5,
            "taker_fills": 40,
            "total_fees": -500.0,
        },
    }
    readiness = _make_readiness(passed=False)
    no_trade = _make_no_trade()

    diagnosis = analyze_truth(metrics, no_trade, readiness)

    assert diagnosis.primary_reason == "Execution Inefficiency"
    assert diagnosis.signals["hypothesis_scores"]["Execution Inefficiency"] > 0.5


def test_success_path_reports_edge_present() -> None:
    """When readiness passes and no major pathologies, primary reason should be 'Edge Present'."""
    metrics = {
        "total_trades": 80,
        "profit_factor": 1.6,
        "total_return_pct": 25.0,
    }
    readiness = _make_readiness(passed=True, positive_regimes=3, total_regimes=4)
    no_trade = _make_no_trade()

    diagnosis = analyze_truth(metrics, no_trade, readiness)

    assert diagnosis.primary_reason == "Edge Present"
    # All failure hypotheses should have low scores in this happy path
    for label, score in diagnosis.signals["hypothesis_scores"].items():
        if label in {
            "No Edge Present",
            "Over-Filtering",
            "Bad Regime Mix",
            "Execution Inefficiency",
        }:
            assert score <= 0.7







