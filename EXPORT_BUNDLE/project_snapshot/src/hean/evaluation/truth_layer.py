"""Truth layer: structured self-explanation of why evaluation passes or fails.

This module takes:
    - aggregate backtest metrics (from BacktestMetrics.calculate())
    - readiness evaluation result (from ReadinessEvaluator.evaluate())
    - no-trade / block statistics (from NoTradeReport.get_summary())

and produces a qualitative, *deterministic* diagnosis of what most likely
explains success or failure.

The goal is not statistical perfection, but a clear, explainable story that
maps to a small vocabulary of canonical reasons:

    - "No Edge Present"
    - "Over-Filtering"
    - "Bad Regime Mix"
    - "Execution Inefficiency"

When the system passes readiness and no major pathologies are detected, the
truth layer will instead surface a success-oriented primary reason:

    - "Edge Present"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hean.evaluation.readiness import EvaluationResult
from hean.observability.no_trade_report import NoTradeSummary


@dataclass
class TruthDiagnosis:
    """Structured explanation produced by the truth layer."""

    primary_reason: str
    secondary_reasons: list[str] = field(default_factory=list)
    suggested_action: str = ""
    # Optional debug / introspection payload
    signals: dict[str, Any] = field(default_factory=dict)


def _safe_get(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = d.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_no_edge(metrics: dict[str, Any]) -> tuple[float, str | None]:
    """Score 'No Edge Present' hypothesis in [0, 1].

    Heuristics:
        - low total trades → cannot be confident (score small)
        - profit factor ~ 1 and small total_return_pct → strong no-edge signal
        - deep negative return → may be regime or execution instead
    """
    total_trades = int(metrics.get("total_trades", 0) or 0)
    profit_factor = _safe_get(metrics, "profit_factor", 1.0)
    total_return_pct = _safe_get(metrics, "total_return_pct", 0.0)

    if total_trades < 10:
        # Too little data to strongly claim no edge
        return 0.1, None

    pf_deviation = abs(profit_factor - 1.0)
    low_edge_band = pf_deviation < 0.15 and -5.0 <= total_return_pct <= 5.0
    if not low_edge_band:
        return 0.0, None

    score = 0.7
    if total_trades >= 50:
        score = 0.9

    explanation = (
        f"Profit factor {profit_factor:.2f} near 1.0 with "
        f"{total_return_pct:.1f}% total return over {total_trades} trades."
    )
    return score, explanation


def _score_over_filtering(
    metrics: dict[str, Any],
    no_trade: NoTradeSummary,
) -> tuple[float, str | None]:
    """Score 'Over-Filtering' hypothesis in [0, 1].

    Indicators:
        - low filter_pass_rate_pct from impulse engine
        - large number of filter_reject no-trade reasons
    """
    strategies = metrics.get("strategies") or {}
    impulse_metrics = {}
    if "impulse_engine" in strategies:
        strat = strategies["impulse_engine"]
        impulse_metrics = strat.get("impulse_engine_metrics") or {}

    filter_pass_rate = _safe_get(impulse_metrics, "filter_pass_rate_pct", 100.0)
    filter_blocked_count = _safe_get(impulse_metrics, "filter_blocked_count", 0.0)

    nt_filter_rejects = no_trade.totals.get("filter_reject", 0)
    nt_edge_rejects = no_trade.totals.get("edge_reject", 0)

    # Require some signals to even consider this
    if nt_filter_rejects + nt_edge_rejects + filter_blocked_count < 5:
        return 0.0, None

    score = 0.0
    if filter_pass_rate < 30.0:
        score += 0.5
    if filter_pass_rate < 15.0:
        score += 0.2

    if nt_filter_rejects > 20:
        score += 0.2
    elif nt_filter_rejects > 5:
        score += 0.1

    if score == 0.0:
        return 0.0, None

    explanation = (
        f"Impulse filter pass rate {filter_pass_rate:.1f}% with "
        f"{int(nt_filter_rejects)} signals rejected by filters."
    )
    return min(score, 1.0), explanation


def _aggregate_regime_pnl(metrics: dict[str, Any]) -> dict[str, float]:
    """Aggregate per-regime PnL across strategies if available."""
    regimes: dict[str, float] = {}
    strategies = metrics.get("strategies") or {}
    for strat_metrics in strategies.values():
        regime_pnl = strat_metrics.get("regime_pnl")
        if not isinstance(regime_pnl, dict):
            continue
        for regime, pnl in regime_pnl.items():
            regimes[regime] = regimes.get(regime, 0.0) + float(pnl)
    return regimes


def _score_bad_regime_mix(
    metrics: dict[str, Any],
    readiness: EvaluationResult,
) -> tuple[float, str | None]:
    """Score 'Bad Regime Mix' hypothesis in [0, 1].

    Indicators:
        - few positive regimes in readiness.criteria["regime_performance"]
        - strongly negative PnL in one or more regimes
    """
    criteria = readiness.criteria.get("regime_performance", {})
    positive_regimes = int(criteria.get("positive_regimes", 0) or 0)
    total_regimes = int(criteria.get("total_regimes", 0) or 0)

    regime_pnl = _aggregate_regime_pnl(metrics)
    worst_regime = None
    worst_pnl = 0.0
    for regime, pnl in regime_pnl.items():
        if pnl < worst_pnl:
            worst_pnl = pnl
            worst_regime = regime

    score = 0.0
    if total_regimes > 0 and positive_regimes < max(1, total_regimes // 2):
        score += 0.4
    if worst_pnl < -0.05 * _safe_get(metrics, "initial_equity", 1.0):
        score += 0.4

    if score == 0.0 or not worst_regime:
        return 0.0, None

    explanation = (
        f"Only {positive_regimes}/{total_regimes} regimes profitable; "
        f"worst regime '{worst_regime}' lost ${worst_pnl:,.2f}."
    )
    return min(score, 1.0), explanation


def _score_execution_inefficiency(metrics: dict[str, Any]) -> tuple[float, str | None]:
    """Score 'Execution Inefficiency' hypothesis in [0, 1].

    Indicators:
        - low maker_fill_rate_pct
        - very high taker_fills vs maker_fills
        - high total fee cost relative to PnL (if available)
    """
    exec_metrics = metrics.get("execution") or {}
    if not exec_metrics:
        return 0.0, None

    maker_fill_rate = _safe_get(exec_metrics, "maker_fill_rate_pct", 0.0)
    maker_fills = _safe_get(exec_metrics, "maker_fills", 0.0)
    taker_fills = _safe_get(exec_metrics, "taker_fills", 0.0)
    total_fees = _safe_get(exec_metrics, "total_fees", 0.0)
    total_return_pct = _safe_get(metrics, "total_return_pct", 0.0)

    score = 0.0
    if maker_fill_rate < 30.0:
        score += 0.4
    if taker_fills > maker_fills * 2 and taker_fills > 10:
        score += 0.3

    if abs(total_return_pct) < 10.0 and total_fees < 0:
        score += 0.2

    if score == 0.0:
        return 0.0, None

    explanation = (
        f"Maker fill rate {maker_fill_rate:.1f}% with "
        f"{int(taker_fills)} taker vs {int(maker_fills)} maker fills."
    )
    return min(score, 1.0), explanation


def analyze_truth(
    metrics: dict[str, Any],
    no_trade: NoTradeSummary,
    readiness: EvaluationResult,
) -> TruthDiagnosis:
    """Run truth-layer analysis and return structured diagnosis."""
    hypotheses: dict[str, tuple[float, str | None]] = {}

    # Core failure-oriented hypotheses
    hypotheses["No Edge Present"] = _score_no_edge(metrics)
    hypotheses["Over-Filtering"] = _score_over_filtering(metrics, no_trade)
    hypotheses["Bad Regime Mix"] = _score_bad_regime_mix(metrics, readiness)
    hypotheses["Execution Inefficiency"] = _score_execution_inefficiency(metrics)

    # Choose primary reason by highest score
    primary_reason = "Edge Present" if readiness.passed else "No Edge Present"
    primary_score = -1.0
    secondary_reasons: list[str] = []
    secondary_details: dict[str, str] = {}

    for label, (score, detail) in hypotheses.items():
        if score <= 0.0:
            continue
        if score > primary_score:
            primary_score = score
            primary_reason = label
        if detail:
            secondary_details[label] = detail

    # Secondary contributors: all non-primary hypotheses with non-trivial scores
    for label, (score, _) in hypotheses.items():
        if label == primary_reason or score <= 0.2:
            continue
        secondary_reasons.append(label)

    # Suggested next action mapped to primary
    if primary_reason == "No Edge Present":
        suggested = (
            "Run focused parameter search and hypothesis redesign; "
            "do not proceed to live until a statistically clear edge emerges."
        )
    elif primary_reason == "Over-Filtering":
        suggested = (
            "Relax the most aggressive filters and re-run evaluation, "
            "especially around impulse detection and regime gating."
        )
    elif primary_reason == "Bad Regime Mix":
        suggested = (
            "Down-weight or disable strategies in losing regimes and "
            "introduce regime-specific variants; re-evaluate per-regime PnL."
        )
    elif primary_reason == "Execution Inefficiency":
        suggested = (
            "Review order routing and maker/taker logic; target higher maker "
            "fill rates and lower fee drag in backtests."
        )
    else:  # Edge Present / success path
        suggested = (
            "Proceed to a small-capital live or paper deployment, while "
            "monitoring the highlighted secondary risk factors."
        )

    diagnosis = TruthDiagnosis(
        primary_reason=primary_reason,
        secondary_reasons=secondary_reasons,
        suggested_action=suggested,
        signals={
            "readiness_passed": readiness.passed,
            "hypothesis_scores": {k: v[0] for k, v in hypotheses.items()},
            "details": secondary_details,
        },
    )
    return diagnosis


def print_truth(diagnosis: TruthDiagnosis) -> None:
    """Pretty-print truth-layer diagnosis."""
    print("\n" + "=" * 80)
    print("TRUTH LAYER DIAGNOSIS")
    print("=" * 80)

    print(f"\nPRIMARY REASON: {diagnosis.primary_reason}")

    if diagnosis.secondary_reasons:
        print("\nSECONDARY CONTRIBUTORS:")
        for label in diagnosis.secondary_reasons:
            detail = diagnosis.signals.get("details", {}).get(label)
            if detail:
                print(f"  - {label}: {detail}")
            else:
                print(f"  - {label}")

    print("\nSUGGESTED NEXT ACTION:")
    print(f"  {diagnosis.suggested_action}")

    print("=" * 80 + "\n")
