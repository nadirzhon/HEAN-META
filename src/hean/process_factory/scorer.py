"""Scoring and ranking functions for opportunities."""

from typing import Callable

from hean.process_factory.schemas import Opportunity


def score_opportunity(
    opportunity: Opportunity,
    confidence_calibration: Callable[[Opportunity], float] | None = None,
) -> float:
    """Score an opportunity for ranking.

    Score formula: (expected_profit_usd * confidence) / max(1e-6, time_hours * risk_factor * complexity)

    Args:
        opportunity: Opportunity to score
        confidence_calibration: Optional function to calibrate confidence based on historical outcomes

    Returns:
        Score (higher is better)
    """
    confidence = opportunity.confidence
    if confidence_calibration:
        confidence = confidence_calibration(opportunity)

    numerator = opportunity.expected_profit_usd * confidence
    denominator = max(1e-6, opportunity.time_hours * opportunity.risk_factor * opportunity.complexity)

    return numerator / denominator


def rank_opportunities(
    opportunities: list[Opportunity],
    confidence_calibration: Callable[[Opportunity], float] | None = None,
    min_score: float = 0.0,
) -> list[tuple[Opportunity, float]]:
    """Rank opportunities by score.

    Args:
        opportunities: List of opportunities to rank
        confidence_calibration: Optional function to calibrate confidence
        min_score: Minimum score threshold (opportunities below this are filtered)

    Returns:
        List of (opportunity, score) tuples sorted by score descending
    """
    scored = [
        (opp, score_opportunity(opp, confidence_calibration))
        for opp in opportunities
    ]
    filtered = [(opp, score) for opp, score in scored if score >= min_score]
    return sorted(filtered, key=lambda x: x[1], reverse=True)


def create_confidence_calibration(
    historical_outcomes: dict[str, dict[str, float]],
) -> Callable[[Opportunity], float]:
    """Create a confidence calibration function based on historical outcomes.

    Args:
        historical_outcomes: Dict mapping process_id or source to outcome stats:
            {"process_id": {"actual_profit": X, "expected_profit": Y, "success_rate": Z}}

    Returns:
        Calibration function that adjusts confidence based on historical performance
    """
    def calibrate(opportunity: Opportunity) -> float:
        """Calibrate confidence based on historical outcomes."""
        base_confidence = opportunity.confidence

        # If we have historical data for this process or source, adjust confidence
        key = opportunity.process_id or opportunity.source.value
        if key in historical_outcomes:
            stats = historical_outcomes[key]
            success_rate = stats.get("success_rate", 0.5)
            if stats.get("expected_profit", 0) > 0:
                accuracy_ratio = stats.get("actual_profit", 0) / stats["expected_profit"]
                # Adjust confidence based on historical accuracy and success rate
                adjusted = base_confidence * success_rate * min(accuracy_ratio, 2.0)
                return min(max(adjusted, 0.0), 1.0)

        return base_confidence

    return calibrate

