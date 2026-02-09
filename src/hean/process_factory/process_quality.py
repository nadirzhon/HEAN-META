"""Process Quality Scorer: Score generated processes for acceptance.

Scores processes by:
- Measurability completeness
- Safety completeness
- Testability (clear kill conditions)
- Capital efficiency (time/risk)
"""


from pydantic import BaseModel, Field

from hean.process_factory.schemas import ProcessDefinition


class ProcessQualityScore(BaseModel):
    """Quality score for a process definition."""

    overall_score: float = Field(
        ..., ge=0, le=1, description="Overall quality score (0-1)"
    )
    measurability_score: float = Field(
        ..., ge=0, le=1, description="Measurability completeness (0-1)"
    )
    safety_score: float = Field(..., ge=0, le=1, description="Safety completeness (0-1)")
    testability_score: float = Field(
        ..., ge=0, le=1, description="Testability score (0-1)"
    )
    capital_efficiency_score: float = Field(
        ..., ge=0, le=1, description="Capital efficiency score (0-1)"
    )
    reasons: list[str] = Field(
        default_factory=list, description="Reasons for score (positive and negative)"
    )
    accepted: bool = Field(
        default=False, description="Whether process meets acceptance threshold"
    )


class ProcessQualityScorer:
    """Scores process definitions for quality and acceptance."""

    def __init__(
        self,
        acceptance_threshold: float = 0.6,
        measurability_weight: float = 0.3,
        safety_weight: float = 0.3,
        testability_weight: float = 0.2,
        capital_efficiency_weight: float = 0.2,
    ) -> None:
        """Initialize process quality scorer.

        Args:
            acceptance_threshold: Minimum score for acceptance (default 0.6)
            measurability_weight: Weight for measurability (default 0.3)
            safety_weight: Weight for safety (default 0.3)
            testability_weight: Weight for testability (default 0.2)
            capital_efficiency_weight: Weight for capital efficiency (default 0.2)
        """
        self.acceptance_threshold = acceptance_threshold
        self.measurability_weight = measurability_weight
        self.safety_weight = safety_weight
        self.testability_weight = testability_weight
        self.capital_efficiency_weight = capital_efficiency_weight

        # Normalize weights
        total_weight = (
            measurability_weight
            + safety_weight
            + testability_weight
            + capital_efficiency_weight
        )
        if total_weight > 0:
            self.measurability_weight /= total_weight
            self.safety_weight /= total_weight
            self.testability_weight /= total_weight
            self.capital_efficiency_weight /= total_weight

    def score(self, process: ProcessDefinition) -> ProcessQualityScore:
        """Score a process definition.

        Args:
            process: Process definition to score

        Returns:
            Quality score with acceptance decision
        """
        reasons: list[str] = []

        # Score measurability
        measurability_score, meas_reasons = self._score_measurability(process)
        reasons.extend(meas_reasons)

        # Score safety
        safety_score, safety_reasons = self._score_safety(process)
        reasons.extend(safety_reasons)

        # Score testability
        testability_score, test_reasons = self._score_testability(process)
        reasons.extend(test_reasons)

        # Score capital efficiency
        capital_efficiency_score, eff_reasons = self._score_capital_efficiency(process)
        reasons.extend(eff_reasons)

        # Compute overall score (weighted average)
        overall_score = (
            measurability_score * self.measurability_weight
            + safety_score * self.safety_weight
            + testability_score * self.testability_weight
            + capital_efficiency_score * self.capital_efficiency_weight
        )

        accepted = overall_score >= self.acceptance_threshold

        if accepted:
            reasons.append(f"Overall score {overall_score:.2f} meets threshold {self.acceptance_threshold}")
        else:
            reasons.append(
                f"Overall score {overall_score:.2f} below threshold {self.acceptance_threshold}"
            )

        return ProcessQualityScore(
            overall_score=overall_score,
            measurability_score=measurability_score,
            safety_score=safety_score,
            testability_score=testability_score,
            capital_efficiency_score=capital_efficiency_score,
            reasons=reasons,
            accepted=accepted,
        )

    def _score_measurability(
        self, process: ProcessDefinition
    ) -> tuple[float, list[str]]:
        """Score measurability completeness.

        Args:
            process: Process definition

        Returns:
            Tuple of (score, reasons)
        """
        score = 0.0
        reasons: list[str] = []

        # Check if measurement spec exists
        if not process.measurement:
            return 0.0, ["Missing measurement spec"]

        # Check metrics
        metrics = process.measurement.metrics or []
        required_metrics = ["capital_delta", "time_hours"]
        has_required = all(m in metrics for m in required_metrics)
        if has_required:
            score += 0.5
            reasons.append("Has required metrics (capital_delta, time_hours)")
        else:
            missing = [m for m in required_metrics if m not in metrics]
            reasons.append(f"Missing required metrics: {missing}")

        # Check for additional useful metrics
        useful_metrics = ["roi", "fail_rate", "fee_drag", "volatility_exposure"]
        has_useful = sum(1 for m in useful_metrics if m in metrics)
        score += min(has_useful / len(useful_metrics), 0.3)
        if has_useful > 0:
            reasons.append(f"Has {has_useful} useful metrics")

        # Check attribution rule
        if process.measurement.attribution_rule:
            score += 0.2
            reasons.append("Has attribution rule defined")
        else:
            reasons.append("Missing attribution rule")

        return min(score, 1.0), reasons

    def _score_safety(
        self, process: ProcessDefinition
    ) -> tuple[float, list[str]]:
        """Score safety completeness.

        Args:
            process: Process definition

        Returns:
            Tuple of (score, reasons)
        """
        score = 0.0
        reasons: list[str] = []

        # Check safety policy
        if not process.safety:
            return 0.0, ["Missing safety policy"]

        # Check max capital
        if process.safety.max_capital_usd > 0:
            score += 0.3
            reasons.append(f"Max capital limit: ${process.safety.max_capital_usd:.2f}")
        else:
            reasons.append("No max capital limit")

        # Check manual approval requirement
        if process.safety.require_manual_approval:
            score += 0.3
            reasons.append("Requires manual approval")
        else:
            reasons.append("No manual approval required (risk)")

        # Check risk factor
        if process.safety.max_risk_factor > 0:
            score += 0.2
            reasons.append(f"Risk factor limit: {process.safety.max_risk_factor}")
        else:
            reasons.append("No risk factor limit")

        # Check for dangerous actions
        dangerous_keywords = [
            "credential",
            "password",
            "api_key",
            "secret",
            "scrape",
            "selenium",
        ]
        has_dangerous = False
        for action in process.actions:
            desc = action.description.lower()
            if any(keyword in desc for keyword in dangerous_keywords):
                has_dangerous = True
                reasons.append(f"Dangerous action detected: {action.description[:50]}")
                break

        if has_dangerous:
            score *= 0.5  # Penalize for dangerous actions
        else:
            score += 0.2
            reasons.append("No dangerous actions detected")

        return min(score, 1.0), reasons

    def _score_testability(
        self, process: ProcessDefinition
    ) -> tuple[float, list[str]]:
        """Score testability (clear kill conditions).

        Args:
            process: Process definition

        Returns:
            Tuple of (score, reasons)
        """
        score = 0.0
        reasons: list[str] = []

        # Check kill conditions
        if not process.kill_conditions:
            return 0.0, ["Missing kill conditions (required for testability)"]

        # Score based on number and quality of kill conditions
        kill_count = len(process.kill_conditions)
        if kill_count >= 2:
            score += 0.5
            reasons.append(f"Has {kill_count} kill conditions")
        elif kill_count == 1:
            score += 0.3
            reasons.append("Has 1 kill condition (should have more)")
        else:
            reasons.append("No kill conditions")

        # Check if kill conditions are specific
        specific_conditions = 0
        for condition in process.kill_conditions:
            if condition.metric and condition.threshold is not None:
                specific_conditions += 1

        if specific_conditions > 0:
            score += 0.3
            reasons.append(f"{specific_conditions} specific kill conditions")
        else:
            reasons.append("Kill conditions lack specificity")

        # Check scale rules (optional but good for testability)
        if process.scale_rules:
            score += 0.2
            reasons.append(f"Has {len(process.scale_rules)} scale rules")
        else:
            reasons.append("No scale rules (optional)")

        return min(score, 1.0), reasons

    def _score_capital_efficiency(
        self, process: ProcessDefinition
    ) -> tuple[float, list[str]]:
        """Score capital efficiency (time/risk).

        Args:
            process: Process definition

        Returns:
            Tuple of (score, reasons)
        """
        score = 0.5  # Start with neutral score
        reasons: list[str] = []

        # Check time estimate (if available in description or actions)
        # Estimate based on number of actions and types
        action_count = len(process.actions)
        human_task_count = sum(
            1 for a in process.actions if a.kind.value == "HUMAN_TASK"
        )

        # More actions = potentially longer time, but also more complete
        if action_count <= 5:
            score += 0.2
            reasons.append("Reasonable action count")
        elif action_count <= 10:
            score += 0.1
            reasons.append("Moderate action count")
        else:
            reasons.append(f"High action count: {action_count} (may be inefficient)")

        # Human tasks are slower but sometimes necessary
        if human_task_count == 0:
            score += 0.1
            reasons.append("No human tasks (fully automatable)")
        elif human_task_count <= 2:
            score += 0.05
            reasons.append(f"Few human tasks: {human_task_count}")
        else:
            reasons.append(f"Many human tasks: {human_task_count} (may be slow)")

        # Check risk factor
        risk_factor = process.safety.max_risk_factor if process.safety else 3.0
        if risk_factor <= 2.0:
            score += 0.1
            reasons.append(f"Low risk factor: {risk_factor}")
        elif risk_factor <= 3.0:
            score += 0.05
            reasons.append(f"Moderate risk factor: {risk_factor}")
        else:
            reasons.append(f"High risk factor: {risk_factor}")

        # Check capital limit
        max_capital = process.safety.max_capital_usd if process.safety else 1000.0
        if max_capital <= 500:
            score += 0.1
            reasons.append(f"Low capital requirement: ${max_capital:.2f}")
        elif max_capital <= 2000:
            score += 0.05
            reasons.append(f"Moderate capital requirement: ${max_capital:.2f}")
        else:
            reasons.append(f"High capital requirement: ${max_capital:.2f}")

        return min(score, 1.0), reasons

