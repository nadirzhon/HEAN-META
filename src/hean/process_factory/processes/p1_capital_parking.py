"""P1: Capital Parking (Earn-like placeholder)."""

from hean.process_factory.schemas import (
    ActionStep,
    ActionStepKind,
    KillCondition,
    MeasurementSpec,
    ProcessDefinition,
    ProcessType,
    SafetyPolicy,
)


def get_process_definition() -> ProcessDefinition:
    """Get Capital Parking process definition."""
    return ProcessDefinition(
        id="p1_capital_parking",
        name="Capital Parking",
        type=ProcessType.EARN,
        description="Placeholder process for capital parking/earning opportunities",
        requirements={"needs_bybit": False, "needs_ui": False},
        inputs_schema={},
        actions=[
            ActionStep(
                step_id="check_earn_availability",
                kind=ActionStepKind.HUMAN_TASK,
                description="Check available Earn products on Bybit",
                params={"checklist": ["Navigate to Earn section", "Check available products", "Note interest rates"]},
            ),
            ActionStep(
                step_id="evaluate_options",
                kind=ActionStepKind.COMPUTE,
                description="Evaluate earn options based on risk/reward",
            ),
        ],
        expected_outputs=["earn_options", "recommended_allocation"],
        safety=SafetyPolicy(
            max_capital_usd=1000.0,
            require_manual_approval=True,
            max_risk_factor=1.0,
        ),
        measurement=MeasurementSpec(
            metrics=["capital_delta", "time_hours", "roi"],
            attribution_rule="direct",
        ),
        kill_conditions=[
            KillCondition(metric="fail_rate", threshold=0.5, comparison=">", window_runs=5),
        ],
    )

