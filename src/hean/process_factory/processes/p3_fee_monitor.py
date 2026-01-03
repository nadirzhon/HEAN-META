"""P3: Fee/Slippage Monitor."""

from hean.process_factory.schemas import (
    ActionStep,
    ActionStepKind,
    MeasurementSpec,
    ProcessDefinition,
    ProcessType,
    SafetyPolicy,
)


def get_process_definition() -> ProcessDefinition:
    """Get Fee Monitor process definition."""
    return ProcessDefinition(
        id="p3_fee_monitor",
        name="Fee/Slippage Monitor",
        type=ProcessType.DATA,
        description="Monitor trading fees and slippage for execution quality tracking",
        requirements={"needs_bybit": True, "needs_ui": False},
        inputs_schema={},
        actions=[
            ActionStep(
                step_id="fetch_fee_structure",
                kind=ActionStepKind.API_CALL,
                description="Fetch current fee structure from Bybit",
            ),
            ActionStep(
                step_id="analyze_recent_trades",
                kind=ActionStepKind.COMPUTE,
                description="Analyze recent trades for fee impact",
            ),
            ActionStep(
                step_id="calculate_slippage_metrics",
                kind=ActionStepKind.COMPUTE,
                description="Calculate slippage metrics",
                depends_on=["fetch_fee_structure", "analyze_recent_trades"],
            ),
        ],
        expected_outputs=["fee_structure", "slippage_metrics", "recommendations"],
        safety=SafetyPolicy(
            max_capital_usd=0.0,  # Read-only
            require_manual_approval=False,
            max_risk_factor=1.0,
        ),
        measurement=MeasurementSpec(
            metrics=["time_hours"],
            attribution_rule="direct",
        ),
        kill_conditions=[],
    )

