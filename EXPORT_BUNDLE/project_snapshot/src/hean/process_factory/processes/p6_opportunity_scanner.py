"""P6: Opportunity Scanner Wrapper."""

from hean.process_factory.schemas import (
    ActionStep,
    ActionStepKind,
    MeasurementSpec,
    ProcessDefinition,
    ProcessType,
    SafetyPolicy,
)


def get_process_definition() -> ProcessDefinition:
    """Get Opportunity Scanner process definition."""
    return ProcessDefinition(
        id="p6_opportunity_scanner",
        name="Opportunity Scanner Wrapper",
        type=ProcessType.ACCESS,
        description="Scan for new opportunities across trading, earn, campaigns, bonuses",
        requirements={"needs_bybit": True, "needs_ui": False},
        inputs_schema={},
        actions=[
            ActionStep(
                step_id="scan_trading_opportunities",
                kind=ActionStepKind.API_CALL,
                description="Scan for trading opportunities",
            ),
            ActionStep(
                step_id="scan_earn_opportunities",
                kind=ActionStepKind.API_CALL,
                description="Scan for earn opportunities",
            ),
            ActionStep(
                step_id="scan_campaign_opportunities",
                kind=ActionStepKind.HUMAN_TASK,
                description="Scan for campaign opportunities",
                params={"checklist": ["Check announcements", "Review eligibility", "Note deadlines"]},
            ),
            ActionStep(
                step_id="aggregate_opportunities",
                kind=ActionStepKind.COMPUTE,
                description="Aggregate and rank opportunities",
                depends_on=["scan_trading_opportunities", "scan_earn_opportunities", "scan_campaign_opportunities"],
            ),
        ],
        expected_outputs=["opportunities_list", "ranked_opportunities"],
        safety=SafetyPolicy(
            max_capital_usd=0.0,  # Read-only scanner
            require_manual_approval=False,
            max_risk_factor=1.0,
        ),
        measurement=MeasurementSpec(
            metrics=["time_hours"],
            attribution_rule="direct",
        ),
        kill_conditions=[],
    )

