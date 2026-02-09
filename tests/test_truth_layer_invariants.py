"""Tests for Truth Layer invariants.

Tests:
1. total = pnl + funding + rewards - fees - opportunity_cost
2. no duplicate ledger entries per trade_id
3. report totals match ledger sums
"""

from datetime import datetime, timedelta

import pytest

from hean.process_factory.schemas import ProcessRun, ProcessRunStatus
from hean.process_factory.truth_layer import LedgerEntryType, TruthLayer


@pytest.fixture
def truth_layer():
    """Create TruthLayer instance."""
    return TruthLayer(
        risk_free_rate_apr=0.05,
        default_maker_fee_bps=0.5,
        default_taker_fee_bps=3.0,
        slippage_proxy_bps=1.0,
    )


@pytest.fixture
def sample_run():
    """Create a sample process run."""
    return ProcessRun(
        run_id="test_run_1",
        process_id="test_process",
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow() + timedelta(hours=1),
        status=ProcessRunStatus.COMPLETED,
        metrics={
            "capital_delta": 100.0,  # Gross PnL
            "fee_drag": -5.0,  # Fees
            "funding_payment_usd": 2.0,  # Funding received
            "reward_usd": 1.0,  # Rewards
            "time_hours": 1.0,
        },
        capital_allocated_usd=1000.0,
        inputs={},
        outputs={},
    )


def test_truth_layer_invariant_total_formula(truth_layer, sample_run):
    """Test invariant: total = pnl + funding + rewards - fees - opportunity_cost."""
    attribution = truth_layer.compute_attribution(sample_run)

    # Calculate expected net from formula
    expected_net = (
        attribution.gross_pnl_usd
        + attribution.total_funding_usd
        + attribution.total_rewards_usd
        - attribution.total_fees_usd
        - attribution.opportunity_cost_usd
    )

    # Net should match formula
    assert abs(attribution.net_pnl_usd - expected_net) < 0.01, (
        f"Net PnL {attribution.net_pnl_usd} does not match formula: "
        f"pnl={attribution.gross_pnl_usd} + funding={attribution.total_funding_usd} "
        f"+ rewards={attribution.total_rewards_usd} - fees={attribution.total_fees_usd} "
        f"- opp_cost={attribution.opportunity_cost_usd} = {expected_net}"
    )


def test_truth_layer_no_duplicate_ledger_entries(truth_layer, sample_run):
    """Test invariant: no duplicate ledger entries per trade_id."""
    attribution = truth_layer.compute_attribution(sample_run)

    # Check for duplicate entry_ids
    entry_ids = [entry.entry_id for entry in attribution.ledger_entries]
    assert len(entry_ids) == len(set(entry_ids)), (
        f"Found duplicate ledger entry IDs: {entry_ids}"
    )

    # Check for duplicate run_id + entry_type combinations (should be unique per run)
    entry_keys = [
        (entry.run_id, entry.entry_type, entry.entry_id)
        for entry in attribution.ledger_entries
    ]
    assert len(entry_keys) == len(set(entry_keys)), (
        f"Found duplicate ledger entry keys: {entry_keys}"
    )


def test_truth_layer_report_totals_match_ledger_sums(truth_layer, sample_run):
    """Test invariant: report totals match ledger sums."""
    attribution = truth_layer.compute_attribution(sample_run)

    # Sum ledger entries by type
    ledger_pnl = sum(
        e.amount_usd
        for e in attribution.ledger_entries
        if e.entry_type == LedgerEntryType.PNL
    )
    ledger_fees = sum(
        abs(e.amount_usd)
        for e in attribution.ledger_entries
        if e.entry_type == LedgerEntryType.FEE
    )
    ledger_funding = sum(
        e.amount_usd
        for e in attribution.ledger_entries
        if e.entry_type == LedgerEntryType.FUNDING
    )
    ledger_rewards = sum(
        e.amount_usd
        for e in attribution.ledger_entries
        if e.entry_type == LedgerEntryType.REWARD
    )
    ledger_opp_cost = sum(
        abs(e.amount_usd)
        for e in attribution.ledger_entries
        if e.entry_type == LedgerEntryType.OPPORTUNITY_COST
    )

    # Report totals should match ledger sums
    assert abs(attribution.gross_pnl_usd - ledger_pnl) < 0.01, (
        f"Gross PnL {attribution.gross_pnl_usd} does not match ledger sum {ledger_pnl}"
    )
    assert abs(attribution.total_fees_usd - ledger_fees) < 0.01, (
        f"Total fees {attribution.total_fees_usd} does not match ledger sum {ledger_fees}"
    )
    assert abs(attribution.total_funding_usd - ledger_funding) < 0.01, (
        f"Total funding {attribution.total_funding_usd} does not match ledger sum {ledger_funding}"
    )
    assert abs(attribution.total_rewards_usd - ledger_rewards) < 0.01, (
        f"Total rewards {attribution.total_rewards_usd} does not match ledger sum {ledger_rewards}"
    )
    assert abs(attribution.opportunity_cost_usd - ledger_opp_cost) < 0.01, (
        f"Opportunity cost {attribution.opportunity_cost_usd} does not match ledger sum {ledger_opp_cost}"
    )


def test_truth_layer_portfolio_attribution_aggregation(truth_layer):
    """Test portfolio attribution aggregation maintains invariants."""
    runs = [
        ProcessRun(
            run_id=f"run_{i}",
            process_id="process_1",
            started_at=datetime.utcnow() - timedelta(hours=i),
            finished_at=datetime.utcnow() - timedelta(hours=i) + timedelta(hours=1),
            status=ProcessRunStatus.COMPLETED,
            metrics={
                "capital_delta": 10.0 * (i + 1),
                "fee_drag": -1.0,
                "time_hours": 1.0,
            },
            capital_allocated_usd=100.0,
            inputs={},
            outputs={},
        )
        for i in range(3)
    ]

    attributions = truth_layer.compute_portfolio_attribution(runs)

    # Check aggregated attribution
    assert "process_1" in attributions
    agg = attributions["process_1"]

    # Aggregated totals should match sum of individual attributions
    individual_attributions = [
        truth_layer.compute_attribution(run) for run in runs
    ]
    expected_gross = sum(a.gross_pnl_usd for a in individual_attributions)
    expected_net = sum(a.net_pnl_usd for a in individual_attributions)
    expected_fees = sum(a.total_fees_usd for a in individual_attributions)
    expected_funding = sum(a.total_funding_usd for a in individual_attributions)
    expected_rewards = sum(a.total_rewards_usd for a in individual_attributions)
    expected_opp_cost = sum(a.opportunity_cost_usd for a in individual_attributions)

    assert abs(agg.gross_pnl_usd - expected_gross) < 0.01
    assert abs(agg.net_pnl_usd - expected_net) < 0.01
    assert abs(agg.total_fees_usd - expected_fees) < 0.01
    assert abs(agg.total_funding_usd - expected_funding) < 0.01
    assert abs(agg.total_rewards_usd - expected_rewards) < 0.01
    assert abs(agg.opportunity_cost_usd - expected_opp_cost) < 0.01

    # Check no duplicate ledger entries in aggregated result
    entry_ids = [entry.entry_id for entry in agg.ledger_entries]
    assert len(entry_ids) == len(set(entry_ids))

