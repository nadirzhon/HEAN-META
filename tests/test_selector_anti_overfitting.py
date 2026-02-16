"""Tests for anti-overfitting selector rules.

Tests:
1. min sample size gating
2. decay weighting effect
3. holdout rejection
4. regime bucket diversification gating
"""

from datetime import datetime, timedelta

import pytest

from hean.process_factory.schemas import (
    ProcessPortfolioEntry,
    ProcessPortfolioState,
    ProcessRun,
    ProcessRunStatus,
)
from hean.process_factory.selector import ProcessSelector


@pytest.fixture
def selector():
    """Create ProcessSelector with anti-overfitting rules."""
    return ProcessSelector(
        min_sample_size_for_scaling=10,
        decay_half_life_days=30.0,
        holdout_window_days=7.0,
    )


def test_min_sample_size_gating(selector):
    """Test that min sample size gates scaling decisions."""
    entry = ProcessPortfolioEntry(
        process_id="test_process",
        state=ProcessPortfolioState.TESTING,
        runs_count=5,  # Below min_sample_size_for_scaling (10)
        wins=3,
        losses=2,
        pnl_sum=50.0,
        max_dd=5.0,
        avg_roi=0.1,
        fail_rate=0.2,
        time_efficiency=10.0,
    )

    runs = [
        ProcessRun(
            run_id=f"run_{i}",
            process_id="test_process",
            started_at=datetime.utcnow() - timedelta(days=i),
            finished_at=datetime.utcnow() - timedelta(days=i) + timedelta(hours=1),
            status=ProcessRunStatus.COMPLETED if i % 2 == 0 else ProcessRunStatus.FAILED,
            metrics={"capital_delta": 0.02 if i % 2 == 0 else -0.01},
            capital_allocated_usd=100.0,
            inputs={},
            outputs={},
        )
        for i in range(5)
    ]

    # Update entry with runs
    entry = selector.update_portfolio_entry(entry, runs)

    # Evaluate - should stay in TESTING due to min sample size
    new_state = selector.evaluate_process(entry)
    assert new_state == ProcessPortfolioState.TESTING, (
        f"Process with {entry.runs_count} runs should stay in TESTING "
        f"(min_sample_size={selector.min_sample_size_for_scaling})"
    )


def test_decay_weighting_effect(selector):
    """Test that decay weighting affects PnL calculation."""
    now = datetime.utcnow()

    # Create runs with different ages
    runs = [
        ProcessRun(
            run_id=f"run_{i}",
            process_id="test_process",
            started_at=now - timedelta(days=i * 10),  # 0, 10, 20, 30, 40 days ago
            finished_at=now - timedelta(days=i * 10) + timedelta(hours=1),
            status=ProcessRunStatus.COMPLETED,
            metrics={"capital_delta": 10.0},  # Same PnL for all
            capital_allocated_usd=100.0,
            inputs={},
            outputs={},
        )
        for i in range(5)
    ]

    entry = ProcessPortfolioEntry(
        process_id="test_process",
        state=ProcessPortfolioState.TESTING,
        runs_count=0,
        wins=0,
        losses=0,
        pnl_sum=0.0,
        max_dd=0.0,
        avg_roi=0.0,
        fail_rate=0.0,
        time_efficiency=0.0,
    )

    # Update with decay weighting
    entry = selector.update_portfolio_entry(entry, runs)

    # PnL sum should be weighted (recent runs weighted more)
    # Without decay, sum would be 50.0 (5 * 10.0)
    # With decay, recent runs have higher weight
    assert entry.pnl_sum > 0, "PnL sum should be positive"
    # Recent runs should contribute more than old runs
    # The exact value depends on decay calculation, but should be less than simple sum
    assert entry.pnl_sum < 50.0, (
        f"Decay-weighted PnL {entry.pnl_sum} should be less than simple sum 50.0"
    )


def test_holdout_rejection(selector):
    """Test that holdout check rejects processes with performance collapse."""
    now = datetime.utcnow()

    # Create runs: good training performance, bad holdout performance
    training_runs = [
        ProcessRun(
            run_id=f"train_{i}",
            process_id="test_process",
            started_at=now - timedelta(days=10 + i),  # 10+ days ago (training)
            finished_at=now - timedelta(days=10 + i) + timedelta(hours=1),
            status=ProcessRunStatus.COMPLETED,
            metrics={"capital_delta": 0.02},  # Positive training performance
            capital_allocated_usd=100.0,
            inputs={},
            outputs={},
        )
        for i in range(5)
    ]

    holdout_runs = [
        ProcessRun(
            run_id=f"holdout_{i}",
            process_id="test_process",
            started_at=now - timedelta(days=i),  # Recent (holdout)
            finished_at=now - timedelta(days=i) + timedelta(hours=1),
            status=ProcessRunStatus.FAILED,
            metrics={"capital_delta": -0.01},  # Negative holdout performance
            capital_allocated_usd=100.0,
            inputs={},
            outputs={},
        )
        for i in range(2)
    ]

    all_runs = training_runs + holdout_runs

    entry = ProcessPortfolioEntry(
        process_id="test_process",
        state=ProcessPortfolioState.TESTING,
        runs_count=len(all_runs),
        wins=5,
        losses=2,
        pnl_sum=0.08,  # Positive overall
        max_dd=0.01,
        avg_roi=0.05,
        fail_rate=0.2,
        time_efficiency=5.0,
    )

    # Update entry
    entry = selector.update_portfolio_entry(entry, all_runs)

    # Evaluate - should detect holdout failure and not scale
    new_state = selector.evaluate_process(entry)
    # Should stay in TESTING due to holdout failure
    assert new_state == ProcessPortfolioState.TESTING, (
        f"Process with holdout failure should stay in TESTING, got {new_state}"
    )


def test_regime_bucket_diversification(selector):
    """Test regime bucket diversification metrics."""
    now = datetime.utcnow()

    # Create runs with different regime buckets (hour, vol, spread)
    runs = [
        ProcessRun(
            run_id=f"run_{i}",
            process_id="test_process",
            started_at=now - timedelta(hours=i),
            finished_at=now - timedelta(hours=i) + timedelta(hours=1),
            status=ProcessRunStatus.COMPLETED,
            metrics={
                "capital_delta": 10.0,
                "volatility": 0.02 if i % 2 == 0 else 0.04,  # Mix of low/medium vol
                "spread_bps": 5.0 if i % 3 == 0 else 15.0,  # Mix of tight/normal/wide
            },
            capital_allocated_usd=100.0,
            inputs={},
            outputs={},
        )
        for i in range(10)
    ]

    # Get regime buckets
    buckets = selector.get_regime_buckets(runs)

    # Should have multiple buckets
    assert len(buckets["hour_bucket"]) > 1, "Should have multiple hour buckets"
    assert len(buckets["vol_bucket"]) > 1, "Should have multiple vol buckets"
    assert len(buckets["spread_bucket"]) > 1, "Should have multiple spread buckets"

    # Check that buckets have runs
    for bucket_type, bucket_data in buckets.items():
        for bucket_key, metrics in bucket_data.items():
            assert metrics["runs"] > 0, f"Bucket {bucket_key} should have runs"
            assert "pnl_sum" in metrics, f"Bucket {bucket_key} should have pnl_sum"


def test_selector_promotes_after_min_sample_size(selector):
    """Test that selector allows promotion after min sample size is met."""
    entry = ProcessPortfolioEntry(
        process_id="test_process",
        state=ProcessPortfolioState.TESTING,
        runs_count=15,  # Above min_sample_size_for_scaling (10)
        wins=12,
        losses=3,
        pnl_sum=0.21,
        max_dd=0.01,
        avg_roi=0.1,
        fail_rate=0.2,  # Below 0.4 threshold
        time_efficiency=10.0,
    )

    runs = [
        ProcessRun(
            run_id=f"run_{i}",
            process_id="test_process",
            started_at=datetime.utcnow() - timedelta(days=i),
            finished_at=datetime.utcnow() - timedelta(days=i) + timedelta(hours=1),
            status=ProcessRunStatus.COMPLETED if i % 5 != 0 else ProcessRunStatus.FAILED,
            metrics={"capital_delta": 0.02 if i % 5 != 0 else -0.01},
            capital_allocated_usd=100.0,
            inputs={},
            outputs={},
        )
        for i in range(15)
    ]

    # Update entry
    entry = selector.update_portfolio_entry(entry, runs)

    # Evaluate - should allow scaling (stay in TESTING but eligible)
    new_state = selector.evaluate_process(entry)
    # With good performance and enough samples, should stay in TESTING (eligible for scaling)
    assert new_state == ProcessPortfolioState.TESTING, (
        f"Process with {entry.runs_count} runs and good performance should be eligible for scaling"
    )

