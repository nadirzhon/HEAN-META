"""Tests for readiness evaluation."""


from hean.evaluation.readiness import EvaluationResult, ReadinessEvaluator


def test_evaluate_returns_structured_result() -> None:
    """Test that evaluate returns structured PASS/FAIL result."""
    evaluator = ReadinessEvaluator()

    # Test PASS case
    pass_metrics = {
        "profit_factor": 1.5,
        "max_drawdown_pct": 15.0,
        "total_return_pct": 10.0,
        "strategies": {
            "strategy1": {
                "regime_pnl": {
                    "range": 100.0,
                    "normal": 50.0,
                    "impulse": 30.0,
                    "high_vol": 20.0,
                }
            }
        }
    }

    result = evaluator.evaluate(pass_metrics)

    assert isinstance(result, EvaluationResult)
    assert result.passed is True
    assert "criteria" in result.__dict__
    assert "recommendations" in result.__dict__
    assert "regime_results" in result.__dict__
    assert isinstance(result.criteria, dict)
    assert isinstance(result.recommendations, list)
    assert isinstance(result.regime_results, dict)


def test_evaluate_fails_on_low_pf() -> None:
    """Test that evaluation fails when profit factor is too low."""
    evaluator = ReadinessEvaluator(min_profit_factor=1.3)

    fail_metrics = {
        "profit_factor": 1.0,  # Below threshold
        "max_drawdown_pct": 10.0,
        "total_return_pct": 5.0,
        "strategies": {
            "strategy1": {
                "regime_pnl": {
                    "range": 100.0,
                    "normal": 50.0,
                    "impulse": 30.0,
                    "high_vol": 20.0,
                }
            }
        }
    }

    result = evaluator.evaluate(fail_metrics)

    assert result.passed is False
    assert result.criteria["profit_factor"]["passed"] is False
    assert len(result.recommendations) > 0
    assert any("Profit Factor" in rec for rec in result.recommendations)


def test_evaluate_fails_on_high_drawdown() -> None:
    """Test that evaluation fails when max drawdown is too high."""
    evaluator = ReadinessEvaluator(max_drawdown_pct=25.0)

    fail_metrics = {
        "profit_factor": 1.5,
        "max_drawdown_pct": 30.0,  # Above threshold
        "total_return_pct": 5.0,
        "strategies": {
            "strategy1": {
                "regime_pnl": {
                    "range": 100.0,
                    "normal": 50.0,
                    "impulse": 30.0,
                    "high_vol": 20.0,
                }
            }
        }
    }

    result = evaluator.evaluate(fail_metrics)

    assert result.passed is False
    assert result.criteria["max_drawdown"]["passed"] is False
    assert len(result.recommendations) > 0
    assert any("Max Drawdown" in rec for rec in result.recommendations)


def test_evaluate_fails_on_insufficient_positive_regimes() -> None:
    """Test that evaluation fails when not enough regimes show positive returns."""
    evaluator = ReadinessEvaluator(min_positive_regimes=3, total_regimes=4)

    fail_metrics = {
        "profit_factor": 1.5,
        "max_drawdown_pct": 15.0,
        "total_return_pct": 5.0,
        "strategies": {
            "strategy1": {
                "regime_pnl": {
                    "range": 100.0,
                    "normal": 50.0,
                    "impulse": -20.0,  # Negative
                    "high_vol": -10.0,  # Negative
                }
            }
        }
    }

    result = evaluator.evaluate(fail_metrics)

    assert result.passed is False
    assert result.criteria["regime_performance"]["passed"] is False
    assert len(result.recommendations) > 0
    assert any("regimes" in rec.lower() for rec in result.recommendations)


def test_evaluate_passes_all_criteria() -> None:
    """Test that evaluation passes when all criteria are met."""
    evaluator = ReadinessEvaluator()

    pass_metrics = {
        "profit_factor": 1.5,
        "max_drawdown_pct": 15.0,
        "total_return_pct": 10.0,
        "strategies": {
            "strategy1": {
                "regime_pnl": {
                    "range": 100.0,
                    "normal": 50.0,
                    "impulse": 30.0,
                    "high_vol": 20.0,
                }
            }
        }
    }

    result = evaluator.evaluate(pass_metrics)

    assert result.passed is True
    assert result.criteria["profit_factor"]["passed"] is True
    assert result.criteria["max_drawdown"]["passed"] is True
    assert result.criteria["regime_performance"]["passed"] is True
    assert any("ready for live trading" in rec.lower() for rec in result.recommendations)


def test_evaluate_handles_missing_regime_data() -> None:
    """Test that evaluation handles missing regime data gracefully."""
    evaluator = ReadinessEvaluator()

    metrics = {
        "profit_factor": 1.5,
        "max_drawdown_pct": 15.0,
        "total_return_pct": 10.0,
        # No strategies or regime_pnl data
    }

    result = evaluator.evaluate(metrics)

    # Should still evaluate PF and DD
    assert "profit_factor" in result.criteria
    assert "max_drawdown" in result.criteria
    # Regime performance may fail if no data, but shouldn't crash
    assert "regime_performance" in result.criteria






