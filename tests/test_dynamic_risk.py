"""Tests for dynamic risk scaling."""

from datetime import datetime, timedelta

from hean.core.regime import Regime
from hean.core.types import Signal
from hean.risk.dynamic_risk import DynamicRiskManager
from hean.risk.position_sizer import PositionSizer


def test_risk_decreases_after_losses() -> None:
    """Test that risk decreases after losses (low PF)."""
    manager = DynamicRiskManager()

    # High PF scenario - should increase risk
    multiplier_high_pf = manager.calculate_risk_multiplier(
        current_regime=Regime.NORMAL,
        rolling_pf=2.0,  # Strong PF
        recent_drawdown=1.0,  # Low drawdown
        volatility_percentile=50.0,  # Normal volatility
    )

    # Low PF scenario - should decrease risk
    multiplier_low_pf = manager.calculate_risk_multiplier(
        current_regime=Regime.NORMAL,
        rolling_pf=0.6,  # Poor PF
        recent_drawdown=1.0,  # Low drawdown
        volatility_percentile=50.0,  # Normal volatility
    )

    # Risk should be lower with poor PF
    assert multiplier_low_pf < multiplier_high_pf
    assert multiplier_low_pf < 1.0  # Should be below baseline
    assert multiplier_high_pf > 1.0  # Should be above baseline


def test_risk_increases_only_with_stable_pf() -> None:
    """Test that risk increases only with stable PF."""
    manager = DynamicRiskManager()

    # Very high PF with low drawdown - should increase risk
    multiplier_stable = manager.calculate_risk_multiplier(
        current_regime=Regime.NORMAL,
        rolling_pf=1.8,  # Strong PF
        recent_drawdown=0.5,  # Very low drawdown
        volatility_percentile=30.0,  # Low volatility
    )

    # High PF but with high drawdown - should not increase as much
    multiplier_unstable = manager.calculate_risk_multiplier(
        current_regime=Regime.NORMAL,
        rolling_pf=1.8,  # Strong PF
        recent_drawdown=6.0,  # High drawdown
        volatility_percentile=30.0,  # Low volatility
    )

    # Stable scenario should have higher risk
    assert multiplier_stable > multiplier_unstable
    assert multiplier_stable > 1.0  # Should be above baseline


def test_drawdown_spike_safeguard() -> None:
    """Test that risk never increases after drawdown spike."""
    manager = DynamicRiskManager()

    # Normal scenario - calculate baseline
    multiplier_before = manager.calculate_risk_multiplier(
        current_regime=Regime.NORMAL,
        rolling_pf=1.5,  # Good PF
        recent_drawdown=1.0,  # Low drawdown
        volatility_percentile=50.0,
    )

    # Simulate drawdown spike (sudden increase from 1% to 4%)
    manager._update_drawdown_tracking(1.0)  # Initial drawdown
    manager._update_drawdown_tracking(4.5)  # Spike detected (>2% change)

    # Try to increase risk with good PF - should be blocked
    multiplier_after_spike = manager.calculate_risk_multiplier(
        current_regime=Regime.NORMAL,
        rolling_pf=2.0,  # Excellent PF
        recent_drawdown=4.5,  # High drawdown
        volatility_percentile=30.0,  # Low volatility
    )

    # Risk should not increase after spike (should be capped at 1.0 or lower)
    assert multiplier_after_spike <= 1.0
    assert manager._drawdown_spike_detected is True
    assert manager._risk_reductions_triggered > 0


def test_impulse_regime_cap() -> None:
    """Test that IMPULSE regime caps max risk at 1.2x."""
    manager = DynamicRiskManager()

    # Calculate multiplier in IMPULSE regime with excellent conditions
    multiplier_impulse = manager.calculate_risk_multiplier(
        current_regime=Regime.IMPULSE,
        rolling_pf=2.5,  # Excellent PF
        recent_drawdown=0.5,  # Very low drawdown
        volatility_percentile=20.0,  # Low volatility
    )

    # Should be capped at max_impulse_multiplier (1.2)
    assert multiplier_impulse <= 1.2
    assert multiplier_impulse == manager._max_impulse_multiplier


def test_risk_multiplier_bounds() -> None:
    """Test that risk multiplier stays within bounds (0.5x - 1.5x)."""
    manager = DynamicRiskManager()

    # Worst case scenario - should hit minimum
    multiplier_min = manager.calculate_risk_multiplier(
        current_regime=Regime.RANGE,
        rolling_pf=0.5,  # Very poor PF
        recent_drawdown=10.0,  # High drawdown
        volatility_percentile=90.0,  # High volatility
    )

    # Best case scenario (non-IMPULSE) - should hit maximum
    multiplier_max = manager.calculate_risk_multiplier(
        current_regime=Regime.NORMAL,
        rolling_pf=2.5,  # Excellent PF
        recent_drawdown=0.1,  # Very low drawdown
        volatility_percentile=10.0,  # Low volatility
    )

    assert multiplier_min >= 0.5
    assert multiplier_max <= 1.5
    assert multiplier_min < multiplier_max


def test_volatility_percentile_calculation() -> None:
    """Test volatility percentile calculation."""
    manager = DynamicRiskManager()

    # Build volatility history (need at least 10 for percentile calculation)
    for vol in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]:
        manager.update_volatility(vol)

    # Current volatility at 50th percentile (around 0.005-0.006)
    percentile = manager.calculate_volatility_percentile(0.005)
    assert 40.0 <= percentile <= 60.0  # Should be around middle

    # Current volatility at high percentile (above all historical values)
    percentile_high = manager.calculate_volatility_percentile(0.015)
    assert percentile_high >= 90.0  # Should be very high

    # Current volatility at low percentile (below all historical values)
    percentile_low = manager.calculate_volatility_percentile(0.0005)
    assert percentile_low <= 10.0  # Should be very low


def test_metrics_tracking() -> None:
    """Test that metrics are tracked correctly."""
    manager = DynamicRiskManager()

    # Calculate several multipliers
    for i in range(5):
        manager.calculate_risk_multiplier(
            current_regime=Regime.NORMAL,
            rolling_pf=1.0 + (i * 0.1),
            recent_drawdown=1.0,
            volatility_percentile=50.0,
        )

    metrics = manager.get_metrics()

    assert "avg_risk_multiplier" in metrics
    assert "risk_reductions_triggered" in metrics
    assert "current_multiplier" in metrics
    assert "drawdown_spike_active" in metrics

    assert metrics["avg_risk_multiplier"] > 0
    assert metrics["risk_reductions_triggered"] >= 0
    assert 0.5 <= metrics["current_multiplier"] <= 1.5


def test_position_sizer_integration() -> None:
    """Test that PositionSizer integrates with DynamicRiskManager."""
    sizer = PositionSizer()

    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        stop_loss=49000.0,  # 2% stop
    )

    # Calculate size without dynamic risk (backward compatible)
    size_no_dynamic = sizer.calculate_size(
        signal, 10000.0, 50000.0, Regime.NORMAL
    )

    # Calculate size with dynamic risk (good conditions)
    size_good = sizer.calculate_size(
        signal, 10000.0, 50000.0, Regime.NORMAL,
        rolling_pf=1.8,
        recent_drawdown=1.0,
        volatility_percentile=30.0,
    )

    # Calculate size with dynamic risk (poor conditions)
    size_poor = sizer.calculate_size(
        signal, 10000.0, 50000.0, Regime.NORMAL,
        rolling_pf=0.6,
        recent_drawdown=6.0,
        volatility_percentile=85.0,
    )

    # All sizes should be positive
    assert size_no_dynamic > 0
    assert size_good > 0
    assert size_poor > 0

    # Good conditions should result in larger size than poor conditions
    assert size_good > size_poor

    # Verify dynamic risk manager is accessible
    dynamic_risk = sizer.get_dynamic_risk_manager()
    assert dynamic_risk is not None
    assert isinstance(dynamic_risk, DynamicRiskManager)


def test_regime_adjustments() -> None:
    """Test that different regimes adjust risk appropriately."""
    manager = DynamicRiskManager()

    # Same conditions, different regimes
    multiplier_range = manager.calculate_risk_multiplier(
        current_regime=Regime.RANGE,
        rolling_pf=1.2,
        recent_drawdown=1.0,
        volatility_percentile=50.0,
    )

    multiplier_normal = manager.calculate_risk_multiplier(
        current_regime=Regime.NORMAL,
        rolling_pf=1.2,
        recent_drawdown=1.0,
        volatility_percentile=50.0,
    )

    multiplier_impulse = manager.calculate_risk_multiplier(
        current_regime=Regime.IMPULSE,
        rolling_pf=1.2,
        recent_drawdown=1.0,
        volatility_percentile=50.0,
    )

    # RANGE should be most conservative
    assert multiplier_range < multiplier_normal
    # IMPULSE should be higher but capped
    assert multiplier_impulse >= multiplier_normal
    assert multiplier_impulse <= 1.2


def test_drawdown_cooldown_expiry() -> None:
    """Test that drawdown spike cooldown expires after time."""
    manager = DynamicRiskManager()

    # Trigger spike
    manager._update_drawdown_tracking(1.0)
    manager._update_drawdown_tracking(4.0)  # Spike

    # Manually set spike time to past (simulate time passing)
    manager._last_spike_time = datetime.utcnow() - timedelta(hours=25)

    # Calculate multiplier - cooldown should be expired
    multiplier = manager.calculate_risk_multiplier(
        current_regime=Regime.NORMAL,
        rolling_pf=1.8,
        recent_drawdown=2.0,
        volatility_percentile=40.0,
    )

    # Should be able to increase risk now
    assert multiplier > 1.0
    assert manager._drawdown_spike_detected is False


def test_high_drawdown_reduction() -> None:
    """Test that high drawdown significantly reduces risk."""
    manager = DynamicRiskManager()

    # Low drawdown
    multiplier_low_dd = manager.calculate_risk_multiplier(
        current_regime=Regime.NORMAL,
        rolling_pf=1.2,
        recent_drawdown=1.0,
        volatility_percentile=50.0,
    )

    # High drawdown
    multiplier_high_dd = manager.calculate_risk_multiplier(
        current_regime=Regime.NORMAL,
        rolling_pf=1.2,
        recent_drawdown=7.0,  # High drawdown
        volatility_percentile=50.0,
    )

    # High drawdown should significantly reduce risk
    assert multiplier_high_dd < multiplier_low_dd
    assert multiplier_high_dd < 1.0  # Should be below baseline


def test_volatility_percentile_impact() -> None:
    """Test that volatility percentile affects risk scaling."""
    manager = DynamicRiskManager()

    # Build volatility history
    for vol in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]:
        manager.update_volatility(vol)

    # Low volatility percentile
    multiplier_low_vol = manager.calculate_risk_multiplier(
        current_regime=Regime.NORMAL,
        rolling_pf=1.2,
        recent_drawdown=1.0,
        volatility_percentile=15.0,  # Low percentile
    )

    # High volatility percentile
    multiplier_high_vol = manager.calculate_risk_multiplier(
        current_regime=Regime.NORMAL,
        rolling_pf=1.2,
        recent_drawdown=1.0,
        volatility_percentile=85.0,  # High percentile
    )

    # High volatility should reduce risk
    assert multiplier_high_vol < multiplier_low_vol


def test_combined_factors() -> None:
    """Test interaction of multiple risk factors."""
    manager = DynamicRiskManager()

    # Excellent conditions across all factors
    multiplier_excellent = manager.calculate_risk_multiplier(
        current_regime=Regime.NORMAL,
        rolling_pf=2.0,  # Excellent PF
        recent_drawdown=0.5,  # Very low drawdown
        volatility_percentile=20.0,  # Low volatility
    )

    # Poor conditions across all factors
    multiplier_poor = manager.calculate_risk_multiplier(
        current_regime=Regime.RANGE,
        rolling_pf=0.6,  # Poor PF
        recent_drawdown=8.0,  # High drawdown
        volatility_percentile=90.0,  # High volatility
    )

    # Should see significant difference
    assert multiplier_excellent > multiplier_poor
    assert multiplier_excellent > 1.0
    assert multiplier_poor < 1.0
    assert multiplier_poor >= 0.5  # Should respect minimum





