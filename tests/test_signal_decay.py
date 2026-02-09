"""Tests for Signal Decay Model."""

import pytest
from datetime import datetime, timedelta

from hean.core.types import Signal
from hean.execution.signal_decay import (
    DecayCurve,
    DecayParameters,
    SignalDecayModel,
    DecayAwareOrderTiming,
)


class TestDecayParameters:
    """Tests for DecayParameters."""

    def test_default_params(self):
        """Test default decay parameters."""
        params = DecayParameters()
        assert params.curve_type == DecayCurve.EXPONENTIAL
        assert params.half_life_seconds == 300.0
        assert params.min_confidence == 0.2

    def test_custom_params(self):
        """Test custom decay parameters."""
        params = DecayParameters(
            curve_type=DecayCurve.LINEAR,
            half_life_seconds=600.0,
            min_confidence=0.3,
        )
        assert params.curve_type == DecayCurve.LINEAR
        assert params.half_life_seconds == 600.0


class TestSignalDecayModel:
    """Tests for SignalDecayModel."""

    @pytest.fixture
    def decay_model(self):
        """Create decay model instance."""
        return SignalDecayModel()

    def test_initialization(self, decay_model):
        """Test model initialization."""
        assert len(decay_model._active_signals) == 0
        assert decay_model._total_decayed == 0

    def test_register_signal(self, decay_model):
        """Test signal registration."""
        decay_model.register_signal(
            signal_id="sig_1",
            initial_confidence=0.8,
            signal_type="momentum",
        )

        assert "sig_1" in decay_model._active_signals
        state = decay_model._active_signals["sig_1"]
        assert state.initial_confidence == 0.8
        assert state.current_confidence == 0.8

    def test_register_signal_custom_params(self, decay_model):
        """Test signal registration with custom params."""
        custom_params = DecayParameters(
            curve_type=DecayCurve.LINEAR,
            half_life_seconds=120.0,
            min_confidence=0.15,
        )

        decay_model.register_signal(
            signal_id="sig_1",
            initial_confidence=0.8,
            custom_params=custom_params,
        )

        state = decay_model._active_signals["sig_1"]
        assert state.decay_params.curve_type == DecayCurve.LINEAR
        assert state.decay_params.half_life_seconds == 120.0

    def test_get_current_confidence_no_decay(self, decay_model):
        """Test getting confidence immediately after registration."""
        decay_model.register_signal(
            signal_id="sig_1",
            initial_confidence=0.8,
            signal_type="default",
        )

        # Immediately after registration
        confidence = decay_model.get_current_confidence("sig_1")
        assert confidence == pytest.approx(0.8, rel=0.01)

    def test_get_current_confidence_with_decay(self, decay_model):
        """Test confidence decay over time."""
        now = datetime.utcnow()
        decay_model.register_signal(
            signal_id="sig_1",
            initial_confidence=0.8,
            signal_type="default",
        )

        # Modify creation time to simulate age
        state = decay_model._active_signals["sig_1"]
        state.creation_time = now - timedelta(seconds=300)  # 5 minutes old (1 half-life)

        # Get confidence (should be ~0.4 after 1 half-life with exponential decay)
        confidence = decay_model.get_current_confidence("sig_1", now=now)
        expected = 0.8 * 0.5  # One half-life
        assert confidence == pytest.approx(expected, rel=0.1)

    def test_exponential_decay_curve(self, decay_model):
        """Test exponential decay curve."""
        params = DecayParameters(
            curve_type=DecayCurve.EXPONENTIAL,
            half_life_seconds=300.0,
            min_confidence=0.2,
        )

        # Test at different ages
        ages_and_expected = [
            (0, 0.8),           # t=0: no decay
            (300, 0.4),         # t=half_life: 50% decay
            (600, 0.2),         # t=2*half_life: 75% decay
        ]

        for age, expected in ages_and_expected:
            confidence = decay_model._calculate_decay(
                initial_confidence=0.8,
                age_seconds=age,
                params=params,
            )
            assert confidence == pytest.approx(expected, rel=0.1)

    def test_linear_decay_curve(self, decay_model):
        """Test linear decay curve."""
        params = DecayParameters(
            curve_type=DecayCurve.LINEAR,
            half_life_seconds=300.0,
            min_confidence=0.2,
        )

        # Linear decay: reaches min at 2*half_life
        confidence_mid = decay_model._calculate_decay(
            initial_confidence=0.8,
            age_seconds=300.0,  # half_life
            params=params,
        )

        confidence_end = decay_model._calculate_decay(
            initial_confidence=0.8,
            age_seconds=600.0,  # 2*half_life
            params=params,
        )

        # Mid should be between initial and min
        assert 0.2 < confidence_mid < 0.8
        # End should be at min
        assert confidence_end == pytest.approx(0.2, rel=0.01)

    def test_logarithmic_decay_curve(self, decay_model):
        """Test logarithmic decay curve (slow decay)."""
        params = DecayParameters(
            curve_type=DecayCurve.LOGARITHMIC,
            half_life_seconds=300.0,
            min_confidence=0.2,
        )

        confidence_early = decay_model._calculate_decay(
            initial_confidence=0.8,
            age_seconds=150.0,
            params=params,
        )

        confidence_late = decay_model._calculate_decay(
            initial_confidence=0.8,
            age_seconds=600.0,
            params=params,
        )

        # Logarithmic decay pattern: early decay is significant, late decay slows down
        # Early should be between min and initial
        assert 0.2 < confidence_early < 0.8
        # Late should be approaching min but still above it
        assert confidence_late >= 0.2

    def test_step_decay_curve(self, decay_model):
        """Test step decay curve (discrete drops)."""
        params = DecayParameters(
            curve_type=DecayCurve.STEP,
            half_life_seconds=300.0,
            min_confidence=0.2,
        )

        # Just before first step
        conf_before = decay_model._calculate_decay(
            initial_confidence=0.8,
            age_seconds=299.0,
            params=params,
        )

        # Just after first step
        conf_after = decay_model._calculate_decay(
            initial_confidence=0.8,
            age_seconds=301.0,
            params=params,
        )

        # Should see a discrete drop
        assert conf_after < conf_before

    def test_adjust_for_market_conditions_volatility(self, decay_model):
        """Test decay adjustment for volatility."""
        decay_model.register_signal(
            signal_id="sig_1",
            initial_confidence=0.8,
            signal_type="default",
        )

        # High volatility should accelerate decay
        decay_model.adjust_for_market_conditions(
            signal_id="sig_1",
            volatility_percentile=80.0,
            regime="NORMAL",
        )

        state = decay_model._active_signals["sig_1"]
        assert state.decay_params.volatility_multiplier > 1.0

    def test_adjust_for_market_conditions_regime(self, decay_model):
        """Test decay adjustment for regime."""
        decay_model.register_signal(
            signal_id="sig_1",
            initial_confidence=0.8,
            signal_type="default",
        )

        # IMPULSE regime should accelerate decay
        decay_model.adjust_for_market_conditions(
            signal_id="sig_1",
            volatility_percentile=50.0,
            regime="IMPULSE",
        )

        state = decay_model._active_signals["sig_1"]
        assert state.decay_params.regime_multiplier > 1.0

    def test_remove_signal(self, decay_model):
        """Test signal removal."""
        decay_model.register_signal(
            signal_id="sig_1",
            initial_confidence=0.8,
        )

        assert "sig_1" in decay_model._active_signals

        decay_model.remove_signal("sig_1")

        assert "sig_1" not in decay_model._active_signals

    def test_get_signal_state(self, decay_model):
        """Test getting signal state."""
        decay_model.register_signal(
            signal_id="sig_1",
            initial_confidence=0.8,
        )

        state = decay_model.get_signal_state("sig_1")
        assert state is not None
        assert state.signal_id == "sig_1"
        assert state.initial_confidence == 0.8

    def test_get_statistics(self, decay_model):
        """Test getting statistics."""
        decay_model.register_signal("sig_1", 0.8, "momentum")
        decay_model.register_signal("sig_2", 0.7, "breakout")

        decay_model.get_current_confidence("sig_1")
        decay_model.get_current_confidence("sig_2")

        stats = decay_model.get_statistics()

        assert stats["active_signals"] == 2
        assert stats["total_decayed"] > 0
        assert "avg_age_seconds" in stats

    def test_cleanup_expired_signals(self, decay_model):
        """Test cleanup of old signals."""
        now = datetime.utcnow()

        # Register old signal
        decay_model.register_signal("sig_old", 0.8)
        state_old = decay_model._active_signals["sig_old"]
        state_old.creation_time = now - timedelta(hours=2)

        # Register recent signal
        decay_model.register_signal("sig_new", 0.8)

        # Cleanup signals older than 1 hour
        removed = decay_model.cleanup_expired_signals(max_age_minutes=60)

        assert removed == 1
        assert "sig_old" not in decay_model._active_signals
        assert "sig_new" in decay_model._active_signals

    def test_signal_type_specific_decay(self, decay_model):
        """Test that different signal types decay at different rates."""
        now = datetime.utcnow()

        # Arbitrage signal (fast decay)
        decay_model.register_signal("sig_arb", 0.8, "arbitrage")
        # Mean reversion signal (slow decay)
        decay_model.register_signal("sig_mr", 0.8, "mean_reversion")

        # Simulate same age for both
        for signal_id in ["sig_arb", "sig_mr"]:
            state = decay_model._active_signals[signal_id]
            state.creation_time = now - timedelta(seconds=300)

        # Get confidence after same time
        conf_arb = decay_model.get_current_confidence("sig_arb", now=now)
        conf_mr = decay_model.get_current_confidence("sig_mr", now=now)

        # Arbitrage should decay faster (lower confidence)
        assert conf_arb < conf_mr


class TestDecayAwareOrderTiming:
    """Tests for DecayAwareOrderTiming."""

    @pytest.fixture
    def mock_timing_optimizer(self):
        """Mock timing optimizer."""
        class MockOptimizer:
            def get_timing_recommendation(self, symbol, side, is_urgent):
                class Rec:
                    urgency = "optimal"
                    wait_minutes = 0
                return Rec()

        return MockOptimizer()

    @pytest.fixture
    def decay_model(self):
        """Create decay model."""
        return SignalDecayModel()

    @pytest.fixture
    def decay_aware_timing(self, mock_timing_optimizer, decay_model):
        """Create decay-aware timing."""
        return DecayAwareOrderTiming(
            timing_optimizer=mock_timing_optimizer,
            decay_model=decay_model,
            decay_threshold=0.4,
        )

    def test_initialization(self, decay_aware_timing):
        """Test initialization."""
        assert decay_aware_timing._decay_threshold == 0.4

    def test_should_execute_high_confidence(self, decay_aware_timing, decay_model):
        """Test execution decision with high confidence."""
        # Register signal with high confidence
        decay_model.register_signal("sig_1", 0.8, "default")

        should_execute, reason = decay_aware_timing.should_execute_now(
            signal_id="sig_1",
            symbol="BTCUSDT",
            side="buy",
        )

        # High confidence + optimal timing = execute
        assert should_execute is True
        assert "optimal" in reason

    def test_should_execute_low_confidence(self, decay_aware_timing, decay_model):
        """Test execution decision with decayed confidence."""
        now = datetime.utcnow()

        # Register signal
        decay_model.register_signal("sig_1", 0.8, "default")

        # Simulate heavy decay
        state = decay_model._active_signals["sig_1"]
        state.creation_time = now - timedelta(seconds=600)  # Old signal

        should_execute, reason = decay_aware_timing.should_execute_now(
            signal_id="sig_1",
            symbol="BTCUSDT",
            side="buy",
        )

        # Low confidence from decay = execute urgently
        assert should_execute is True
        assert "decay_urgent" in reason

    def test_get_adjusted_confidence(self, decay_aware_timing, decay_model):
        """Test confidence adjustment for decay."""
        now = datetime.utcnow()

        # Register signal
        decay_model.register_signal("sig_1", 0.8, "default")

        # Simulate some decay
        state = decay_model._active_signals["sig_1"]
        state.creation_time = now - timedelta(seconds=300)

        # Get adjusted confidence
        adjusted = decay_aware_timing.get_adjusted_confidence(
            signal_id="sig_1",
            base_confidence=0.9,
        )

        # Should be reduced due to decay
        assert adjusted < 0.9


@pytest.mark.asyncio
async def test_decay_integration():
    """Integration test for signal decay."""
    decay_model = SignalDecayModel()

    # Register multiple signals with different types
    decay_model.register_signal("sig_momentum", 0.8, "momentum")
    decay_model.register_signal("sig_arb", 0.8, "arbitrage")
    decay_model.register_signal("sig_mr", 0.8, "mean_reversion")

    # Simulate time passing
    now = datetime.utcnow()
    for signal_id in ["sig_momentum", "sig_arb", "sig_mr"]:
        state = decay_model._active_signals[signal_id]
        state.creation_time = now - timedelta(seconds=300)

    # Adjust for market conditions
    for signal_id in ["sig_momentum", "sig_arb", "sig_mr"]:
        decay_model.adjust_for_market_conditions(
            signal_id=signal_id,
            volatility_percentile=80.0,  # High volatility
            regime="IMPULSE",
        )

    # Get final confidences
    confidences = {
        signal_id: decay_model.get_current_confidence(signal_id, now=now)
        for signal_id in ["sig_momentum", "sig_arb", "sig_mr"]
    }

    # All should be decayed
    assert all(c < 0.8 for c in confidences.values())

    # Arbitrage should decay most (fastest half-life)
    assert confidences["sig_arb"] < confidences["sig_momentum"]
    assert confidences["sig_arb"] < confidences["sig_mr"]

    # Get statistics
    stats = decay_model.get_statistics()
    assert stats["active_signals"] == 3
    assert stats["total_decayed"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
