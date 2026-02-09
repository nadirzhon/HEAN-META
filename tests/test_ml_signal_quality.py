"""Tests for ML Signal Quality Scorer.

Tests feature extraction and signal quality scoring functionality.
"""

import pytest
from datetime import datetime, timedelta

from hean.core.types import Signal
from hean.ml.feature_extraction import FeatureExtractor, MarketFeatures
from hean.ml.signal_quality_scorer import SignalQualityScorer


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance."""
        return FeatureExtractor(window_size=100)

    def test_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor._window_size == 100
        assert len(extractor._price_history) == 0
        assert len(extractor._volume_history) == 0

    def test_update_price(self, extractor):
        """Test price update."""
        symbol = "BTCUSDT"
        now = datetime.utcnow()

        # Update prices
        extractor.update_price(symbol, 50000.0, now)
        extractor.update_price(symbol, 50100.0, now + timedelta(seconds=1))
        extractor.update_price(symbol, 50050.0, now + timedelta(seconds=2))

        assert symbol in extractor._price_history
        assert len(extractor._price_history[symbol]) == 3
        assert extractor._price_history[symbol][-1] == 50050.0

        # Check tick direction
        assert extractor._last_tick_direction[symbol] == -1  # Price went down

    def test_update_volume(self, extractor):
        """Test volume update."""
        symbol = "BTCUSDT"

        extractor.update_volume(symbol, 100.0)
        extractor.update_volume(symbol, 150.0)
        extractor.update_volume(symbol, 120.0)

        assert symbol in extractor._volume_history
        assert len(extractor._volume_history[symbol]) == 3

    def test_update_orderbook(self, extractor):
        """Test orderbook update."""
        symbol = "BTCUSDT"

        extractor.update_orderbook(symbol, 49900.0, 50100.0)
        extractor.update_orderbook(symbol, 49950.0, 50050.0)

        assert symbol in extractor._bid_history
        assert symbol in extractor._ask_history
        assert len(extractor._bid_history[symbol]) == 2

    def test_extract_features_basic(self, extractor):
        """Test basic feature extraction."""
        symbol = "BTCUSDT"
        now = datetime.utcnow()

        # Setup market data
        for i in range(10):
            price = 50000.0 + (i * 10)
            extractor.update_price(symbol, price, now - timedelta(minutes=10-i))
            extractor.update_volume(symbol, 100.0 + i)
            extractor.update_orderbook(symbol, price - 50, price + 50)

        # Create signal
        signal = Signal(
            strategy_id="test_strategy",
            symbol=symbol,
            side="buy",
            entry_price=50100.0,
            stop_loss=49900.0,
            take_profit=50500.0,
        )

        context = {
            "regime": "NORMAL",
            "volatility_percentile": 50.0,
        }

        # Extract features
        features = extractor.extract_features(signal, context)

        assert isinstance(features, MarketFeatures)
        assert features.hour_of_day >= 0 and features.hour_of_day < 24
        assert features.regime_normal is True
        assert features.signal_strength > 0  # Has take profit

    def test_feature_array_conversion(self):
        """Test feature to array conversion."""
        features = MarketFeatures()
        array = features.to_array()

        assert len(array) == len(MarketFeatures.get_feature_names())
        assert array.dtype == float

    def test_momentum_calculation(self, extractor):
        """Test momentum calculation."""
        symbol = "BTCUSDT"
        now = datetime.utcnow()

        # Create upward price trend
        prices = [50000.0, 50100.0, 50200.0, 50300.0]
        timestamps = [now - timedelta(minutes=i) for i in range(len(prices)-1, -1, -1)]

        for price, ts in zip(prices, timestamps):
            extractor.update_price(symbol, price, ts)

        # Calculate momentum
        momentum = extractor._calculate_momentum(
            prices=list(extractor._price_history[symbol]),
            timestamps=list(extractor._price_timestamps[symbol]),
            now=now,
            minutes=5,
        )

        # Should be positive (upward trend)
        assert momentum > 0

    def test_volatility_calculation(self, extractor):
        """Test volatility calculation."""
        symbol = "BTCUSDT"
        now = datetime.utcnow()

        # Create volatile prices
        prices = [50000.0, 51000.0, 49500.0, 50500.0, 49000.0]
        timestamps = [now - timedelta(minutes=i) for i in range(len(prices)-1, -1, -1)]

        for price, ts in zip(prices, timestamps):
            extractor.update_price(symbol, price, ts)

        # Calculate volatility
        volatility = extractor._calculate_volatility(
            prices=list(extractor._price_history[symbol]),
            timestamps=list(extractor._price_timestamps[symbol]),
            now=now,
            minutes=5,
        )

        # Should be non-zero for volatile prices
        assert volatility > 0


class TestSignalQualityScorer:
    """Tests for SignalQualityScorer."""

    @pytest.fixture
    def extractor(self):
        """Create feature extractor."""
        return FeatureExtractor(window_size=100)

    @pytest.fixture
    def scorer(self, extractor):
        """Create signal quality scorer."""
        return SignalQualityScorer(
            feature_extractor=extractor,
            online_learning=False,  # Disable for tests
            min_training_samples=50,
        )

    def test_initialization(self, scorer):
        """Test scorer initialization."""
        assert scorer._predictions_made == 0
        assert scorer._model_version == 0
        assert len(scorer._training_buffer) == 0

    def test_score_signal_basic(self, scorer, extractor):
        """Test basic signal scoring."""
        symbol = "BTCUSDT"
        now = datetime.utcnow()

        # Setup minimal market data
        extractor.update_price(symbol, 50000.0, now)
        extractor.update_volume(symbol, 100.0)
        extractor.update_orderbook(symbol, 49950.0, 50050.0)

        # Create signal
        signal = Signal(
            strategy_id="test_strategy",
            symbol=symbol,
            side="buy",
            entry_price=50000.0,
            stop_loss=49800.0,
            take_profit=50400.0,
        )

        context = {
            "regime": "NORMAL",
            "volatility_percentile": 50.0,
        }

        # Score signal
        score = scorer.score_signal(signal, context)

        assert 0.0 <= score <= 1.0
        assert scorer._predictions_made == 1

    def test_score_signal_multiple(self, scorer, extractor):
        """Test multiple signal scores."""
        symbol = "BTCUSDT"
        now = datetime.utcnow()

        # Setup market data
        extractor.update_price(symbol, 50000.0, now)
        extractor.update_volume(symbol, 100.0)
        extractor.update_orderbook(symbol, 49950.0, 50050.0)

        context = {
            "regime": "NORMAL",
            "volatility_percentile": 50.0,
        }

        # Score multiple signals
        scores = []
        for i in range(5):
            signal = Signal(
                strategy_id=f"strategy_{i}",
                symbol=symbol,
                side="buy",
                entry_price=50000.0 + (i * 10),
                stop_loss=49800.0,
                take_profit=50400.0,
            )
            score = scorer.score_signal(signal, context)
            scores.append(score)

        assert len(scores) == 5
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert scorer._predictions_made == 5

    def test_record_outcome_no_learning(self, scorer, extractor):
        """Test outcome recording without online learning."""
        symbol = "BTCUSDT"
        signal = Signal(
            strategy_id="test_strategy",
            symbol=symbol,
            side="buy",
            entry_price=50000.0,
            stop_loss=49800.0,
            take_profit=50400.0,
        )

        context = {"regime": "NORMAL"}

        # Record outcome (should not add to buffer since learning disabled)
        scorer.record_outcome(
            signal_id="sig_1",
            signal=signal,
            context=context,
            success=True,
            pnl_pct=0.8,
        )

        assert len(scorer._training_buffer) == 0  # Learning disabled

    def test_record_outcome_with_learning(self, extractor):
        """Test outcome recording with online learning."""
        scorer = SignalQualityScorer(
            feature_extractor=extractor,
            online_learning=True,  # Enable learning
            min_training_samples=50,
        )

        symbol = "BTCUSDT"
        now = datetime.utcnow()
        extractor.update_price(symbol, 50000.0, now)

        signal = Signal(
            strategy_id="test_strategy",
            symbol=symbol,
            side="buy",
            entry_price=50000.0,
            stop_loss=49800.0,
            take_profit=50400.0,
        )

        context = {"regime": "NORMAL"}

        # Record outcomes
        for i in range(10):
            scorer.record_outcome(
                signal_id=f"sig_{i}",
                signal=signal,
                context=context,
                success=(i % 2 == 0),  # Alternate wins/losses
                pnl_pct=0.5 if (i % 2 == 0) else -0.5,
            )

        assert len(scorer._training_buffer) == 10

    def test_get_model_stats(self, scorer):
        """Test getting model statistics."""
        stats = scorer.get_model_stats()

        assert "model_version" in stats
        assert "predictions_made" in stats
        assert "training_samples" in stats
        assert stats["online_learning"] is False

    def test_get_feature_importance(self, scorer):
        """Test getting feature importance."""
        importance = scorer.get_feature_importance()

        assert len(importance) == len(MarketFeatures.get_feature_names())
        assert all(0.0 <= v <= 1.0 for v in importance.values())
        # Should sum to ~1.0 (normalized)
        assert abs(sum(importance.values()) - 1.0) < 0.01


class TestEnhancedMultiFactorConfirmation:
    """Tests for enhanced multi-factor confirmation with ML."""

    @pytest.fixture
    def mock_base_confirmation(self):
        """Mock base confirmation that always returns confirmed."""
        class MockConfirmation:
            def confirm(self, signal, context):
                class Result:
                    confidence = 0.7
                    confirmed = True
                    metadata = {}
                return Result()

        return MockConfirmation()

    @pytest.fixture
    def scorer(self):
        """Create scorer."""
        extractor = FeatureExtractor()
        return SignalQualityScorer(extractor, online_learning=False)

    def test_enhanced_confirmation(self, mock_base_confirmation, scorer):
        """Test enhanced confirmation combines base and ML scores."""
        from hean.ml.signal_quality_scorer import EnhancedMultiFactorConfirmation

        enhanced = EnhancedMultiFactorConfirmation(
            base_confirmation=mock_base_confirmation,
            signal_quality_scorer=scorer,
            ml_weight=0.3,
        )

        signal = Signal(
            strategy_id="test",
            symbol="BTCUSDT",
            side="buy",
            entry_price=50000.0,
            stop_loss=49800.0,
            take_profit=50400.0,
        )

        context = {"regime": "NORMAL"}

        result = enhanced.confirm(signal, context)

        # Result should have combined confidence
        assert hasattr(result, 'confidence')
        assert 0.0 <= result.confidence <= 1.0
        assert 'ml_quality_score' in result.metadata
        assert 'base_confidence' in result.metadata


@pytest.mark.asyncio
async def test_phase2_integration():
    """Integration test for Phase 2 components."""
    # Setup
    extractor = FeatureExtractor(window_size=50)
    scorer = SignalQualityScorer(extractor, online_learning=False)

    # Add market data
    symbol = "BTCUSDT"
    now = datetime.utcnow()

    for i in range(20):
        price = 50000.0 + (i * 5)
        extractor.update_price(symbol, price, now - timedelta(seconds=20-i))
        extractor.update_volume(symbol, 100.0 + i)
        extractor.update_orderbook(symbol, price - 25, price + 25)

    # Create and score signal
    signal = Signal(
        strategy_id="integration_test",
        symbol=symbol,
        side="buy",
        entry_price=50100.0,
        stop_loss=49900.0,
        take_profit=50500.0,
    )

    context = {
        "regime": "NORMAL",
        "volatility_percentile": 60.0,
    }

    score = scorer.score_signal(signal, context)

    # Verify
    assert 0.0 <= score <= 1.0
    assert scorer._predictions_made == 1

    # Check feature extraction worked
    features = extractor.extract_features(signal, context)
    assert features.regime_normal is True
    assert features.volatility_percentile == 60.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
