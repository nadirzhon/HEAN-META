#!/usr/bin/env python3
"""Verification script for Phase 2 implementation.

Checks that all Phase 2 components are properly installed and functional.
"""

import sys
from datetime import datetime


def check_imports():
    """Check that all Phase 2 modules can be imported."""
    print("=" * 60)
    print("Phase 2 Implementation Verification")
    print("=" * 60)
    print()

    modules_to_check = [
        ("ML Feature Extraction", "hean.ml.feature_extraction"),
        ("ML Signal Quality Scorer", "hean.ml.signal_quality_scorer"),
        ("ML Package", "hean.ml"),
        ("Enhanced Adaptive TTL", "hean.execution.adaptive_ttl"),
        ("Signal Decay Model", "hean.execution.signal_decay"),
        ("Phase 2 Metrics", "hean.observability.phase2_metrics"),
    ]

    all_passed = True
    print("1. Import Tests")
    print("-" * 60)

    for name, module_path in modules_to_check:
        try:
            __import__(module_path)
            print(f"✓ {name}: OK")
        except ImportError as e:
            print(f"✗ {name}: FAILED - {e}")
            all_passed = False

    print()
    return all_passed


def check_basic_functionality():
    """Check basic functionality of Phase 2 components."""
    print("2. Functionality Tests")
    print("-" * 60)

    all_passed = True

    # Test FeatureExtractor
    try:
        from hean.ml import FeatureExtractor, MarketFeatures

        extractor = FeatureExtractor(window_size=50)
        extractor.update_price("BTCUSDT", 50000.0, datetime.utcnow())

        features = MarketFeatures()
        array = features.to_array()
        assert len(array) > 0
        print("✓ FeatureExtractor: OK")
    except Exception as e:
        print(f"✗ FeatureExtractor: FAILED - {e}")
        all_passed = False

    # Test SignalQualityScorer
    try:
        from hean.ml import SignalQualityScorer
        from hean.core.types import Signal

        extractor = FeatureExtractor(window_size=50)
        scorer = SignalQualityScorer(extractor, online_learning=False)

        signal = Signal(
            strategy_id="test",
            symbol="BTCUSDT",
            side="buy",
            entry_price=50000.0,
            stop_loss=49800.0,
            take_profit=50400.0,
        )

        score = scorer.score_signal(signal, {"regime": "NORMAL"})
        assert 0.0 <= score <= 1.0
        print("✓ SignalQualityScorer: OK")
    except Exception as e:
        print(f"✗ SignalQualityScorer: FAILED - {e}")
        all_passed = False

    # Test EnhancedAdaptiveTTL
    try:
        from hean.execution.adaptive_ttl import EnhancedAdaptiveTTL

        ttl = EnhancedAdaptiveTTL(base_ttl_ms=500.0)
        optimal_ttl = ttl.calculate_ttl(
            symbol="BTCUSDT",
            spread_bps=5.0,
            volatility_regime="medium",
            current_hour=12,
        )
        assert 200.0 <= optimal_ttl <= 3000.0
        print("✓ EnhancedAdaptiveTTL: OK")
    except Exception as e:
        print(f"✗ EnhancedAdaptiveTTL: FAILED - {e}")
        all_passed = False

    # Test SignalDecayModel
    try:
        from hean.execution.signal_decay import SignalDecayModel

        decay = SignalDecayModel()
        decay.register_signal("sig_1", 0.8, "momentum")
        confidence = decay.get_current_confidence("sig_1")
        assert 0.0 <= confidence <= 1.0
        print("✓ SignalDecayModel: OK")
    except Exception as e:
        print(f"✗ SignalDecayModel: FAILED - {e}")
        all_passed = False

    # Test Phase2Metrics
    try:
        from hean.observability.phase2_metrics import phase2_metrics

        phase2_metrics.record_ml_prediction(0.75, 1, False)
        summary = phase2_metrics.get_summary()
        assert "ml_predictions_made" in summary
        print("✓ Phase2Metrics: OK")
    except Exception as e:
        print(f"✗ Phase2Metrics: FAILED - {e}")
        all_passed = False

    print()
    return all_passed


def check_tests():
    """Check that tests can be discovered."""
    print("3. Test Discovery")
    print("-" * 60)

    import subprocess

    test_files = [
        "tests/test_ml_signal_quality.py",
        "tests/test_enhanced_ttl.py",
        "tests/test_signal_decay.py",
        "tests/test_phase2_metrics.py",
    ]

    all_passed = True
    for test_file in test_files:
        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", test_file, "--collect-only", "-q"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Count collected tests
                lines = result.stdout.split("\n")
                for line in lines:
                    if "test" in line:
                        print(f"✓ {test_file}: Found tests")
                        break
            else:
                print(f"✗ {test_file}: FAILED to collect")
                all_passed = False
        except Exception as e:
            print(f"✗ {test_file}: FAILED - {e}")
            all_passed = False

    print()
    return all_passed


def main():
    """Run all verification checks."""
    results = []

    results.append(("Import Tests", check_imports()))
    results.append(("Functionality Tests", check_basic_functionality()))
    results.append(("Test Discovery", check_tests()))

    print("=" * 60)
    print("Verification Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("🎉 All Phase 2 components verified successfully!")
        print()
        print("Next steps:")
        print("1. Run full test suite: pytest tests/test_*phase2*.py -v")
        print("2. Review PHASE_2_IMPLEMENTATION_SUMMARY.md")
        print("3. Check PHASE_2_QUICK_REFERENCE.md for usage")
        return 0
    else:
        print("⚠️  Some verification checks failed.")
        print("Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
