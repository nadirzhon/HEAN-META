#!/usr/bin/env python3
"""
Verification script for Symbiont X stub fixes.

Demonstrates that test worlds and stress tests now return honest
zero results instead of fake positive metrics.
"""

from src.hean.symbiont_x.adversarial_twin.test_worlds import (
    ReplayWorld,
    PaperWorld,
    MicroRealWorld,
    WorldType
)
from src.hean.symbiont_x.adversarial_twin.stress_tests import StressTestSuite


def verify_replay_world():
    """Verify ReplayWorld returns honest zero results"""
    print("\n=== Testing ReplayWorld ===")

    world = ReplayWorld(
        historical_data={'BTCUSDT': []},
        initial_capital=10000
    )

    strategy_config = {
        'strategy_id': 'test_strategy_1',
        'name': 'Test Strategy'
    }

    result = world.run_test(strategy_config, duration_seconds=60)

    print(f"World Type: {result.world_type}")
    print(f"Passed: {result.passed}")
    print(f"Failure Reason: {result.failure_reason}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate}")
    print(f"Total PnL: {result.total_pnl}")
    print(f"Sharpe Ratio: {result.sharpe_ratio}")
    print(f"Survival Score: {result.get_survival_score()}")
    print(f"Metrics: {result.metrics}")

    # Assertions
    assert result.passed is False, "Should not pass"
    assert result.total_pnl == 0.0, "PnL should be zero"
    assert result.win_rate == 0.0, "Win rate should be zero"
    assert result.sharpe_ratio == 0.0, "Sharpe should be zero"
    assert result.get_survival_score() == 0.0, "Survival score should be zero"
    assert result.metrics.get('is_not_implemented') is True, "Should be flagged as not implemented"

    print("✅ ReplayWorld verification PASSED")


def verify_paper_world():
    """Verify PaperWorld returns honest zero results"""
    print("\n=== Testing PaperWorld ===")

    world = PaperWorld(
        exchange_connector=None,  # Not used in stub
        initial_capital=10000
    )

    strategy_config = {
        'strategy_id': 'test_strategy_2',
        'name': 'Test Strategy'
    }

    result = world.run_test(strategy_config, duration_seconds=60)

    print(f"World Type: {result.world_type}")
    print(f"Passed: {result.passed}")
    print(f"Failure Reason: {result.failure_reason}")
    print(f"Total PnL: {result.total_pnl}")
    print(f"Survival Score: {result.get_survival_score()}")
    print(f"Metrics: {result.metrics}")

    # Assertions
    assert result.passed is False, "Should not pass"
    assert result.total_pnl == 0.0, "PnL should be zero"
    assert result.get_survival_score() == 0.0, "Survival score should be zero"
    assert result.metrics.get('is_not_implemented') is True, "Should be flagged as not implemented"

    print("✅ PaperWorld verification PASSED")


def verify_micro_real_world():
    """Verify MicroRealWorld returns honest zero results"""
    print("\n=== Testing MicroRealWorld ===")

    world = MicroRealWorld(
        exchange_connector=None,  # Not used in stub
        initial_capital=100,
        max_position_size=50
    )

    strategy_config = {
        'strategy_id': 'test_strategy_3',
        'name': 'Test Strategy'
    }

    result = world.run_test(strategy_config, duration_seconds=60)

    print(f"World Type: {result.world_type}")
    print(f"Passed: {result.passed}")
    print(f"Failure Reason: {result.failure_reason}")
    print(f"Total PnL: {result.total_pnl}")
    print(f"Survival Score: {result.get_survival_score()}")
    print(f"Metrics: {result.metrics}")

    # Test place_order with small order (should reach stub rejection)
    order = {'quantity': 0.001, 'price': 30000, 'side': 'buy'}  # $30 order
    order_result = world.place_order(order)
    print(f"Order Status: {order_result['status']}")
    print(f"Order Rejection Reason: {order_result['reason']}")
    print(f"Is Stub: {order_result.get('is_stub')}")

    # Assertions
    assert result.passed is False, "Should not pass"
    assert result.total_pnl == 0.0, "PnL should be zero"
    assert result.get_survival_score() == 0.0, "Survival score should be zero"
    assert result.metrics.get('is_not_implemented') is True, "Should be flagged as not implemented"
    assert order_result['status'] == 'rejected', "Orders should be rejected"
    assert order_result['reason'] == 'real_trading_not_implemented', "Should be stub rejection"
    assert order_result.get('is_stub') is True, "Order should be flagged as stub"

    print("✅ MicroRealWorld verification PASSED")


def verify_stress_tests():
    """Verify StressTestSuite returns honest zero results"""
    print("\n=== Testing StressTestSuite ===")

    suite = StressTestSuite()

    strategy_config = {
        'strategy_id': 'test_strategy_4',
        'name': 'Test Strategy'
    }

    results = suite.run_all_tests(strategy_config)

    print(f"Total Stress Tests: {len(results)}")

    # Check first result in detail
    first_result = results[0]
    print(f"\nFirst Test ({first_result.test_type.value}):")
    print(f"Survived: {first_result.survived}")
    print(f"Failure Reason: {first_result.failure_reason}")
    print(f"Max Loss: {first_result.max_loss_during_stress}")
    print(f"Is Simulated: {first_result.is_simulated}")
    print(f"Robustness Score: {first_result.get_robustness_score()}")

    # Get statistics
    stats = suite.get_statistics()
    print(f"\nStatistics:")
    print(f"Total Tests: {stats['total_tests']}")
    print(f"Passed Tests: {stats['passed_tests']}")
    print(f"Simulated Tests: {stats['simulated_tests']}")
    print(f"Overall Robustness: {stats['overall_robustness']}")
    print(f"Implementation Status: {stats['implementation_status']}")

    # Assertions
    for result in results:
        assert result.survived is False, "No test should survive"
        assert result.is_simulated is True, "All should be flagged as simulated"
        assert result.get_robustness_score() == 0.0, "Robustness score should be zero"

    assert stats['simulated_tests'] == stats['total_tests'], "All tests should be simulated"
    assert stats['overall_robustness'] == 0.0, "Overall robustness should be zero"
    assert stats['implementation_status'] == 'stub', "Should be flagged as stub"

    print("✅ StressTestSuite verification PASSED")


def main():
    """Run all verifications"""
    print("=" * 60)
    print("SYMBIONT X STUB FIXES VERIFICATION")
    print("=" * 60)

    try:
        verify_replay_world()
        verify_paper_world()
        verify_micro_real_world()
        verify_stress_tests()

        print("\n" + "=" * 60)
        print("ALL VERIFICATIONS PASSED ✅")
        print("=" * 60)
        print("\nConclusion:")
        print("- All test worlds return honest zero results")
        print("- All stress tests return honest zero robustness scores")
        print("- Clear flags identify stub implementations")
        print("- Evolution engine will NOT promote untested strategies")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
