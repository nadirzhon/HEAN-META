"""
Тестовый скрипт для проверки SYMBIONT X

Проверяет импорты и базовую функциональность
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Тест импортов всех компонентов"""

    print("🧪 Testing imports...")

    tests = []

    # Test 1: Main Symbiont
    try:
        from hean.symbiont_x import HEANSymbiontX
        print("✅ Main Symbiont imported")
        tests.append(("Main Symbiont", True, None))
    except Exception as e:
        print(f"❌ Main Symbiont import failed: {e}")
        tests.append(("Main Symbiont", False, str(e)))

    # Test 2: Nervous System
    try:
        from hean.symbiont_x.nervous_system import BybitWSConnector, HealthSensorArray
        print("✅ Nervous System imported")
        tests.append(("Nervous System", True, None))
    except Exception as e:
        print(f"❌ Nervous System import failed: {e}")
        tests.append(("Nervous System", False, str(e)))

    # Test 3: Regime Brain
    try:
        from hean.symbiont_x.regime_brain import MarketRegime, FeatureExtractor, RegimeClassifier
        print("✅ Regime Brain imported")
        tests.append(("Regime Brain", True, None))
    except Exception as e:
        print(f"❌ Regime Brain import failed: {e}")
        tests.append(("Regime Brain", False, str(e)))

    # Test 4: Genome Lab
    try:
        from hean.symbiont_x.genome_lab import StrategyGenome, MutationEngine, CrossoverEngine, EvolutionEngine
        print("✅ Genome Lab imported")
        tests.append(("Genome Lab", True, None))
    except Exception as e:
        print(f"❌ Genome Lab import failed: {e}")
        tests.append(("Genome Lab", False, str(e)))

    # Test 5: Adversarial Twin
    try:
        from hean.symbiont_x.adversarial_twin import ReplayWorld, PaperWorld, MicroRealWorld, StressTestSuite
        print("✅ Adversarial Twin imported")
        tests.append(("Adversarial Twin", True, None))
    except Exception as e:
        print(f"❌ Adversarial Twin import failed: {e}")
        tests.append(("Adversarial Twin", False, str(e)))

    # Test 6: Capital Allocator
    try:
        from hean.symbiont_x.capital_allocator import Portfolio, CapitalAllocator, PortfolioRebalancer
        print("✅ Capital Allocator imported")
        tests.append(("Capital Allocator", True, None))
    except Exception as e:
        print(f"❌ Capital Allocator import failed: {e}")
        tests.append(("Capital Allocator", False, str(e)))

    # Test 7: Immune System
    try:
        from hean.symbiont_x.immune_system import RiskConstitution, ReflexSystem, CircuitBreakerSystem
        print("✅ Immune System imported")
        tests.append(("Immune System", True, None))
    except Exception as e:
        print(f"❌ Immune System import failed: {e}")
        tests.append(("Immune System", False, str(e)))

    # Test 8: Decision Ledger
    try:
        from hean.symbiont_x.decision_ledger import Decision, DecisionLedger, DecisionAnalyzer
        print("✅ Decision Ledger imported")
        tests.append(("Decision Ledger", True, None))
    except Exception as e:
        print(f"❌ Decision Ledger import failed: {e}")
        tests.append(("Decision Ledger", False, str(e)))

    # Test 9: Execution Kernel
    try:
        from hean.symbiont_x.execution_kernel import ExecutionKernel, OrderRequest
        print("✅ Execution Kernel imported")
        tests.append(("Execution Kernel", True, None))
    except Exception as e:
        print(f"❌ Execution Kernel import failed: {e}")
        tests.append(("Execution Kernel", False, str(e)))

    # Test 10: KPI System
    try:
        from hean.symbiont_x.kpi_system import KPISystem
        print("✅ KPI System imported")
        tests.append(("KPI System", True, None))
    except Exception as e:
        print(f"❌ KPI System import failed: {e}")
        tests.append(("KPI System", False, str(e)))

    return tests


def test_basic_functionality():
    """Тест базовой функциональности"""

    print("\n🧪 Testing basic functionality...")

    tests = []

    # Test: Create random genome
    try:
        from hean.symbiont_x.genome_lab import create_random_genome
        genome = create_random_genome("TestStrategy")
        assert genome.name == "TestStrategy"
        assert len(genome.genes) > 0
        print("✅ Random genome creation works")
        tests.append(("Random Genome", True, None))
    except Exception as e:
        print(f"❌ Random genome creation failed: {e}")
        tests.append(("Random Genome", False, str(e)))

    # Test: Portfolio creation
    try:
        from hean.symbiont_x.capital_allocator import Portfolio
        portfolio = Portfolio(
            portfolio_id="test",
            name="Test Portfolio",
            total_capital=10000
        )
        assert portfolio.total_capital == 10000
        print("✅ Portfolio creation works")
        tests.append(("Portfolio", True, None))
    except Exception as e:
        print(f"❌ Portfolio creation failed: {e}")
        tests.append(("Portfolio", False, str(e)))

    # Test: KPI System
    try:
        from hean.symbiont_x.kpi_system import KPISystem
        kpi = KPISystem()
        kpi.update_survival_score(75.0, {'portfolio_sharpe': 1.5})
        assert kpi.survival_score is not None
        print("✅ KPI System works")
        tests.append(("KPI System", True, None))
    except Exception as e:
        print(f"❌ KPI System failed: {e}")
        tests.append(("KPI System", False, str(e)))

    # Test: Decision Ledger
    try:
        from hean.symbiont_x.decision_ledger import DecisionLedger, Decision, DecisionType
        import uuid
        ledger = DecisionLedger()
        decision = Decision(
            decision_id=str(uuid.uuid4()),
            decision_type=DecisionType.OPEN_POSITION,
            reason="Test decision"
        )
        ledger.record_decision(decision)
        assert ledger.total_decisions == 1
        print("✅ Decision Ledger works")
        tests.append(("Decision Ledger", True, None))
    except Exception as e:
        print(f"❌ Decision Ledger failed: {e}")
        tests.append(("Decision Ledger", False, str(e)))

    return tests


def print_summary(import_tests, func_tests):
    """Выводит итоговую статистику"""

    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

    all_tests = import_tests + func_tests
    passed = sum(1 for t in all_tests if t[1])
    failed = sum(1 for t in all_tests if not t[1])
    total = len(all_tests)

    print(f"\nTotal Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

    if failed > 0:
        print("\n❌ FAILED TESTS:")
        for name, success, error in all_tests:
            if not success:
                print(f"  - {name}: {error}")

    print("\n" + "="*60)

    return failed == 0


def main():
    """Главная функция"""

    print("="*60)
    print("🧬 HEAN SYMBIONT X - TEST SUITE")
    print("="*60)

    try:
        # Run import tests
        import_tests = test_imports()

        # Run functionality tests
        func_tests = test_basic_functionality()

        # Print summary
        success = print_summary(import_tests, func_tests)

        if success:
            print("\n🎉 ALL TESTS PASSED! System is ready.")
            return 0
        else:
            print("\n⚠️  SOME TESTS FAILED. Check errors above.")
            return 1

    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
