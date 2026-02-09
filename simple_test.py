"""
Простая проверка структуры проекта без внешних зависимостей
"""

import sys
from pathlib import Path

def check_file_structure():
    """Проверка структуры файлов"""

    print("="*60)
    print("🔍 Checking HEAN SYMBIONT X file structure...")
    print("="*60)

    base_path = Path("src/hean/symbiont_x")

    required_files = [
        # Main
        "__init__.py",
        "symbiont.py",
        "kpi_system.py",

        # Nervous System
        "nervous_system/__init__.py",
        "nervous_system/event_envelope.py",
        "nervous_system/ws_connectors.py",
        "nervous_system/health_sensors.py",

        # Regime Brain
        "regime_brain/__init__.py",
        "regime_brain/regime_types.py",
        "regime_brain/features.py",
        "regime_brain/classifier.py",

        # Genome Lab
        "genome_lab/__init__.py",
        "genome_lab/genome_types.py",
        "genome_lab/mutation_engine.py",
        "genome_lab/crossover.py",
        "genome_lab/evolution_engine.py",

        # Adversarial Twin
        "adversarial_twin/__init__.py",
        "adversarial_twin/test_worlds.py",
        "adversarial_twin/stress_tests.py",
        "adversarial_twin/survival_score.py",

        # Capital Allocator
        "capital_allocator/__init__.py",
        "capital_allocator/portfolio.py",
        "capital_allocator/allocator.py",
        "capital_allocator/rebalancer.py",

        # Immune System
        "immune_system/__init__.py",
        "immune_system/constitution.py",
        "immune_system/reflexes.py",
        "immune_system/circuit_breakers.py",

        # Decision Ledger
        "decision_ledger/__init__.py",
        "decision_ledger/decision_types.py",
        "decision_ledger/ledger.py",
        "decision_ledger/replay.py",
        "decision_ledger/analysis.py",

        # Execution Kernel
        "execution_kernel/__init__.py",
        "execution_kernel/executor.py",
    ]

    missing = []
    found = []

    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            found.append(file_path)
            print(f"  ✅ {file_path}")
        else:
            missing.append(file_path)
            print(f"  ❌ MISSING: {file_path}")

    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    print(f"Total files required: {len(required_files)}")
    print(f"✅ Found: {len(found)}")
    print(f"❌ Missing: {len(missing)}")

    if missing:
        print("\n❌ MISSING FILES:")
        for f in missing:
            print(f"  - {f}")
        return False
    else:
        print("\n🎉 ALL FILES PRESENT!")
        return True


def check_syntax():
    """Проверка синтаксиса Python файлов"""

    print("\n" + "="*60)
    print("🔍 Checking Python syntax...")
    print("="*60)

    import py_compile

    base_path = Path("src/hean/symbiont_x")
    py_files = list(base_path.rglob("*.py"))

    errors = []
    success = []

    for py_file in py_files:
        try:
            py_compile.compile(str(py_file), doraise=True)
            success.append(py_file)
            print(f"  ✅ {py_file.relative_to('src/hean/symbiont_x')}")
        except py_compile.PyCompileError as e:
            errors.append((py_file, str(e)))
            print(f"  ❌ {py_file.relative_to('src/hean/symbiont_x')}: {e}")

    print("\n" + "="*60)
    print("📊 SYNTAX CHECK RESULTS")
    print("="*60)
    print(f"Total Python files: {len(py_files)}")
    print(f"✅ Valid syntax: {len(success)}")
    print(f"❌ Syntax errors: {len(errors)}")

    if errors:
        print("\n❌ SYNTAX ERRORS:")
        for file, error in errors:
            print(f"  - {file}: {error}")
        return False
    else:
        print("\n🎉 ALL FILES HAVE VALID SYNTAX!")
        return True


def count_lines():
    """Подсчёт строк кода"""

    print("\n" + "="*60)
    print("📏 Counting lines of code...")
    print("="*60)

    base_path = Path("src/hean/symbiont_x")
    py_files = list(base_path.rglob("*.py"))

    total_lines = 0
    component_lines = {}

    for py_file in py_files:
        with open(py_file, 'r', encoding='utf-8') as f:
            lines = len(f.readlines())
            total_lines += lines

            # Get component name
            parts = py_file.relative_to(base_path).parts
            if len(parts) > 1:
                component = parts[0]
            else:
                component = "main"

            if component not in component_lines:
                component_lines[component] = 0
            component_lines[component] += lines

    print("\n📊 Lines of code by component:")
    for component, lines in sorted(component_lines.items(), key=lambda x: x[1], reverse=True):
        print(f"  {component:30s}: {lines:5d} lines")

    print("\n" + "="*60)
    print(f"📊 TOTAL LINES OF CODE: {total_lines}")
    print("="*60)


def main():
    """Main function"""

    print("\n" + "="*60)
    print("🧬 HEAN SYMBIONT X - STRUCTURE CHECK")
    print("="*60)

    # Check file structure
    structure_ok = check_file_structure()

    # Check syntax
    syntax_ok = check_syntax()

    # Count lines
    count_lines()

    # Final summary
    print("\n" + "="*60)
    print("🏁 FINAL SUMMARY")
    print("="*60)

    if structure_ok and syntax_ok:
        print("✅ ALL CHECKS PASSED!")
        print("\nProject structure is VALID ✨")
        print("All Python files have valid syntax ✅")
        print("\nREADY for:")
        print("  1. Dependency installation")
        print("  2. Integration with Bybit API")
        print("  3. Unit testing")
        print("  4. Docker deployment")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        if not structure_ok:
            print("  - File structure incomplete")
        if not syntax_ok:
            print("  - Syntax errors found")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
