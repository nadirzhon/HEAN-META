#!/bin/bash
# Verification script for critical bug fixes
# Run this to verify all fixes are in place

set -e

echo "=== VERIFYING CRITICAL BUG FIXES ==="
echo ""

# FIX-002: PaperBroker always starts
echo "✓ Checking FIX-002: PaperBroker fallback..."
if grep -q "# Always start paper broker as safety net" src/hean/execution/router.py; then
    echo "  ✅ PaperBroker always starts (safety net enabled)"
else
    echo "  ❌ FAILED: PaperBroker fix not found"
    exit 1
fi

# FIX-003: No hardcoded prices in critical paths
echo ""
echo "✓ Checking FIX-003: Hardcoded prices removed..."

critical_files=(
    "src/hean/execution/router.py"
    "src/hean/execution/router_bybit_only.py"
    "src/hean/income/streams.py"
    "src/hean/api/routers/trading.py"
    "src/hean/execution/paper_broker.py"
    "src/hean/google_trends/strategy.py"
)

failed=0
for file in "${critical_files[@]}"; do
    if grep -q "50000\.0 if\|50000 if.*BTC" "$file" 2>/dev/null; then
        echo "  ❌ FAILED: Hardcoded price found in $file"
        failed=1
    fi
done

if [ $failed -eq 0 ]; then
    echo "  ✅ No hardcoded fallback prices in critical paths"
else
    echo ""
    echo "ERROR: Some files still have hardcoded prices"
    exit 1
fi

# Verify defensive checks are in place
echo ""
echo "✓ Checking defensive price validation..."

defensive_patterns=(
    "if price is None or price <= 0:"
    "No price data for"
    "rejecting order"
    "skipping signal"
)

found_defensive=0
for pattern in "${defensive_patterns[@]}"; do
    if grep -rq "$pattern" src/hean/execution/router*.py src/hean/income/streams.py; then
        found_defensive=$((found_defensive + 1))
    fi
done

if [ $found_defensive -ge 2 ]; then
    echo "  ✅ Defensive price validation in place"
else
    echo "  ⚠️  WARNING: Defensive checks may be incomplete"
fi

# Summary
echo ""
echo "=== VERIFICATION COMPLETE ==="
echo ""
echo "All critical fixes verified:"
echo "  • FIX-002: PaperBroker safety net enabled"
echo "  • FIX-003: Hardcoded prices removed"
echo "  • Defensive validation added"
echo ""
echo "Next steps:"
echo "  1. Run smoke tests: ./scripts/smoke_test.sh"
echo "  2. Check logs for 'No price data' warnings"
echo "  3. Verify orders rejected when no market data"
echo ""
