#!/bin/bash
# HEAN Smoke Test - validates all critical endpoints and features after deployment
# Usage: ./scripts/smoke_test.sh [host] [port]

set -e  # Exit on error

HOST="${1:-localhost}"
PORT="${2:-8000}"
BASE_URL="http://${HOST}:${PORT}"

echo "==================================================================="
echo "HEAN SMOKE TEST"
echo "==================================================================="
echo "Target: $BASE_URL"
echo "Started: $(date)"
echo ""

# Test counter
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Helper function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    echo -n "[TEST $TOTAL_TESTS] $test_name ... "

    if eval "$test_command" > /dev/null 2>&1; then
        echo "✓ PASS"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo "✗ FAIL"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Helper function to check JSON response
check_json() {
    local url="$1"
    local expected_field="$2"

    response=$(curl -s -f "$url" 2>/dev/null || echo "{}")
    echo "$response" | grep -q "\"$expected_field\"" && return 0 || return 1
}

# Helper function to check HTTP status
check_http() {
    local url="$1"
    local expected_status="${2:-200}"

    status=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    [ "$status" = "$expected_status" ] && return 0 || return 1
}

echo "-------------------------------------------------------------------"
echo "1. CORE REST ENDPOINTS"
echo "-------------------------------------------------------------------"

run_test "Health check" "check_http '$BASE_URL/health' 200"
run_test "Telemetry ping" "check_json '$BASE_URL/telemetry/ping' 'status'"
run_test "Telemetry summary" "check_json '$BASE_URL/telemetry/summary' 'total'"
run_test "Trading why" "check_json '$BASE_URL/trading/why' 'engine_state'"
run_test "Portfolio summary" "check_http '$BASE_URL/portfolio/summary' 200"

echo ""
echo "-------------------------------------------------------------------"
echo "2. AI CATALYST ENDPOINTS"
echo "-------------------------------------------------------------------"

run_test "System changelog/today" "check_json '$BASE_URL/system/changelog/today' 'items'"
run_test "System agents" "check_json '$BASE_URL/system/agents' 'agents'"

echo ""
echo "-------------------------------------------------------------------"
echo "3. MARKET DATA"
echo "-------------------------------------------------------------------"

run_test "Market ticker" "check_http '$BASE_URL/market/ticker?symbol=BTCUSDT' 200"

echo ""
echo "-------------------------------------------------------------------"
echo "4. RISK GOVERNOR"
echo "-------------------------------------------------------------------"

run_test "Risk governor status" "check_json '$BASE_URL/risk/governor/status' 'risk_state'"

echo ""
echo "-------------------------------------------------------------------"
echo "5. WEBSOCKET CONNECTION"
echo "-------------------------------------------------------------------"

# Test WebSocket connection (basic ping)
ws_test() {
    # Use websocat or wscat if available, otherwise skip
    if command -v websocat &> /dev/null; then
        echo '{"action":"ping"}' | timeout 3 websocat "ws://${HOST}:${PORT}/ws" 2>/dev/null | grep -q "pong" && return 0
    elif command -v wscat &> /dev/null; then
        echo '{"action":"ping"}' | timeout 3 wscat -c "ws://${HOST}:${PORT}/ws" 2>/dev/null | grep -q "pong" && return 0
    else
        # Skip if no WS client available
        echo "SKIP (no websocat/wscat)"
        return 0
    fi
    return 1
}

run_test "WebSocket connection" "ws_test"

echo ""
echo "-------------------------------------------------------------------"
echo "6. ENGINE CONTROL (if available)"
echo "-------------------------------------------------------------------"

# Try pause (may fail if not running, that's OK for smoke test)
pause_test() {
    status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/engine/pause" 2>/dev/null || echo "000")
    # Accept 200 (success) or 409 (conflict - already paused/stopped)
    [ "$status" = "200" ] || [ "$status" = "409" ] && return 0 || return 1
}

run_test "Engine pause endpoint" "pause_test"

echo ""
echo "-------------------------------------------------------------------"
echo "7. MULTI-SYMBOL SUPPORT"
echo "-------------------------------------------------------------------"

# Check if trading/why returns multi_symbol data
multi_symbol_test() {
    response=$(curl -s -f "$BASE_URL/trading/why" 2>/dev/null || echo "{}")
    echo "$response" | grep -q "\"multi_symbol\"" && return 0 || return 1
}

run_test "Multi-symbol data in /trading/why" "multi_symbol_test"

echo ""
echo "==================================================================="
echo "SMOKE TEST SUMMARY"
echo "==================================================================="
echo "Total tests:  $TOTAL_TESTS"
echo "Passed:       $PASSED_TESTS"
echo "Failed:       $FAILED_TESTS"
echo "Completed:    $(date)"
echo ""

if [ "$FAILED_TESTS" -eq 0 ]; then
    echo "✅ ALL TESTS PASSED - System is operational"
    echo ""
    exit 0
else
    echo "❌ SOME TESTS FAILED - Review output above"
    echo ""
    exit 1
fi
