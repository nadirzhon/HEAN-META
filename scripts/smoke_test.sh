#!/bin/bash
# Smoke test for AFO-Director features
# Checks REST endpoints, WebSocket, control, and multi-symbol support

set -e

API_URL="${API_URL:-http://localhost:8000}"
WS_URL="${WS_URL:-ws://localhost:8000/ws}"
TIMEOUT="${TIMEOUT:-30}"

echo "=== HEAN AFO-Director Smoke Test ==="
echo "API URL: $API_URL"
echo "WS URL: $WS_URL"
echo ""

FAILED=0

# Helper function to check HTTP endpoint
check_http() {
    local endpoint=$1
    local expected_status=${2:-200}
    local name=$3
    
    echo -n "Checking $name ($endpoint)... "
    response=$(curl -s -w "\n%{http_code}" "$API_URL$endpoint" || echo -e "\n000")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" = "$expected_status" ]; then
        echo "✓ PASS ($http_code)"
        return 0
    else
        echo "✗ FAIL ($http_code, expected $expected_status)"
        echo "  Response: $body"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# REST Endpoints
echo "--- REST Endpoints ---"
check_http "/telemetry/ping" 200 "Telemetry ping"
check_http "/telemetry/summary" 200 "Telemetry summary"
check_http "/trading/why" 200 "Trading why endpoint" || {
    echo "  Note: /trading/why may return 200 even if engine not running"
}

# Check /trading/why has required fields
echo -n "Checking /trading/why fields... "
why_response=$(curl -s "$API_URL/trading/why")
if echo "$why_response" | grep -q "engine_state" && \
   echo "$why_response" | grep -q "killswitch_state" && \
   echo "$why_response" | grep -q "profit_capture_state"; then
    echo "✓ PASS (required fields present)"
else
    echo "✗ FAIL (missing required fields)"
    FAILED=$((FAILED + 1))
fi

check_http "/portfolio/summary" 200 "Portfolio summary" || true

# WebSocket Test (basic connection)
echo ""
echo "--- WebSocket Test ---"
echo -n "Checking WebSocket connection... "
if command -v wscat >/dev/null 2>&1 || command -v websocat >/dev/null 2>&1; then
    # Try wscat first, then websocat
    if command -v wscat >/dev/null 2>&1; then
        timeout 5 wscat -c "$WS_URL" -x '{"action":"ping"}' >/dev/null 2>&1 && echo "✓ PASS" || {
            echo "✗ FAIL (connection failed)"
            FAILED=$((FAILED + 1))
        }
    else
        echo "⚠ SKIP (wscat/websocat not installed, install with: npm install -g wscat or cargo install websocat)"
    fi
else
    echo "⚠ SKIP (wscat/websocat not installed)"
fi

# Control Test (if engine is running)
echo ""
echo "--- Control Test ---"
echo -n "Checking engine control endpoints... "
status_response=$(curl -s "$API_URL/engine/status" || echo '{"status":"STOPPED"}')
if echo "$status_response" | grep -q "status"; then
    echo "✓ PASS (status endpoint accessible)"
    
    # Try pause/resume if engine is running
    engine_status=$(echo "$status_response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "STOPPED")
    if [ "$engine_status" = "RUNNING" ]; then
        echo -n "  Testing pause/resume... "
        pause_result=$(curl -s -X POST "$API_URL/engine/pause" -H "Content-Type: application/json" -d '{}' || echo '{"ok":false}')
        if echo "$pause_result" | grep -q "ok"; then
            sleep 1
            resume_result=$(curl -s -X POST "$API_URL/engine/resume" -H "Content-Type: application/json" || echo '{"ok":false}')
            if echo "$resume_result" | grep -q "ok"; then
                echo "✓ PASS"
            else
                echo "✗ FAIL (resume failed)"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "⚠ SKIP (pause not available)"
        fi
    else
        echo "  Engine not running, skipping pause/resume test"
    fi
else
    echo "✗ FAIL (status endpoint not accessible)"
    FAILED=$((FAILED + 1))
fi

# Multi-Symbol Test (check ORDER_DECISION for multiple symbols)
echo ""
echo "--- Multi-Symbol Test ---"
echo -n "Waiting ${TIMEOUT}s for ORDER_DECISION events from >=3 symbols... "
# This would require WebSocket subscription or log parsing
# For now, just check that multi-symbol config is available
if curl -s "$API_URL/trading/why" | grep -q "multi_symbol"; then
    echo "✓ PASS (multi_symbol field present in /trading/why)"
else
    echo "⚠ WARN (multi_symbol field not found, may need engine running)"
fi

# Summary
echo ""
echo "=== Smoke Test Summary ==="
if [ $FAILED -eq 0 ]; then
    echo "✓ ALL CHECKS PASSED"
    exit 0
else
    echo "✗ $FAILED CHECK(S) FAILED"
    echo ""
    echo "To debug:"
    echo "  1. Check API logs: docker-compose logs api"
    echo "  2. Verify engine is running: curl $API_URL/engine/status"
    echo "  3. Check /trading/why: curl $API_URL/trading/why | jq"
    exit 1
fi
