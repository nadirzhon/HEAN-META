#!/bin/bash

# HEAN Integration Test Script - The Bug Hunter
# Tests connectivity and latency between all system components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

API_URL="${API_URL:-http://localhost:8000}"
WS_URL="${WS_URL:-ws://localhost:8000}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   HEAN INTEGRATION TEST - THE BUG HUNTER             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Test counters
PASSED=0
FAILED=0

# Function to print test result
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗${NC} $2"
        FAILED=$((FAILED + 1))
    fi
}

# 1. Check if Redis is alive
echo -e "${YELLOW}[1/5] Testing Redis connectivity...${NC}"
if command -v redis-cli &> /dev/null; then
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping > /dev/null 2>&1; then
        print_result 0 "Redis is alive and responding"
    else
        print_result 1 "Redis is not responding"
    fi
else
    # Try Python fallback
    if python3 -c "import redis; r = redis.Redis(host='$REDIS_HOST', port=$REDIS_PORT); r.ping()" 2>/dev/null; then
        print_result 0 "Redis is alive and responding (via Python)"
    else
        print_result 1 "Redis is not responding"
    fi
fi
echo ""

# 2. Verify C++ to Python Shared Memory connectivity
echo -e "${YELLOW}[2/5] Testing C++ to Python shared memory...${NC}"
if python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from hean.core.intelligence.graph_engine import GraphEngine
    engine = GraphEngine()
    print('GraphEngine initialized successfully')
    sys.exit(0)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" 2>/dev/null; then
    print_result 0 "C++ GraphEngine Python bindings are working"
else
    print_result 1 "C++ GraphEngine Python bindings failed"
fi
echo ""

# 3. Test API response times
echo -e "${YELLOW}[3/5] Testing API response times...${NC}"

# Check if API is reachable
if curl -s -f "$API_URL/health" > /dev/null 2>&1; then
    print_result 0 "API is reachable"
    
    # Measure response time
    RESPONSE_TIME=$(curl -s -o /dev/null -w "%{time_total}" "$API_URL/health" 2>/dev/null)
    RESPONSE_TIME_MS=$(echo "$RESPONSE_TIME * 1000" | bc)
    
    echo -e "   Response time: ${RESPONSE_TIME_MS}ms"
    
    # Check if response time is acceptable (< 100ms for health check)
    if (( $(echo "$RESPONSE_TIME_MS < 100" | bc -l) )); then
        print_result 0 "API response time is acceptable (< 100ms)"
    else
        print_result 1 "API response time is slow (> 100ms)"
    fi
    
    # Test multiple endpoints
    ENDPOINTS=("/health" "/api/v1/dashboard")
    for endpoint in "${ENDPOINTS[@]}"; do
        if curl -s -f "$API_URL$endpoint" > /dev/null 2>&1; then
            print_result 0 "Endpoint $endpoint is accessible"
        else
            print_result 1 "Endpoint $endpoint is not accessible"
        fi
    done
else
    print_result 1 "API is not reachable at $API_URL"
fi
echo ""

# 4. Validate that Frontend can receive data packets < 20ms (WebSocket test)
echo -e "${YELLOW}[4/5] Testing WebSocket latency...${NC}"

if command -v python3 &> /dev/null; then
    # Create a simple WebSocket latency test
    python3 << 'EOF' 2>/dev/null
import asyncio
import websockets
import time
import sys
import os

API_URL = os.environ.get('API_URL', 'http://localhost:8000')
WS_URL = API_URL.replace('http://', 'ws://').replace('https://', 'wss://')

async def test_websocket_latency():
    try:
        start_time = time.time()
        async with websockets.connect(f"{WS_URL}/ws") as websocket:
            connect_time = (time.time() - start_time) * 1000
            
            # Subscribe to a topic
            await websocket.send('{"action": "subscribe", "topic": "test"}')
            subscribe_start = time.time()
            response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            subscribe_time = (time.time() - subscribe_start) * 1000
            
            # Send ping
            ping_start = time.time()
            await websocket.send('{"action": "ping"}')
            pong = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            ping_time = (time.time() - ping_start) * 1000
            
            print(f"   WebSocket connect: {connect_time:.2f}ms")
            print(f"   Subscribe response: {subscribe_time:.2f}ms")
            print(f"   Ping-pong latency: {ping_time:.2f}ms")
            
            if ping_time < 20:
                print("   ✓ WebSocket latency < 20ms requirement met")
                sys.exit(0)
            else:
                print("   ✗ WebSocket latency exceeds 20ms requirement")
                sys.exit(1)
    except Exception as e:
        print(f"   ✗ WebSocket test failed: {e}")
        sys.exit(1)

result = asyncio.run(test_websocket_latency())
sys.exit(result)
EOF

    if [ $? -eq 0 ]; then
        print_result 0 "WebSocket latency test passed (< 20ms)"
    else
        print_result 1 "WebSocket latency test failed"
    fi
else
    print_result 1 "Python3 not available for WebSocket test"
fi
echo ""

# 5. Test Emergency Kill-Switch endpoint
echo -e "${YELLOW}[5/5] Testing Emergency Kill-Switch endpoint...${NC}"

if curl -s -X POST "$API_URL/api/v1/emergency/killswitch" -H "Content-Type: application/json" > /dev/null 2>&1; then
    KILLSWITCH_TIME=$(curl -s -o /dev/null -w "%{time_total}" -X POST "$API_URL/api/v1/emergency/killswitch" -H "Content-Type: application/json" 2>/dev/null)
    KILLSWITCH_TIME_MS=$(echo "$KILLSWITCH_TIME * 1000" | bc)
    
    echo -e "   Kill-Switch response time: ${KILLSWITCH_TIME_MS}ms"
    
    # Kill-switch should respond quickly (< 100ms)
    if (( $(echo "$KILLSWITCH_TIME_MS < 100" | bc -l) )); then
        print_result 0 "Emergency Kill-Switch endpoint is responsive"
    else
        print_result 1 "Emergency Kill-Switch endpoint is slow"
    fi
else
    print_result 1 "Emergency Kill-Switch endpoint is not accessible"
fi
echo ""

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   TEST SUMMARY                                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All integration tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some integration tests failed!${NC}"
    exit 1
fi