#!/bin/bash

echo "🚀 Starting Multi-Language HFT System..."
echo "================================================"

# Start Order Router (Rust) in background
echo "Starting Order Router (Rust)..."
./rust_order_router/target/release/order-router &
ORDER_ROUTER_PID=$!
sleep 2

# Start Risk Engine (Rust) in background
echo "Starting Risk Engine (Rust)..."
./rust_risk_engine/target/release/risk-engine &
RISK_ENGINE_PID=$!
sleep 1

# Start Python Orchestrator
echo "Starting Strategy Orchestrator (Python)..."
python3 python_orchestrator/strategy_orchestrator.py &
ORCHESTRATOR_PID=$!

echo ""
echo "================================================"
echo "✅ ALL SERVICES STARTED!"
echo "================================================"
echo "PIDs:"
echo "  - Order Router: $ORDER_ROUTER_PID"
echo "  - Risk Engine: $RISK_ENGINE_PID"
echo "  - Orchestrator: $ORCHESTRATOR_PID"
echo ""
echo "Press Ctrl+C to stop all services..."
echo "================================================"

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping all services..."
    kill $ORDER_ROUTER_PID 2>/dev/null
    kill $RISK_ENGINE_PID 2>/dev/null
    kill $ORCHESTRATOR_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait
wait
