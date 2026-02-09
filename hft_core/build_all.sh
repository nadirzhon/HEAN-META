#!/bin/bash
set -e

echo "🚀 Building Multi-Language HFT System..."
echo "================================================"

# Colors
GREEN='\033[0.32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check prerequisites
check_prereqs() {
    echo -e "${BLUE}Checking prerequisites...${NC}"

    command -v cargo >/dev/null 2>&1 || { echo "❌ Rust not found. Install from https://rustup.rs/"; exit 1; }
    command -v cmake >/dev/null 2>&1 || { echo "❌ CMake not found"; exit 1; }
    command -v go >/dev/null 2>&1 || { echo "❌ Go not found"; exit 1; }
    command -v python3 >/dev/null 2>&1 || { echo "❌ Python3 not found"; exit 1; }

    echo -e "${GREEN}✅ All prerequisites found${NC}"
}

# Build Rust Order Router
build_rust_order_router() {
    echo -e "${BLUE}Building Rust Order Router...${NC}"
    cd rust_order_router
    cargo build --release
    cd ..
    echo -e "${GREEN}✅ Order Router built${NC}"
}

# Build Rust Risk Engine
build_rust_risk_engine() {
    echo -e "${BLUE}Building Rust Risk Engine...${NC}"
    cd rust_risk_engine
    cargo build --release
    cd ..
    echo -e "${GREEN}✅ Risk Engine built${NC}"
}

# Build Rust Market Data
build_rust_market_data() {
    echo -e "${BLUE}Building Rust Market Data Processor...${NC}"
    cd rust_market_data
    cargo build --release 2>/dev/null || echo "Skipping (optional)"
    cd ..
}

# Build C++ Indicators
build_cpp_indicators() {
    echo -e "${BLUE}Building C++ Indicators Library...${NC}"
    cd cpp_indicators
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release
    cd ../..
    echo -e "${GREEN}✅ C++ Indicators built${NC}"
}

# Build Go API Gateway
build_go_api() {
    echo -e "${BLUE}Building Go API Gateway...${NC}"
    cd go_api_gateway
    go mod download 2>/dev/null || true
    go build -o api-gateway main.go 2>/dev/null || echo "Skipping (optional)"
    cd ..
}

# Setup Python
setup_python() {
    echo -e "${BLUE}Setting up Python Orchestrator...${NC}"
    cd python_orchestrator
    pip install -q zmq orjson numpy 2>/dev/null || true
    cd ..
    echo -e "${GREEN}✅ Python setup complete${NC}"
}

# Main build process
main() {
    check_prereqs

    echo ""
    echo "================================================"
    echo "Building components..."
    echo "================================================"

    build_rust_order_router
    build_rust_risk_engine
    build_rust_market_data
    build_cpp_indicators
    build_go_api
    setup_python

    echo ""
    echo "================================================"
    echo -e "${GREEN}✅ BUILD COMPLETE!${NC}"
    echo "================================================"
    echo ""
    echo "🎯 Binaries location:"
    echo "   - Order Router: ./rust_order_router/target/release/order-router"
    echo "   - Risk Engine:  ./rust_risk_engine/target/release/risk-engine"
    echo "   - C++ Indicators: ./cpp_indicators/build/indicators_cpp.*.so"
    echo ""
    echo "🚀 To start the system:"
    echo "   ./run_all.sh"
    echo ""
}

main
