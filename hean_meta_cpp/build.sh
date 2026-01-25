#!/bin/bash

set -e

echo "🔨 Building HEAN-META C++ Order Engine"
echo "======================================"

# Check prerequisites
echo ""
echo "📋 Checking prerequisites..."

# Check C++ compiler
if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo "❌ No C++ compiler found. Install g++ or clang++."
    exit 1
fi

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo "❌ CMake not found. Install with: sudo apt-get install cmake"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found."
    exit 1
fi

echo "✅ All prerequisites met"

# Install pybind11 if not present
echo ""
echo "📦 Installing pybind11..."
python3 -m pip install --quiet pybind11 || {
    echo "⚠️  Failed to install pybind11 via pip"
    echo "    Trying system package..."
    sudo apt-get install -y python3-pybind11 || {
        echo "❌ Failed to install pybind11"
        exit 1
    }
}

# Create build directory
echo ""
echo "📁 Creating build directory..."
mkdir -p build
cd build

# Configure
echo ""
echo "⚙️  Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -flto" \
    || {
        echo "❌ CMake configuration failed"
        exit 1
    }

# Build
echo ""
echo "🔨 Building (this may take a minute)..."
make -j$(nproc) || {
    echo "❌ Build failed"
    exit 1
}

# Run tests
echo ""
echo "🧪 Running C++ tests..."
./test_order_engine || {
    echo "⚠️  Some tests failed"
}

# Install Python module
echo ""
echo "📦 Installing Python module..."
cd ..
pip install -e . || {
    echo "⚠️  Failed to install Python module"
    echo "    You can still use the module from build directory"
}

# Verify installation
echo ""
echo "✅ Verifying installation..."
python3 -c "import hean_meta_cpp; print('  ✅ Module imported successfully')" || {
    echo "❌ Failed to import module"
    exit 1
}

echo ""
echo "======================================"
echo "✅ Build completed successfully!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Run C++ tests: ./build/test_order_engine"
echo "  2. Run Python example: python3 python/example.py"
echo "  3. Import in your code: import hean_meta_cpp"
echo ""
