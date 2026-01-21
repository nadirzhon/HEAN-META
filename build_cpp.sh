#!/bin/bash
# Build script for C++ Graph Engine and Python bindings

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="${SCRIPT_DIR}/src/hean/core/cpp"
BUILD_DIR="${CPP_DIR}/build"
INSTALL_DIR="${SCRIPT_DIR}/src/hean/core/cpp"

echo "Building HEAN Graph Engine (C++)..."

# Check for required dependencies
command -v cmake >/dev/null 2>&1 || { echo "CMake is required but not installed. Aborting." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Python3 is required but not installed. Aborting." >&2; exit 1; }

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Find Python3
PYTHON3_EXECUTABLE=$(which python3)
PYTHON3_VERSION=$(${PYTHON3_EXECUTABLE} --version | cut -d' ' -f2 | cut -d'.' -f1,2)

echo "Using Python: ${PYTHON3_EXECUTABLE} (version ${PYTHON3_VERSION})"

# Check for pybind11
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo "Installing pybind11..."
    pip3 install pybind11
fi

# Find pybind11 path
PYBIND11_PATH=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null || echo "")
if [ -z "$PYBIND11_PATH" ]; then
    echo "Warning: Could not find pybind11 CMake directory. Trying default locations..."
    PYBIND11_PATH=""
fi

# Check for ONNX Runtime (optional)
ONNX_AVAILABLE=OFF
if pkg-config --exists onnxruntime 2>/dev/null; then
    ONNX_AVAILABLE=ON
    echo "ONNX Runtime found via pkg-config"
elif [ -d "/usr/local/include/onnxruntime" ] || [ -d "/opt/onnxruntime/include" ]; then
    ONNX_AVAILABLE=ON
    echo "ONNX Runtime found in system directories"
else
    echo "Warning: ONNX Runtime not found. Volatility prediction will be disabled."
    echo "  To enable: Install ONNX Runtime C++ library"
    echo "  See: https://onnxruntime.ai/docs/install/"
fi

# Configure CMake
echo "Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DPython3_EXECUTABLE="${PYTHON3_EXECUTABLE}" \
    ${PYBIND11_PATH:+-Dpybind11_DIR="${PYBIND11_PATH}"} \
    -DENABLE_ONNX=${ONNX_AVAILABLE}

# Build
echo "Building..."
cmake --build . --config Release -j$(nproc 2>/dev/null || echo 4)

# Install (copy to Python package directory)
echo "Installing..."
if [ -f "${BUILD_DIR}/graph_engine_py.so" ] || [ -f "${BUILD_DIR}/graph_engine_py.cpython-*.so" ]; then
    # Find the built module
    MODULE_FILE=$(find "${BUILD_DIR}" -name "graph_engine_py*.so" | head -n 1)
    if [ -n "$MODULE_FILE" ]; then
        echo "Found module: ${MODULE_FILE}"
        # Copy to parent directory so Python can import it
        cp "${MODULE_FILE}" "${CPP_DIR}/graph_engine_py.so"
        echo "Installed to: ${CPP_DIR}/graph_engine_py.so"
    fi
else
    echo "Warning: Built module not found. Build may have failed."
    exit 1
fi

echo "Build complete!"
echo ""
echo "To use the graph engine in Python:"
echo "  import sys"
echo "  sys.path.insert(0, '${CPP_DIR}')"
echo "  import graph_engine_py"
