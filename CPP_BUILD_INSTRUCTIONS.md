# C++ Core Modules Build Instructions

## Prerequisites

### macOS
```bash
brew install cmake python@3.11
pip install nanobind
```

### Linux (Docker)
```dockerfile
# Already included in multi-stage docker build
```

## Building C++ Modules

### Local Development (macOS)

```bash
cd cpp_core
mkdir -p build
cd build

# Configure with Release optimizations
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build modules
make -j$(sysctl -n hw.ncpu)

# Install to Python package location
make install

# Verify modules are installed
ls -la ../src/hean/cpp_modules/
# Should see: indicators_cpp.so, order_router_cpp.so
```

### Docker Production Build

The `Dockerfile` should include a multi-stage build:

```dockerfile
# Stage 1: Build C++ modules
FROM python:3.11-slim as cpp-builder

RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY cpp_core/ ./cpp_core/
COPY pyproject.toml ./

RUN pip install nanobind
RUN cd cpp_core && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install

# Stage 2: Runtime image
FROM python:3.11-slim

# Copy built C++ modules from builder
COPY --from=cpp-builder /build/src/hean/cpp_modules/ /app/src/hean/cpp_modules/

# ... rest of docker build
```

## Verification

After building, verify C++ modules are loaded:

```python
# In Python REPL or script
try:
    import hean.cpp_modules.indicators_cpp as indicators_cpp
    print("✓ indicators_cpp loaded successfully")
except ImportError as e:
    print(f"✗ indicators_cpp NOT loaded: {e}")

try:
    import hean.cpp_modules.order_router_cpp as order_router_cpp
    print("✓ order_router_cpp loaded successfully")
except ImportError as e:
    print(f"✗ order_router_cpp NOT loaded: {e}")
```

## Expected Performance Improvements

With C++ modules loaded:
- **Indicators calculation**: 50-100x faster than pandas/numpy
- **Order routing decisions**: sub-microsecond latency
- **Oracle/Triangular scanning**: 10-20x faster

## Fallback Behavior

If C++ modules are not available, HEAN will automatically fall back to pure Python implementations with warnings logged.

Check `/system/cpp/status` API endpoint to verify C++ module availability.
