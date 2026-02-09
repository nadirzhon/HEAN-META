cmake_minimum_required(VERSION 3.15)
project(hean_cpp_core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ultra-performance optimizations
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -finline-functions")

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Get nanobind using FetchContent
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v1.9.0
)
FetchContent_MakeAvailable(nanobind)

# Indicators module (50-100x faster than Python/pandas)
nanobind_add_module(indicators_cpp indicators.cpp)

# Apply aggressive optimizations for hot path
target_compile_options(indicators_cpp PRIVATE
    -O3
    -march=native
    -mtune=native
    -ffast-math
    -finline-functions
    -funroll-loops
    -flto
)

# Order router module (sub-microsecond latency)
nanobind_add_module(order_router_cpp order_router.cpp)

target_compile_options(order_router_cpp PRIVATE
    -O3
    -march=native
    -mtune=native
    -fomit-frame-pointer
    -finline-functions
    -flto
)

# Install to Python package location
install(TARGETS indicators_cpp order_router_cpp
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../src/hean/cpp_modules)
