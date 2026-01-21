cmake_minimum_required(VERSION 3.15)
project(hean_graph_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ultra-Low Latency Optimizations
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -flto -finline-functions -funroll-loops -ffast-math")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native -mtune=native -flto -finline-functions -funroll-loops -ffast-math")

# Link-Time Optimization (LTO)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "Link-Time Optimization (LTO) enabled")
endif()

# Pre-compiled Headers (PCH) - CMake 3.16+ native support
# Will be configured per-target below
message(STATUS "Pre-compiled Headers (PCH) will be configured per-target")

# Find Python3
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find Boost (for Boost.Interprocess)
find_package(Boost COMPONENTS system filesystem REQUIRED)
if(Boost_FOUND)
    message(STATUS "Boost found: ${Boost_VERSION}")
    set(BOOST_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
    set(BOOST_LIBRARIES ${Boost_LIBRARIES})
    # Note: Boost.Interprocess is header-only, no linking needed
else()
    message(WARNING "Boost not found. Shared memory bridge will be disabled.")
    message(STATUS "Install Boost with: apt-get install libboost-dev (Linux) or brew install boost (macOS)")
    set(ENABLE_SHARED_MEMORY FALSE)
endif()

# Try to find Boost.Interprocess (header-only)
find_path(BOOST_INTERPROCESS_INCLUDE_DIR
    NAMES boost/interprocess/shared_memory_object.hpp
    PATHS ${Boost_INCLUDE_DIRS}
    PATH_SUFFIXES boost
)
if(BOOST_INTERPROCESS_INCLUDE_DIR)
    message(STATUS "Boost.Interprocess found")
    set(ENABLE_SHARED_MEMORY TRUE)
else()
    message(WARNING "Boost.Interprocess not found. Shared memory bridge will be disabled.")
    set(ENABLE_SHARED_MEMORY FALSE)
endif()

# ONNX Runtime (optional - will use system install or download)
find_package(onnxruntime QUIET)
if(NOT onnxruntime_FOUND)
    message(WARNING "ONNX Runtime not found. Volatility prediction will be disabled.")
    set(ENABLE_ONNX FALSE)
else()
    set(ENABLE_ONNX TRUE)
endif()

# simdjson library for ultra-fast JSON parsing (Bybit V5)
# Try to find system installation first
find_path(SIMDJSON_INCLUDE_DIR
    NAMES simdjson.h
    PATHS /usr/include /usr/local/include /opt/homebrew/include
)
find_library(SIMDJSON_LIBRARY
    NAMES simdjson
    PATHS /usr/lib /usr/local/lib /opt/homebrew/lib
)

if(SIMDJSON_INCLUDE_DIR AND SIMDJSON_LIBRARY)
    message(STATUS "simdjson found: ${SIMDJSON_INCLUDE_DIR}, ${SIMDJSON_LIBRARY}")
    set(ENABLE_SIMDJSON TRUE)
else()
    # Try to use FetchContent to download simdjson
    include(FetchContent)
    FetchContent_Declare(
        simdjson
        GIT_REPOSITORY https://github.com/simdjson/simdjson.git
        GIT_TAG v3.7.0
    )
    set(SIMDJSON_SANITIZE_UNDEFINED OFF CACHE BOOL "" FORCE)
    set(SIMDJSON_SANITIZE_ADDRESS OFF CACHE BOOL "" FORCE)
    set(SIMDJSON_SANITIZE_THREAD OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(simdjson)
    if(simdjson_SOURCE_DIR OR TARGET simdjson)
        message(STATUS "simdjson downloaded via FetchContent: ${simdjson_SOURCE_DIR}")
        set(SIMDJSON_INCLUDE_DIR ${simdjson_SOURCE_DIR}/include)
        set(ENABLE_SIMDJSON TRUE)
        # simdjson is header-only, no library to link
        set(SIMDJSON_LIBRARY "")
    else()
        message(WARNING "simdjson not found. Install with: apt-get install libsimdjson-dev (Linux) or brew install simdjson (macOS)")
        message(STATUS "Falling back to basic SIMD parser")
        set(ENABLE_SIMDJSON FALSE)
    endif()
endif()

# Graph Engine library
add_library(graph_engine STATIC
    GraphEngine.cpp
)

# OrderFlow AI library (VPIN and Spoofing Detection)
# Move orderflow_ai.cpp to cpp directory for compilation
add_library(orderflow_ai STATIC
    orderflow_ai.cpp
)

target_include_directories(orderflow_ai PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../intelligence
)

# TDA Engine library
add_library(tda_engine STATIC
    TDA_Engine.cpp
    FastWarden.cpp
)

# Add FastMath header to TDA engine (for inline math functions)
target_include_directories(tda_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Triangular Arbitrage Scanner library
add_library(triangular_scanner STATIC
    TriangularScanner.cpp
)

# Swarm Manager library
add_library(swarm_manager STATIC
    swarm_manager.cpp
)

target_include_directories(swarm_manager PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# OFI Monitor library
add_library(ofi_monitor STATIC
    ofi_monitor.cpp
)

target_include_directories(ofi_monitor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Algorithmic Fingerprinting Engine library
add_library(fingerprinter STATIC
    fingerprinter.cpp
)

# Find ZeroMQ for MetamorphicCore communication
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ZMQ QUIET libzmq)
endif()

if(NOT ZMQ_FOUND)
    # Try to find ZeroMQ manually
    find_path(ZMQ_INCLUDE_DIR
        NAMES zmq.h
        PATHS /usr/include /usr/local/include /opt/homebrew/include
    )
    find_library(ZMQ_LIBRARY
        NAMES zmq
        PATHS /usr/lib /usr/local/lib /opt/homebrew/lib
    )
    if(ZMQ_INCLUDE_DIR AND ZMQ_LIBRARY)
        set(ZMQ_FOUND TRUE)
        set(ZMQ_INCLUDE_DIRS ${ZMQ_INCLUDE_DIR})
        set(ZMQ_LIBRARIES ${ZMQ_LIBRARY})
    endif()
endif()

if(ZMQ_FOUND)
    message(STATUS "ZeroMQ found: ${ZMQ_INCLUDE_DIRS}, ${ZMQ_LIBRARIES}")
    set(ENABLE_ZMQ TRUE)
else()
    message(WARNING "ZeroMQ not found. MetamorphicCore will use shared memory only.")
    message(STATUS "Install ZeroMQ with: apt-get install libzmq3-dev (Linux) or brew install zeromq (macOS)")
    set(ENABLE_ZMQ FALSE)
endif()

# Metamorphic Engine library
add_library(metamorphic_engine STATIC
    metamorphic_engine.cpp
)

target_include_directories(metamorphic_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# MetamorphicCore library (C++ Body for Absolute+)
add_library(metamorphic_core STATIC
    MetamorphicCore.cpp
)

target_include_directories(metamorphic_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(ENABLE_ZMQ)
    target_include_directories(metamorphic_core PRIVATE
        ${ZMQ_INCLUDE_DIRS}
    )
    target_link_libraries(metamorphic_core PRIVATE
        ${ZMQ_LIBRARIES}
    )
    target_compile_definitions(metamorphic_core PRIVATE ENABLE_ZMQ=1)
else()
    target_compile_definitions(metamorphic_core PRIVATE ENABLE_ZMQ=0)
endif()

# Create shared library for Python bindings
add_library(metamorphic_core_shared SHARED
    MetamorphicCore.cpp
)

target_include_directories(metamorphic_core_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(ENABLE_ZMQ)
    target_include_directories(metamorphic_core_shared PRIVATE
        ${ZMQ_INCLUDE_DIRS}
    )
    target_link_libraries(metamorphic_core_shared PRIVATE
        ${ZMQ_LIBRARIES}
    )
    target_compile_definitions(metamorphic_core_shared PRIVATE ENABLE_ZMQ=1)
else()
    target_compile_definitions(metamorphic_core_shared PRIVATE ENABLE_ZMQ=0)
endif()

set_target_properties(metamorphic_core_shared PROPERTIES
    OUTPUT_NAME "metamorphic"
    PREFIX ""
)

# SIMD JSON Parser library (ultra-fast Bybit V5 parsing)
add_library(simdjson_parser STATIC
    SIMDJSONParser.cpp
)

target_include_directories(simdjson_parser PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(ENABLE_SIMDJSON)
    target_include_directories(simdjson_parser PRIVATE
        ${SIMDJSON_INCLUDE_DIR}
    )
    if(SIMDJSON_LIBRARY)
        target_link_libraries(simdjson_parser PRIVATE ${SIMDJSON_LIBRARY})
    elseif(TARGET simdjson)
        # simdjson via FetchContent - link to the target
        target_link_libraries(simdjson_parser PRIVATE simdjson)
    endif()
    target_compile_definitions(simdjson_parser PRIVATE USE_REAL_SIMDJSON=1)
    message(STATUS "SIMD JSON Parser: Using simdjson library from ${SIMDJSON_INCLUDE_DIR}")
else()
    target_compile_definitions(simdjson_parser PRIVATE USE_REAL_SIMDJSON=0)
    message(STATUS "SIMD JSON Parser: Using basic SIMD parser (simdjson library not available)")
endif()

# HFT Engine library (The Sniper, ELM, Toxicity Detector, Scalper)
add_library(hft_engine STATIC
    Sniper.cpp
    ELM_Regressor.cpp
    ToxicityDetector.cpp
    Scalper.cpp
    OptimizedWebSocket.cpp
)

target_include_directories(hft_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add simdjson parser to HFT engine for Bybit V5 parsing
target_link_libraries(hft_engine PRIVATE simdjson_parser)

# Phase 16: Feed Handler library (shared memory bridge)
if(ENABLE_SHARED_MEMORY)
    add_library(feed_handler STATIC
        FeedHandler.cpp
    )
    
    target_include_directories(feed_handler PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Boost_INCLUDE_DIRS}
    )
    
    # Boost.Interprocess is header-only, no linking needed
    message(STATUS "Feed Handler library enabled (shared memory bridge)")
else()
    message(STATUS "Feed Handler library disabled (Boost.Interprocess not found)")
endif()

target_include_directories(graph_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(tda_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(triangular_scanner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(fingerprinter PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Swarm Manager library
add_library(swarm_manager STATIC
    swarm_manager.cpp
)

target_include_directories(swarm_manager PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# OFI Monitor library
add_library(ofi_monitor STATIC
    ofi_monitor.cpp
)

target_include_directories(ofi_monitor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Python bindings
pybind11_add_module(graph_engine_py
    python_bindings.cpp
)

target_link_libraries(graph_engine_py PRIVATE
    graph_engine
    tda_engine
    triangular_scanner
    fingerprinter
    hft_engine
    swarm_manager
    ofi_monitor
    orderflow_ai
    simdjson_parser
    metamorphic_engine
    metamorphic_core
)

# Pre-compiled Headers for hot-path headers
target_precompile_headers(graph_engine_py PRIVATE
    <atomic>
    <chrono>
    <cmath>
    <cstdint>
    <string>
    <vector>
    "FastMath.h"
)

# Apply compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(graph_engine_py PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -Wno-missing-field-initializers
        # Aggressive inlining for hot-path
        -finline-limit=10000
        -finline-functions
        -finline-small-functions
        # CPU-specific optimizations
        -march=native
        -mtune=native
        # Remove frame pointers for faster function calls
        -fomit-frame-pointer
        # Fast math (trade precision for speed where acceptable)
        -ffast-math
        -ffinite-math-only
        -fno-math-errno
        # Link-time optimization
        -flto
    )
endif()

# Link Feed Handler if enabled
if(ENABLE_SHARED_MEMORY)
    target_link_libraries(graph_engine_py PRIVATE
        feed_handler
    )
    target_compile_definitions(graph_engine_py PRIVATE ENABLE_SHARED_MEMORY=1)
endif()

if(ENABLE_ONNX)
    target_link_libraries(graph_engine_py PRIVATE
        onnxruntime::onnxruntime
    )
    target_compile_definitions(graph_engine_py PRIVATE ENABLE_ONNX=1)
endif()

# Install Python module
install(TARGETS graph_engine_py
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/src/hean/core/cpp
)
