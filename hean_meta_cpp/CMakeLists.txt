cmake_minimum_required(VERSION 3.15)
project(hean_meta_cpp VERSION 1.0.0 LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags for production
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -pedantic")

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via CMake, trying to find via pip...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_FIND_RESULT
    )

    if(PYBIND11_FIND_RESULT EQUAL 0)
        set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/order_engine.cpp
    src/position_manager.cpp
)

# Create static library for testing
add_library(hean_core STATIC ${SOURCES})
target_include_directories(hean_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Python module
pybind11_add_module(hean_meta_cpp
    src/bindings.cpp
    ${SOURCES}
)

target_include_directories(hean_meta_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link options for better performance
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_link_options(hean_meta_cpp PRIVATE -flto)
endif()

# Install targets
install(TARGETS hean_meta_cpp
    LIBRARY DESTINATION ${Python3_SITEARCH}
)

# Tests (optional)
option(BUILD_TESTS "Build C++ tests" ON)

if(BUILD_TESTS)
    enable_testing()

    # Simple test executable
    add_executable(test_order_engine tests/test_orders.cpp)
    target_link_libraries(test_order_engine PRIVATE hean_core)

    add_test(NAME OrderEngineTest COMMAND test_order_engine)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler:    ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ flags:       ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "  Python version:  ${Python3_VERSION}")
message(STATUS "  Python include:  ${Python3_INCLUDE_DIRS}")
message(STATUS "  pybind11 version: ${pybind11_VERSION}")
message(STATUS "  Install prefix:  ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
