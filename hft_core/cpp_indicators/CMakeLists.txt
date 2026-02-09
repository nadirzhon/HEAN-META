cmake_minimum_required(VERSION 3.20)
project(indicators_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mavx2 -mfma -DNDEBUG")

# Find Python
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)

# Find nanobind
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE NB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Create Python module
nanobind_add_module(indicators_cpp
    src/indicators.cpp
)

target_include_directories(indicators_cpp PRIVATE include)

# Install
install(TARGETS indicators_cpp LIBRARY DESTINATION .)
