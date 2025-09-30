cmake_minimum_required(VERSION 3.18)
project(categorical_sampling LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# CUDA architecture (adjust based on your GPU) Common values: 70 (V100), 75
# (T4), 80 (A100), 86 (RTX 30xx), 89 (RTX 40xx)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

# Source files
set(CUDA_SOURCES src/categorical_sampling.cu)

# Create library
add_library(categorical_sampling STATIC ${CUDA_SOURCES})

# Set CUDA flags
target_compile_options(
  categorical_sampling
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr --use_fast_math
          -Xcompiler=-Wall >)

# Test executable
add_executable(test_categorical_sampling test/test_categorical_sampling.cu)
target_link_libraries(test_categorical_sampling categorical_sampling
                      ${CUDA_LIBRARIES} curand)

# Enable testing
enable_testing()
add_test(NAME CategoricalSamplingTest COMMAND test_categorical_sampling)

# Print configuration
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Include Dirs: ${CUDA_INCLUDE_DIRS}")
