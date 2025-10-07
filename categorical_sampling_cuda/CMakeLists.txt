cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CategoricalSamplingPlugin CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find TensorRT
find_path(
  TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)

find_library(
  TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

find_library(
  TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY_INFER)
  message(
    FATAL_ERROR
      "Cannot find TensorRT. Please set TENSORRT_ROOT or ensure TensorRT is in CUDA toolkit."
  )
endif()

include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Plugin library
add_library(
  categorical_sampling_plugin SHARED CategoricalSamplingPlugin.cpp
                                     ../sampling/src/categorical_sampling.cu)

target_link_libraries(categorical_sampling_plugin ${TENSORRT_LIBRARY_INFER}
                      ${CUDA_LIBRARIES} curand)

set_target_properties(
  categorical_sampling_plugin PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                         POSITION_INDEPENDENT_CODE ON)

# Optional: Build example
option(BUILD_EXAMPLE "Build example application" ON)
if(BUILD_EXAMPLE)
  add_executable(categorical_sampling_example example.cpp)
  target_link_libraries(
    categorical_sampling_example categorical_sampling_plugin
    ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN}
    ${CUDA_LIBRARIES})
endif()

# Install
install(
  TARGETS categorical_sampling_plugin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES CategoricalSamplingPlugin.h DESTINATION include)
