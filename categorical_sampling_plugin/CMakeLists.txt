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
  categorical_sampling_plugin SHARED
  CategoricalSamplingPlugin.cpp
  ../categorical_sampling_cuda_kernel/src/categorical_sampling.cu)

target_link_libraries(categorical_sampling_plugin ${TENSORRT_LIBRARY_INFER}
                      ${CUDA_LIBRARIES} curand)

set_target_properties(
  categorical_sampling_plugin PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                         POSITION_INDEPENDENT_CODE ON)

# # Install install( TARGETS categorical_sampling_plugin LIBRARY DESTINATION lib
# ARCHIVE DESTINATION lib)

# Install to custom build directory
install(
  TARGETS categorical_sampling_plugin
  LIBRARY DESTINATION /code/tensorrt_llm/build
  ARCHIVE DESTINATION /code/tensorrt_llm/build)

install(FILES CategoricalSamplingPlugin.h DESTINATION /code/tensorrt_llm/build)
