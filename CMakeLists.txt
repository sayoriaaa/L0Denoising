cmake_minimum_required(VERSION 3.6)
find_package(CUDA REQUIRED)
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ./pybind11/include
    )

link_directories(
    /usr/local/cuda/lib64
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(cuda_utils SHARED
  cuda_utils.cu)

target_link_libraries(cuda_utils
  ${PYTHON_LIBRARIES}
  cudart
  cufft)

set_target_properties(cuda_utils PROPERTIES PREFIX "")