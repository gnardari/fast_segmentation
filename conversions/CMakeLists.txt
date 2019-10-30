cmake_minimum_required(VERSION 3.10)

find_package(CUDA REQUIRED)
include_directories(${CMAKE_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 11)

add_executable(uff_to_plan src/uff_to_plan.cpp)
target_link_libraries(uff_to_plan nvinfer nvparsers)
