cmake_minimum_required(VERSION 3.1)
project(trt_inference)

#include_directories(${CMAKE_SOURCE_DIR})

# set(CUDA_NVCC_FLAGS --std=c++11)
set(CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  roscpp
  roslib
  tensorflow_ros_cpp
  pcl_conversions
  pcl_ros
)
find_package(tensorflow_ros_cpp REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(CUDA REQUIRED)

catkin_package(
   INCLUDE_DIRS
   include
   LIBRARIES
   ${PROJECT_NAME}
   CATKIN_DEPENDS
   roscpp)

include_directories(
  # needs to be first so that the tensorflow versions of includes are always treated first (not needed on all systems, but sometimes it is).
  ${tensorflow_ros_cpp_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

# CPU
# again, tensorflow_ros_cpp_LIBRARIES need to go first in some setups
add_library(inferenceEngineCPU src/inferenceEngineCPU.cpp)
add_dependencies(inferenceEngineCPU ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_include_directories(inferenceEngineCPU BEFORE PRIVATE ${tensorflow_ros_cpp_INCLUDE_DIRS}
                          PUBLIC include ${catkin_INCLUDE_DIRS})
target_link_libraries(inferenceEngineCPU ${tensorflow_ros_cpp_LIBRARIES} ${catkin_LIBRARIES})
target_compile_definitions(inferenceEngineCPU PRIVATE ${tensorflow_ros_cpp_CMAKE_CXX_FLAGS_PRIVATE})

add_library(inferenceNodeCPU
    src/segNodeCPU.cpp
    src/segNodeletCPU.cpp)
target_include_directories(inferenceNodeCPU PUBLIC include ${catkin_INCLUDE_DIRS})
target_link_libraries(inferenceNodeCPU inferenceEngineCPU ${OpenCV_LIBS} ${catkin_LIBRARIES})

# GPU
cuda_add_library(inferenceEngine src/inferenceEngine.cu)
target_include_directories(inferenceEngine PUBLIC include)
target_link_libraries(inferenceEngine ${OpenCV_LIBS} nvinfer nvparsers)

# cuda_add_executable(test_inf src/test_inf.cu)
# target_link_libraries(test_inf ${OpenCV_LIBS} inferenceEngine nvinfer nvparsers)

# if you don't want to use a nodelet, use add_executable instead of add library
# add_executable(${PROJECT_NAME} src/segNode.cpp)
add_library(inferenceNode
    src/segNode.cpp
    src/segNodelet.cpp)
target_include_directories(inferenceNode PUBLIC include ${catkin_INCLUDE_DIRS})
target_link_libraries(inferenceNode inferenceEngine ${OpenCV_LIBS} ${catkin_LIBRARIES})
