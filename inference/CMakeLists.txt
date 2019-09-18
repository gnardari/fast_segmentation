cmake_minimum_required(VERSION 3.1)
project(trt_inference)

#include_directories(${CMAKE_SOURCE_DIR})

set(CUDA_NVCC_FLAGS --std=c++11)
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
find_package(OpenCV REQUIRED)
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

# again, tensorflow_ros_cpp_LIBRARIES need to go first in some setups
add_library(inferenceEngineCPU src/inferenceEngineCPU.cpp)
add_dependencies(inferenceEngineCPU ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_include_directories(inferenceEngineCPU BEFORE PRIVATE ${tensorflow_ros_cpp_INCLUDE_DIRS}
                          PUBLIC include ${catkin_INCLUDE_DIRS})
target_link_libraries(inferenceEngineCPU ${tensorflow_ros_cpp_LIBRARIES} ${catkin_LIBRARIES})
target_compile_definitions(inferenceEngineCPU PRIVATE ${tensorflow_ros_cpp_CMAKE_CXX_FLAGS_PRIVATE})

### FOR PROFILING (need to set after catkin_package)
###add_compile_options(-pg)
###set(catkin_LIBRARIES ${catkin_LIBRARIES} -pg)
###

# cuda_add_library(inferenceEngine src/inferenceEngine.cu)
# target_include_directories(inferenceEngine PUBLIC include)
# target_link_libraries(inferenceEngine ${OpenCV_LIBS} nvinfer nvparsers)
#
# cuda_add_executable(run_plan src/run_plan.cu)
# target_link_libraries(run_plan ${OpenCV_LIBS} nvinfer nvparsers)
# add_executable(${PROJECT_NAME} src/segNode.cpp)

add_library(${PROJECT_NAME}
            src/segNode.cpp
            src/segNodelet.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include ${catkin_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} inferenceEngineCPU ${OpenCV_LIBS} ${catkin_LIBRARIES})
