cmake_minimum_required(VERSION 2.8.3)
project(monitors)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
find_package(InferenceEngine 2.0 REQUIRED)
find_package(ngraph REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES monitors
  CATKIN_DEPENDS roscpp rospy
#  DEPENDS system_lib
)


include_directories(
 include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${InferenceEngine_INCLUDE_DIRS}
)

add_library(monitors src/cpu_monitor.cpp
                     src/memory_monitor.cpp
                     src/presenter.cpp)
target_link_libraries(monitors opencv_core opencv_imgproc ${catkin_LIBRARIES})

