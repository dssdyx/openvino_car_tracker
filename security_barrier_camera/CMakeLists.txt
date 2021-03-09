cmake_minimum_required(VERSION 2.8.3)
project(security_barrier_camera)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
  monitors
  rect_msgs
  geometry_msgs
  message_generation
  message_filters
)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
find_package(InferenceEngine 2.0 REQUIRED)
find_package(ngraph REQUIRED)



catkin_package(
  INCLUDE_DIRS include
  LIBRARIES security_barrier_camera
  CATKIN_DEPENDS cv_bridge image_transport roscpp rospy sensor_msgs std_msgs monitors rect_msgs
#  DEPENDS system_lib
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${InferenceEngine_INCLUDE_DIRS}
)


add_executable(${PROJECT_NAME}_node
   src/main.cpp
)
target_link_libraries(${PROJECT_NAME}_node ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES} ${catkin_LIBRARIES} pthread)


