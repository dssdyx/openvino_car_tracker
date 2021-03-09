cmake_minimum_required(VERSION 3.0.2)
project(object_pose)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_generation
  rect_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  nav_msgs
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)



catkin_package(
  INCLUDE_DIRS include
  LIBRARIES object_pose
  CATKIN_DEPENDS cv_bridge image_transport rect_msgs roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(${PROJECT_NAME}_node
   src/main.cpp
)
target_link_libraries(${PROJECT_NAME}_node ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} pthread)

