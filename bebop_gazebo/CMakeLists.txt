cmake_minimum_required(VERSION 2.8.3)
project(bebop_gazebo)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  geometry_msgs
  gazebo_dev
  gazebo_msgs
  hector_quadrotor_interface
  hector_quadrotor_actions
  hector_uav_msgs
  std_msgs
  roscpp
  tf2
  tf2_geometry_msgs
  laser_assembler
  message_generation
  )

find_package(gazebo REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")
include_directories(${catkin_INCLUDE_DIRS})
include_directories(${GAZEBO_INCLUDE_DIRS})
include_directories(${gazebo_ros_INDLUDE_DIRECTORIES})
link_directories(${GAZEBO_LIBRARY_DIRS})

add_service_files(DIRECTORY srv
  FILES
  RequestUavPose.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
    LIBRARIES 
    CATKIN_DEPENDS roscpp
    DEPENDS
)

add_executable(emergency_action
  src/emergency_action.cpp
  )

target_link_libraries(emergency_action ${catkin_LIBRARIES})
add_dependencies(emergency_action ${catkin_EXPORTED_TARGETS})

add_library(plugin SHARED src/plugin.cpp)
target_link_libraries(plugin ${GAZEBO_LIBRARIES} ${catkin_LIBRARIES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

install(TARGETS
  plugin
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)
