cmake_minimum_required(VERSION 3.0.2)
project(rbkairos_etf_services)


## Find catkin macros and libraries

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  cv_bridge
  image_transport
  vision_msgs
  tf2_ros
  tf2_geometry_msgs
  message_filters
  message_generation
)

catkin_python_setup()

add_service_files(
  FILES
    MoveArm.srv
    MoveBase.srv
    ActionServer.srv
)

generate_messages(
  DEPENDENCIES
    std_msgs
    geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
    sensor_msgs
    cv_bridge
    image_transport
    vision_msgs
    tf2_ros
    tf2_geometry_msgs
    message_filters
    message_runtime
)

catkin_install_python(PROGRAMS
  scripts/move_moveit_fr3.py
  scripts/move_rbkairos.py
  scripts/service_caller.py
  scripts/action_manager.py
  scripts/yolo_node_laptop.py
  scripts/detect_and_grasp.py
  scripts/visualize_detections.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)



