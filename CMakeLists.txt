cmake_minimum_required(VERSION 3.1.0)

project(image_classification)

#set(CMAKE_PREFIX_PATH /home/vladimir/Work/libtorch)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(Boost REQUIRED)
find_package(CURL REQUIRED) 

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${CURL_INCLUDE_DIR})
add_executable(${PROJECT_NAME} "image_classification.cpp")

message(STATUS "OpenCV library status:")
message(STATUS " config: ${OPENCV_DIR}")
message(STATUS " version: ${OpenV_VERSION}")
message(STATUS " libraries: ${OpenCV_LIBS}")
message(STATUS " include_path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "Torch library status:")
message(STATUS " include_path: ${TORCH_INCLUDE_DIRS}")
message(STATUS " TORCHLIB: ${TORCH_LIBRARIES}")

message(STATUS " CURLLIB: ${CURL_LIBRARIES}")


target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${CURL_LIBRARIES})
