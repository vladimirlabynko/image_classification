cmake_minimum_required(VERSION 3.1)
project(image_classification)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)



find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")

include_directories(image_classification ${OpenCV_INCLUDE_DIRS} )

add_executable(image_classification image_classification.cpp)
target_link_libraries(image_classification ${OpenCV_LIBS} )

message(YOBA_LIBS ${OpenCV_LIBS} )
