cmake_minimum_required(VERSION 3.10)
project(ass2)

find_package(OpenCV REQUIRED)

add_executable(assign2_task1 assign2_task1.cpp)
add_executable(assign2_task2 assign2_task2.cpp)

target_link_libraries(assign2_task1 ${OpenCV_LIBS})
target_link_libraries(assign2_task2 ${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})
