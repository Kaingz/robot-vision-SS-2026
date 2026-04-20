cmake_minimum_required(VERSION 3.10)
project(camera_calibration)

find_package(OpenCV REQUIRED)

add_executable(camera_calibration camera_calibration.cpp)
target_link_libraries(camera_calibration ${OpenCV_LIBS})
target_include_directories(camera_calibration PRIVATE ${OpenCV_INCLUDE_DIRS})