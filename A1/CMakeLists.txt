cmake_minimum_required(VERSION 3.10)
project(ass1)

find_package(OpenCV REQUIRED)

add_executable(camera_calibration camera_calibration.cpp)
add_executable(stereo_calib stereo_calib.cpp)

target_link_libraries(camera_calibration ${OpenCV_LIBS})
target_link_libraries(stereo_calib ${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})
