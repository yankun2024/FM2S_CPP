cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(FM2S_CPP)

set(OpenCV_PREFIX_PATH "./opencv/build") # fix your opencv path
list(APPEND CMAKE_PREFIX_PATH ${OpenCV_PREFIX_PATH})
find_package(OpenCV REQUIRED)

set(CMAKE_PREFIX_PATH "./libtorch") # fix your libtorch path
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(FM2S_CPP main.cpp FM2S.h)
target_link_libraries(FM2S_CPP "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

set_property(TARGET FM2S_CPP PROPERTY CXX_STANDARD 17)
