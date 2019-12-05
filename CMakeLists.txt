cmake_minimum_required(VERSION 3.15)
project(CS243)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake-3.15/Modules/)

find_package(TBB)
# Add include dirctory and compile definitions for all targets
include_directories(${TBB_INCLUDE_DIRS})
add_definitions(${TBB_DEFINITIONS})
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_COMPILER /usr/local/bin/g++-9)
set(CMAKE_C_COMPILER /usr/local/bin/gcc-9)
# enable openmp using -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenacc")

add_executable(CS243 main.cpp)
target_link_libraries(CS243 ${TBB_LIBRARIES})
