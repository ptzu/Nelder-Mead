cmake_minimum_required(VERSION 2.8)
project(Nelder-Mead)
add_definitions(-std=c++11)

find_package(Eigen3)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

add_executable(nelder-mead main.cpp)
