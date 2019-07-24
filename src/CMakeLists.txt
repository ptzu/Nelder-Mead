cmake_minimum_required(VERSION 2.8)
project(Nelder-Mead)

find_package(Eigen3)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

add_executable(nelder-mead main.cpp)
