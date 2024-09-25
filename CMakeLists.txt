cmake_minimum_required(VERSION 3.19)

project(LMA_Eigen VERSION 0.1.0 LANGUAGES CXX C)

find_package(Eigen3 CONFIG REQUIRED)

add_executable(LMA_Eigen 
  main.cpp)

target_link_libraries(LMA_Eigen PRIVATE Eigen3::Eigen)

