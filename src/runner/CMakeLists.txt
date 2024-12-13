cmake_minimum_required(VERSION 3.16)
project(ctrl_runner)
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)


add_executable(${PROJECT_NAME} ctrl_runner.cpp)

target_link_libraries(${PROJECT_NAME}
  simulate_engine
  # ${CMAKE_SOURCE_DIR}/lib/libmujoco.so
  # ${CMAKE_SOURCE_DIR}/lib/libsimulate.a
  # pthread
  # glfw
)


