cmake_minimum_required(VERSION 3.16)
project(basic_render)
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# find_package(glfw3 REQUIRED)
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
  ${CMAKE_SOURCE_DIR}/lib/libmujoco.so
  pthread
  glfw
)