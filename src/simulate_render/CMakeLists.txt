cmake_minimum_required(VERSION 3.16)
project(simulate_render)
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

include(FetchContent)
set(MUJOCO_DEP_VERSION_lodepng
    b4ed2cd7ecf61d29076169b49199371456d4f90b
    CACHE STRING "Version of `lodepng` to be fetched."
)

# Fetch lodepng dependency.
if(NOT TARGET lodepng)
  FetchContent_Declare(
    lodepng
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/third_party/lodepng
    # GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
    # GIT_TAG ${MUJOCO_DEP_VERSION_lodepng}
  )

  FetchContent_GetProperties(lodepng)
  if(NOT lodepng_POPULATED)
    FetchContent_Populate(lodepng)
    # This is not a CMake project.
    set(LODEPNG_SRCS ${lodepng_SOURCE_DIR}/lodepng.cpp)
    set(LODEPNG_HEADERS ${lodepng_SOURCE_DIR}/lodepng.h)
    add_library(lodepng STATIC ${LODEPNG_HEADERS} ${LODEPNG_SRCS})
    target_compile_options(lodepng PRIVATE ${MUJOCO_MACOS_COMPILE_OPTIONS})
    target_link_options(lodepng PRIVATE ${MUJOCO_MACOS_LINK_OPTIONS})
    target_include_directories(lodepng PUBLIC ${lodepng_SOURCE_DIR})
  endif()
endif()


# find_package(glfw3 REQUIRED)
add_executable(${PROJECT_NAME} main.cc)

target_link_libraries(${PROJECT_NAME}
  ${CMAKE_SOURCE_DIR}/lib/libmujoco.so
  ${CMAKE_SOURCE_DIR}/lib/libsimulate.a
  pthread
  glfw
  lodepng
)