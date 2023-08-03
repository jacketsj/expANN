cmake_minimum_required(VERSION 3.10)

project(expann LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(SRC ${CMAKE_SOURCE_DIR}/src)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
include_directories("${SRC}")

#set(CMAKE_CXX_FLAGS "-Wall -pthread -O3 -mavx -fopenmp -g -fsanitize=address")
set(CMAKE_CXX_FLAGS "-Wall -pthread -O3 -mavx -g ${OpenMP_CXX_FLAGS}")

include(./cmake/CPM.cmake)  # Package manager for simple requirements.

# run `conan install .. --build=missing` in `./cmake`
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

#CPMAddPackage("gh:fmtlib/fmt#9.1.0")  # fmt for nice strings

find_package(Threads)
find_package(nlohmann_json)

file(GLOB SOURCES ${SRC}/*.cpp)

add_executable(expann ${SOURCES})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
add_library(plt INTERFACE)
target_include_directories(plt INTERFACE plt)

#target_link_libraries(expann PUBLIC fmt::fmt)
target_link_libraries(expann PUBLIC plt ${PYTHON_LIBRARIES} nlohmann_json::nlohmann_json)
