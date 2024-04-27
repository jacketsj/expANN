cmake_minimum_required(VERSION 3.10)

project(expann LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(SRC ${CMAKE_SOURCE_DIR}/src)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
include_directories("${SRC}")

#set(CMAKE_CXX_FLAGS "-Wall -pthread -O3 -mavx -fopenmp -g -fsanitize=address")
#set(CMAKE_CXX_FLAGS "-Wall -pthread -O3 -mavx -ggdb3 ${OpenMP_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "-Wall -pthread -O3 -march=native -fno-omit-frame-pointer -ggdb3 ${OpenMP_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "-Wall -pthread -Og -mavx -ggdb3 ${OpenMP_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "-Wall -pthread -O3 -mavx -g ${OpenMP_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "-Wall -pthread -O3 -ggdb3 ${OpenMP_CXX_FLAGS}") # For profiling
#set(CMAKE_CXX_FLAGS "-Wall -pthread -O3 -march=native -ftree-vectorize -funroll-loops -frename-registers -ggdb3 ${OpenMP_CXX_FLAGS}") # The real one
#set(CMAKE_CXX_FLAGS "-Wall -pthread -O3 -march=native -ggdb3 ${OpenMP_CXX_FLAGS}") # Improve the instruction cache...?
set(CMAKE_CXX_FLAGS "-Wall -pthread -O3 -march=native -ggdb3 -frename-registers ${OpenMP_CXX_FLAGS}") # Non-standard options
#set(CMAKE_CXX_FLAGS "-Wall -pthread -O3 -march=native -fno-omit-frame-pointer -g ${OpenMP_CXX_FLAGS}") # For running perf
#set(CMAKE_CXX_FLAGS "-Wall -pthread -Og -mavx -fprofile-arcs -ftest-coverage -ggdb3 ${OpenMP_CXX_FLAGS}") # For running line coverage


# Different build options for profiling of various kinds
option(ENABLE_STACK_INFO "Enable stack info" OFF)
option(ENABLE_GCOV "Enable gcov" OFF)
option(ENABLE_PGO_GENERATE "Enable PGO - Profile generation phase" OFF)
option(ENABLE_PGO_USE "Enable PGO - Use profile data" OFF)

#set(CMAKE_CXX_FLAGS "-Wall -pthread -O3 ${OpenMP_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "-Wall -pthread -Ofast -ftree-vectorize -ftree-vectorizer-verbose=0 ${OpenMP_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "-Wall -pthread -Ofast -ftree-vectorize ${OpenMP_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "-Wall -pthread -Ofast -ftree-vectorize ${OpenMP_CXX_FLAGS}") # Normal
#set(CMAKE_CXX_FLAGS "-Wall -pthread -O0 -ftree-vectorize ${OpenMP_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "-Wall -pthread -O0 ${OpenMP_CXX_FLAGS}") # Segfault debugging
#set(CMAKE_CXX_FLAGS "-Wall -Ofast ${OpenMP_CXX_FLAGS}")

if(ENABLE_ASAN)
    message("Enabling asan.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -fno-omit-frame-pointer")
else()
    message("Disabling asan.")
endif()

if(ENABLE_STACK_INFO)
    message("Enabling stack info (-march=native omitted).")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mno-avx -fno-omit-frame-pointer -g")
else()
    message("Disabling stack info.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ggdb3 -funroll-loops")
endif()

if(ENABLE_GCOV)
    message("Enabling gcov.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
else()
    message("Disabling gcov.")
endif()

if (ENABLE_PGO_GENERATE)
    message("Enabling PGO: profile generation.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-generate")
elseif (ENABLE_PGO_USE)
    message("Enabling PGO: use profile data.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-use")
endif()


#include(./cmake/CPM.cmake)  # Package manager for simple requirements.

# run `conan install .. --build=missing` in `./cmake`
#list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
#list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

#CPMAddPackage("gh:fmtlib/fmt#9.1.0")  # fmt for nice strings

find_package(Threads)
find_package(nlohmann_json)

add_subdirectory(eigen)

file(GLOB SOURCES ${SRC}/main.cpp)

add_executable(expann ${SOURCES})
target_compile_definitions(expann PRIVATE DIM=128)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

#target_link_libraries(expann PUBLIC fmt::fmt)
target_link_libraries(expann PUBLIC ${PYTHON_LIBRARIES} nlohmann_json::nlohmann_json Threads::Threads eigen)

# Pybinds
add_subdirectory(pybind11)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
file(GLOB SOURCES ${SRC}/pyrunner.cpp)

# Significant speedups are obtained by choosing the dimension at compile-time (seemingly due to a dumb optimizer, rather than because of additional possible compile-time optimizations)
# Only allowable dimensionalities are multiples of 64 (required by scalar quantization method used)
pybind11_add_module(expann_py MODULE src/pyrunner.cpp)
target_compile_definitions(expann_py PRIVATE MODULE_NAME=expann_py)
target_link_libraries(expann_py PRIVATE ${PYTHON_LIBRARIES} nlohmann_json::nlohmann_json Threads::Threads Eigen3::Eigen pybind11::headers)
set_target_properties(expann_py PROPERTIES
                                INTERPROCEDURAL_OPTIMIZATION ON
                                CXX_VISIBILITY_PRESET default
                                VISIBILITY_INLINES_HIDDEN ON)

pybind11_add_module(expann_py_64 MODULE src/pyrunner.cpp)
target_compile_definitions(expann_py_64 PRIVATE MODULE_NAME=expann_py_64)
target_compile_definitions(expann_py_64 PRIVATE DIM=64)
target_link_libraries(expann_py_64 PRIVATE ${PYTHON_LIBRARIES} nlohmann_json::nlohmann_json Threads::Threads Eigen3::Eigen pybind11::headers)
set_target_properties(expann_py_64 PROPERTIES
                                INTERPROCEDURAL_OPTIMIZATION ON
                                CXX_VISIBILITY_PRESET default
                                VISIBILITY_INLINES_HIDDEN ON)

pybind11_add_module(expann_py_128 MODULE src/pyrunner.cpp)
target_compile_definitions(expann_py_128 PRIVATE MODULE_NAME=expann_py_128)
target_compile_definitions(expann_py_128 PRIVATE DIM=128)
target_link_libraries(expann_py_128 PRIVATE ${PYTHON_LIBRARIES} nlohmann_json::nlohmann_json Threads::Threads Eigen3::Eigen pybind11::headers)
set_target_properties(expann_py_128 PROPERTIES
                                INTERPROCEDURAL_OPTIMIZATION ON
                                CXX_VISIBILITY_PRESET default
                                VISIBILITY_INLINES_HIDDEN ON)

pybind11_add_module(expann_py_256 MODULE src/pyrunner.cpp)
target_compile_definitions(expann_py_256 PRIVATE MODULE_NAME=expann_py_256)
target_compile_definitions(expann_py_256 PRIVATE DIM=256)
target_link_libraries(expann_py_256 PRIVATE ${PYTHON_LIBRARIES} nlohmann_json::nlohmann_json Threads::Threads Eigen3::Eigen pybind11::headers)
set_target_properties(expann_py_256 PROPERTIES
                                INTERPROCEDURAL_OPTIMIZATION ON
                                CXX_VISIBILITY_PRESET default
                                VISIBILITY_INLINES_HIDDEN ON)

pybind11_add_module(expann_py_768 MODULE src/pyrunner.cpp)
target_compile_definitions(expann_py_768 PRIVATE MODULE_NAME=expann_py_768)
target_compile_definitions(expann_py_768 PRIVATE DIM=768)
target_link_libraries(expann_py_768 PRIVATE ${PYTHON_LIBRARIES} nlohmann_json::nlohmann_json Threads::Threads Eigen3::Eigen pybind11::headers)
set_target_properties(expann_py_768 PROPERTIES
                                INTERPROCEDURAL_OPTIMIZATION ON
                                CXX_VISIBILITY_PRESET default
                                VISIBILITY_INLINES_HIDDEN ON)

pybind11_add_module(expann_py_832 MODULE src/pyrunner.cpp)
target_compile_definitions(expann_py_832 PRIVATE MODULE_NAME=expann_py_832)
target_compile_definitions(expann_py_832 PRIVATE DIM=832)
target_link_libraries(expann_py_832 PRIVATE ${PYTHON_LIBRARIES} nlohmann_json::nlohmann_json Threads::Threads Eigen3::Eigen pybind11::headers)
set_target_properties(expann_py_832 PROPERTIES
                                INTERPROCEDURAL_OPTIMIZATION ON
                                CXX_VISIBILITY_PRESET default
                                VISIBILITY_INLINES_HIDDEN ON)

pybind11_add_module(expann_py_1024 MODULE src/pyrunner.cpp)
target_compile_definitions(expann_py_1024 PRIVATE MODULE_NAME=expann_py_1024)
target_compile_definitions(expann_py_1024 PRIVATE DIM=1024)
target_link_libraries(expann_py_1024 PRIVATE ${PYTHON_LIBRARIES} nlohmann_json::nlohmann_json Threads::Threads Eigen3::Eigen pybind11::headers)
set_target_properties(expann_py_1024 PROPERTIES
                                INTERPROCEDURAL_OPTIMIZATION ON
                                CXX_VISIBILITY_PRESET default
                                VISIBILITY_INLINES_HIDDEN ON)
