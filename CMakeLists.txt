# Created by SwiftWare Lab on 9/24.
# CE 4SP4 - High Performance Programming
# Copyright (c) 2024 SwiftWare Lab

cmake_minimum_required(VERSION 3.22)

option(PROFILING_ENABLED "Enables Modules for Profiling, needs LIBFPM4" ON)
option(OPENMP "Enables OpenMP" ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx")
include(FetchContent)

FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)


include(FetchContent)
FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.0
)
# enable BENCHMARK_DOWNLOAD_DEPENDENCIES
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
if (PROFILING_ENABLED)
    # enable PFM
    set(BENCHMARK_ENABLE_LIBPFM ON)
endif()



FetchContent_MakeAvailable(googlebenchmark)

project(gpu C CXX )

#check CUDA version
find_package(CUDA OPTIONAL_COMPONENTS)
find_package(CUDAToolkit OPTIONAL_COMPONENTS)

if (CUDA_FOUND AND CUDAToolkit_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    add_definitions(-D__CUDA__)
    enable_language(CUDA)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G --ptxas-options=-v")
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
    message(STATUS "CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

    include_directories(${CUDA_INCLUDE_DIRS})
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")


else()
    message(STATUS "CUDA not found, proceeding without CUDA support")
    # Set the project languages to only CXX if CUDA is not found
    set(PROJECT_LANGUAGES CXX)
endif()


find_package(OpenCL OPTIONAL_COMPONENTS)

find_package(OpenMP REQUIRED)

include_directories(
        ${CMAKE_SOURCE_DIR}/include
)

set(CMAKE_CXX_STANDARD 17)

if (CUDA_FOUND AND CUDAToolkit_FOUND)
    add_executable(matmul
            ${CMAKE_SOURCE_DIR}/src/gemm.cu
            ${CMAKE_SOURCE_DIR}/matmul_driver.cpp
            ${CMAKE_SOURCE_DIR}/src/gemm.cpp
    )
else()
    message(FATAL_ERROR "CUDA is required to build this project. Please install CUDA and try again.")

endif()

target_link_libraries(matmul
        benchmark::benchmark
        OpenMP::OpenMP_CXX
)

add_subdirectory(test)