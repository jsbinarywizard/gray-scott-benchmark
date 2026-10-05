cmake_minimum_required(VERSION 3.28)

project(
    GrayScottCommBenchmark
    VERSION 0.2.1
    LANGUAGES CXX
)

# ------------------------------------------------------------------------------
# Build configuration
# ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------

set(GS_ENABLE_CCL "AUTO" CACHE STRING "CCL backend (NCCL on CUDA, RCCL on HIP): AUTO, ON or OFF")
set_property(CACHE GS_ENABLE_CCL PROPERTY STRINGS AUTO ON OFF)

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

find_package(MPI REQUIRED)
find_package(Kokkos REQUIRED)

include(FetchContent)

FetchContent_Declare(
    KokkosComm
    GIT_REPOSITORY https://github.com/kokkos/kokkos-comm.git
    GIT_TAG develop
)

FetchContent_MakeAvailable(KokkosComm)

# ------------------------------------------------------------------------------
# Target
# ------------------------------------------------------------------------------

add_executable(
    gray_scott_benchmark
    src/main.cpp
)

target_link_libraries(
    gray_scott_benchmark
    PRIVATE
    MPI::MPI_CXX
    Kokkos::kokkos
    KokkosComm::KokkosComm
)

# KokkosComm generates config.hpp in its binary directory.
get_target_property(KOKKOSCOMM_BINARY_DIR KokkosComm::KokkosComm BINARY_DIR)

target_include_directories(
    gray_scott_benchmark
    PRIVATE
    ${KOKKOSCOMM_BINARY_DIR}/src
)

target_compile_features(
    gray_scott_benchmark
    PRIVATE
    cxx_std_20
)

# ------------------------------------------------------------------------------
# Optional CCL backend (NCCL / RCCL)
# ------------------------------------------------------------------------------

set(GS_CCL_BACKEND "NONE")

if(NOT GS_ENABLE_CCL STREQUAL "OFF")
    kokkos_check(DEVICES CUDA RESULT_VARIABLE GS_KOKKOS_HAS_CUDA)
    kokkos_check(DEVICES HIP  RESULT_VARIABLE GS_KOKKOS_HAS_HIP)

    if(GS_KOKKOS_HAS_CUDA)
        find_path(NCCL_INCLUDE_DIR nccl.h
            HINTS $ENV{NCCL_HOME}/include ${NCCL_ROOT}/include)
        find_library(NCCL_LIBRARY nccl
            HINTS $ENV{NCCL_HOME}/lib ${NCCL_ROOT}/lib)
        if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
            set(GS_CCL_BACKEND "NCCL")
            target_include_directories(gray_scott_benchmark PRIVATE ${NCCL_INCLUDE_DIR})
            target_link_libraries(gray_scott_benchmark PRIVATE ${NCCL_LIBRARY})
            target_compile_definitions(gray_scott_benchmark PRIVATE GS_CCL_BACKEND_NCCL)
        endif()
    elseif(GS_KOKKOS_HAS_HIP)
        find_package(rccl QUIET)
        if(rccl_FOUND)
            set(GS_CCL_BACKEND "RCCL")
            target_link_libraries(gray_scott_benchmark PRIVATE roc::rccl)
            target_compile_definitions(gray_scott_benchmark PRIVATE GS_CCL_BACKEND_RCCL)
        endif()
    endif()

    if(GS_ENABLE_CCL STREQUAL "ON" AND GS_CCL_BACKEND STREQUAL "NONE")
        message(FATAL_ERROR
            "GS_ENABLE_CCL=ON, but Kokkos has no CUDA/HIP backend, "
            "or NCCL/RCCL was not found (try -DNCCL_ROOT=... or NCCL_HOME).")
    endif()
endif()

message(STATUS "GS CCL backend: ${GS_CCL_BACKEND}")