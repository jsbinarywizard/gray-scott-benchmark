cmake_minimum_required(VERSION 3.28)

project(
    GrayScottCommBenchmark
    VERSION 0.1.2
    LANGUAGES CXX
)

# ------------------------------------------------------------------------------
# Build configuration
# ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CXXFLAGS = -O3 _DNDEBUG)
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
get_target_property(KOKOSCOMM_BINARY_DIR KokkosComm::KokkosComm BINARY_DIR)

target_include_directories(
    gray_scott_benchmark
    PRIVATE
        ${KOKOSCOMM_BINARY_DIR}/src
)

target_compile_features(
    gray_scott_benchmark
    PRIVATE
        cxx_std_20
)
