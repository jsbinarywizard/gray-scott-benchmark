cmake_minimum_required(VERSION 3.28)

project(
    GrayScottCommBenchmark
    VERSION 0.1.2
    LANGUAGES CXX
)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------

option(
    ENABLE_DOWNLOAD_FALLBACK
    "Download dependencies if they cannot be found"
    ON
)

# ------------------------------------------------------------------------------
# Build configuration
# ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

find_package(MPI REQUIRED)

find_package(Kokkos 5.1.1 QUIET)

if(NOT Kokkos_FOUND)
    if(NOT ENABLE_DOWNLOAD_FALLBACK)
        message(FATAL_ERROR
            "Kokkos >= 5.1.1 was not found and "
            "ENABLE_DOWNLOAD_FALLBACK is OFF."
        )
    endif()

    message(STATUS "Kokkos not found; downloading Kokkos 5.1.1")

    include(FetchContent)

    FetchContent_Declare(
        Kokkos
        URL
            https://github.com/kokkos/kokkos/releases/download/5.1.1/kokkos-5.1.1.zip
        URL_HASH
            SHA256=1de49ac660b5ccc619e22f5b3215194b1ea2ac005aa6130ab28d300ca8f9e5ff
    )

    FetchContent_MakeAvailable(Kokkos)
endif()

# KokkosComm is currently required by the executable.
# find_package(KokkosComm REQUIRED)

# ------------------------------------------------------------------------------
# Target
# ------------------------------------------------------------------------------

add_executable(gray_scott_benchmark)

target_sources(
    gray_scott_benchmark
    PRIVATE
        src/main.cpp
)

target_link_libraries(
    gray_scott_benchmark
    PRIVATE
        MPI::MPI_CXX
        Kokkos::kokkos
        # KokkosComm::KokkosComm
)

