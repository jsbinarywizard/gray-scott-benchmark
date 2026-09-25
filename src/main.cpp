#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../common/benchmark.hpp"
#include "../common/arguments.hpp"
#include "../common/parameters.hpp"
#include "../common/output.hpp"

#include "../mpi/gs_mpi_blocking.hpp"
#include "../mpi/gs_mpi_async.hpp"

#include "../kc/gs_kc_blocking.hpp"
#include "../kc/gs_kc_async.hpp"

namespace {

// -----------------------------------------------------------------------------
// The one place that maps a Backend enum value to an actual benchmark
// class. Adding backend 3 and 4 means adding a case here (plus the enum
// entry, backend_name()/all_backends() and --backend parsing in
// benchmark_cli.hpp) -- nothing else in main.cpp changes.
// -----------------------------------------------------------------------------
template <typename real>
std::unique_ptr<benchmark<real>> make_benchmark(Backend backend, const Parameters& parameters) {
    switch (backend) {
        case Backend::MPI_BLOCKING:
            return std::make_unique<GS_MPI_Blocking<real>>(parameters);
        case Backend::MPI_NONBLOCKING:
            return std::make_unique<GS_MPI_NonBlocking<real>>(parameters);

        case Backend::KC_BLOCKING:
            return std::make_unique<GS_KC_Blocking<real>>(parameters);
        case Backend::KC_NONBLOCKING:
            return std::make_unique<GS_KC_NonBlocking<real>>(parameters);
    }

    throw std::runtime_error("Unknown backend.");
}

// -----------------------------------------------------------------------------
// Rough per-rank memory footprint of the four fields (u, v, u_temp, v_temp)
// at a given local size, used only to warn when --memory-limit is exceeded.
// -----------------------------------------------------------------------------
template <typename real>
double estimated_local_gb(int local_rows, int local_columns, int process_count) {
    constexpr int n_fields = 4;
    const double cells =
        static_cast<double>(local_rows + 2) * static_cast<double>(local_columns + 2);
    return n_fields * cells * sizeof(real) / ((1024.0 * 1024.0 * 1024.0) / static_cast<double>(process_count));
}

// Scaling::BOTH is a sweep instruction, not a value run() understands --
// expand it here, once, rather than every caller having to remember to.
std::vector<Scaling> scalings_to_run(Scaling scaling) {
    if (scaling == Scaling::BOTH) {
        return {Scaling::STRONG, Scaling::WEAK};
    }
    return {scaling};
}

// -----------------------------------------------------------------------------
// Build Parameters for one (backend, scaling, size, precision) combination
// and execute it.
// -----------------------------------------------------------------------------
template <typename real>
bool run_one(const BenchmarkConfig& config, Backend backend, Scaling scal,
             int size, int rank, ResultsWriter& writer) {

    writer.note("Running backend " + std::string(backend_name(backend)) +
                ", precision " + std::string(precision_name<real>()) +
                ", scaling " + std::string(scaling_name(scal)) +
                ", size " + std::to_string(size) + "...");

    Parameters parameters;
    parameters.warmup_iters = config.warmup_iters;
    parameters.benchmark_iters = config.benchmark_iters;
    parameters.rows = size;
    parameters.columns = size;
    parameters.measure_cell_updates = config.measure_cell_updates;
    parameters.measure_comm_bandwidth = config.measure_comm_bandwidth;
    parameters.strong_scaling = (scal == Scaling::STRONG);

    const bool strong_scaling = parameters.strong_scaling;
    int process_count = 1;
    if (strong_scaling)
        MPI_Comm_size(MPI_COMM_WORLD, &process_count);    // For strong scaling, the field size is fixed and spread across all ranks, so we need to check the memory limit against the per-rank size. 
    // For weak scaling, each rank has its own local size, so we only need to check the memory limit against the local size.


    const double gb = estimated_local_gb<real>(size, size, process_count);
    if (gb > config.memory_limit_gb) {
        writer.skip(backend, precision_name<real>(), scal, size,
                 "estimated " + std::to_string(gb) + " GB per rank exceeds --memory-limit " +
                 std::to_string(config.memory_limit_gb) + " GB");
        return false;
    }

    auto bm = make_benchmark<real>(backend, parameters);

    results r;
    bm->run(r);

    writer.write(BenchmarkRow::from_results(r, backend, precision_name<real>(), scal, process_count, size));

    writer.note("  [" + std::string(backend_name(backend)) + "/" +
             std::string(scaling_name(scal)) + "/" +
             std::string(precision_name<real>()) + "] size=" +
             std::to_string(size) + " Finished \n");

    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

#if defined(KOKKOSCOMM_ENABLE_NCCL)
    {
        // Bind each rank to a distinct GPU on its node before Kokkos/NCCL
        // initialize, based on node-local rank. Harmless (and unused) for
        // backends that don't need a GPU; only built into NCCL-enabled
        // binaries in the first place.
        int local_rank = 0;
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                             MPI_INFO_NULL, &local_comm);
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_free(&local_comm);
        cudaSetDevice(local_rank);
    }
#endif

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int exit_code = 0;

    try {
        Kokkos::ScopeGuard kokkos{argc, argv};

        BenchmarkConfig config;
        BenchmarkParser parser("gs_benchmark");

        bool should_run = true;

        try {
            should_run = parser.parse(argc, argv, config, rank);
        }
        catch (const std::exception& e) {
            if (rank == 0) {
                std::cerr << e.what() << '\n';
            }
            should_run = false;
            exit_code = 1;
        }

        if (should_run) {
            ResultsWriter writer(config.output_file, rank == 0);

            MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (exit_code != 0) {
                return exit_code;
            }

            const std::vector<Scaling> scalings = scalings_to_run(config.scaling);

            for (int size : config.sizes) {
                for (Scaling scaling : scalings) {
                    for (Backend backend : config.backends) {

                        if (config.precision == Precision::SINGLE ||
                            config.precision == Precision::BOTH) {
                            run_one<float>(config, backend, scaling, size, rank, writer);
                        }

                        if (config.precision == Precision::DOUBLE ||
                            config.precision == Precision::BOTH) {
                            run_one<double>(config, backend, scaling, size, rank, writer);
                        }
                    }
                }
            }
            writer.note("All benchmarks completed successfully.");
            writer.note("Results written to " + config.output_file);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Rank " << rank << ": " << e.what() << '\n';
        exit_code = 1;
    }

    MPI_Finalize();

    return exit_code;
}