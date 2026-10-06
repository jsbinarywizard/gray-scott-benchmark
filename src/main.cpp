#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../benchmarks/benchmark.hpp"
#include "../common/arguments.hpp"
#include "../common/parameters.hpp"
#include "../common/output.hpp"

#include "../benchmarks/gs_mpi.hpp"
#include "../benchmarks/gs_kc.hpp"
#include "../benchmarks/gs_ccl.hpp"

namespace {

// -----------------------------------------------------------------------------
// The one place that maps a Backend enum value to an actual benchmark
// class. Adding backend 3 and 4 means adding a case here (plus the enum
// entry, backend_name()/all_backends() and --backend parsing in
// benchmark_cli.hpp) -- nothing else in main.cpp changes.
// -----------------------------------------------------------------------------
template <typename real>
std::unique_ptr<benchmark<real>> make_benchmark(
    Backend backend, const Parameters& parameters, const Decomposition& decomposition) {
    switch (backend) {
        case Backend::MPI_BLOCKING:
            return std::make_unique<GS_MPI_Blocking<real>>(parameters, decomposition);
        case Backend::MPI_NONBLOCKING:
            return std::make_unique<GS_MPI_NonBlocking<real>>(parameters, decomposition);

        case Backend::CCL_BLOCKING:
            return std::make_unique<GS_CCL_Blocking<real>>(parameters, decomposition);
        case Backend::CCL_NONBLOCKING:
            return std::make_unique<GS_CCL_NonBlocking<real>>(parameters, decomposition);

        case Backend::KC_MPI_BLOCKING:
            return std::make_unique<GS_KC_Blocking<real>>(parameters, decomposition);
        case Backend::KC_MPI_NONBLOCKING:
            return std::make_unique<GS_KC_NonBlocking<real>>(parameters, decomposition);

        case Backend::KC_CCL_BLOCKING:
            return std::make_unique<GS_KC_Blocking<real, true>>(parameters, decomposition);
        case Backend::KC_CCL_NONBLOCKING:
            return std::make_unique<GS_KC_NonBlocking<real, true>>(parameters, decomposition);
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

// -----------------------------------------------------------------------------
// Build Parameters for one (backend, scaling, size, precision) combination
// and execute it.
// -----------------------------------------------------------------------------
template <typename real>
bool run_one(const BenchmarkConfig& config, Backend backend, Scaling scal,
             int problem_size, const Decomposition& decomposition, ResultsWriter& writer) {

    writer.note("Running backend " + std::string(backend_name(backend)) +
                ", precision " + std::string(precision_name<real>()) +
                ", scaling " + std::string(scaling_name(scal)) +
                ", size " + std::to_string(problem_size) + "...");

    Parameters parameters;
    parameters.warmup_time = config.warmup_time;
    parameters.benchmark_time = config.benchmark_time;
    parameters.rows = problem_size;
    parameters.columns = problem_size;
    parameters.strong_scaling = (scal == Scaling::STRONG);

    const bool strong_scaling = parameters.strong_scaling;

    int process_count = 1;
    if (strong_scaling) {
        // For strong scaling, the field size is fixed and spread across all
        // ranks, so the memory limit must be checked against the per-rank
        // size once that's accounted for.
        MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    }
    // For weak scaling, each rank has its own local size, so the memory
    // limit is checked against the local size directly (process_count == 1).

    const double gb = estimated_local_gb<real>(problem_size, problem_size, process_count);
    if (gb > config.memory_limit_gb) {
        writer.skip(backend, precision_name<real>(), scal, problem_size,
                 "estimated " + std::to_string(gb) + " GB per rank exceeds --memory-limit " +
                 std::to_string(config.memory_limit_gb) + " GB");
        return false;
    }

    auto bm = make_benchmark<real>(backend, parameters, decomposition);

    results r;
    bm->run(r);

    writer.write(BenchmarkRow::from_results(r, backend, precision_name<real>(), scal, decomposition.size, problem_size));

    writer.note("  [" + std::string(backend_name(backend)) + "/" +
             std::string(scaling_name(scal)) + "/" +
             std::string(precision_name<real>()) + "] size=" +
             std::to_string(problem_size) + " Finished \n");

    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

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

        // `should_run`/`exit_code` are computed independently per rank (each
        // rank calls parse() itself) and can legitimately disagree -- e.g.
        // rank 0 fails to open the output file while other ranks' local
        // parse succeeds. If ranks then took different branches below, the
        // collectives inside the benchmark loop (MPI_Comm_size, and whatever
        // GS_MPI/GS_KC do internally) would hang waiting on ranks that
        // already returned. Reduce to a single, rank-agnostic decisrank
        int local_should_run = should_run ? 1 : 0;
        int global_should_run = 0;
        MPI_Allreduce(&local_should_run, &global_should_run, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        should_run = (global_should_run != 0);

        int local_exit_code = exit_code;
        int global_exit_code = 0;
        MPI_Allreduce(&local_exit_code, &global_exit_code, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        exit_code = global_exit_code;

        Decomposition decomposition = Decomposition(config.periodic); // Currently not supported.
        // Need to add argument. Stuck to non-periodic for now.

        if (should_run) {
            ResultsWriter writer(config.output_file, decomposition.rank == 0);

            for (Scaling scaling : config.scaling) {
                auto sizes = (scaling == Scaling::STRONG) ? config.strong_sizes : config.weak_sizes;

                // A timing estimate learned under one scaling mode doesn't
                // transfer to the other (different local problem size per
                // rank), so start each sweep with a clean estimate.
                benchmark<float>::reset_timing();
                benchmark<double>::reset_timing();

                for (int problem_size : sizes) {
                    for (Precision precision : config.precision) {
                        for (Backend backend : config.backends) {

                            if (precision == Precision::SINGLE) {
                                run_one<float>(config, backend, scaling, problem_size, decomposition, writer);
                            }

                            if (precision == Precision::DOUBLE) {
                                run_one<double>(config, backend, scaling, problem_size, decomposition, writer);
                            }
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