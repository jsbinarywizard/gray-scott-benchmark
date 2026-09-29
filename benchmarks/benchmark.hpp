#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include "../common/coefficients.hpp"
#include "../common/decomposition.hpp"
#include "../common/kernel.hpp"
#include "../common/parameters.hpp"
#include "../common/timer.hpp"

struct results {
    double total_time = 0.0;
    double updates_per_second = 0.0;
    int iterations = 0;
    double communication_time = 0.0;
};

template <typename real>
class benchmark {
public:
    using View = gs_view<real>;
    using StridedRow = Kokkos::View<real*, Kokkos::LayoutStride>;

    explicit benchmark(const Parameters& parameters,
                        const Decomposition& decomposition,
                        const coefficients<real>& coeffs = coefficients<real>())
        : parameters(parameters), decomposition(decomposition), coeffs(coeffs) {
        if (parameters.strong_scaling) {
            local_rows = parameters.rows / decomposition.dims[0];
            local_columns = parameters.columns / decomposition.dims[1];
        } else {
            local_rows = parameters.rows;
            local_columns = parameters.columns;
        }

        u = View("u", local_rows + 2, local_columns + 2);
        v = View("v", local_rows + 2, local_columns + 2);
        u_temp = View("u_temp", local_rows + 2, local_columns + 2);
        v_temp = View("v_temp", local_rows + 2, local_columns + 2);
    }

    virtual ~benchmark() = default;
    benchmark() = delete;
    benchmark(const benchmark&) = delete;
    benchmark& operator=(const benchmark&) = delete;

    void run(results& results) {
        int w_iterations = iteration_count(parameters.warmup_time);
        MPI_Bcast(&w_iterations, 1, MPI_INT, 0, decomposition.comm);
        const int warmup_iterations = w_iterations;

        iteration_timer.reset();
        for (int i = 0; i < warmup_iterations; ++i) iteration();
        Kokkos::fence();

        const double warmup_time = iteration_timer.elapsed();
        time_per_iteration = warmup_time / static_cast<double>(warmup_iterations);


        int b_iterations = iteration_count(parameters.benchmark_time);
        MPI_Bcast(&b_iterations, 1, MPI_INT, 0, decomposition.comm);
        const int benchmark_iterations = b_iterations;

        communication_time = 0.0;
        iteration_timer.reset();
        for (int it = 0; it < benchmark_iterations; ++it) iteration();
        Kokkos::fence();

        results.iterations = benchmark_iterations;
        results.total_time = iteration_timer.elapsed();

        int process_count = 1;
        if (!parameters.strong_scaling) {
            MPI_Comm_size(MPI_COMM_WORLD, &process_count);
        }
        const double total_updates = static_cast<double>(benchmark_iterations) *
                                      static_cast<double>(parameters.rows) *
                                      static_cast<double>(parameters.columns) *
                                      static_cast<double>(process_count);
        results.updates_per_second = total_updates / results.total_time;

        results.communication_time = communication_time;
    }

    static void reset_timing() {first_run = true;}

protected:
    virtual void iteration() = 0;

    static StridedRow pack_subview(int dir, const View& field, int r, int c) {
        switch (dir) {
            case N:  return Kokkos::subview(field, 1, Kokkos::make_pair(int(1), c + 1));
            case S:  return Kokkos::subview(field, r, Kokkos::make_pair(int(1), c + 1));
            case W:  return Kokkos::subview(field, Kokkos::make_pair(int(1), r + 1), 1);
            case E:  return Kokkos::subview(field, Kokkos::make_pair(int(1), r + 1), c);
            case NW: return Kokkos::subview(field, 1, Kokkos::make_pair(int(1), int(2)));
            case NE: return Kokkos::subview(field, 1, Kokkos::make_pair(c, c + 1));
            case SW: return Kokkos::subview(field, r, Kokkos::make_pair(int(1), int(2)));
            case SE: return Kokkos::subview(field, r, Kokkos::make_pair(c, c + 1));
            default: throw std::logic_error("pack_subview: invalid direction");
        }
    }

    static StridedRow halo_subview(int dir, View& field, int r, int c) {
        switch (dir) {
            case N:  return Kokkos::subview(field, 0,     Kokkos::make_pair(int(1), c + 1));
            case S:  return Kokkos::subview(field, r + 1, Kokkos::make_pair(int(1), c + 1));
            case W:  return Kokkos::subview(field, Kokkos::make_pair(int(1), r + 1), 0);
            case E:  return Kokkos::subview(field, Kokkos::make_pair(int(1), r + 1), c + 1);
            case NW: return Kokkos::subview(field, 0,     Kokkos::make_pair(int(0), int(1)));
            case NE: return Kokkos::subview(field, 0,     Kokkos::make_pair(c + 1, c + 2));
            case SW: return Kokkos::subview(field, r + 1, Kokkos::make_pair(int(0), int(1)));
            case SE: return Kokkos::subview(field, r + 1, Kokkos::make_pair(c + 1, c + 2));
            default: throw std::logic_error("halo_subview: invalid direction");
        }
    }

    const Parameters& parameters;
    const Decomposition& decomposition;
    coefficients<real> coeffs;

    // Local field dimensions (excluding halo).
    int local_rows;
    int local_columns;


    timer iteration_timer;
    timer communication_timer;
    double communication_time = 0.0;

    View u, v, u_temp, v_temp;

private:
    static int estimate_iterations(int rows, int columns) {
        const long long cells = static_cast<long long>(rows) * static_cast<long long>(columns);
        return cells > 0 ? static_cast<int>(std::max<long long>(1'000'000LL / cells, 10))
                         : 10;
    }

    int iteration_count(double time) {
        if (first_run) {
            first_run = false;
            previous_cells = static_cast<long long>(parameters.rows) * static_cast<long long>(parameters.columns);
            return estimate_iterations(parameters.rows, parameters.columns);
        }
        const long long cells = static_cast<long long>(parameters.rows) * static_cast<long long>(parameters.columns);
        const double scal_cells = static_cast<double>(previous_cells) / static_cast<double>(cells);
        previous_cells = cells;
        return std::max(static_cast<int>((time / time_per_iteration) * scal_cells), 10);
    }

    // Deliberately static: shared across successive benchmark<real> objects
    // (e.g. a sweep of sub-benchmarks) so each new run reuses the previous
    // one's measured rate instead of re-estimating from scratch. Scoped per
    // `real` so benchmark<float> and benchmark<double> — which generally
    // have different achievable rates — don't share one estimate.
    static inline double time_per_iteration = 0.0;
    static inline long previous_cells = 0;
    static inline bool first_run = true;
};
