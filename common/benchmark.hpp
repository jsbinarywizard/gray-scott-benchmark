#pragma once

#include <cstddef>

#include <Kokkos_Core.hpp>

#include "coefficients.hpp"
#include "parameters.hpp"
#include "timer.hpp"

struct results {
    double total_time = 0.0;
    double communication_time = 0.0;
    double updates_per_second = 0.0;
};

template <typename real>
class benchmark {

public:
    explicit benchmark(const Parameters& parameters,
                        const coefficients<real>& coeffs = coefficients<real>())
        : parameters(parameters), coeffs(coeffs) {}

    virtual ~benchmark() = default;

    benchmark() = delete;
    benchmark(const benchmark&) = delete;
    benchmark& operator=(const benchmark&) = delete;

    void run(results& results) {

        for (int warmup_iteration = 0;
             warmup_iteration < parameters.warmup_iters;
             ++warmup_iteration) {
            iteration();
        }

        Kokkos::fence();

        iteration_timer.reset();

        for (int it = 0; it < parameters.benchmark_iters; ++it) {
            iteration();
        }

        Kokkos::fence();

        results.total_time = iteration_timer.elapsed();
        results.communication_time = communication_seconds;
        results.updates_per_second =
            static_cast<double>(parameters.benchmark_iters) / results.total_time;
    }

protected:
    // Derived classes implement exactly one iteration (halo exchange +
    // compute + field swap) using whatever member state they own
    // (fields, decomposition, communication buffers, ...).
    virtual void iteration() = 0;

    const Parameters& parameters;
    coefficients<real> coeffs;

    timer iteration_timer;
    timer communication_timer;

    // Derived classes accumulate time spent in communication here so
    // that run() can report it back through results.communication_time.
    double communication_seconds = 0.0;
};