#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
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

public:
    static StridedRow pack_subview(int dir, const View& field) {
        const int r = static_cast<int>(field.extent(0)) - 2;
        const int c = static_cast<int>(field.extent(1)) - 2;
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

    static StridedRow halo_subview(int dir, const View& field) {
        const int r = static_cast<int>(field.extent(0)) - 2;
        const int c = static_cast<int>(field.extent(1)) - 2;
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

protected:
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

template <typename real>
class blocking_benchmark : public benchmark<real> {
public:
    using View = typename benchmark<real>::View;

    struct Buffers {
        using Buffer = Kokkos::View<real*>;

        std::array<Buffer, n_directions> send;
        std::array<Buffer, n_directions> recv;

        Buffers(int rows, int columns) {
            for (int direction = 0; direction < n_directions; ++direction) {
                const int length = halo_length(direction, rows, columns);
                send[direction] = Buffer("send", 2 * length);
                recv[direction] = Buffer("recv", 2 * length);
            }
        }
    };

    explicit blocking_benchmark(const Parameters& parameters,
                                const Decomposition& decomposition,
                                const coefficients<real>& coeffs = coefficients<real>())
                : benchmark<real>(parameters, decomposition, coeffs),
                    buffers(this->local_rows, this->local_columns) {}

protected:
    void iteration() final {
        this->communication_timer.reset();
        pack(this->u, this->v, buffers);
        Kokkos::fence();
        exchange(buffers, 100);
        unpack(this->u, this->v, buffers);
        Kokkos::fence();
        this->communication_time += this->communication_timer.elapsed();

        gs_compute(this->u, this->v, this->u_temp, this->v_temp, this->coeffs);
        Kokkos::fence();
        std::swap(this->u, this->u_temp);
        std::swap(this->v, this->v_temp);
    }

    virtual void exchange(Buffers& buffers, int tag_base) = 0;

    static int halo_length(int direction, int rows, int columns) {
        switch (direction) {
            case N:
            case S:
                return columns;
            case W:
            case E:
                return rows;
            case NW:
            case NE:
            case SW:
            case SE:
                return 1;
            default:
                throw std::logic_error("halo_length: invalid direction");
        }
    }

    void pack(const View u, const View v, Buffers& buffers) {
        for (int direction = 0; direction < n_directions; ++direction) {
            if (this->decomposition.neighbors[direction] == MPI_PROC_NULL) continue;
            const int length = half_length(direction, u);
            auto u_target = Kokkos::subview(buffers.send[direction],
                                            Kokkos::make_pair(0, length));
            auto v_target = Kokkos::subview(buffers.send[direction],
                                            Kokkos::make_pair(length, 2 * length));
            Kokkos::deep_copy(u_target, benchmark<real>::pack_subview(direction, u));
            Kokkos::deep_copy(v_target, benchmark<real>::pack_subview(direction, v));
        }
    }

    void unpack(const View u, const View v, Buffers& buffers) {
        for (int direction = 0; direction < n_directions; ++direction) {
            if (this->decomposition.neighbors[direction] == MPI_PROC_NULL) continue;
            const int length = half_length(direction, u);
            auto u_source = Kokkos::subview(buffers.recv[direction],
                                            Kokkos::make_pair(0, length));
            auto v_source = Kokkos::subview(buffers.recv[direction],
                                            Kokkos::make_pair(length, 2 * length));
            Kokkos::deep_copy(benchmark<real>::halo_subview(direction, u), u_source);
            Kokkos::deep_copy(benchmark<real>::halo_subview(direction, v), v_source);
        }
    }

    static int half_length(int direction, const View& field) {
        const int rows = static_cast<int>(field.extent(0)) - 2;
        const int columns = static_cast<int>(field.extent(1)) - 2;
        return halo_length(direction, rows, columns);
    }

    Buffers buffers;
};

template <typename real>
class NonBlockingBenchmark : public benchmark<real> {
public:
    using View = typename benchmark<real>::View;

    struct Buffers {
        using Buffer = Kokkos::View<real*>;

        std::array<Buffer, n_directions> send;
        std::array<Buffer, n_directions> recv;

        Buffers(int rows, int) {
            send[W] = Buffer("send west", rows);
            send[E] = Buffer("send east", rows);
            recv[W] = Buffer("recv west", rows);
            recv[E] = Buffer("recv east", rows);
        }
    };

    static bool uses_buffer(int direction) {
        return direction == W || direction == E;
    }

    class ExchangeHandleBase {
    public:
        virtual ~ExchangeHandleBase() = default;
        virtual void wait_receives() = 0;
        virtual void wait_sends() = 0;
    };
    using ExchangeHandle = std::unique_ptr<ExchangeHandleBase>;

    explicit NonBlockingBenchmark(const Parameters& parameters,
                                  const Decomposition& decomposition,
                                  const coefficients<real>& coeffs = coefficients<real>())
                : benchmark<real>(parameters, decomposition, coeffs),
                    u_buffers(this->local_rows, this->local_columns),
                    v_buffers(this->local_rows, this->local_columns) {}

protected:
    void iteration() final {
        pack(this->u, u_buffers);
        pack(this->v, v_buffers);
        Kokkos::fence();
        ExchangeHandle u_handle = start_exchange(this->u, u_buffers, 100);
        ExchangeHandle v_handle = start_exchange(this->v, v_buffers, 200);

        gs_compute_interior(this->u, this->v, this->u_temp, this->v_temp, this->coeffs);

        u_handle->wait_receives();
        v_handle->wait_receives();

        Kokkos::fence();

        gs_compute_ring(this->u, this->v, this->u_temp, this->v_temp, this->coeffs);
        Kokkos::fence();

        u_handle->wait_sends();
        v_handle->wait_sends();

        std::swap(this->u, this->u_temp);
        std::swap(this->v, this->v_temp);
    }

    virtual ExchangeHandle start_exchange(View& field, Buffers& buffers, int tag_base) = 0;

    void pack(const View& field, Buffers& buffers) {
        for (int direction = 0; direction < n_directions; ++direction) {
            if (!uses_buffer(direction) ||
                this->decomposition.neighbors[direction] == MPI_PROC_NULL) continue;
            Kokkos::deep_copy(buffers.send[direction], benchmark<real>::pack_subview(
                direction, field));
        }
    }

    Buffers u_buffers;
    Buffers v_buffers;
};
