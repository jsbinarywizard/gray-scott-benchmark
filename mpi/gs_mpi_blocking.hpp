#pragma once

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "../common/benchmark.hpp"
#include "../common/coefficients.hpp"
#include "../common/kernel.hpp"
#include "../common/parameters.hpp"
#include "../common/timer.hpp"

template <typename real>
class GS_MPI_Blocking : public benchmark<real> {

public:
    explicit GS_MPI_Blocking(const Parameters& parameters)
        : benchmark<real>(parameters),
          decomposition(parameters.rows, parameters.columns, parameters.strong_scaling),
          u("u", decomposition.local_rows + 2, decomposition.local_columns + 2),
          v("v", decomposition.local_rows + 2, decomposition.local_columns + 2),
          u_temp("u_temp", decomposition.local_rows + 2, decomposition.local_columns + 2),
          v_temp("v_temp", decomposition.local_rows + 2, decomposition.local_columns + 2),
          u_buffers(decomposition.local_rows, decomposition.local_columns),
          v_buffers(decomposition.local_rows, decomposition.local_columns) {

        initialize_fields(u, v, u_temp, v_temp, decomposition, parameters);

        MPI_Barrier(decomposition.comm);
    }

    GS_MPI_Blocking() = delete;
    GS_MPI_Blocking(const GS_MPI_Blocking&) = delete;
    GS_MPI_Blocking& operator=(const GS_MPI_Blocking&) = delete;

private:
    using View = Kokkos::View<real**, Kokkos::LayoutRight>;

    enum Direction {
        NW = 0,
        N,
        NE,
        W,
        E,
        SW,
        S,
        SE
    };

    static constexpr int n_directions = 8;

    // -------------------------------------------------------------------
    // MPI datatype corresponding to `real`.
    // -------------------------------------------------------------------
    static MPI_Datatype mpi_real_type() {
        if constexpr (std::is_same_v<real, float>) {
            return MPI_FLOAT;
        } else if constexpr (std::is_same_v<real, double>) {
            return MPI_DOUBLE;
        } else {
            static_assert(std::is_same_v<real, float> || std::is_same_v<real, double>,
                          "Unsupported real type for MPI communication.");
        }
    }

    struct CartesianDecomposition {
        MPI_Comm comm = MPI_COMM_NULL;

        int rank = MPI_PROC_NULL;
        int size = 0;

        int dims[2] = {0, 0};
        int coords[2] = {0, 0};

        int neighbors[n_directions] = {
            MPI_PROC_NULL, MPI_PROC_NULL,
            MPI_PROC_NULL, MPI_PROC_NULL,
            MPI_PROC_NULL, MPI_PROC_NULL,
            MPI_PROC_NULL, MPI_PROC_NULL
        };

        std::size_t global_rows = 0;
        std::size_t global_columns = 0;

        std::size_t local_rows = 0;
        std::size_t local_columns = 0;

        // `rows_`/`columns_` mean different things depending on
        // `strong_scaling`:
        //   - strong scaling: they are the GLOBAL problem size, fixed
        //     regardless of rank count, so more ranks means less work
        //     per rank. They must divide evenly across the Cartesian
        //     grid, otherwise ranks would end up with mismatched local
        //     sizes.
        //   - weak scaling: they are the LOCAL (per-rank) problem size,
        //     fixed regardless of rank count, so more ranks means a
        //     larger global problem. The global size is simply
        //     local * dims and always divides evenly by construction.
        explicit CartesianDecomposition(std::size_t rows_,
                                        std::size_t columns_,
                                        bool strong_scaling) {
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            int dims_tmp[2] = {0, 0};
            MPI_Dims_create(size, 2, dims_tmp);

            dims[0] = dims_tmp[0];  // rows / north-south
            dims[1] = dims_tmp[1];  // columns / west-east

            if (strong_scaling) {
                global_rows = rows_;
                global_columns = columns_;

                if (global_rows % static_cast<std::size_t>(dims[0]) != 0 ||
                    global_columns % static_cast<std::size_t>(dims[1]) != 0) {
                    throw std::runtime_error(
                        "Global field dimensions must be divisible by the "
                        "MPI Cartesian decomposition for strong scaling.");
                }

                local_rows = global_rows / static_cast<std::size_t>(dims[0]);
                local_columns = global_columns / static_cast<std::size_t>(dims[1]);
            } else {
                local_rows = rows_;
                local_columns = columns_;

                global_rows = local_rows * static_cast<std::size_t>(dims[0]);
                global_columns = local_columns * static_cast<std::size_t>(dims[1]);
            }

            if (local_rows == 0 || local_columns == 0) {
                throw std::runtime_error(
                    "Local field dimensions must be greater than zero.");
            }

            const int periods[2] = {0, 0};

            MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm);

            if (comm == MPI_COMM_NULL) {
                throw std::runtime_error("MPI_Cart_create returned MPI_COMM_NULL.");
            }

            MPI_Comm_rank(comm, &rank);
            MPI_Cart_coords(comm, rank, 2, coords);

            auto neighbor = [&](int dr, int dc) {
                const int c[2] = {coords[0] + dr, coords[1] + dc};

                // Outside the global domain -> physical boundary.
                if (c[0] < 0 || c[0] >= dims[0] || c[1] < 0 || c[1] >= dims[1]) {
                    return MPI_PROC_NULL;
                }

                int r = MPI_PROC_NULL;
                MPI_Cart_rank(comm, c, &r);
                return r;
            };

            neighbors[NW] = neighbor(-1, -1);
            neighbors[N] = neighbor(-1, 0);
            neighbors[NE] = neighbor(-1, +1);

            neighbors[W] = neighbor(0, -1);
            neighbors[E] = neighbor(0, +1);

            neighbors[SW] = neighbor(+1, -1);
            neighbors[S] = neighbor(+1, 0);
            neighbors[SE] = neighbor(+1, +1);
        }

        ~CartesianDecomposition() {
            if (comm != MPI_COMM_NULL) {
                MPI_Comm_free(&comm);
            }
        }

        CartesianDecomposition(const CartesianDecomposition&) = delete;
        CartesianDecomposition& operator=(const CartesianDecomposition&) = delete;
    };

    // -------------------------------------------------------------------
    // Communication buffers
    // -------------------------------------------------------------------

    struct CommBuffers {
        std::array<Kokkos::View<real*>, n_directions> send;
        std::array<Kokkos::View<real*>, n_directions> recv;

        CommBuffers(std::size_t rows, std::size_t columns) {
            send[NW] = Kokkos::View<real*>("send_NW", 1);
            recv[NW] = Kokkos::View<real*>("recv_NW", 1);

            send[N] = Kokkos::View<real*>("send_N", columns);
            recv[N] = Kokkos::View<real*>("recv_N", columns);

            send[NE] = Kokkos::View<real*>("send_NE", 1);
            recv[NE] = Kokkos::View<real*>("recv_NE", 1);

            send[W] = Kokkos::View<real*>("send_W", rows);
            recv[W] = Kokkos::View<real*>("recv_W", rows);

            send[E] = Kokkos::View<real*>("send_E", rows);
            recv[E] = Kokkos::View<real*>("recv_E", rows);

            send[SW] = Kokkos::View<real*>("send_SW", 1);
            recv[SW] = Kokkos::View<real*>("recv_SW", 1);

            send[S] = Kokkos::View<real*>("send_S", columns);
            recv[S] = Kokkos::View<real*>("recv_S", columns);

            send[SE] = Kokkos::View<real*>("send_SE", 1);
            recv[SE] = Kokkos::View<real*>("recv_SE", 1);
        }
    };

    // -------------------------------------------------------------------
    // Direction helper
    // -------------------------------------------------------------------

    static int opposite(int dir) {
        static constexpr int opposite_dir[n_directions] = {
            SE, S, SW,
            E, W,
            NE, N, NW
        };

        return opposite_dir[dir];
    }

    // -------------------------------------------------------------------
    // Pack one field into device-resident communication buffers.
    // -------------------------------------------------------------------

    static void pack(const View& field, CommBuffers& b, const CartesianDecomposition& d) {
        const std::size_t r = field.extent(0) - 2;
        const std::size_t c = field.extent(1) - 2;

        if (d.neighbors[N] != MPI_PROC_NULL) {
            Kokkos::parallel_for(
                "pack north halo", Kokkos::RangePolicy<std::size_t>(0, c),
                KOKKOS_LAMBDA(const std::size_t j) { b.send[N][j] = field(1, j + 1); });
        }

        if (d.neighbors[S] != MPI_PROC_NULL) {
            Kokkos::parallel_for(
                "pack south halo", Kokkos::RangePolicy<std::size_t>(0, c),
                KOKKOS_LAMBDA(const std::size_t j) { b.send[S][j] = field(r, j + 1); });
        }

        if (d.neighbors[W] != MPI_PROC_NULL) {
            Kokkos::parallel_for(
                "pack west halo", Kokkos::RangePolicy<std::size_t>(0, r),
                KOKKOS_LAMBDA(const std::size_t i) { b.send[W][i] = field(i + 1, 1); });
        }

        if (d.neighbors[E] != MPI_PROC_NULL) {
            Kokkos::parallel_for(
                "pack east halo", Kokkos::RangePolicy<std::size_t>(0, r),
                KOKKOS_LAMBDA(const std::size_t i) { b.send[E][i] = field(i + 1, c); });
        }

        if (d.neighbors[NW] != MPI_PROC_NULL) {
            Kokkos::parallel_for(
                "pack NW corner", Kokkos::RangePolicy<std::size_t>(0, 1),
                KOKKOS_LAMBDA(const std::size_t) { b.send[NW][0] = field(1, 1); });
        }
        if (d.neighbors[NE] != MPI_PROC_NULL) {
            Kokkos::parallel_for(
                "pack NE corner", Kokkos::RangePolicy<std::size_t>(0, 1),
                KOKKOS_LAMBDA(const std::size_t) { b.send[NE][0] = field(1, c); });
        }
        if (d.neighbors[SW] != MPI_PROC_NULL) {
            Kokkos::parallel_for(
                "pack SW corner", Kokkos::RangePolicy<std::size_t>(0, 1),
                KOKKOS_LAMBDA(const std::size_t) { b.send[SW][0] = field(r, 1); });
        }
        if (d.neighbors[SE] != MPI_PROC_NULL) {
            Kokkos::parallel_for(
                "pack SE corner", Kokkos::RangePolicy<std::size_t>(0, 1),
                KOKKOS_LAMBDA(const std::size_t) { b.send[SE][0] = field(r, c); });
        }
    }

    // -------------------------------------------------------------------
    // Unpack one field. For an internal boundary, use the received MPI
    // halo. Physical (global) boundaries are left untouched here — they
    // are set once in initialize_fields() and compute() never writes to
    // them, so there is nothing to unpack on that side.
    // -------------------------------------------------------------------

    static void unpack(View& field, const CommBuffers& b, const CartesianDecomposition& d) {
        const std::size_t r = field.extent(0) - 2;
        const std::size_t c = field.extent(1) - 2;

        if (d.neighbors[N] != MPI_PROC_NULL) {
            Kokkos::parallel_for(
                "unpack north halo", Kokkos::RangePolicy<std::size_t>(0, c),
                KOKKOS_LAMBDA(const std::size_t j) { field(0, j + 1) = b.recv[N][j]; });
        }

        if (d.neighbors[S] != MPI_PROC_NULL) {
            Kokkos::parallel_for(
                "unpack south halo", Kokkos::RangePolicy<std::size_t>(0, c),
                KOKKOS_LAMBDA(const std::size_t j) { field(r + 1, j + 1) = b.recv[S][j]; });
        }

        if (d.neighbors[W] != MPI_PROC_NULL) {
            Kokkos::parallel_for(
                "unpack west halo", Kokkos::RangePolicy<std::size_t>(0, r),
                KOKKOS_LAMBDA(const std::size_t i) { field(i + 1, 0) = b.recv[W][i]; });
        }

        if (d.neighbors[E] != MPI_PROC_NULL) {
            Kokkos::parallel_for(
                "unpack east halo", Kokkos::RangePolicy<std::size_t>(0, r),
                KOKKOS_LAMBDA(const std::size_t i) { field(i + 1, c + 1) = b.recv[E][i]; });
        }

        if (d.neighbors[NW] != MPI_PROC_NULL) {
            Kokkos::parallel_for(
                "unpack NW corner", Kokkos::RangePolicy<std::size_t>(0, 1),
                KOKKOS_LAMBDA(const std::size_t) { field(0, 0) = b.recv[NW][0]; });
        }

        if (d.neighbors[NE] != MPI_PROC_NULL) {
            Kokkos::parallel_for(
                "unpack NE corner", Kokkos::RangePolicy<std::size_t>(0, 1),
                KOKKOS_LAMBDA(const std::size_t) { field(0, c + 1) = b.recv[NE][0]; });
        }

        if (d.neighbors[SW] != MPI_PROC_NULL) {
            Kokkos::parallel_for(
                "unpack SW corner", Kokkos::RangePolicy<std::size_t>(0, 1),
                KOKKOS_LAMBDA(const std::size_t) { field(r + 1, 0) = b.recv[SW][0]; });
        }

        if (d.neighbors[SE] != MPI_PROC_NULL) {
            Kokkos::parallel_for(
                "unpack SE corner", Kokkos::RangePolicy<std::size_t>(0, 1),
                KOKKOS_LAMBDA(const std::size_t) { field(r + 1, c + 1) = b.recv[SE][0]; });
        }
    }

    // -------------------------------------------------------------------
    // Halo exchange: explicit pack, post all 8 neighbors, Waitall, unpack.
    // -------------------------------------------------------------------

    void exchange(View& field, CommBuffers& b, int tag_base) {
        pack(field, b, decomposition);

        std::array<MPI_Request, 16> requests{};
        int nreq = 0;

        for (int dir = 0; dir < n_directions; ++dir) {
            if (decomposition.neighbors[dir] == MPI_PROC_NULL) {
                continue;
            }
            MPI_Irecv(b.recv[dir].data(),
                      static_cast<int>(b.recv[dir].size()),
                      mpi_real_type(),
                      decomposition.neighbors[dir], tag_base + dir, decomposition.comm,
                      &requests[nreq++]);
        }

        for (int dir = 0; dir < n_directions; ++dir) {
            if (decomposition.neighbors[dir] == MPI_PROC_NULL) {
                continue;
            }
            MPI_Isend(b.send[dir].data(),
                      static_cast<int>(b.send[dir].size()),
                      mpi_real_type(),
                      decomposition.neighbors[dir], tag_base + opposite(dir), decomposition.comm,
                      &requests[nreq++]);
        }

        MPI_Waitall(nreq, requests.data(), MPI_STATUSES_IGNORE);
        unpack(field, b, decomposition);
    }

    // -------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------

    static void initialize_fields(
        const View& u,
        const View& v,
        const View& u_temp,
        const View& v_temp,
        const CartesianDecomposition& d,
        const Parameters& parameters) {

        Kokkos::deep_copy(u, 1);
        Kokkos::deep_copy(v, 0);

        Kokkos::deep_copy(u_temp, 1);
        Kokkos::deep_copy(v_temp, 0);

        const std::size_t local_rows = d.local_rows;
        const std::size_t local_columns = d.local_columns;

        // Global center square where the initial "drop" is placed.
        const std::size_t global_i_center = d.global_rows / 2;
        const std::size_t global_j_center = d.global_columns / 2;
        const std::size_t global_i_drop_first = global_i_center - 1;
        const std::size_t global_i_drop_last = global_i_center + 1;
        const std::size_t global_j_drop_first = global_j_center - 1;
        const std::size_t global_j_drop_last = global_j_center + 1;

        const std::size_t first_i = static_cast<std::size_t>(d.coords[0]) * local_rows;
        const std::size_t first_j = static_cast<std::size_t>(d.coords[1]) * local_columns;

        const std::size_t last_i = first_i + local_rows;
        const std::size_t last_j = first_j + local_columns;

        // Could be solved better with a single kernel, but this is easier.
        for (std::size_t gi = global_i_drop_first; gi < global_i_drop_last; ++gi) {
            for (std::size_t gj = global_j_drop_first; gj < global_j_drop_last; ++gj) {

                if (gi >= first_i && gi < last_i && gj >= first_j && gj < last_j) {

                    const std::size_t local_i = gi - first_i + 1;
                    const std::size_t local_j = gj - first_j + 1;

                    Kokkos::parallel_for(
                        "add drop", Kokkos::RangePolicy<>(0, 1),
                        KOKKOS_LAMBDA(const int) {
                            u(local_i, local_j) = 0;
                            v(local_i, local_j) = 1;
                        });
                }
            }
        }

        // Global physical boundary conditions for u and u_temp. v starts
        // at zero everywhere and compute() never writes the halo, so its
        // global physical boundary stays zero without further work.

        Kokkos::parallel_for(
            "initialize vertical boundary", Kokkos::RangePolicy<>(0, local_rows + 2),
            KOKKOS_LAMBDA(const std::size_t i) {
                u(i, 0) = 0;
                u_temp(i, 0) = 0;

                u(i, local_columns + 1) = 0;
                u_temp(i, local_columns + 1) = 0;
            });

        Kokkos::parallel_for(
            "initialize horizontal boundary", Kokkos::RangePolicy<>(0, local_columns + 2),
            KOKKOS_LAMBDA(const std::size_t j) {
                u(0, j) = 0;
                u_temp(0, j) = 0;

                u(local_rows + 1, j) = 0;
                u_temp(local_rows + 1, j) = 0;
            });
    }

    // -------------------------------------------------------------------
    // Compute
    // -------------------------------------------------------------------

    void compute() {
        const std::size_t nr = u.extent(0);
        const std::size_t nc = u.extent(1);

        const auto& coefficients = this->coeffs;

        Kokkos::parallel_for(
            "compute",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {nr - 1, nc - 1}),
            KOKKOS_LAMBDA(const int i, const int j) {
                gs_kernel(i, j, u, v, u_temp, v_temp, coefficients);
            });
    }

    // -------------------------------------------------------------------
    // benchmark<real> interface
    // -------------------------------------------------------------------

    void iteration() override {
        timer comm_timer;

        exchange(u, u_buffers, 100);
        exchange(v, v_buffers, 200);
        Kokkos::fence();

        this->communication_seconds += comm_timer.elapsed();

        compute();

        std::swap(u, u_temp);
        std::swap(v, v_temp);
    }

    // -------------------------------------------------------------------
    // Members. Declaration order matters: decomposition must exist
    // before it is used to size the fields and buffers below it.
    // -------------------------------------------------------------------

    CartesianDecomposition decomposition;

    View u;
    View v;
    View u_temp;
    View v_temp;

    CommBuffers u_buffers;
    CommBuffers v_buffers;
};