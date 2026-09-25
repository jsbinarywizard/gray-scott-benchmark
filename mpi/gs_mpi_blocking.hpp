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

    using View = Kokkos::View<real**, Kokkos::LayoutRight>;
    using StridedRow = Kokkos::View<real*, Kokkos::LayoutStride>;

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

        explicit CartesianDecomposition(std::size_t rows_,
                                        std::size_t columns_,
                                        bool strong_scaling) {
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            int dims_tmp[2] = {0, 0};
            MPI_Dims_create(size, 2, dims_tmp);

            dims[0] = dims_tmp[0];
            dims[1] = dims_tmp[1];

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
private:
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

    static int opposite(int dir) {
        static constexpr int opposite_dir[n_directions] = {
            SE, S, SW,
            E, W,
            NE, N, NW
        };

        return opposite_dir[dir];
    }

    // -------------------------------------------------------------------
    // Per-direction pack/unpack subviews. Each is a 1-D slice of the
    // interior edge/corner (pack) or halo ring (unpack); packing/unpacking
    // a direction is then a single deep_copy instead of a bespoke kernel.
    // -------------------------------------------------------------------

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
        }
        throw std::logic_error("pack_subview: invalid direction");
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
        }
        throw std::logic_error("halo_subview: invalid direction");
    }

    static void pack_direction(int dir, const View& field, CommBuffers& b) {
        const int r = field.extent(0) - 2;
        const int c = field.extent(1) - 2;
        Kokkos::deep_copy(b.send[dir], pack_subview(dir, field, r, c));
    }

    static void unpack_direction(int dir, View& field, const CommBuffers& b) {
        const int r = field.extent(0) - 2;
        const int c = field.extent(1) - 2;
        Kokkos::deep_copy(halo_subview(dir, field, r, c), b.recv[dir]);
    }

    static void pack(const View& field, CommBuffers& b, const CartesianDecomposition& d) {
        for (int dir = 0; dir < n_directions; ++dir) {
            if (d.neighbors[dir] != MPI_PROC_NULL) {
                pack_direction(dir, field, b);
            }
        }
        Kokkos::fence();
    }

    static void unpack(View& field, const CommBuffers& b, const CartesianDecomposition& d) {
        for (int dir = 0; dir < n_directions; ++dir) {
            if (d.neighbors[dir] != MPI_PROC_NULL) {
                unpack_direction(dir, field, b);
            }
        }
        Kokkos::fence();
    }

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

public:
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

    void compute() {
        const std::size_t nr = u.extent(0);
        const std::size_t nc = u.extent(1);

        const View u_      = this->u;
        const View v_      = this->v;
        View u_temp_ = this->u_temp;
        View v_temp_ = this->v_temp;
        auto coeffs_ = this->coeffs;

        Kokkos::parallel_for(
            "compute",
            Kokkos::MDRangePolicy<
                Kokkos::Rank<2, Kokkos::Iterate::Default, Kokkos::Iterate::Right>
            >({1, 1}, {nr - 1, nc - 1}),
            KOKKOS_LAMBDA(const int i, const int j) {
                gs_kernel(i, j, u_, v_, u_temp_, v_temp_, coeffs_);
            });
    }

private:
    void iteration() override {
        this->communication_timer.reset();

        exchange(u, u_buffers, 100);
        exchange(v, v_buffers, 200);
        Kokkos::fence();

        this->communication_seconds += this->communication_timer.elapsed();

        compute();

        Kokkos::fence();

        std::swap(u, u_temp);
        std::swap(v, v_temp);
    }

    CartesianDecomposition decomposition;

    View u;
    View v;
    View u_temp;
    View v_temp;

    CommBuffers u_buffers;
    CommBuffers v_buffers;
};