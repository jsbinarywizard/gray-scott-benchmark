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
class GS_MPI_NonBlocking : public benchmark<real> {

public:
    explicit GS_MPI_NonBlocking(const Parameters& parameters)
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

    GS_MPI_NonBlocking() = delete;
    GS_MPI_NonBlocking(const GS_MPI_NonBlocking&) = delete;
    GS_MPI_NonBlocking& operator=(const GS_MPI_NonBlocking&) = delete;

    using View = Kokkos::View<real**, Kokkos::LayoutRight>;

    // Every direction is a 1-D slice of the field: contiguous for N/S (fixed
    // row, ranged column; LayoutRight makes a row contiguous), strided for
    // W/E (fixed column, ranged row), and size-1 for the four corners either
    // way. A single LayoutStride view type covers all cases.
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

    // -------------------------------------------------------------------
    // Cartesian decomposition. Identical to GS_MPI_Blocking's.
    // -------------------------------------------------------------------

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

        int global_rows = 0;
        int global_columns = 0;

        int local_rows = 0;
        int local_columns = 0;

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
        explicit CartesianDecomposition(int rows_,
                                        int columns_,
                                        bool strong_scaling) {
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            int dims_tmp[2] = {0, 0};
            MPI_Dims_create(size, 2, dims_tmp);

            dims[0] = dims_tmp[0];  // rows / north-south
            dims[1] = dims_tmp[1];  // columns / west-east

            if (strong_scaling) {
                global_rows = rows_;
                global_columns = columns_;

                if (global_rows % static_cast<int>(dims[0]) != 0 ||
                    global_columns % static_cast<int>(dims[1]) != 0) {
                    throw std::runtime_error(
                        "Global field dimensions must be divisible by the "
                        "MPI Cartesian decomposition for strong scaling.");
                }

                local_rows = global_rows / static_cast<int>(dims[0]);
                local_columns = global_columns / static_cast<int>(dims[1]);
            } else {
                local_rows = rows_;
                local_columns = columns_;

                global_rows = local_rows * static_cast<int>(dims[0]);
                global_columns = local_columns * static_cast<int>(dims[1]);
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
    // Communication buffers. Identical to GS_MPI_Blocking's: packed,
    // contiguous 1-D device buffers, one send/recv pair per direction.
    // -------------------------------------------------------------------

    struct CommBuffers {
        std::array<Kokkos::View<real*>, n_directions> send;
        std::array<Kokkos::View<real*>, n_directions> recv;

        CommBuffers(int rows, int columns) {
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

    // -------------------------------------------------------------------
    // Non-blocking, pipelined halo exchange.
    //
    // begin_exchange() posts all receives, then packs and sends one
    // direction at a time. end_recvs() unpacks each direction as soon as
    // its receive completes. end_sends() is deferred until the caller has
    // done other useful work, since the sent buffers aren't touched again
    // until the next begin_exchange() overwrites them.
    // -------------------------------------------------------------------

    struct ExchangeHandle {
        std::array<MPI_Request, n_directions> recv_requests{};
        std::array<int, n_directions> recv_dir{};
        int n_recv = 0;

        std::array<MPI_Request, n_directions> send_requests{};
        int n_send = 0;
    };

    ExchangeHandle begin_exchange(const View& field, CommBuffers& b, int tag_base) {
        ExchangeHandle h;

        // Post all receives first so incoming messages always have
        // somewhere to land as soon as they arrive.
        for (int dir = 0; dir < n_directions; ++dir) {
            if (decomposition.neighbors[dir] == MPI_PROC_NULL) continue;

            h.recv_dir[h.n_recv] = dir;
            MPI_Irecv(b.recv[dir].data(),
                      static_cast<int>(b.recv[dir].size()),
                      mpi_real_type(),
                      decomposition.neighbors[dir], tag_base + dir, decomposition.comm,
                      &h.recv_requests[h.n_recv++]);
        }

        // Pack and send one direction at a time, so the Isend for a
        // direction fires as soon as that direction is packed rather than
        // after every direction has been packed.
        for (int dir = 0; dir < n_directions; ++dir) {
            if (decomposition.neighbors[dir] == MPI_PROC_NULL) continue;

            pack_direction(dir, field, b);

            // The Isend below reads b.send[dir] directly, so the copy
            // into it must have actually completed on the device first.
            Kokkos::fence("pack_direction fence");

            MPI_Isend(b.send[dir].data(),
                      static_cast<int>(b.send[dir].size()),
                      mpi_real_type(),
                      decomposition.neighbors[dir], tag_base + opposite(dir), decomposition.comm,
                      &h.send_requests[h.n_send++]);
        }

        return h;
    }

    // Unpacks each direction's halo as soon as its receive completes.
    static void end_recvs(View& field, const CommBuffers& b, ExchangeHandle& h) {
        for (int done = 0; done < h.n_recv; ++done) {
            int index = MPI_UNDEFINED;
            MPI_Waitany(h.n_recv, h.recv_requests.data(), &index, MPI_STATUS_IGNORE);

            unpack_direction(h.recv_dir[index], field, b);
        }
    }

    // Waits for this exchange's sends to complete. Safe to call late
    // (e.g. after this iteration's compute is done), since the sent data
    // isn't touched again until the next iteration's pack overwrites it.
    static void end_sends(ExchangeHandle& h) {
        MPI_Waitall(h.n_send, h.send_requests.data(), MPI_STATUSES_IGNORE);
    }

    // -------------------------------------------------------------------
    // Initialization. Identical to GS_MPI_Blocking's.
    // -------------------------------------------------------------------
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

        const int local_rows = d.local_rows;
        const int local_columns = d.local_columns;

        // Global center square where the initial "drop" is placed.
        const int global_i_center = d.global_rows / 2;
        const int global_j_center = d.global_columns / 2;
        const int global_i_drop_first = global_i_center - 1;
        const int global_i_drop_last = global_i_center + 1;
        const int global_j_drop_first = global_j_center - 1;
        const int global_j_drop_last = global_j_center + 1;

        const int first_i = static_cast<int>(d.coords[0]) * local_rows;
        const int first_j = static_cast<int>(d.coords[1]) * local_columns;

        const int last_i = first_i + local_rows;
        const int last_j = first_j + local_columns;

        // Could be solved better with a single kernel, but this is easier.
        for (int gi = global_i_drop_first; gi < global_i_drop_last; ++gi) {
            for (int gj = global_j_drop_first; gj < global_j_drop_last; ++gj) {

                if (gi >= first_i && gi < last_i && gj >= first_j && gj < last_j) {

                    const int local_i = gi - first_i + 1;
                    const int local_j = gj - first_j + 1;

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
            KOKKOS_LAMBDA(const int i) {
                u(i, 0) = 0;
                u_temp(i, 0) = 0;

                u(i, local_columns + 1) = 0;
                u_temp(i, local_columns + 1) = 0;
            });

        Kokkos::parallel_for(
            "initialize horizontal boundary", Kokkos::RangePolicy<>(0, local_columns + 2),
            KOKKOS_LAMBDA(const int j) {
                u(0, j) = 0;
                u_temp(0, j) = 0;

                u(local_rows + 1, j) = 0;
                u_temp(local_rows + 1, j) = 0;
            });
    }

    // -------------------------------------------------------------------
    // Compute, split into interior (halo-independent) and ring
    // (halo-dependent) so the interior can run while the halo exchange is
    // still in flight. Both share the same gs_kernel used by
    // GS_MPI_Blocking, so the two implementations compute bit-identical
    // updates.
    // -------------------------------------------------------------------

    // Cells whose 3x3 stencil does NOT touch the halo, i.e. local index
    // range [2, local_rows-1] x [2, local_columns-1].
    void compute_interior() {
        const int local_rows = decomposition.local_rows;
        const int local_columns = decomposition.local_columns;

        if (local_rows < 3 || local_columns < 3) return;  // no safe interior

        const auto coefficients = this->coeffs;
        const auto u_ = u;
        const auto v_ = v;
        auto u_temp_ = u_temp;
        auto v_temp_ = v_temp;

        Kokkos::parallel_for(
            "compute interior",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {2, 2},
                {static_cast<int>(local_rows - 1), static_cast<int>(local_columns - 1)}),
            KOKKOS_LAMBDA(const int i, const int j) {
                gs_kernel(i, j, u_, v_, u_temp_, v_temp_, coefficients);
            });
    }

    // The one-cell-wide ring of interior cells adjacent to the halo,
    // computed after the halo has actually arrived. Corners are covered
    // exactly once via the top/bottom rows; left/right columns exclude
    // those two rows.
    void compute_ring() {
        const int local_rows = decomposition.local_rows;
        const int local_columns = decomposition.local_columns;

        const auto coefficients = this->coeffs;

        const int nr = static_cast<int>(local_rows);
        const int nc = static_cast<int>(local_columns);

        const auto u_ = u;
        const auto v_ = v;
        const auto u_temp_ = u_temp;
        const auto v_temp_ = v_temp;

        // Top row (i = 1) and bottom row (i = nr), full width incl. corners.
        Kokkos::parallel_for(
            "compute ring top/bottom",
            Kokkos::RangePolicy<int>(1, nc + 1),
            KOKKOS_LAMBDA(const int j) {
                gs_kernel(1, j, u_, v_, u_temp_, v_temp_, coefficients);
                gs_kernel(nr, j, u_, v_, u_temp_, v_temp_, coefficients);
            });

        // Left column (j = 1) and right column (j = nc), excluding the
        // corners already done above, i.e. i in [2, nr-1]. Only
        // meaningful if nr > 2.
        if (nr > 2) {
            Kokkos::parallel_for(
                "compute ring left/right",
                Kokkos::RangePolicy<int>(2, nr),
                KOKKOS_LAMBDA(const int i) {
                    gs_kernel(i, 1, u_, v_, u_temp_, v_temp_, coefficients);
                    gs_kernel(i, nc, u_, v_, u_temp_, v_temp_, coefficients);
                });
        }
    }

    // -------------------------------------------------------------------
    // benchmark<real> interface
    // -------------------------------------------------------------------
private:
    void iteration() override {
        this->communication_timer.reset();

        // Post receives, then pack+send one direction at a time.
        ExchangeHandle u_handle = begin_exchange(u, u_buffers, 100);
        ExchangeHandle v_handle = begin_exchange(v, v_buffers, 200);

        // Overlap: interior compute runs while the halo is in flight.
        compute_interior();

        end_recvs(u, u_buffers, u_handle);
        end_recvs(v, v_buffers, v_handle);
        this->communication_seconds += this->communication_timer.elapsed();

        // Finish the cells that depend on the freshly-received halo.
        compute_ring();

        Kokkos::fence();

        // The sends are no longer needed by this rank once handed to MPI,
        // so waiting for them is deferred until after compute_ring
        // instead of blocking compute on them.
        end_sends(u_handle);
        end_sends(v_handle);

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