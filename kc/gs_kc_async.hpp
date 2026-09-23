#pragma once

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <KokkosComm/KokkosComm.hpp>

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include <nccl.h>
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../common/benchmark.hpp"
#include "../common/coefficients.hpp"
#include "../common/kernel.hpp"
#include "../common/parameters.hpp"
#include "../common/timer.hpp"

template <typename real>
class GS_KC_NonBlocking : public benchmark<real> {

public:
    explicit GS_KC_NonBlocking(const Parameters& parameters)
        : decomposition(parameters.rows, parameters.columns, parameters.strong_scaling),
          benchmark<real>(parameters),
          u("u", decomposition.local_rows + 2, decomposition.local_columns + 2),
          v("v", decomposition.local_rows + 2, decomposition.local_columns + 2),
          u_temp("u_temp", decomposition.local_rows + 2, decomposition.local_columns + 2),
          v_temp("v_temp", decomposition.local_rows + 2, decomposition.local_columns + 2),
          u_buffers(u),
          v_buffers(v) {

        initialize_fields(u, v, u_temp, v_temp, decomposition, parameters);

        MPI_Barrier(decomposition.cart_comm);
    }

    GS_KC_NonBlocking() = delete;
    GS_KC_NonBlocking(const GS_KC_NonBlocking&) = delete;
    GS_KC_NonBlocking& operator=(const GS_KC_NonBlocking&) = delete;

private:
    using View = Kokkos::View<real**, Kokkos::LayoutRight>;
    using HaloView = Kokkos::View<real*, Kokkos::LayoutStride>;

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
    static constexpr int KC_PROC_NULL = -1;

#if defined(KOKKOSCOMM_ENABLE_NCCL)
    using CommSpace = KokkosComm::Experimental::NcclSpace;
    using ExecSpace = Kokkos::Cuda;
    static_assert(
        std::is_same_v<ExecSpace, Kokkos::DefaultExecutionSpace>,
        "Kokkos::DefaultExecutionSpace must be Kokkos::Cuda when "
        "KOKKOSCOMM_ENABLE_NCCL is defined.");
#else
    using CommSpace = KokkosComm::MpiSpace;
    using ExecSpace = Kokkos::DefaultExecutionSpace;
#endif

    using CommHandle = KokkosComm::Communicator<CommSpace, ExecSpace>;

    // -------------------------------------------------------------------
    // Cartesian decomposition. Identical to GS_KokkosComm's: MPI is used
    // purely as a topology helper (Cartesian grid, neighbor discovery);
    // the KokkosComm communicator built on top is what actually moves
    // data, whether that is MpiSpace or a bootstrapped NcclSpace.
    // -------------------------------------------------------------------

    struct Communicator {
        std::optional<CommHandle> handle;

        Communicator() = default;

        explicit Communicator(CommHandle h)
            : handle(std::move(h)) {}

        CommHandle* operator->() {
            return &*handle;
        }

        const CommHandle* operator->() const {
            return &*handle;
        }

        CommHandle& operator*() {
            return *handle;
        }

        const CommHandle& operator*() const {
            return *handle;
        }
    };

    struct CartesianDecomposition {
        MPI_Comm cart_comm = MPI_COMM_NULL;

#if defined(KOKKOSCOMM_ENABLE_NCCL)
        ncclComm_t nccl_comm{};
#endif

        Communicator comm;

        int rank = MPI_PROC_NULL;
        int size = 0;

        int dims[2] = {0, 0};
        int coords[2] = {0, 0};

        int neighbors[n_directions] = {
            KC_PROC_NULL, KC_PROC_NULL,
            KC_PROC_NULL, KC_PROC_NULL,
            KC_PROC_NULL, KC_PROC_NULL,
            KC_PROC_NULL, KC_PROC_NULL
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

            MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

            if (cart_comm == MPI_COMM_NULL) {
                throw std::runtime_error("MPI_Cart_create returned MPI_COMM_NULL.");
            }

            MPI_Comm_rank(cart_comm, &rank);
            MPI_Cart_coords(cart_comm, rank, 2, coords);

            auto neighbor = [&](int dr, int dc) {
                const int c[2] = {coords[0] + dr, coords[1] + dc};

                // Outside the global domain -> physical boundary.
                if (c[0] < 0 || c[0] >= dims[0] || c[1] < 0 || c[1] >= dims[1]) {
                    return KC_PROC_NULL;
                }

                int r = MPI_PROC_NULL;
                MPI_Cart_rank(cart_comm, c, &r);
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

#if defined(KOKKOSCOMM_ENABLE_NCCL)
            ncclUniqueId id;
            if (rank == 0) {
                ncclGetUniqueId(&id);
            }
            MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, cart_comm);
            ncclCommInitRank(&nccl_comm, size, id, rank);

            comm = Communicator(CommHandle::from_raw(nccl_comm, ExecSpace()));
#else
            comm = Communicator(CommHandle::from_raw(cart_comm, ExecSpace()));
#endif
        }

        ~CartesianDecomposition() {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
            ncclCommDestroy(nccl_comm);
#endif
            if (cart_comm != MPI_COMM_NULL) {
                MPI_Comm_free(&cart_comm);
            }
        }

        CartesianDecomposition(const CartesianDecomposition&) = delete;
        CartesianDecomposition& operator=(const CartesianDecomposition&) = delete;
    };

    // -------------------------------------------------------------------
    // Communication buffers: recv[dir] aliases the field's own halo cells
    // directly, so there is no separate unpack step -- once a receive
    // completes, the halo is already in place. Rebuilt every iteration
    // after u/v are swapped with u_temp/v_temp, since a subview only
    // stays valid for the View it was taken from.
    // -------------------------------------------------------------------

    struct CommBuffers {
        std::array<HaloView, n_directions> send;
        std::array<HaloView, n_directions> recv;

        CommBuffers() = default;

        explicit CommBuffers(const View& field) {
            const std::size_t nr = field.extent(0);
            const std::size_t nc = field.extent(1);

            using Range = Kokkos::pair<std::size_t, std::size_t>;

            send[NW] = Kokkos::subview(field, Range(1, 2), 1);
            recv[NW] = Kokkos::subview(field, Range(0, 1), 0);

            send[NE] = Kokkos::subview(field, Range(1, 2), nc - 2);
            recv[NE] = Kokkos::subview(field, Range(0, 1), nc - 1);

            send[SW] = Kokkos::subview(field, Range(nr - 2, nr - 1), 1);
            recv[SW] = Kokkos::subview(field, Range(nr - 1, nr), 0);

            send[SE] = Kokkos::subview(field, Range(nr - 2, nr - 1), nc - 2);
            recv[SE] = Kokkos::subview(field, Range(nr - 1, nr), nc - 1);

            send[N] = Kokkos::subview(field, 1, Kokkos::ALL());
            recv[N] = Kokkos::subview(field, 0, Kokkos::ALL());

            send[S] = Kokkos::subview(field, nr - 2, Kokkos::ALL());
            recv[S] = Kokkos::subview(field, nr - 1, Kokkos::ALL());

            send[W] = Kokkos::subview(field, Kokkos::ALL(), 1);
            recv[W] = Kokkos::subview(field, Kokkos::ALL(), 0);

            send[E] = Kokkos::subview(field, Kokkos::ALL(), nc - 2);
            recv[E] = Kokkos::subview(field, Kokkos::ALL(), nc - 1);
        }
    };

    // -------------------------------------------------------------------
    // Non-blocking halo exchange, split into begin/end.
    //
    // begin_exchange() only posts the recv()/send() operations and
    // returns immediately, without waiting on anything, so the caller can
    // overlap compute_interior() with the in-flight transfers.
    // end_exchange() is where wait_all() actually blocks.
    // -------------------------------------------------------------------

    struct ExchangeHandle {
        std::vector<KokkosComm::Request<>> requests;
    };

    ExchangeHandle begin_exchange(CommBuffers& b) {
        ExchangeHandle h;
        h.requests.reserve(2 * n_directions);

        // Post all receives first, non-blocking.
        for (int dir = 0; dir < n_directions; ++dir) {
            if (decomposition.neighbors[dir] == KC_PROC_NULL) {
                continue;
            }
            h.requests.push_back(
                KokkosComm::recv(*decomposition.comm, b.recv[dir], decomposition.neighbors[dir]));
        }

        // Post all sends, also non-blocking -- do not wait on anything here.
        for (int dir = 0; dir < n_directions; ++dir) {
            if (decomposition.neighbors[dir] == KC_PROC_NULL) {
                continue;
            }
            h.requests.push_back(
                KokkosComm::send(*decomposition.comm, b.send[dir], decomposition.neighbors[dir]));
        }

        return h;
    }

    static void end_exchange(ExchangeHandle& h) {
        KokkosComm::wait_all(h.requests);
    }

    // -------------------------------------------------------------------
    // Initialization. Identical to GS_KokkosComm's / GS_MPI_Blocking's.
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

        // v starts at zero everywhere and compute() never writes the
        // halo, so v's global physical boundary stays zero without
        // further work.
    }

    // -------------------------------------------------------------------
    // Compute, split into interior (halo-independent) and ring
    // (halo-dependent), same split as GS_MPI_NonBlocking, sharing the
    // same gs_kernel/coefficients used by GS_KokkosComm so all four
    // variants compute bit-identical updates.
    // -------------------------------------------------------------------

    // Cells whose 3x3 stencil does NOT touch the halo, i.e. local index
    // range [2, local_rows-1] x [2, local_columns-1].
    void compute_interior() {
        const std::size_t local_rows = decomposition.local_rows;
        const std::size_t local_columns = decomposition.local_columns;

        if (local_rows < 3 || local_columns < 3) return;  // no safe interior

        const auto& coefficients = this->coeffs;

        Kokkos::parallel_for(
            "compute interior",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {2, 2},
                {static_cast<std::int64_t>(local_rows), static_cast<std::int64_t>(local_columns)}),
            KOKKOS_LAMBDA(const int i, const int j) {
                gs_kernel(i, j, u, v, u_temp, v_temp, coefficients);
            });
    }

    // The one-cell-wide ring of interior cells adjacent to the halo, only
    // safe to run once end_exchange() has returned. Corners are covered
    // exactly once via the top/bottom rows; left/right columns exclude
    // those two rows.
    void compute_ring() {
        const std::size_t local_rows = decomposition.local_rows;
        const std::size_t local_columns = decomposition.local_columns;

        const auto& coefficients = this->coeffs;

        const int nr = static_cast<int>(local_rows);
        const int nc = static_cast<int>(local_columns);

        // Top row (i = 1) and bottom row (i = nr), full width incl. corners.
        Kokkos::parallel_for(
            "compute ring top/bottom",
            Kokkos::RangePolicy<int>(1, nc + 1),
            KOKKOS_LAMBDA(const int j) {
                gs_kernel(1, j, u, v, u_temp, v_temp, coefficients);
                gs_kernel(nr, j, u, v, u_temp, v_temp, coefficients);
            });

        // Left column (j = 1) and right column (j = nc), excluding the
        // corners already done above, i.e. i in [2, nr-1]. Only
        // meaningful if nr > 2.
        if (nr > 2) {
            Kokkos::parallel_for(
                "compute ring left/right",
                Kokkos::RangePolicy<int>(2, nr),
                KOKKOS_LAMBDA(const int i) {
                    gs_kernel(i, 1, u, v, u_temp, v_temp, coefficients);
                    gs_kernel(i, nc, u, v, u_temp, v_temp, coefficients);
                });
        }
    }

    // -------------------------------------------------------------------
    // benchmark<real> interface
    // -------------------------------------------------------------------

    void iteration() override {
        timer post_timer;

        // Kick off both halo exchanges without blocking.
        ExchangeHandle u_handle = begin_exchange(u_buffers);
        ExchangeHandle v_handle = begin_exchange(v_buffers);

        this->communication_seconds += post_timer.elapsed();

        // Overlap: interior compute runs while the halo is in flight.
        compute_interior();

        timer wait_timer;

        // Block until the halo has actually arrived. Because recv[dir]
        // aliases the field's own halo cells, there is nothing left to
        // unpack once this returns.
        end_exchange(u_handle);
        end_exchange(v_handle);

        this->communication_seconds += wait_timer.elapsed();

        // Finish the cells that depend on the freshly-received halo.
        compute_ring();

        std::swap(u, u_temp);
        std::swap(v, v_temp);

        // u_buffers/v_buffers are subviews of the pre-swap u/v, so they
        // must be rebuilt against the (now current) u/v before the next
        // begin_exchange().
        u_buffers = CommBuffers(u);
        v_buffers = CommBuffers(v);
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