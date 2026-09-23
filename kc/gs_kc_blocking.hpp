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

#include "../common/benchmark.hpp"
#include "../common/coefficients.hpp"
#include "../common/kernel.hpp"
#include "../common/parameters.hpp"
#include "../common/timer.hpp"


template <typename real>
class GS_KC_Blocking : public benchmark<real> {

public:
    explicit GS_KC_Blocking(const Parameters& parameters)
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

    GS_KC_Blocking() = delete;
    GS_KC_Blocking(const GS_KC_Blocking&) = delete;
    GS_KC_Blocking& operator=(const GS_KC_Blocking&) = delete;

private:
    using View = Kokkos::View<real**, Kokkos::LayoutRight>;

    // Halo cells are exchanged as subviews of u/v/u_temp/v_temp directly
    // (edges are contiguous rows/strided columns, corners are 1-element
    // strided windows), so the buffer type must accept a general stride.
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
    // Cartesian decomposition. Same shape and same strong/weak scaling
    // semantics as GS_MPI_Blocking::CartesianDecomposition; the extra
    // piece here is building (and, for NCCL, bootstrapping) the
    // KokkosComm communicator on top of the Cartesian topology.
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

        // Same meaning as in GS_MPI_Blocking: under strong scaling
        // `rows_`/`columns_` are the fixed GLOBAL problem size (must
        // divide evenly across the Cartesian grid); under weak scaling
        // they are the fixed LOCAL (per-rank) problem size, and the
        // global size grows with the rank count.
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

            // Build the KokkosComm communicator on top of the Cartesian
            // topology just established. NCCL ranks are bootstrapped to
            // match the MPI Cartesian ranks 1:1, so the neighbor ranks
            // above remain valid peer arguments to KokkosComm::send/recv
            // regardless of which backend is active.
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
    // Communication buffers: subviews of the halo cells of a single field
    // instance. Rebuilt every iteration after u/v are swapped with
    // u_temp/v_temp, since a subview only stays valid for the View it was
    // taken from. Corners are collapsed to 1x1 windows via a length-1
    // Kokkos::pair range on one axis so they stay rank-1.
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

            send[N] = Kokkos::subview(field, 1, Range(1, nc - 1));
            recv[N] = Kokkos::subview(field, 0, Range(1, nc - 1));

            send[S] = Kokkos::subview(field, nr - 2, Range(1, nc - 1));
            recv[S] = Kokkos::subview(field, nr - 1, Range(1, nc - 1));

            send[W] = Kokkos::subview(field, Range(1, nr - 1), 1);
            recv[W] = Kokkos::subview(field, Range(1, nr - 1), 0);

            send[E] = Kokkos::subview(field, Range(1, nr - 1), nc - 2);
            recv[E] = Kokkos::subview(field, Range(1, nr - 1), nc - 1);
        }
    };

    // -------------------------------------------------------------------
    // Halo exchange, written entirely against KokkosComm's generic
    // send()/recv()/wait_all() API. There is no explicit pack/unpack and
    // no MPI_Datatype/MPI_Request bookkeeping: CommBuffers already holds
    // the strided halo subviews to send from / receive into directly, and
    // that stays identical whether CommSpace is MpiSpace or NcclSpace.
    // -------------------------------------------------------------------

    void exchange(CommBuffers& b) {
        std::vector<KokkosComm::Request<>> recv_requests{};
        int n_recv = 0;

        for (int dir = 0; dir < n_directions; ++dir) {
            if (decomposition.neighbors[dir] == KC_PROC_NULL) {
                continue;
            }
            recv_requests.push_back(KokkosComm::recv(*decomposition.comm, b.recv[dir], decomposition.neighbors[dir]));
        }

        for (int dir = 0; dir < n_directions; ++dir) {
            if (decomposition.neighbors[dir] == KC_PROC_NULL) {
                continue;
            }
            KokkosComm::send(*decomposition.comm, b.send[dir], decomposition.neighbors[dir]);
        }

        KokkosComm::wait_all(recv_requests);
    }

    // -------------------------------------------------------------------
    // Initialization. Identical to GS_MPI_Blocking::initialize_fields:
    // global physical boundary set once here for u/u_temp; v starts at
    // zero everywhere and compute() never writes the halo, so v's global
    // physical boundary stays zero without further work.
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
    }

    // -------------------------------------------------------------------
    // Compute. Identical to GS_MPI_Blocking::compute -- same MDRangePolicy
    // over interior cells, same shared gs_kernel + coefficients from the
    // common benchmark infrastructure.
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

        exchange(u_buffers);
        exchange(v_buffers);
        Kokkos::fence();

        this->communication_seconds += comm_timer.elapsed();

        compute();

        std::swap(u, u_temp);
        std::swap(v, v_temp);

        // u_buffers/v_buffers are subviews of the pre-swap u/v, so they
        // must be rebuilt against the (now current) u/v before the next
        // exchange().
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