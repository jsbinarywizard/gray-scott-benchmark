#pragma once

#include <array>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>
#include <KokkosComm/KokkosComm.hpp>

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include <nccl.h>
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
#include <rccl.h>
#endif

#include "benchmark.hpp"
#include "../common/kernel.hpp"

// CCL selects the device communication space. With CCL=false, this benchmark
// explicitly uses MPI; with CCL=true, it uses KokkosComm's configured default
// CCL space, such as NCCL or RCCL. The halo exchange and iteration code below
// is independent of that choice.
template <typename real, bool CCL = false>
class GS_KC : public benchmark<real> {
protected:
    using Base = benchmark<real>;
    using View = typename Base::View;
    using ExecSpace = Kokkos::DefaultExecutionSpace;

    // KokkosComm makes its CCL backend the default communication space when
    // one is enabled. MPI must therefore be selected explicitly for the
    // non-CCL specialization.
    using CommSpace = std::conditional_t<CCL,
                                          KokkosComm::DefaultCommunicationSpace,
                                          KokkosComm::MpiSpace>;

    using CommHandle = KokkosComm::Communicator<CommSpace, ExecSpace>;
    using RequestType = KokkosComm::Request<CommSpace>;

    explicit GS_KC(const Parameters& parameters, const Decomposition& decomposition)
        : Base(parameters, decomposition),
          communicator(make_communicator(decomposition)) {
    }

    static CommHandle make_communicator(const Decomposition& decomposition) {
        if constexpr (!CCL) {
            return CommHandle::from_raw(decomposition.comm, ExecSpace{});
        } else {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
            static_assert(std::is_same_v<ExecSpace, Kokkos::Cuda>,
                          "NCCL communication requires Kokkos::DefaultExecutionSpace to be CUDA.");
            ncclUniqueId id{};
            if (decomposition.rank == 0) {
                ncclGetUniqueId(&id);
            }
            MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, decomposition.comm);

            ncclComm_t nccl_comm{};
            ncclCommInitRank(&nccl_comm, decomposition.size, id, decomposition.rank);
            return CommHandle::from_raw(nccl_comm, ExecSpace{});
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
            static_assert(std::is_same_v<ExecSpace, Kokkos::HIP>,
                          "RCCL communication requires Kokkos::DefaultExecutionSpace to be HIP.");
            rcclUniqueId id{};
            if (decomposition.rank == 0) {
                rcclGetUniqueId(&id);
            }
            MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, decomposition.comm);

            rcclComm_t rccl_comm{};
            rcclCommInitRank(&rccl_comm, decomposition.size, id, decomposition.rank);
            return CommHandle::from_raw(rccl_comm, ExecSpace{});
#else
            throw std::runtime_error("CCL communication was requested but no CCL backend is enabled.");
#endif
        }
    }

    CommHandle communicator;
};

template <typename real, bool CCL = false>
class GS_KC_Blocking : public GS_KC<real, CCL> {
    // GS_KC<real, CCL> is a dependent base: its members are invisible to
    // unqualified lookup, so pull in what's needed explicitly.
    using Base = GS_KC<real, CCL>;
    using View = typename Base::View;
    using RequestType = typename Base::RequestType;

public:
    explicit GS_KC_Blocking(const Parameters& parameters, const Decomposition& decomposition)
        : Base(parameters, decomposition) {}

private:
    void exchange(View& b) {
        std::vector<RequestType> recv_requests{};
        std::vector<RequestType> send_requests{};
        recv_requests.reserve(n_directions);
        send_requests.reserve(n_directions);

        for (int dir = 0; dir < n_directions; ++dir) {
            if (this->decomposition.neighbors[dir] == MPI_PROC_NULL) {
                continue;
            }
            auto recv_subview = this->halo_subview(dir, b, this->local_rows, this->local_columns);
            recv_requests.push_back(
                KokkosComm::recv(
                    this->communicator,
                    recv_subview,
                    this->decomposition.neighbors[dir]));
        }

        for (int dir = 0; dir < n_directions; ++dir) {
            if (this->decomposition.neighbors[dir] == MPI_PROC_NULL) {
                continue;
            }
            auto send_subview = this->pack_subview(dir, b, this->local_rows, this->local_columns);
            send_requests.push_back(
                KokkosComm::send(
                    this->communicator,
                    send_subview,
                    this->decomposition.neighbors[dir]));
        }

        KokkosComm::wait_all(recv_requests);
        KokkosComm::wait_all(send_requests);
    }

    void iteration() override {
        timer comm_timer;
        exchange(this->u);
        exchange(this->v);
        this->communication_time += comm_timer.elapsed();
        gs_compute(this->u, this->v, this->u_temp, this->v_temp, this->coeffs);
        Kokkos::fence();
        std::swap(this->u, this->u_temp);
        std::swap(this->v, this->v_temp);
    }
};

template <typename real, bool CCL = false>
class GS_KC_NonBlocking : public GS_KC<real, CCL> {
    using Base = GS_KC<real, CCL>;
    using View = typename Base::View;
    using RequestType = typename Base::RequestType;

    struct ExchangeHandle {
        std::vector<RequestType> requests;
    };

    ExchangeHandle begin_exchange(View& b) {
        ExchangeHandle h;
        h.requests.reserve(2 * n_directions);

        // Post all receives first, non-blocking.
        for (int dir = 0; dir < n_directions; ++dir) {
            if (this->decomposition.neighbors[dir] == MPI_PROC_NULL) {
                continue;
            }
            auto recv_subview = this->halo_subview(dir, b, this->local_rows, this->local_columns);
            h.requests.push_back(
                KokkosComm::recv(
                    this->communicator,
                    recv_subview,
                    this->decomposition.neighbors[dir]));
        }

        // Post all sends, also non-blocking -- do not wait on anything here.
        for (int dir = 0; dir < n_directions; ++dir) {
            if (this->decomposition.neighbors[dir] == MPI_PROC_NULL) {
                continue;
            }
            auto send_subview = this->pack_subview(dir, b, this->local_rows, this->local_columns);
            h.requests.push_back(
                KokkosComm::send(
                    this->communicator,
                    send_subview,
                    this->decomposition.neighbors[dir]));
        }

        return h;
    }

    static void end_exchange(ExchangeHandle& h) {
        KokkosComm::wait_all(h.requests);
    }

public:
    explicit GS_KC_NonBlocking(const Parameters& parameters, const Decomposition& decomposition)
        : Base(parameters, decomposition) {}

private:
    void iteration() override {
        ExchangeHandle u_handle = begin_exchange(this->u);
        ExchangeHandle v_handle = begin_exchange(this->v);

        // Overlap: interior compute runs while the halo is in flight.
        gs_compute_interior(this->u, this->v, this->u_temp, this->v_temp, this->coeffs);

        // Block until the halo has actually arrived. Because recv[dir]
        // writes directly into the field's own halo cells, there is
        // nothing left to unpack once this returns.
        end_exchange(u_handle);
        end_exchange(v_handle);

        // Finish the cells that depend on the freshly-received halo.
        gs_compute_ring(this->u, this->v, this->u_temp, this->v_temp, this->coeffs);

        Kokkos::fence();

        std::swap(this->u, this->u_temp);
        std::swap(this->v, this->v_temp);
    }
};