#pragma once

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include <KokkosComm/KokkosComm.hpp>

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include <nccl.h>
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
#include <rccl.h>
#endif

#include "benchmark.hpp"

template <typename real, bool CCL = false>
class GS_KC {
protected:
    using BlockingBuffers = typename blocking_benchmark<real>::Buffers;
    using Buffers = typename NonBlockingBenchmark<real>::Buffers;
    using View = typename benchmark<real>::View;
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using CommSpace = std::conditional_t<CCL,
                                          KokkosComm::DefaultCommunicationSpace,
                                          KokkosComm::MpiSpace>;
    using CommHandle = KokkosComm::Communicator<CommSpace, ExecSpace>;
    using RequestType = KokkosComm::Request<CommSpace>;

    explicit GS_KC(const Decomposition& decomposition)
        : decomposition(decomposition), communicator(make_communicator(decomposition)) {}

    static CommHandle make_communicator(const Decomposition& decomposition) {
        if constexpr (!CCL) {
            return CommHandle::from_raw(decomposition.comm, ExecSpace{});
        } else {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
            static_assert(std::is_same_v<ExecSpace, Kokkos::Cuda>,
                          "NCCL communication requires Kokkos::DefaultExecutionSpace to be CUDA.");
            ncclUniqueId id{};
            if (decomposition.rank == 0) ncclGetUniqueId(&id);
            MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, decomposition.comm);
            ncclComm_t nccl_comm{};
            ncclCommInitRank(&nccl_comm, decomposition.size, id, decomposition.rank);
            return CommHandle::from_raw(nccl_comm, ExecSpace{});
#elif defined(KOKKOSCOMM_ENABLE_RCCL)
            static_assert(std::is_same_v<ExecSpace, Kokkos::HIP>,
                          "RCCL communication requires Kokkos::DefaultExecutionSpace to be HIP.");
            rcclUniqueId id{};
            if (decomposition.rank == 0) rcclGetUniqueId(&id);
            MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, decomposition.comm);
            rcclComm_t rccl_comm{};
            rcclCommInitRank(&rccl_comm, decomposition.size, id, decomposition.rank);
            return CommHandle::from_raw(rccl_comm, ExecSpace{});
#else
            throw std::runtime_error("CCL communication was requested but no CCL backend is enabled.");
#endif
        }
    }

    void exchange(BlockingBuffers& buffers, int) {
        std::vector<RequestType> receives;
        std::vector<RequestType> sends;
        receives.reserve(n_directions);
        sends.reserve(n_directions);
        for (int direction = 0; direction < n_directions; ++direction) {
            const int neighbor = decomposition.neighbors[direction];
            if (neighbor == MPI_PROC_NULL) continue;
            receives.push_back(KokkosComm::recv(
                communicator, buffers.recv[direction], neighbor));
        }
        for (int direction = 0; direction < n_directions; ++direction) {
            const int neighbor = decomposition.neighbors[direction];
            if (neighbor == MPI_PROC_NULL) continue;
            sends.push_back(KokkosComm::send(
                communicator, buffers.send[direction], neighbor));
        }
        KokkosComm::wait_all(receives);
        KokkosComm::wait_all(sends);
    }

    struct ExchangeState {
        std::vector<RequestType> receives;
        std::vector<RequestType> sends;
    };

    typename NonBlockingBenchmark<real>::ExchangeHandle start_exchange(
        View& field, Buffers& buffers, int) {
        auto state = std::make_shared<ExchangeState>();
        state->receives.reserve(n_directions);
        state->sends.reserve(n_directions);
        for (int direction = 0; direction < n_directions; ++direction) {
            const int neighbor = decomposition.neighbors[direction];
            if (neighbor == MPI_PROC_NULL) continue;
            if (NonBlockingBenchmark<real>::uses_buffer(direction)) {
                state->receives.push_back(KokkosComm::recv(
                    communicator, buffers.recv[direction], neighbor));
            } else {
                auto target = benchmark<real>::halo_subview(
                    direction, field);
                state->receives.push_back(KokkosComm::recv(communicator, target, neighbor));
            }
        }
        for (int direction = 0; direction < n_directions; ++direction) {
            const int neighbor = decomposition.neighbors[direction];
            if (neighbor == MPI_PROC_NULL) continue;
            if (NonBlockingBenchmark<real>::uses_buffer(direction)) {
                state->sends.push_back(KokkosComm::send(
                    communicator, buffers.send[direction], neighbor));
            } else {
                auto source = benchmark<real>::pack_subview(
                    direction, field);
                state->sends.push_back(KokkosComm::send(communicator, source, neighbor));
            }
        }

        typename NonBlockingBenchmark<real>::ExchangeHandle handle;
        handle.finish_receives = [this, state, &field, &buffers] {
            KokkosComm::wait_all(state->receives);
            for (int direction = 0; direction < n_directions; ++direction) {
            if (this->decomposition.neighbors[direction] == MPI_PROC_NULL) continue;
                if (NonBlockingBenchmark<real>::uses_buffer(direction)) {
                    Kokkos::deep_copy(benchmark<real>::halo_subview(
                        direction, field),
                        buffers.recv[direction]);
                }
            }
        };
        handle.finish_sends = [state] { KokkosComm::wait_all(state->sends); };
        return handle;
    }

    const Decomposition& decomposition;
    CommHandle communicator;
};

template <typename real, bool CCL = false>
class GS_KC_Blocking : public blocking_benchmark<real>, protected GS_KC<real, CCL> {
public:
    explicit GS_KC_Blocking(const Parameters& parameters, const Decomposition& decomposition)
        : blocking_benchmark<real>(parameters, decomposition), GS_KC<real, CCL>(decomposition) {}

protected:
    void exchange(typename blocking_benchmark<real>::Buffers& buffers, int tag_base) override {
        GS_KC<real, CCL>::exchange(buffers, tag_base);
    }
};

template <typename real, bool CCL = false>
class GS_KC_NonBlocking : public NonBlockingBenchmark<real>, protected GS_KC<real, CCL> {
public:
    explicit GS_KC_NonBlocking(const Parameters& parameters, const Decomposition& decomposition)
        : NonBlockingBenchmark<real>(parameters, decomposition), GS_KC<real, CCL>(decomposition) {}

protected:
    typename NonBlockingBenchmark<real>::ExchangeHandle start_exchange(
        typename NonBlockingBenchmark<real>::View& field,
        typename NonBlockingBenchmark<real>::Buffers& buffers, int tag_base) override {
        return GS_KC<real, CCL>::start_exchange(field, buffers, tag_base);
    }
};
