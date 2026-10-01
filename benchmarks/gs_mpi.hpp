#pragma once

#include <array>
#include <memory>
#include <type_traits>

#include <mpi.h>

#include "benchmark.hpp"

template <typename real>
class GS_MPI {
protected:
    using BlockingBuffers = typename blocking_benchmark<real>::Buffers;
    using Buffers = typename NonBlockingBenchmark<real>::Buffers;

    explicit GS_MPI(const Decomposition& decomposition)
        : decomposition(decomposition) {}

    static MPI_Datatype mpi_real_type() {
        static_assert(std::is_same_v<real, float> || std::is_same_v<real, double>,
                      "Unsupported real type for MPI communication.");
        if constexpr (std::is_same_v<real, float>) return MPI_FLOAT;
        return MPI_DOUBLE;
    }

    static int opposite(int direction) {
        static constexpr int opposite_dir[n_directions] = {SE, S, SW, E, W, NE, N, NW};
        return opposite_dir[direction];
    }

    void exchange(BlockingBuffers& buffers, int tag_base) {
        std::array<MPI_Request, 2 * n_directions> requests{};
        int count = 0;
        for (int direction = 0; direction < n_directions; ++direction) {
            const int neighbor = decomposition.neighbors[direction];
            if (neighbor == MPI_PROC_NULL) continue;
            MPI_Irecv(buffers.recv[direction].data(),
                      static_cast<int>(buffers.recv[direction].size()), mpi_real_type(),
                      neighbor, tag_base + direction, decomposition.comm, &requests[count++]);
        }
        for (int direction = 0; direction < n_directions; ++direction) {
            const int neighbor = decomposition.neighbors[direction];
            if (neighbor == MPI_PROC_NULL) continue;
            MPI_Isend(buffers.send[direction].data(),
                      static_cast<int>(buffers.send[direction].size()), mpi_real_type(),
                      neighbor, tag_base + opposite(direction), decomposition.comm,
                      &requests[count++]);
        }
        MPI_Waitall(count, requests.data(), MPI_STATUSES_IGNORE);
    }

    struct ExchangeState {
        std::array<MPI_Request, n_directions> recv_requests{};
        std::array<int, n_directions> recv_directions{};
        int recv_count = 0;
        std::array<MPI_Request, n_directions> send_requests{};
        int send_count = 0;
    };

    typename NonBlockingBenchmark<real>::ExchangeHandle start_exchange(
        typename NonBlockingBenchmark<real>::View& field, Buffers& buffers, int tag_base) {
        auto state = std::make_shared<ExchangeState>();
        for (int direction = 0; direction < n_directions; ++direction) {
            const int neighbor = decomposition.neighbors[direction];
            if (neighbor == MPI_PROC_NULL) continue;
            state->recv_directions[state->recv_count] = direction;
            if (NonBlockingBenchmark<real>::uses_buffer(direction)) {
                MPI_Irecv(buffers.recv[direction].data(),
                          static_cast<int>(buffers.recv[direction].size()), mpi_real_type(),
                          neighbor, tag_base + direction, decomposition.comm,
                          &state->recv_requests[state->recv_count++]);
            } else {
                auto target = benchmark<real>::halo_subview(
                    direction, field);
                MPI_Irecv(target.data(), static_cast<int>(target.size()), mpi_real_type(),
                          neighbor, tag_base + direction, decomposition.comm,
                          &state->recv_requests[state->recv_count++]);
            }
        }
        for (int direction = 0; direction < n_directions; ++direction) {
            const int neighbor = decomposition.neighbors[direction];
            if (neighbor == MPI_PROC_NULL) continue;
            if (NonBlockingBenchmark<real>::uses_buffer(direction)) {
                MPI_Isend(buffers.send[direction].data(),
                          static_cast<int>(buffers.send[direction].size()), mpi_real_type(),
                          neighbor, tag_base + opposite(direction), decomposition.comm,
                          &state->send_requests[state->send_count++]);
            } else {
                auto source = benchmark<real>::pack_subview(
                    direction, field);
                MPI_Isend(source.data(), static_cast<int>(source.size()), mpi_real_type(),
                          neighbor, tag_base + opposite(direction), decomposition.comm,
                          &state->send_requests[state->send_count++]);
            }
        }

        typename NonBlockingBenchmark<real>::ExchangeHandle handle;
        handle.finish_receives = [state, &field, &buffers] {
            for (int done = 0; done < state->recv_count; ++done) {
                int index = MPI_UNDEFINED;
                MPI_Waitany(state->recv_count, state->recv_requests.data(), &index,
                            MPI_STATUS_IGNORE);
                const int direction = state->recv_directions[index];
                if (NonBlockingBenchmark<real>::uses_buffer(direction)) {
                    Kokkos::deep_copy(benchmark<real>::halo_subview(
                        direction, field),
                        buffers.recv[direction]);
                }
            }
        };
        handle.finish_sends = [state] {
            MPI_Waitall(state->send_count, state->send_requests.data(), MPI_STATUSES_IGNORE);
        };
        return handle;
    }

    const Decomposition& decomposition;
};

template <typename real>
class GS_MPI_Blocking : public blocking_benchmark<real>, protected GS_MPI<real> {
public:
    explicit GS_MPI_Blocking(const Parameters& parameters, const Decomposition& decomposition)
        : blocking_benchmark<real>(parameters, decomposition), GS_MPI<real>(decomposition) {}

protected:
    void exchange(typename blocking_benchmark<real>::Buffers& buffers, int tag_base) override {
        GS_MPI<real>::exchange(buffers, tag_base);
    }
};

template <typename real>
class GS_MPI_NonBlocking : public NonBlockingBenchmark<real>, protected GS_MPI<real> {
public:
    explicit GS_MPI_NonBlocking(const Parameters& parameters, const Decomposition& decomposition)
        : NonBlockingBenchmark<real>(parameters, decomposition), GS_MPI<real>(decomposition) {}

protected:
    typename NonBlockingBenchmark<real>::ExchangeHandle start_exchange(
        typename NonBlockingBenchmark<real>::View& field,
        typename NonBlockingBenchmark<real>::Buffers& buffers, int tag_base) override {
        return GS_MPI<real>::start_exchange(field, buffers, tag_base);
    }
};
