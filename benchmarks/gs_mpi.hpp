#pragma once

#include <array>
#include <memory>
#include <type_traits>
#include <vector>

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

    class MPIExchangeHandle : public NonBlockingBenchmark<real>::ExchangeHandleBase {
    public:
        MPIExchangeHandle(typename NonBlockingBenchmark<real>::View& field,
                          Buffers& buffers)
            : field(field), buffers(buffers) {}

        void wait_receives() override {
            for (std::size_t done = 0; done < recv_requests.size(); ++done) {
                int index = MPI_UNDEFINED;
                MPI_Waitany(static_cast<int>(recv_requests.size()), recv_requests.data(),
                            &index, MPI_STATUS_IGNORE);
                if (index == MPI_UNDEFINED) continue;
                const int direction = recv_directions[index];
                if (NonBlockingBenchmark<real>::uses_buffer(direction)) {
                    Kokkos::deep_copy(benchmark<real>::halo_subview(
                        direction, field),
                        buffers.recv[direction]);
                }
            }
        }

        void wait_sends() override {
            MPI_Waitall(static_cast<int>(send_requests.size()), send_requests.data(),
                        MPI_STATUSES_IGNORE);
        }

        void push_send_request(MPI_Request request) {
            send_requests.push_back(request);
        }

        void push_recv_request(MPI_Request request, int direction) {
            recv_requests.push_back(request);
            recv_directions.push_back(direction);
        }

    private:
        typename NonBlockingBenchmark<real>::View& field;
        Buffers& buffers;
        std::vector<MPI_Request> recv_requests;
        std::vector<MPI_Request> send_requests;
        std::vector<int> recv_directions;
    };

    typename NonBlockingBenchmark<real>::ExchangeHandle start_exchange(
        typename NonBlockingBenchmark<real>::View& field, Buffers& buffers, int tag_base) {
        auto state = std::make_unique<MPIExchangeHandle>(field, buffers);

        for (int direction = 0; direction < n_directions; ++direction) {
            const int neighbor = decomposition.neighbors[direction];
            if (neighbor == MPI_PROC_NULL) continue;
            MPI_Request request;
            if (NonBlockingBenchmark<real>::uses_buffer(direction)) {
                MPI_Irecv(buffers.recv[direction].data(),
                          static_cast<int>(buffers.recv[direction].size()), mpi_real_type(),
                          neighbor, tag_base + direction, decomposition.comm,
                          &request);
            } else {
                auto target = benchmark<real>::halo_subview(
                    direction, field);
                MPI_Irecv(target.data(), static_cast<int>(target.size()), mpi_real_type(),
                          neighbor, tag_base + direction, decomposition.comm,
                          &request);
            }
            state->push_recv_request(request, direction);
        }
        for (int direction = 0; direction < n_directions; ++direction) {
            const int neighbor = decomposition.neighbors[direction];
            if (neighbor == MPI_PROC_NULL) continue;
            MPI_Request request;
            if (NonBlockingBenchmark<real>::uses_buffer(direction)) {
                MPI_Isend(buffers.send[direction].data(),
                          static_cast<int>(buffers.send[direction].size()), mpi_real_type(),
                          neighbor, tag_base + opposite(direction), decomposition.comm,
                          &request);
            } else {
                auto source = benchmark<real>::pack_subview(
                    direction, field);
                MPI_Isend(source.data(), static_cast<int>(source.size()), mpi_real_type(),
                          neighbor, tag_base + opposite(direction), decomposition.comm,
                          &request);
            }
            state->push_send_request(request);
        }

        return state;
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
