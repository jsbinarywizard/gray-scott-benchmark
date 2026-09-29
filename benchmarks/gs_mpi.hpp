#pragma once

#include <array>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include "benchmark.hpp"
#include "../common/kernel.hpp"

template <typename real>
class GS_MPI : public benchmark<real> {
protected:
    using Base = benchmark<real>;
    using View = typename Base::View;

    struct CommBuffers {
        std::array<Kokkos::View<real*>, n_directions> send;
        std::array<Kokkos::View<real*>, n_directions> recv;

        CommBuffers(int rows, int columns) {
            const int lengths[n_directions] = {1, columns, 1, rows, rows, 1, columns, 1};
            for (int direction = 0; direction < n_directions; ++direction) {
                send[direction] = Kokkos::View<real*>("send", lengths[direction]);
                recv[direction] = Kokkos::View<real*>("recv", lengths[direction]);
            }
        }
    };

    explicit GS_MPI(const Parameters& parameters, const Decomposition& decomposition)
        : Base(parameters, decomposition),
          u_buffers(this->local_rows, this->local_columns),
          v_buffers(this->local_rows, this->local_columns) {
    }

    static MPI_Datatype mpi_real_type() {
        static_assert(std::is_same_v<real, float> || std::is_same_v<real, double>,
                      "Unsupported real type for MPI communication.");
        if constexpr (std::is_same_v<real, float>) {
            return MPI_FLOAT;
        } else {
            return MPI_DOUBLE;
        }
    }

    static int opposite(int direction) {
        static constexpr int opposite_dir[n_directions] = {SE, S, SW, E, W, NE, N, NW};
        return opposite_dir[direction];
    }

    void pack(int direction, const View& field, CommBuffers& buffers) {
        Kokkos::deep_copy(buffers.send[direction], this->pack_subview(
            direction, field, this->local_rows, this->local_columns));
    }

    void unpack(int direction, View& field, const CommBuffers& buffers) {
        Kokkos::deep_copy(this->halo_subview(
            direction, field, this->local_rows, this->local_columns), buffers.recv[direction]);
    }

    CommBuffers u_buffers;
    CommBuffers v_buffers;
};

template <typename real>
class GS_MPI_Blocking : public GS_MPI<real> {
    // GS_MPI<real> is a dependent base, so these names (types and static
    // members) are invisible to unqualified lookup until explicitly pulled
    // in — same reasoning GS_MPI_NonBlocking already follows below.
    using Base = GS_MPI<real>;
    using View = typename Base::View;
    using CommBuffers = typename Base::CommBuffers;

public:
    explicit GS_MPI_Blocking(const Parameters& parameters, const Decomposition& decomposition)
        : Base(parameters, decomposition) {}

private:
    void exchange(View& field, CommBuffers& buffers, int tag_base) {
        for (int direction = 0; direction < n_directions; ++direction) {
            if (this->decomposition.neighbors[direction] != MPI_PROC_NULL) {
                this->pack(direction, field, buffers);
            }
        }
        Kokkos::fence();

        std::array<MPI_Request, 2 * n_directions> requests{};
        int count = 0;
        for (int direction = 0; direction < n_directions; ++direction) {
            const int neighbor = this->decomposition.neighbors[direction];
            if (neighbor == MPI_PROC_NULL) continue;
            MPI_Irecv(buffers.recv[direction].data(), static_cast<int>(buffers.recv[direction].size()),
                      Base::mpi_real_type(), neighbor, tag_base + direction,
                      this->decomposition.comm, &requests[count++]);
        }
        for (int direction = 0; direction < n_directions; ++direction) {
            const int neighbor = this->decomposition.neighbors[direction];
            if (neighbor == MPI_PROC_NULL) continue;
            MPI_Isend(buffers.send[direction].data(), static_cast<int>(buffers.send[direction].size()),
                      Base::mpi_real_type(), neighbor, tag_base + Base::opposite(direction),
                      this->decomposition.comm, &requests[count++]);
        }
        MPI_Waitall(count, requests.data(), MPI_STATUSES_IGNORE);

        for (int direction = 0; direction < n_directions; ++direction) {
            if (this->decomposition.neighbors[direction] != MPI_PROC_NULL) {
                this->unpack(direction, field, buffers);
            }
        }
        Kokkos::fence();
    }

    void iteration() override {
        timer comm_timer;
        exchange(this->u, this->u_buffers, 100);
        exchange(this->v, this->v_buffers, 200);
        this->communication_time += comm_timer.elapsed();
        gs_compute(this->u, this->v, this->u_temp, this->v_temp, this->coeffs);
        Kokkos::fence();
        std::swap(this->u, this->u_temp);
        std::swap(this->v, this->v_temp);
    }
};

template <typename real>
class GS_MPI_NonBlocking : public GS_MPI<real> {
    using Base = GS_MPI<real>;
    using View = typename Base::View;
    using CommBuffers = typename Base::CommBuffers;

    struct ExchangeHandle {
        std::array<MPI_Request, n_directions> recv_requests{};
        std::array<int, n_directions> recv_directions{};
        int recv_count = 0;
        std::array<MPI_Request, n_directions> send_requests{};
        int send_count = 0;
    };

    ExchangeHandle begin_exchange(const View& field, CommBuffers& buffers, int tag_base) {
        ExchangeHandle handle;
        for (int direction = 0; direction < n_directions; ++direction) {
            const int neighbor = this->decomposition.neighbors[direction];
            if (neighbor == MPI_PROC_NULL) continue;
            handle.recv_directions[handle.recv_count] = direction;
            MPI_Irecv(buffers.recv[direction].data(), static_cast<int>(buffers.recv[direction].size()),
                      Base::mpi_real_type(), neighbor, tag_base + direction,
                      this->decomposition.comm, &handle.recv_requests[handle.recv_count++]);
        }
        for (int direction = 0; direction < n_directions; ++direction) {
            if (this->decomposition.neighbors[direction] == MPI_PROC_NULL) continue;
            this->pack(direction, field, buffers);
            Kokkos::fence();
            MPI_Isend(buffers.send[direction].data(), static_cast<int>(buffers.send[direction].size()),
                      Base::mpi_real_type(), this->decomposition.neighbors[direction],
                      tag_base + Base::opposite(direction), this->decomposition.comm,
                      &handle.send_requests[handle.send_count++]);
        }
        return handle;
    }

    void finish_receives(View& field, CommBuffers& buffers, ExchangeHandle& handle) {
        for (int done = 0; done < handle.recv_count; ++done) {
            int index = MPI_UNDEFINED;
            MPI_Waitany(handle.recv_count, handle.recv_requests.data(), &index, MPI_STATUS_IGNORE);
            this->unpack(handle.recv_directions[index], field, buffers);
        }
    }

public:
    explicit GS_MPI_NonBlocking(const Parameters& parameters, const Decomposition& decomposition)
        : GS_MPI<real>(parameters, decomposition) {}

private:
    void iteration() override {
        auto u_handle = begin_exchange(this->u, this->u_buffers, 100);
        auto v_handle = begin_exchange(this->v, this->v_buffers, 200);
        gs_compute_interior(this->u, this->v, this->u_temp, this->v_temp, this->coeffs);
        finish_receives(this->u, this->u_buffers, u_handle);
        finish_receives(this->v, this->v_buffers, v_handle);
        gs_compute_ring(this->u, this->v, this->u_temp, this->v_temp, this->coeffs);
        Kokkos::fence();
        MPI_Waitall(u_handle.send_count, u_handle.send_requests.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(v_handle.send_count, v_handle.send_requests.data(), MPI_STATUSES_IGNORE);
        std::swap(this->u, this->u_temp);
        std::swap(this->v, this->v_temp);
    }
};