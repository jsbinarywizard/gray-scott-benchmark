#pragma once

#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include "benchmark.hpp"
#include "gs_mpi.hpp"   // fallback; adjust to your header name

// ---------------------------------------------------------------------------
// Backend selection (set by CMake): NCCL on CUDA, RCCL on HIP, else MPI fallback.
// ---------------------------------------------------------------------------
#if defined(GS_CCL_BACKEND_NCCL)
  #include <cuda_runtime.h>
  #include <nccl.h>
  #define GS_CCL_NAME "NCCL"
  namespace gs_ccl {
      using error_t  = cudaError_t;
      using stream_t = cudaStream_t;
      constexpr error_t success = cudaSuccess;
      inline const char* error_string(error_t e) { return cudaGetErrorString(e); }
      inline error_t stream_create(stream_t* s)  { return cudaStreamCreateWithFlags(s, cudaStreamNonBlocking); }
      inline error_t stream_destroy(stream_t s)  { return cudaStreamDestroy(s); }
      inline error_t stream_sync(stream_t s)     { return cudaStreamSynchronize(s); }
  }
#elif defined(GS_CCL_BACKEND_RCCL)
  #include <hip/hip_runtime.h>
  #if __has_include(<rccl/rccl.h>)
    #include <rccl/rccl.h>
  #else
    #include <rccl.h>      // older ROCm
  #endif
  #define GS_CCL_NAME "RCCL"
  namespace gs_ccl {
      using error_t  = hipError_t;
      using stream_t = hipStream_t;
      constexpr error_t success = hipSuccess;
      inline const char* error_string(error_t e) { return hipGetErrorString(e); }
      inline error_t stream_create(stream_t* s)  { return hipStreamCreateWithFlags(s, hipStreamNonBlocking); }
      inline error_t stream_destroy(stream_t s)  { return hipStreamDestroy(s); }
      inline error_t stream_sync(stream_t s)     { return hipStreamSynchronize(s); }
  }
#endif

#if defined(GS_CCL_BACKEND_NCCL) || defined(GS_CCL_BACKEND_RCCL)
  #define GS_CCL_ACTIVE 1
#else
  #define GS_CCL_ACTIVE 0
#endif

inline constexpr bool gs_ccl_active = GS_CCL_ACTIVE;

#if GS_CCL_ACTIVE
// ===========================================================================
// Real implementation. RCCL exposes the nccl* API, so this part is shared.
// ===========================================================================

#define GS_GPU_CHECK(call)                                                       \
    do {                                                                         \
        gs_ccl::error_t err_ = (call);                                           \
        if (err_ != gs_ccl::success) {                                           \
            std::fprintf(stderr, "%s runtime error: %s at %s:%d\n", GS_CCL_NAME, \
                         gs_ccl::error_string(err_), __FILE__, __LINE__);        \
            std::abort();                                                        \
        }                                                                        \
    } while (0)

#define GS_CCL_CHECK(call)                                                       \
    do {                                                                         \
        ncclResult_t res_ = (call);                                              \
        if (res_ != ncclSuccess) {                                               \
            std::fprintf(stderr, "%s error: %s at %s:%d\n", GS_CCL_NAME,         \
                         ncclGetErrorString(res_), __FILE__, __LINE__);          \
            std::abort();                                                        \
        }                                                                        \
    } while (0)

template <typename real>
class GS_CCL {
protected:
    using BlockingBuffers = typename blocking_benchmark<real>::Buffers;
    using Buffers = typename NonBlockingBenchmark<real>::Buffers;

    explicit GS_CCL(const Decomposition& decomposition) : decomposition(decomposition) {
        ncclUniqueId id{};
        if (decomposition.rank == 0) GS_CCL_CHECK(ncclGetUniqueId(&id));
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, decomposition.comm);
        GS_CCL_CHECK(ncclCommInitRank(&ccl_comm, decomposition.size, id, decomposition.rank));
        GS_GPU_CHECK(gs_ccl::stream_create(&recv_stream));
        GS_GPU_CHECK(gs_ccl::stream_create(&send_stream));
    }

    ~GS_CCL() {
        gs_ccl::stream_sync(recv_stream);
        gs_ccl::stream_sync(send_stream);
        gs_ccl::stream_destroy(recv_stream);
        gs_ccl::stream_destroy(send_stream);
        ncclCommDestroy(ccl_comm);
    }

    GS_CCL(const GS_CCL&) = delete;
    GS_CCL& operator=(const GS_CCL&) = delete;

    static ncclDataType_t ccl_real_type() {
        static_assert(std::is_same_v<real, float> || std::is_same_v<real, double>,
                      "Unsupported real type for CCL communication.");
        if constexpr (std::is_same_v<real, float>) return ncclFloat;
        return ncclDouble;
    }

    static int opposite(int direction) {
        static constexpr int opposite_dir[n_directions] = {SE, S, SW, E, W, NE, N, NW};
        return opposite_dir[direction];
    }

    // No tags: messages between a rank pair match in issue order. Sends go out
    // in ascending d; the peer receives them as opposite(d), so receives are
    // issued in ascending opposite(d).
    template <typename RecvFn, typename SendFn>
    void post_group(gs_ccl::stream_t recv_stream, gs_ccl::stream_t send_stream,
                    RecvFn&& recv_fn, SendFn&& send_fn) {
        GS_CCL_CHECK(ncclGroupStart());
        for (int d = 0; d < n_directions; ++d) {
            const int r = opposite(d);
            const int neighbor = decomposition.neighbors[r];
            if (neighbor == MPI_PROC_NULL) continue;
            auto [ptr, count] = recv_fn(r);
            GS_CCL_CHECK(ncclRecv(ptr, count, ccl_real_type(), neighbor, ccl_comm,
                                  recv_stream));
        }
        for (int d = 0; d < n_directions; ++d) {
            const int neighbor = decomposition.neighbors[d];
            if (neighbor == MPI_PROC_NULL) continue;
            auto [ptr, count] = send_fn(d);
            GS_CCL_CHECK(ncclSend(ptr, count, ccl_real_type(), neighbor, ccl_comm,
                                  send_stream));
        }
        GS_CCL_CHECK(ncclGroupEnd());
    }

    void exchange(BlockingBuffers& buffers, int) {
        Kokkos::fence();
        post_group(recv_stream, recv_stream,
            [&](int d) { return std::make_pair(static_cast<void*>(buffers.recv[d].data()),
                                               static_cast<size_t>(buffers.recv[d].size())); },
            [&](int d) { return std::make_pair(static_cast<const void*>(buffers.send[d].data()),
                                               static_cast<size_t>(buffers.send[d].size())); });
        GS_GPU_CHECK(gs_ccl::stream_sync(recv_stream));
    }

    class NCCLExchangeHandle : public NonBlockingBenchmark<real>::ExchangeHandleBase {
    public:
        NCCLExchangeHandle(typename NonBlockingBenchmark<real>::View& field,
                           Buffers& buffers,
                           const Decomposition& decomposition,
                           gs_ccl::stream_t recv_stream,
                           gs_ccl::stream_t send_stream)
            : field(field), buffers(buffers), decomposition(decomposition),
              recv_stream(recv_stream), send_stream(send_stream) {}

        void wait_receives() override {
            GS_GPU_CHECK(gs_ccl::stream_sync(recv_stream));
            for (int direction : {W, E}) {
                if (decomposition.neighbors[direction] == MPI_PROC_NULL) continue;
                Kokkos::deep_copy(benchmark<real>::halo_subview(
                    direction, field),
                    buffers.recv[direction]);
            }
        }

        void wait_sends() override {
            GS_GPU_CHECK(gs_ccl::stream_sync(send_stream));
        }

    private:
        typename NonBlockingBenchmark<real>::View& field;
        Buffers& buffers;
        const Decomposition& decomposition;
        gs_ccl::stream_t recv_stream;
        gs_ccl::stream_t send_stream;
    };

    typename NonBlockingBenchmark<real>::ExchangeHandle start_exchange(
        typename NonBlockingBenchmark<real>::View& field, Buffers& buffers, int) {
        Kokkos::fence();
        auto handle = std::make_unique<NCCLExchangeHandle>(
            field, buffers, decomposition, recv_stream, send_stream);
        post_group(
            recv_stream, send_stream,
            [&](int d) {
                if (NonBlockingBenchmark<real>::uses_buffer(d))
                    return std::make_pair(static_cast<void*>(buffers.recv[d].data()),
                                          static_cast<size_t>(buffers.recv[d].size()));
                auto target = benchmark<real>::halo_subview(d, field);
                return std::make_pair(static_cast<void*>(target.data()),
                                      static_cast<size_t>(target.size()));
            },
            [&](int d) {
                if (NonBlockingBenchmark<real>::uses_buffer(d))
                    return std::make_pair(static_cast<const void*>(buffers.send[d].data()),
                                          static_cast<size_t>(buffers.send[d].size()));
                auto source = benchmark<real>::pack_subview(d, field);
                return std::make_pair(static_cast<const void*>(source.data()),
                                      static_cast<size_t>(source.size()));
            });


        return handle;
    }

    const Decomposition& decomposition;
    ncclComm_t ccl_comm{};
    gs_ccl::stream_t recv_stream{};
    gs_ccl::stream_t send_stream{};
};

#else
// ===========================================================================
// Fallback: CCL not enabled/available -> behave exactly like the MPI variant.
// ===========================================================================

template <typename real>
class GS_CCL : protected GS_MPI<real> {
protected:
    explicit GS_CCL(const Decomposition& decomposition) : GS_MPI<real>(decomposition) {
        static bool warned = false;
        if (!warned && decomposition.rank == 0) {
            std::fprintf(stderr,
                         "[GS] CCL backend not enabled; falling back to plain MPI. "
                         "Results are NOT CCL results.\n");
        }
        warned = true;
    }
    using GS_MPI<real>::exchange;
    using GS_MPI<real>::start_exchange;
};

#endif  // GS_CCL_ACTIVE

template <typename real>
class GS_CCL_Blocking : public blocking_benchmark<real>, protected GS_CCL<real> {
public:
    explicit GS_CCL_Blocking(const Parameters& parameters, const Decomposition& decomposition)
        : blocking_benchmark<real>(parameters, decomposition), GS_CCL<real>(decomposition) {}

protected:
    void exchange(typename blocking_benchmark<real>::Buffers& buffers, int tag_base) override {
        GS_CCL<real>::exchange(buffers, tag_base);
    }
};

template <typename real>
class GS_CCL_NonBlocking : public NonBlockingBenchmark<real>, protected GS_CCL<real> {
public:
    explicit GS_CCL_NonBlocking(const Parameters& parameters, const Decomposition& decomposition)
        : NonBlockingBenchmark<real>(parameters, decomposition), GS_CCL<real>(decomposition) {}

protected:
    typename NonBlockingBenchmark<real>::ExchangeHandle start_exchange(
        typename NonBlockingBenchmark<real>::View& field,
        typename NonBlockingBenchmark<real>::Buffers& buffers, int tag_base) override {
        return GS_CCL<real>::start_exchange(field, buffers, tag_base);
    }
};