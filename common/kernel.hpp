#pragma once

#include <Kokkos_Core.hpp>

#include "coefficients.hpp"

template <typename real>
using gs_view = Kokkos::View<real**, Kokkos::LayoutRight>;

template <typename real, bool full_stencil = true>
KOKKOS_INLINE_FUNCTION
void gs_kernel(
    const int i,
    const int j,
    const gs_view<real> u,
    const gs_view<real> v,
    const gs_view<real> u_out,
    const gs_view<real> v_out,
    const coefficients<real>& c)
{
    const real u_ij = u(i, j);
    const real v_ij = v(i, j);

    if constexpr (full_stencil) {

        // 8-neighbor Laplacian.
        const real lap_u =
            u(i - 1, j - 1) + u(i - 1, j) + u(i - 1, j + 1) +
            u(i,     j - 1) - real(8) * u_ij + u(i,     j + 1) +
            u(i + 1, j - 1) + u(i + 1, j) + u(i + 1, j + 1);

        const real lap_v =
            v(i - 1, j - 1) + v(i - 1, j) + v(i - 1, j + 1) +
            v(i,     j - 1) - real(8) * v_ij + v(i,     j + 1) +
            v(i + 1, j - 1) + v(i + 1, j) + v(i + 1, j + 1);
    }else{
        
        // 4-neighbor Laplacian.
        const real lap_u =
            u(i - 1, j) + u(i, j - 1) - real(4) * u_ij + u(i, j + 1) + u(i + 1, j);

        const real lap_v =
            v(i - 1, j) + v(i, j - 1) - real(4) * v_ij + v(i, j + 1) + v(i + 1, j);
    }

    // Gray-Scott reaction term: u * v^2.
    const real uvv = u_ij * v_ij * v_ij;

    const real du =
        c.diffusion_rate_u * lap_u
        - uvv
        + c.feed_rate * (real(1) - u_ij);

    const real dv =
        c.diffusion_rate_v * lap_v
        + uvv
        - (c.feed_rate + c.kill_rate) * v_ij;

    const real dt = c.dt;

    u_out(i, j) = u_ij + dt * du;
    v_out(i, j) = v_ij + dt * dv;
}

template <typename real, bool full_stencil = true>
void gs_compute(
    const gs_view<real>& u,
    const gs_view<real>& v,
    const gs_view<real>& u_out,
    const gs_view<real>& v_out,
    const coefficients<real>& c)
{
    const int rows = static_cast<int>(u.extent(0));
    const int columns = static_cast<int>(u.extent(1));
    const auto u_ = u;
    const auto v_ = v;
    const auto u_out_ = u_out;
    const auto v_out_ = v_out;
    const auto c_ = c;

    Kokkos::parallel_for(
        "compute",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {1, 1}, {rows - 1, columns - 1}),
        KOKKOS_LAMBDA(const int i, const int j) {
            gs_kernel<real, full_stencil>(i, j, u_, v_, u_out_, v_out_, c_);
        });
}

template <typename real, bool full_stencil = true>
void gs_compute_interior(
    const gs_view<real>& u,
    const gs_view<real>& v,
    const gs_view<real>& u_out,
    const gs_view<real>& v_out,
    const coefficients<real>& c)
{
    const int rows = static_cast<int>(u.extent(0)) - 2;
    const int columns = static_cast<int>(u.extent(1)) - 2;
    if (rows < 3 || columns < 3) return;

    const auto u_ = u;
    const auto v_ = v;
    const auto u_out_ = u_out;
    const auto v_out_ = v_out;
    const auto c_ = c;

    Kokkos::parallel_for(
        "compute interior",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {2, 2}, {rows, columns}),
        KOKKOS_LAMBDA(const int i, const int j) {
            gs_kernel<real, full_stencil>(i, j, u_, v_, u_out_, v_out_, c_);
        });
}

template <typename real, bool full_stencil = true>
void gs_compute_ring(
    const gs_view<real>& u,
    const gs_view<real>& v,
    const gs_view<real>& u_out,
    const gs_view<real>& v_out,
    const coefficients<real>& c)
{
    const int rows = static_cast<int>(u.extent(0)) - 2;
    const int columns = static_cast<int>(u.extent(1)) - 2;
    const auto u_ = u;
    const auto v_ = v;
    const auto u_out_ = u_out;
    const auto v_out_ = v_out;
    const auto c_ = c;

    // Fuse Kernel calls, as the kernel is very small and the overhead of launching a kernel is significant.
    Kokkos::parallel_for(
        "compute ring top/bottom",
        Kokkos::RangePolicy<int>(1, columns + 1),
        KOKKOS_LAMBDA(const int j) {
            gs_kernel<real, full_stencil>(1, j, u_, v_, u_out_, v_out_, c_);
            gs_kernel<real, full_stencil>(rows, j, u_, v_, u_out_, v_out_, c_);
        });

    if (rows > 2) {
        Kokkos::parallel_for(
            "compute ring left/right",
            Kokkos::RangePolicy<int>(2, rows),
            KOKKOS_LAMBDA(const int i) {
                gs_kernel<real, full_stencil>(i, 1, u_, v_, u_out_, v_out_, c_);
                gs_kernel<real, full_stencil>(i, columns, u_, v_, u_out_, v_out_, c_);
            });
    }
}