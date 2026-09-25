#pragma once

#include <Kokkos_Core.hpp>

#include "coefficients.hpp"

template <typename real>
KOKKOS_INLINE_FUNCTION
void gs_kernel(
    const int i,
    const int j,
    const Kokkos::View<real**, Kokkos::LayoutRight> u,
    const Kokkos::View<real**, Kokkos::LayoutRight> v,
    const Kokkos::View<real**, Kokkos::LayoutRight> u_out,
    const Kokkos::View<real**, Kokkos::LayoutRight> v_out,
    const coefficients<real>& c)
{
    const real u_ij = u(i, j);
    const real v_ij = v(i, j);

    // 8-neighbor Laplacian.
    const real lap_u =
        u(i - 1, j - 1) + u(i - 1, j) + u(i - 1, j + 1) +
        u(i,     j - 1) - real(8) * u_ij + u(i,     j + 1) +
        u(i + 1, j - 1) + u(i + 1, j) + u(i + 1, j + 1);

    const real lap_v =
        v(i - 1, j - 1) + v(i - 1, j) + v(i - 1, j + 1) +
        v(i,     j - 1) - real(8) * v_ij + v(i,     j + 1) +
        v(i + 1, j - 1) + v(i + 1, j) + v(i + 1, j + 1);

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
