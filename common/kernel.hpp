#pragma once

#include <Kokkos_Core.hpp>

#include "coefficients.hpp"

template<typename real>
KOKKOS_INLINE_FUNCTION
void gs_kernel(
    const int i, const int j,
    const Kokkos::View<real**, Kokkos::LayoutRight>& u,
    const Kokkos::View<real**, Kokkos::LayoutRight>& v,
    Kokkos::View<real**, Kokkos::LayoutRight>& u_temp,
    Kokkos::View<real**, Kokkos::LayoutRight>& v_temp,
    const coefficients<real>& constants)
{
    const real u_full =
        u(i - 1, j - 1) + u(i - 1, j) + u(i - 1, j + 1) +
        u(i, j - 1) - real(8) * u(i, j) + u(i, j + 1) +
        u(i + 1, j - 1) + u(i + 1, j) + u(i + 1, j + 1);

    const real v_full =
        v(i - 1, j - 1) + v(i - 1, j) + v(i - 1, j + 1) +
        v(i, j - 1) - real(8) * v(i, j) + v(i, j + 1) +
        v(i + 1, j - 1) + v(i + 1, j) + v(i + 1, j + 1);

    const real uvv = u(i, j) * v(i, j) * v(i, j);

    const real u_delta =
        constants.diffusion_rate_u * u_full
        - uvv
        + constants.feed_rate * (real(1) - u(i, j));

    const real v_delta =
        constants.diffusion_rate_v * v_full
        + uvv
        - (constants.feed_rate + constants.kill_rate) * v(i, j);

    u_temp(i, j) = u(i, j) + u_delta * constants.dt;
    v_temp(i, j) = v(i, j) + v_delta * constants.dt;
}