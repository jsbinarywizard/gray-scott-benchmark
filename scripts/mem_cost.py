#!/usr/bin/env python3

import math


BYTES_PER_DOUBLE = 8
NUM_FIELDS = 4


def memory_cost(grid_size, decomposition_factor=1):
    """
    Calculate memory requirements for a single global grid.

    Parameters
    ----------
    grid_size : int or float
        Global grid size N, resulting in an N x N grid.
    decomposition_factor : int
        Number of domains/processes used for domain decomposition.

    Returns
    -------
    tuple[float, float]
        Total memory and memory per domain, in MiB.
    """
    memory_bytes = grid_size**2 * NUM_FIELDS * BYTES_PER_DOUBLE
    memory_per_domain = memory_bytes / decomposition_factor

    return (
        memory_bytes / 1024**2,
        memory_per_domain / 1024**2,
    )


def strong_scaling(
    grid_exponents,
    nodes,
    processes_per_node,
):
    """
    Strong scaling.

    The global problem size is fixed while the number of processes
    is increased.

    The global grid size is always a power of two.
    """

    print("=" * 90)
    print("STRONG SCALING")
    print("=" * 90)

    for exponent in grid_exponents:
        grid_size = 2**exponent

        total_memory, _ = memory_cost(grid_size)

        print(f"\nGlobal grid: N = {grid_size}")
        print(f"Total memory: {total_memory:.2f} MiB")

        print(
            f"{'Nodes':>8} "
            f"{'Proc/node':>10} "
            f"{'Processes':>10} "
            f"{'Memory/proc':>15}"
        )
        print("-" * 55)

        for num_nodes in nodes:
            for ppn in processes_per_node:
                total_processes = num_nodes * ppn

                _, memory_per_process = memory_cost(
                    grid_size,
                    decomposition_factor=total_processes,
                )

                print(
                    f"{num_nodes:8d} "
                    f"{ppn:10d} "
                    f"{total_processes:10d} "
                    f"{memory_per_process:15.2f} MiB"
                )


def weak_scaling(
    local_grid_exponents,
    nodes,
    processes_per_node,
):
    """
    Weak scaling.

    The local problem size per process is fixed while the number
    of processes is increased.

    The local grid size is always a power of two.

    The global grid size is:

        N_global = N_local * sqrt(P)

    Therefore, N_global does not need to be a power of two.
    """

    print("=" * 90)
    print("WEAK SCALING")
    print("=" * 90)

    for exponent in local_grid_exponents:
        local_grid_size = 2**exponent

        print(f"\nLocal grid: N_local = {local_grid_size}")

        print(
            f"{'Nodes':>8} "
            f"{'Proc/node':>10} "
            f"{'Processes':>10} "
            f"{'Global N':>12} "
            f"{'Memory/proc':>15} "
            f"{'Total memory':>15}"
        )
        print("-" * 85)

        for num_nodes in nodes:
            for ppn in processes_per_node:
                total_processes = num_nodes * ppn

                # Keep the local problem size constant:
                #
                #     N_global^2 / P = N_local^2
                #
                # Therefore:
                #
                #     N_global = N_local * sqrt(P)
                #
                global_grid_size = (
                    local_grid_size * math.sqrt(total_processes)
                )

                total_memory, memory_per_process = memory_cost(
                    global_grid_size,
                    decomposition_factor=total_processes,
                )

                print(
                    f"{num_nodes:8d} "
                    f"{ppn:10d} "
                    f"{total_processes:10d} "
                    f"{global_grid_size:12.2f} "
                    f"{memory_per_process:15.2f} MiB "
                    f"{total_memory:15.2f} MiB"
                )


if __name__ == "__main__":

    # ---------------------------------------------------------------
    # Machine configuration
    # ---------------------------------------------------------------

    nodes = [1, 2, 4]

    # Processes per node.
    processes_per_node = [4]

    # ---------------------------------------------------------------
    # Strong scaling
    #
    # Global grid sizes:
    #     2^7, 2^8, ..., 2^15
    # ---------------------------------------------------------------

    strong_scaling(
        grid_exponents=range(7, 17),
        nodes=nodes,
        processes_per_node=processes_per_node,
    )

    # ---------------------------------------------------------------
    # Weak scaling
    #
    # Local grid sizes:
    #     2^7, 2^8, 2^9
    #
    # Each is an independent weak-scaling series.
    # ---------------------------------------------------------------

    weak_scaling(
        local_grid_exponents=range(7, 13),
        nodes=nodes,
        processes_per_node=processes_per_node,
    )