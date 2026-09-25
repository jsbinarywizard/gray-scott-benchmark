#!/usr/bin/env python3

def memory_cost(grid_sizes, decomposition_factor=1):
    """
    Calculate memory requirements for a list of grid sizes.

    Parameters
    ----------
    grid_sizes : list[int]
        List of integer grid sizes.
    decomposition_factor : int
        Number of domains used for domain decomposition.

    Each field has:
        grid_size^2 elements * 8 bytes (double precision)

    There are 4 fields in total.
    """

    BYTES_PER_DOUBLE = 8
    NUM_FIELDS = 4

    for n in grid_sizes:
        # Total memory without domain decomposition
        memory_bytes = n**2 * NUM_FIELDS * BYTES_PER_DOUBLE

        # Memory per domain after decomposition
        memory_per_domain = memory_bytes / decomposition_factor

        print(
            f"N = {n:6d} | "
            f"Total = {memory_bytes / 1024**2:10.2f} MiB | "
            f"Per domain ({decomposition_factor}) = "
            f"{memory_per_domain / 1024**2:10.2f} MiB"
        )


if __name__ == "__main__":
    grid_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    decomposition_factor = 8

    memory_cost(
        grid_sizes,
        decomposition_factor=decomposition_factor,
    )