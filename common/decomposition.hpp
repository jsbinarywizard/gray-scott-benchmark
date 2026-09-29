#pragma once

#include <array>
#include <mpi.h>

enum Direction {
    NW = 0,
    N,
    NE,
    W,
    E,
    SW,
    S,
    SE
};

constexpr int num_directions = 8;
constexpr int n_directions = num_directions;

struct Decomposition {
    int rank;
    int size;

    MPI_Comm comm = MPI_COMM_WORLD;

    int neighbors[num_directions] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL,
                    MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};

    
    int coords[2]; // Cartesian coordinates of this rank in the 2D grid.
    int dims[2]; // Dimensions of the Cartesian grid.

    Decomposition(std::array<bool, 2> periodic = {false, false}) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        dims[0] = 0;
        dims[1] = 0;

        MPI_Dims_create(size, 2, dims);

        int periods[2] = {periodic[0] ? 1 : 0, periodic[1] ? 1 : 0};

        MPI_Comm cart_comm;

        MPI_Cart_create(comm, 2, dims, periods, 0, &cart_comm);
        comm = cart_comm;

        coords[0] = 0;
        coords[1] = 0;
        MPI_Cart_coords(comm, rank, 2, coords); 

        auto neighbor = [&](int dr, int dc) {
            const int c[2] = {coords[0] + dr, coords[1] + dc};

            if (c[0] < 0 || c[0] >= dims[0] || c[1] < 0 || c[1] >= dims[1]) {
                return MPI_PROC_NULL;
            }

            int r = MPI_PROC_NULL;
            MPI_Cart_rank(comm, c, &r);
            return r;
        };
        neighbors[NW] = neighbor(-1, -1);
        neighbors[N] = neighbor(-1, 0);
        neighbors[NE] = neighbor(-1, +1);

        neighbors[W] = neighbor(0, -1);
        neighbors[E] = neighbor(0, +1);

        neighbors[SW] = neighbor(+1, -1);
        neighbors[S] = neighbor(+1, 0);
        neighbors[SE] = neighbor(+1, +1);
    }

    Decomposition(const Decomposition&) = delete;
    Decomposition& operator=(const Decomposition&) = delete;

    ~Decomposition() {
        if (comm != MPI_COMM_NULL) {
            MPI_Comm_free(&comm);
        }
    }
};