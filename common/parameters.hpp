#pragma once

struct Parameters {
    int warmup_iters = 500;
    int benchmark_iters = 5000;

    int rows = 0;     // Either local or global, depending on strong_scaling.
    int columns = 0;

    bool measure_cell_updates = true;
    bool measure_comm_bandwidth = true;

    bool strong_scaling = true;
};