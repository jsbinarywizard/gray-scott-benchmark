#pragma once

struct Parameters {
    int warmup_time = 1; // seconds
    int benchmark_time = 5; // seconds

    int rows = 0;     // Either local or global, depending on strong_scaling.
    int columns = 0;

    bool measure_cell_updates = true; // Deleted the option to disable cell-update measurements, as it is not used in the benchmark.

    bool strong_scaling = true;
};