#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../benchmarks/benchmark.hpp" // Needed for the definition of `results`.
#include "arguments.hpp" // Needed for the defintions of Backend, Scaling, and Precision enums.

template <typename real>
const char* precision_name() {
    return std::is_same_v<real, float> ? "single" : "double";
}

// -----------------------------------------------------------------------------
// One row of benchmark output: everything needed to identify a run plus its
// measured results. Building one from a `results` (benchmark.hpp) plus run
// metadata is the only place that needs to know how those two things
// combine into something printable/writable.
// -----------------------------------------------------------------------------
struct BenchmarkRow {
    Backend backend;
    std::string precision;  // "single" / "double"
    Scaling scaling;
    int ranks = 0;
    int size = 0;

    double total_time = 0.0;
    double updates_per_second = 0.0;
    int iterations = 0;
    double communication_time = 0.0;

    static BenchmarkRow from_results(
        const results& r,
        Backend backend,
        std::string precision,
        Scaling scaling,
        int ranks,
        int size) {

        BenchmarkRow row;
        row.backend = backend;
        row.precision = std::move(precision);
        row.scaling = scaling;
        row.ranks = ranks;
        row.size = size;
        row.iterations = r.iterations;
        row.total_time = r.total_time;
        row.updates_per_second = r.updates_per_second;
        row.communication_time = r.communication_time;
        return row;
    }
};

// -----------------------------------------------------------------------------
// Owns the CSV file: writes its header once at construction, one line per
// call to write(). Also prints the matching summary line to stdout/stderr,
// so main.cpp's sweep loop doesn't need to keep two output formats in sync
// by hand, and every "how do we report a result/skip/status line" decision
// lives in one file instead of being spread across main.cpp.
//
// Only rank 0 should construct one of these with `active = true`; every
// other rank should pass `active = false`, after which every method below
// is a silent no-op. That lets the sweep loop in main.cpp call write() /
// skip() / note() unconditionally, rather than guarding every call site
// with `if (rank == 0)`.
//
// Throws std::runtime_error if `active` is true and the file cannot be
// opened; callers on rank 0 should catch this and MPI_Bcast the failure to
// other ranks before they block on any collective the benchmark relies on
// (see main.cpp).
// -----------------------------------------------------------------------------
class ResultsWriter {
public:
    ResultsWriter(const std::string& path, bool active) : active_(active) {
        if (!active_) {
            return;
        }

        csv_.open(path);

        if (!csv_) {
            throw std::runtime_error("Could not open output file '" + path + "'");
        }

        csv_ << header() << '\n';
    }

    static std::string header() {
        return "backend,precision,scaling,ranks,size,iterations,total_time,updates_per_second,communication_time";
    }

    void write(const BenchmarkRow& row) {
        if (!active_) {
            return;
        }

        csv_
            << backend_name(row.backend) << ','
            << row.precision << ','
            << scaling_name(row.scaling) << ','
            << row.ranks << ','
            << row.size << ','
            << row.iterations << ','
            << row.total_time << ','
            << row.updates_per_second << ','
            << row.communication_time << '\n';
    }

    void skip(Backend backend, const std::string& precision, Scaling scaling,
              int size, const std::string& reason) {
        if (!active_) {
            return;
        }

        std::cerr
            << "  [" << backend_name(backend) << "/" << scaling_name(scaling)
            << "/" << precision << "] size=" << size << " skipped: " << reason << '\n';
    }

    void note(const std::string& message) {
        if (!active_) {
            return;
        }

        std::cout << message << '\n';
    }

private:
    bool active_ = false;
    std::ofstream csv_;
};