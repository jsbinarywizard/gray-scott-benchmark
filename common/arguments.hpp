// ============================================================================
// Common Benchmark Infrastructure
// ============================================================================

#pragma once

#include <cctype>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// Benchmark parameter enums
// ============================================================================

enum class Precision {
    SINGLE,
    DOUBLE,
    BOTH
};

enum class Scaling {
    STRONG,
    WEAK,
    BOTH
};

// One entry per benchmark class (GS_MPI_Blocking, GS_KokkosComm, ...). Add
// new backends here and in backend_name()/all_backends()/the --backend
// parsing below; main.cpp's dispatch (make_benchmark()) is the only other
// place that needs to know about a new benchmark class.
enum class Backend {
    MPI_BLOCKING,
    MPI_NONBLOCKING,
    KC_BLOCKING,
    KC_NONBLOCKING
};

inline const char* backend_name(Backend backend) {
    switch (backend) {
        case Backend::MPI_BLOCKING: return "mpi_blocking";
        case Backend::MPI_NONBLOCKING: return "mpi_nonblocking";
        case Backend::KC_BLOCKING:   return "kc_blocking";
        case Backend::KC_NONBLOCKING: return "kc_nonblocking";
    }
    return "unknown";
}

// Every implemented backend, in the order "--backend all" runs them.
inline const std::vector<Backend>& all_backends() {
    static const std::vector<Backend> backends = {
        Backend::MPI_BLOCKING,
        Backend::MPI_NONBLOCKING,
        Backend::KC_BLOCKING,
        Backend::KC_NONBLOCKING
    };
    return backends;
}

// ============================================================================
// Benchmark configuration
// ============================================================================

struct BenchmarkConfig {

    // ------------------------------------------------------------------------
    // Iteration control
    // ------------------------------------------------------------------------
    int warmup_iters = 500;
    int benchmark_iters = 5000;

    // ------------------------------------------------------------------------
    // Resource limits
    // ------------------------------------------------------------------------
    double memory_limit_gb = 30.0;

    // ------------------------------------------------------------------------
    // Benchmark configuration
    // ------------------------------------------------------------------------
    Precision precision = Precision::BOTH;
    Scaling scaling = Scaling::STRONG;
    std::vector<Backend> backends = {Backend::MPI_BLOCKING};

    bool measure_cell_updates = true;
    bool measure_comm_bandwidth = true;

    std::vector<int> sizes = {
        32, 64, 128, 256, 512, 1024, 2048, 4096
    };

    // ------------------------------------------------------------------------
    // Output
    // ------------------------------------------------------------------------
    std::string output_file = "benchmark_results.csv";
};


// ============================================================================
// Benchmark argument parser
// ============================================================================

class BenchmarkParser {

private:

    std::string benchmark_name;

    // ------------------------------------------------------------------------
    // Check whether an argument matches one of the provided flags.
    // ------------------------------------------------------------------------
    static bool matches_flag(
        const char* arg,
        const std::vector<std::string>& flags)
    {
        for (const auto& flag : flags) {
            if (std::strcmp(arg, flag.c_str()) == 0) {
                return true;
            }
        }

        return false;
    }

    // ------------------------------------------------------------------------
    // Trim leading and trailing whitespace.
    // ------------------------------------------------------------------------
    static std::string trim_copy(const std::string& input)
    {
        std::size_t start = 0;

        while (start < input.size() &&
               std::isspace(static_cast<unsigned char>(input[start]))) {
            ++start;
        }

        std::size_t end = input.size();

        while (end > start &&
               std::isspace(static_cast<unsigned char>(input[end - 1]))) {
            --end;
        }

        return input.substr(start, end - start);
    }

    // ------------------------------------------------------------------------
    // Lowercase a string (ASCII).
    // ------------------------------------------------------------------------
    static std::string lowercase_copy(const std::string& input)
    {
        std::string result = input;

        for (char& c : result) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }

        return result;
    }

    // ------------------------------------------------------------------------
    // Parse an integer argument.
    // ------------------------------------------------------------------------
    static bool parse_int(
        const char* value,
        int& result)
    {
        if (value == nullptr || *value == '\0') {
            return false;
        }

        try {
            std::size_t consumed = 0;
            const std::string str(value);

            const int parsed = std::stoi(str, &consumed);

            if (consumed != str.size()) {
                return false;
            }

            result = parsed;
            return true;
        }
        catch (const std::exception&) {
            return false;
        }
    }

    // ------------------------------------------------------------------------
    // Parse a floating-point argument.
    // ------------------------------------------------------------------------
    static bool parse_double(
        const char* value,
        double& result)
    {
        if (value == nullptr || *value == '\0') {
            return false;
        }

        try {
            std::size_t consumed = 0;
            const std::string str(value);

            const double parsed = std::stod(str, &consumed);

            if (consumed != str.size()) {
                return false;
            }

            result = parsed;
            return true;
        }
        catch (const std::exception&) {
            return false;
        }
    }

    // ------------------------------------------------------------------------
    // Parse a comma-separated list of backend names, or "all".
    // ------------------------------------------------------------------------
    static bool parse_backends(
        const char* value,
        std::vector<Backend>& backends)
    {
        if (value == nullptr || *value == '\0') {
            return false;
        }

        const std::string input(value);

        if (lowercase_copy(trim_copy(input)) == "all") {
            backends = all_backends();
            return true;
        }

        std::vector<Backend> parsed_backends;

        std::size_t begin = 0;

        while (begin <= input.size()) {
            const std::size_t comma = input.find(',', begin);

            const std::string token =
                lowercase_copy(trim_copy(input.substr(
                    begin,
                    comma == std::string::npos
                        ? std::string::npos
                        : comma - begin)));

            if (token.empty()) {
                return false;
            }

            if (token == "mpi-blocking" || token == "mpi_blocking") {
                parsed_backends.push_back(Backend::MPI_BLOCKING);
            }
            else if (token == "kc-blocking" || token == "kc_blocking") {
                parsed_backends.push_back(Backend::KC_BLOCKING);
            }
            else {
                return false;
            }

            if (comma == std::string::npos) {
                break;
            }

            begin = comma + 1;
        }

        if (parsed_backends.empty()) {
            return false;
        }

        backends = std::move(parsed_backends);
        return true;
    }

    // ------------------------------------------------------------------------
    // Parse a comma-separated list of positive integers.
    // ------------------------------------------------------------------------
    static bool parse_sizes(
        const char* value,
        std::vector<int>& sizes)
    {
        if (value == nullptr || *value == '\0') {
            return false;
        }

        std::vector<int> parsed_sizes;
        std::string input(value);

        std::size_t begin = 0;

        while (begin <= input.size()) {
            const std::size_t comma = input.find(',', begin);

            const std::string token =
                trim_copy(input.substr(
                    begin,
                    comma == std::string::npos
                        ? std::string::npos
                        : comma - begin));

            if (token.empty()) {
                return false;
            }

            int size = 0;

            if (!parse_int(token.c_str(), size) || size <= 0) {
                return false;
            }

            parsed_sizes.push_back(size);

            if (comma == std::string::npos) {
                break;
            }

            begin = comma + 1;
        }

        if (parsed_sizes.empty()) {
            return false;
        }

        sizes = std::move(parsed_sizes);
        return true;
    }

    // ------------------------------------------------------------------------
    // Parse standard benchmark arguments.
    //
    // Returns false when parsing should terminate:
    //   - --help
    //   - invalid argument
    //
    // Returns true otherwise.
    // ------------------------------------------------------------------------
    bool parse_arg(
        const char* arg,
        int argc,
        char* argv[],
        int& i,
        BenchmarkConfig& config,
        int rank)
    {
        // --------------------------------------------------------------------
        // Help
        // --------------------------------------------------------------------
        if (std::strcmp(arg, "--help") == 0 ||
            std::strcmp(arg, "-h") == 0) {

            if (rank == 0) {
                print_usage(argv[0]);
            }

            return false;
        }

        // --------------------------------------------------------------------
        // Warmup iterations
        // --------------------------------------------------------------------
        if (std::strcmp(arg, "--warmup-iters") == 0 ||
            std::strcmp(arg, "-w") == 0) {

            if (i + 1 >= argc) {
                if (rank == 0) {
                    std::cerr
                        << "Error: " << arg
                        << " requires a value\n";
                }
                return false;
            }

            int value = 0;

            if (!parse_int(argv[++i], value) || value < 0) {
                if (rank == 0) {
                    std::cerr
                        << "Error: Warmup iterations must be "
                        << "a non-negative integer\n";
                }
                return false;
            }

            config.warmup_iters = value;
            return true;
        }

        // --------------------------------------------------------------------
        // Benchmark iterations
        // --------------------------------------------------------------------
        if (std::strcmp(arg, "--bench-iters") == 0 ||
            std::strcmp(arg, "-b") == 0) {

            if (i + 1 >= argc) {
                if (rank == 0) {
                    std::cerr
                        << "Error: " << arg
                        << " requires a value\n";
                }
                return false;
            }

            int value = 0;

            if (!parse_int(argv[++i], value) || value <= 0) {
                if (rank == 0) {
                    std::cerr
                        << "Error: Benchmark iterations must be "
                        << "a positive integer\n";
                }
                return false;
            }

            config.benchmark_iters = value;
            return true;
        }

        // --------------------------------------------------------------------
        // Memory limit
        // --------------------------------------------------------------------
        if (std::strcmp(arg, "--memory-limit") == 0 ||
            std::strcmp(arg, "-m") == 0) {

            if (i + 1 >= argc) {
                if (rank == 0) {
                    std::cerr
                        << "Error: " << arg
                        << " requires a value\n";
                }
                return false;
            }

            double value = 0.0;

            if (!parse_double(argv[++i], value) || value <= 0.0) {
                if (rank == 0) {
                    std::cerr
                        << "Error: Memory limit must be "
                        << "a positive number\n";
                }
                return false;
            }

            config.memory_limit_gb = value;
            return true;
        }

        // --------------------------------------------------------------------
        // Output file
        // --------------------------------------------------------------------
        if (std::strcmp(arg, "--output") == 0 ||
            std::strcmp(arg, "-o") == 0) {

            if (i + 1 >= argc) {
                if (rank == 0) {
                    std::cerr
                        << "Error: " << arg
                        << " requires a filename\n";
                }
                return false;
            }

            config.output_file = argv[++i];

            if (config.output_file.empty()) {
                if (rank == 0) {
                    std::cerr
                        << "Error: Output filename must not be empty\n";
                }
                return false;
            }

            return true;
        }

        // --------------------------------------------------------------------
        // Precision
        // --------------------------------------------------------------------
        if (std::strcmp(arg, "--precision") == 0) {

            if (i + 1 >= argc) {
                if (rank == 0) {
                    std::cerr
                        << "Error: --precision requires a value\n";
                }
                return false;
            }

            const std::string value = lowercase_copy(argv[++i]);

            if (value == "single") {
                config.precision = Precision::SINGLE;
            }
            else if (value == "double") {
                config.precision = Precision::DOUBLE;
            }
            else if (value == "both") {
                config.precision = Precision::BOTH;
            }
            else {
                if (rank == 0) {
                    std::cerr
                        << "Error: Precision must be "
                        << "'single', 'double', or 'both'\n";
                }
                return false;
            }

            return true;
        }

        // --------------------------------------------------------------------
        // Scaling
        // --------------------------------------------------------------------
        if (std::strcmp(arg, "--scaling") == 0) {

            if (i + 1 >= argc) {
                if (rank == 0) {
                    std::cerr
                        << "Error: --scaling requires a value\n";
                }
                return false;
            }

            const std::string value = lowercase_copy(argv[++i]);

            if (value == "strong") {
                config.scaling = Scaling::STRONG;
            }
            else if (value == "weak") {
                config.scaling = Scaling::WEAK;
            }
            else if (value == "both") {
                config.scaling = Scaling::BOTH;
            }
            else {
                if (rank == 0) {
                    std::cerr
                        << "Error: Scaling must be "
                        << "'strong', 'weak', or 'both'\n";
                }
                return false;
            }

            return true;
        }

        // --------------------------------------------------------------------
        // Backend
        // --------------------------------------------------------------------
        if (std::strcmp(arg, "--backend") == 0) {

            if (i + 1 >= argc) {
                if (rank == 0) {
                    std::cerr
                        << "Error: --backend requires a value\n";
                }
                return false;
            }

            std::vector<Backend> backends;

            if (!parse_backends(argv[++i], backends)) {
                if (rank == 0) {
                    std::cerr
                        << "Error: --backend must be 'all' or a "
                        << "comma-separated list of: "
                        << "'mpi-blocking', 'kokkoscomm'\n";
                }
                return false;
            }

            config.backends = std::move(backends);
            return true;
        }

        // --------------------------------------------------------------------
        // Grid sizes
        // --------------------------------------------------------------------
        if (std::strcmp(arg, "--sizes") == 0) {

            if (i + 1 >= argc) {
                if (rank == 0) {
                    std::cerr
                        << "Error: --sizes requires a "
                        << "comma-separated list\n";
                }
                return false;
            }

            std::vector<int> sizes;

            if (!parse_sizes(argv[++i], sizes)) {
                if (rank == 0) {
                    std::cerr
                        << "Error: --sizes must contain "
                        << "positive integers separated by commas\n";
                }
                return false;
            }

            config.sizes = std::move(sizes);
            return true;
        }

        // --------------------------------------------------------------------
        // Benchmark metrics
        // --------------------------------------------------------------------
        if (std::strcmp(arg, "--cell-updates") == 0) {
            config.measure_cell_updates = true;
            return true;
        }

        if (std::strcmp(arg, "--no-cell-updates") == 0) {
            config.measure_cell_updates = false;
            return true;
        }

        if (std::strcmp(arg, "--comm-bandwidth") == 0) {
            config.measure_comm_bandwidth = true;
            return true;
        }

        if (std::strcmp(arg, "--no-comm-bandwidth") == 0) {
            config.measure_comm_bandwidth = false;
            return true;
        }

        // --------------------------------------------------------------------
        // Not a standard argument.
        // --------------------------------------------------------------------
        return false;
    }

public:

    explicit BenchmarkParser(
        const std::string& name = "benchmark")
        : benchmark_name(name)
    {
    }


    // ------------------------------------------------------------------------
    // Print command-line usage.
    // ------------------------------------------------------------------------
    void print_usage(const char* program_name) const
    {
        std::cout
            << "Usage: " << program_name << " [options]\n\n";

        // --------------------------------------------------------------------
        // General
        // --------------------------------------------------------------------
        std::cout
            << "General options:\n"
            << "  -h, --help\n"
            << "        Show this help message and exit.\n\n";

        // --------------------------------------------------------------------
        // Iteration control
        // --------------------------------------------------------------------
        std::cout
            << "Iteration control:\n"
            << "  -w, --warmup-iters N\n"
            << "        Number of warmup iterations before timing.\n"
            << "        (default: 500)\n\n"

            << "  -b, --bench-iters N\n"
            << "        Number of timed benchmark iterations.\n"
            << "        (default: 10000)\n\n";

        // --------------------------------------------------------------------
        // Benchmark configuration
        // --------------------------------------------------------------------
        std::cout
            << "Benchmark configuration:\n"
            << "  --backend LIST\n"
            << "        Communication backend(s) / implementation(s) to run,\n"
            << "        comma-separated, or 'all':\n"
            << "            mpi-blocking  -> blocking MPI Isend/Irecv/Waitall\n"
            << "            kc-blocking   -> blocking KokkosComm (MPI or NCCL transport,\n"
            << "                             chosen at compile time)\n"
            << "        Example: --backend mpi-blocking,kc-blocking\n"
            << "        (default: mpi-blocking)\n\n"

            << "  --precision TYPE\n"
            << "        Floating-point precision:\n"
            << "            single  -> single precision\n"
            << "            double  -> double precision\n"
            << "            both    -> both precisions\n"
            << "        (default: both)\n\n"

            << "  --scaling TYPE\n"
            << "        Scaling behavior:\n"
            << "            strong  -> --sizes gives the fixed GLOBAL\n"
            << "                       problem size, split across ranks\n"
            << "            weak    -> --sizes gives the fixed LOCAL\n"
            << "                       (per-rank) problem size\n"
            << "            both    -> run both strong and weak scaling\n"
            << "        (default: strong)\n\n"

            << "  --sizes LIST\n"
            << "        Comma-separated grid sizes.\n"
            << "        Example: 32,64,128,256\n"
            << "        (default: 32,64,128,256,512,1024,2048,4096)\n\n";

        // --------------------------------------------------------------------
        // Metrics
        // --------------------------------------------------------------------
        std::cout
            << "Metrics:\n"
            << "  --cell-updates\n"
            << "        Enable cell-update measurements.\n"
            << "        (default: enabled)\n\n"

            << "  --no-cell-updates\n"
            << "        Disable cell-update measurements.\n\n"

            << "  --comm-bandwidth\n"
            << "        Enable communication-bandwidth measurements.\n"
            << "        (default: enabled)\n\n"

            << "  --no-comm-bandwidth\n"
            << "        Disable communication-bandwidth measurements.\n\n";

        // --------------------------------------------------------------------
        // Resource limits
        // --------------------------------------------------------------------
        std::cout
            << "Resource limits:\n"
            << "  -m, --memory-limit GB\n"
            << "        Maximum memory budget in GB.\n"
            << "        (default: 128 GB)\n\n";

        // --------------------------------------------------------------------
        // Output
        // --------------------------------------------------------------------
        std::cout
            << "Output:\n"
            << "  -o, --output FILE\n"
            << "        CSV file used to store benchmark results.\n"
            << "        (default: benchmark_results.csv)\n";
    }

    // ------------------------------------------------------------------------
    // Parse all arguments.
    //
    // Returns:
    //   true  -> parsing succeeded
    //   false -> help requested
    //
    // Throws std::runtime_error on an invalid argument.
    // ------------------------------------------------------------------------
    bool parse(
        int argc,
        char* argv[],
        BenchmarkConfig& config,
        int rank = 0)
    {
        for (int i = 1; i < argc; ++i) {
            const bool is_help =
                std::strcmp(argv[i], "--help") == 0 ||
                std::strcmp(argv[i], "-h") == 0;

            if (!parse_arg(
                    argv[i],
                    argc,
                    argv,
                    i,
                    config,
                    rank)) {

                if (is_help) {
                    return false;
                }

                throw std::runtime_error(
                    "Error: Invalid argument: " + std::string(argv[i]));
            }
        }

        return true;
    }
};