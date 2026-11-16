// SPDX-License-Identifier: Apache-2.0
// Command-line options of the native radio runner, parsed and validated once.
#pragma once
#include "xbeam/cuda.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace xbeam {

struct Options {
    std::string job, output, index_dir, scan, profile;
    std::int64_t first_integration = 0, integration_count = 0, fft_tile = 400;
    int device = 0;
    ReadMode read_mode = ReadMode::Buffered;
    bool shared_staging = false;
    bool prefetch = true;
    std::int64_t chunk_bytes = 16 * 1024 * 1024;
    int upload_workers = 4;
    int xmac_pack_products = 4;
    bool xmac_reuse_sources = true;
    // Dynamic scheduling: workers claim guided-size chunks through `claim_dir`.
    std::string claim_dir;  // empty = static, single-output mode
    int workers_total = 1;
    std::int64_t min_chunk = 4;

    bool dynamic() const {
        return !claim_dir.empty();
    }
    // argv[0] is the program name, argv[1] the job file, then option/value pairs.
    static Options parse(const std::vector<std::string>& argv);
};

const char* io_name(ReadMode mode);

}  // namespace xbeam
