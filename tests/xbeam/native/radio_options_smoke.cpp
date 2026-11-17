// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
#include "radio_options.hpp"
#include "xbeam/radio.hpp"
#include <utility>

int main() {
    const std::vector<std::string> arguments{"native",
        "job.input",
        "--output",
        "output",
        "--index-dir",
        "sidecars",
        "--scan",
        "scan",
        "--profile",
        "interop-real-usb-buffered-window-replay-v1",
        "--integration-count",
        "2"};
    const auto options = xbeam::Options::parse(arguments);
    if (options.integration_count != 2 || options.fft_tile != 1250
        || options.read_mode != xbeam::ReadMode::Buffered) {
        return 1;
    }
    for (const auto& [key, value] :
        std::vector<std::pair<std::string, std::string>>{
            {"--integration-count", "2junk"},
            {"--integration-count", "9223372036854775808"},
            {"--first-integration", "1.5"},
            {"--fft-tile", "1250junk"},
            {"--device", "4294967296"},
            {"--upload-workers", "4294967300"},
            {"--xmac-pack-products", "4294967300"},
            {"--workers-total", "4294967297"}}) {
        auto invalid = arguments;
        if (key == "--integration-count") {
            invalid.back() = value;
        } else {
            invalid.insert(invalid.end(), {key, value});
        }
        if (key == "--workers-total") {
            invalid.insert(invalid.end(), {"--claim-dir", "claims"});
        }
        bool rejected = false;
        try {
            xbeam::Options::parse(invalid);
        } catch (const xbeam::RadioError&) {
            rejected = true;
        }
        if (!rejected) {
            return 2;
        }
    }
    return 0;
}
