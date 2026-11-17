// SPDX-License-Identifier: Apache-2.0
#include "xbeam/radio.hpp"
#include <array>
#include <iostream>
int main() {
    std::array<unsigned char, 32> h{};
    h[8] = 4;
    h[11] = 0;
    h[15] = 4;  // 32-byte frame, two-bit samples.
    try {
        const auto parsed = xbeam::parse_vdif_header(h.data(), h.size());
        if (parsed.frame_length_bytes != 32 || parsed.samples_per_frame != 0) {
            return 1;
        }
        for (std::size_t n = 0; n < 32; ++n) {
            bool threw = false;
            try {
                xbeam::parse_vdif_header(h.data(), n);
            } catch (const xbeam::RadioError&) {
                threw = true;
            }
            if (!threw) {
                return 2;
            }
        }
        h[3] = 64;
        try {
            xbeam::parse_vdif_header(h.data(), 32);
            return 3;
        } catch (const xbeam::RadioError&) {
        }
    } catch (const std::exception& e) {
        std::cerr << e.what();
        return 4;
    }
    return 0;
}
