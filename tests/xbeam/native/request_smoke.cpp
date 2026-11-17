// SPDX-License-Identifier: Apache-2.0
#include "xbeam/radio.hpp"
#include <limits>
int main() {
    xbeam::RequestGeometry wide{4, 1, 262144, 50, 128, 2};
    auto result = xbeam::request_cursor(-262144, 8e-7, wide, 4096000000.);
    if (!result.valid || result.origin_bytes != 65408
        || result.skipped_ffts != 2 || result.decoder_delta_samples != 381) {
        return 1;
    }
    if (xbeam::request_cursor(-50 * 262144, 0, wide, 4096000000.).valid) {
        return 2;
    }
    xbeam::RequestGeometry packed{1, 8, 32, 50, 32, 2};
    result = xbeam::request_cursor(-1, 0, packed, 128.);
    if (!result.valid || result.origin_bytes != 224 || result.skipped_ffts != 1
        || result.decoder_delta_samples != 3) {
        return 3;
    }
    bool rejected = false;
    try {
        xbeam::request_cursor(
            std::numeric_limits<std::int64_t>::max(), 0, packed, 128.);
    } catch (const xbeam::RadioError&) {
        rejected = true;
    }
    return rejected ? 0 : 4;
}
